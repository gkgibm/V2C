"""
Post-ASR transcription refinement.

The raw Whisper transcript frequently suffers from:
  - Phonetic drift  (e.g. "async" → "a sink")
  - CamelCase splitting ("getUserInfo" → "get use run foe")
  - Symbol loss  ("!=" → "not equal" or dropped entirely)
  - Keyword ambiguity ("is not" interpreted as prose)

This module provides two refinement strategies, selected automatically:

1. **LLM Refiner** — sends the transcript + active file AST context to an
   OpenAI-compatible API for intelligent correction.  Requires a valid API key.

2. **Rule Refiner** — a deterministic fallback that applies:
   - A curated symbol-substitution table (spoken form → code token)
   - Phonetic matching via Double Metaphone + banded Levenshtein distance
     to snap misrecognised identifiers to known in-scope names.

Both refiners expose the same interface::

    async def refine(transcript: str, context: RefinementContext) -> str
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from v2c.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context passed to refiners
# ---------------------------------------------------------------------------

@dataclass
class RefinementContext:
    """
    Contextual information extracted from the active editor.

    Attributes:
        identifiers:   List of identifiers (variables, functions, classes, etc.)
                       currently visible in the open file's AST scope.
        active_file:   Name of the currently open file, used as a hint to the LLM.
        language:      Programming language of the active file.
    """

    identifiers: list[str] = field(default_factory=list)
    active_file: str = ""
    language: str = "python"

    @classmethod
    def empty(cls) -> "RefinementContext":
        return cls()


# ---------------------------------------------------------------------------
# Protocol / interface
# ---------------------------------------------------------------------------

class Refiner(Protocol):
    async def refine(self, transcript: str, context: RefinementContext) -> str:
        ...


# ---------------------------------------------------------------------------
# Symbol substitution table
# ---------------------------------------------------------------------------

_SYMBOL_SUBS: list[tuple[re.Pattern[str], str]] = [
    # Operators
    (re.compile(r"\bnot equals?\b|\bbang equals?\b|\bnot equal to\b", re.I), "!="),
    (re.compile(r"\bequals equals?\b|\bdouble equals?\b|\bis equal to\b", re.I), "=="),
    (re.compile(r"\bgreater than or equals?\b", re.I), ">="),
    (re.compile(r"\bless than or equals?\b", re.I), "<="),
    (re.compile(r"\bgreater than\b", re.I), ">"),
    (re.compile(r"\bless than\b", re.I), "<"),
    (re.compile(r"\bplus equals?\b", re.I), "+="),
    (re.compile(r"\bminus equals?\b", re.I), "-="),
    (re.compile(r"\btimes equals?\b|\bmultiply equals?\b", re.I), "*="),
    (re.compile(r"\bdivide equals?\b", re.I), "/="),
    (re.compile(r"\barrow\b|\bright arrow\b", re.I), "->"),
    (re.compile(r"\bwalrus\b|\bcolon equals?\b", re.I), ":="),
    (re.compile(r"\bdouble star\b|\bpower\b|\bexponent\b", re.I), "**"),
    (re.compile(r"\bdouble slash\b|\bfloor divide?\b", re.I), "//"),
    # Python keywords often misheard
    (re.compile(r"\ba sink\b|\basync?\b", re.I), "async"),
    (re.compile(r"\bask key\b|\bASCII\b", re.I), "ascii"),
    (re.compile(r"\blambda\b", re.I), "lambda"),
    (re.compile(r"\byield from\b", re.I), "yield from"),
    # Formatting shortcuts spoken aloud
    # These use a lookahead/behind to join directly to adjacent tokens
    # without inserting a space character.  When spoken in isolation
    # (e.g. "underscore init") the substitution still leaves a space
    # BEFORE the underscore which Python allows (e.g. `_ init` is
    # normally invalid, but we strip the space in post-processing).
    # We instead replace the word boundary directly so "underscore"
    # becomes "_" and then tighten: collapse "_ word" → "_word".
    (re.compile(r"\bdouble underscore\b|\bdunder\b", re.I), "__"),
    (re.compile(r"\bunderscore\b", re.I), "_"),
    (re.compile(r"\bdot\b", re.I), "."),
    (re.compile(r"\bcolon\b", re.I), ":"),
    (re.compile(r"\bopen paren\b|\bleft paren\b", re.I), "("),
    (re.compile(r"\bclose paren\b|\bright paren\b", re.I), ")"),
    (re.compile(r"\bopen bracket\b|\bleft bracket\b", re.I), "["),
    (re.compile(r"\bclose bracket\b|\bright bracket\b", re.I), "]"),
    (re.compile(r"\bopen brace\b|\bleft brace\b", re.I), "{"),
    (re.compile(r"\bclose brace\b|\bright brace\b", re.I), "}"),
]


# Post-processing: collapse "_ word" → "_word" and "__ word" → "__word"
_UNDERSCORE_COMPACT = re.compile(r"(_+)\s+(\w)")


def _apply_symbol_subs(text: str) -> str:
    """Replace spoken symbol forms with their code equivalents."""
    for pattern, replacement in _SYMBOL_SUBS:
        text = pattern.sub(replacement, text)
    # Collapse underscore + space + word into a single token
    text = _UNDERSCORE_COMPACT.sub(lambda m: m.group(1) + m.group(2), text)
    return text


# ---------------------------------------------------------------------------
# Phonetic matching (Double Metaphone + banded Levenshtein)
# ---------------------------------------------------------------------------

def _double_metaphone_key(word: str) -> tuple[str, str]:
    """Return the Double Metaphone primary and secondary keys for ``word``."""
    try:
        import jellyfish  # type: ignore[import]
        primary = jellyfish.metaphone(word)
        return primary, primary
    except ImportError:
        return word.upper(), word.upper()


def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    try:
        import Levenshtein  # type: ignore[import]
        return Levenshtein.distance(a, b)
    except ImportError:
        # Pure-Python fallback (acceptable for short strings)
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i, ca in enumerate(a, 1):
            prev = dp[:]
            dp[0] = i
            for j, cb in enumerate(b, 1):
                dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + (ca != cb))
        return dp[n]


def _snap_to_known_identifiers(word: str, identifiers: list[str]) -> str:
    """
    If *word* is phonetically similar to a known identifier and within
    an acceptable edit distance, return the identifier; else return *word*.
    """
    if not identifiers or not word:
        return word

    word_key, _ = _double_metaphone_key(word)
    best_match = word
    best_dist = 3  # maximum edit distance we're willing to accept

    for ident in identifiers:
        ident_key, _ = _double_metaphone_key(ident)
        if word_key == ident_key:
            dist = _levenshtein(word.lower(), ident.lower())
            if dist < best_dist:
                best_dist = dist
                best_match = ident

    return best_match


# ---------------------------------------------------------------------------
# Rule-based refiner
# ---------------------------------------------------------------------------

class RuleRefiner:
    """
    Deterministic, offline refiner.

    Applies symbol substitution and phonetic snapping without any network
    calls.  Suitable as a fallback when no LLM key is configured or when
    the LLM call fails.
    """

    async def refine(self, transcript: str, context: RefinementContext) -> str:
        text = transcript.strip()

        # 1. Symbol substitution
        text = _apply_symbol_subs(text)

        # 2. Phonetic identifier snapping — tokenise by whitespace, try to
        #    snap each word to a known identifier in the current scope.
        if context.identifiers:
            tokens = text.split()
            tokens = [_snap_to_known_identifiers(t, context.identifiers) for t in tokens]
            text = " ".join(tokens)

        logger.debug("RuleRefiner: %r → %r", transcript, text)
        return text


# ---------------------------------------------------------------------------
# LLM-based refiner
# ---------------------------------------------------------------------------

_REFINER_SYSTEM_PROMPT = """\
You are a precise transcription correction engine for a voice-to-code system.
The user has dictated Python code or a coding command into a microphone.
The raw ASR transcript may contain:
  - Phonetic drift (e.g. "a sink" instead of "async")
  - CamelCase splitting ("get use run foe" instead of "getUserInfo")
  - Symbol loss ("not equal" instead of "!=")
  - Filler words ("um", "uh", "like")

Your task is to return ONLY the corrected string — no explanations, no markdown.
Use the AST identifiers provided to snap misheard variable/function names.
Do NOT invent new identifiers not present in the context.
"""


def _build_refiner_user_prompt(transcript: str, context: RefinementContext) -> str:
    idents = ", ".join(context.identifiers[:40]) if context.identifiers else "(none)"
    return (
        f"File: {context.active_file or 'unknown'} ({context.language})\n"
        f"Known identifiers: {idents}\n"
        f"Raw transcript: {transcript!r}\n\n"
        "Corrected transcript:"
    )


class LLMRefiner:
    """
    Post-ASR refiner that calls an OpenAI-compatible API for intelligent
    correction.  Falls back to :class:`RuleRefiner` if the API call fails.
    """

    def __init__(self) -> None:
        self._rule_refiner = RuleRefiner()

    async def refine(self, transcript: str, context: RefinementContext) -> str:
        try:
            import openai  # type: ignore[import]

            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": _REFINER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_refiner_user_prompt(transcript, context),
                    },
                ],
                temperature=settings.llm_temperature,
                max_tokens=256,
                timeout=settings.llm_timeout,
            )
            refined = response.choices[0].message.content or transcript
            refined = refined.strip()
            logger.debug("LLMRefiner: %r → %r", transcript, refined)
            return refined

        except Exception as exc:
            logger.warning(
                "LLMRefiner API call failed (%s), falling back to RuleRefiner.",
                exc,
            )
            return await self._rule_refiner.refine(transcript, context)


# ---------------------------------------------------------------------------
# Factory — returns the best available refiner
# ---------------------------------------------------------------------------

def get_refiner() -> Refiner:
    """
    Return an :class:`LLMRefiner` if an OpenAI key is configured and
    refinement is enabled; otherwise return a :class:`RuleRefiner`.
    """
    if settings.refine_enabled and settings.use_llm:
        logger.info("Using LLMRefiner for post-ASR correction.")
        return LLMRefiner()
    logger.info("Using RuleRefiner (offline) for post-ASR correction.")
    return RuleRefiner()
