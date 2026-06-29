"""
Intent Router — classifies a refined transcript as either:

  * DICTATION — raw text that should be inserted at the cursor position.
  * COMMAND   — a structural or semantic coding instruction.

The router uses a lightweight keyword/pattern matcher first (zero latency).
Only if the pattern fails to classify with high confidence does it defer
to a heavier downstream component (the LLM command parser).

Decision flow:

    transcript
        │
        ▼
    _pattern_classify()  ─── DICTATION ──▶ return IntentType.DICTATION
        │
        │ COMMAND / AMBIGUOUS
        ▼
    _keyword_score()  ─── high score ──▶ return IntentType.COMMAND
        │
        │ low score (ambiguous)
        ▼
    IntentType.AMBIGUOUS  (caller can escalate to LLM)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent type
# ---------------------------------------------------------------------------


class IntentType(Enum):
    DICTATION = auto()   # insert transcript verbatim at cursor
    COMMAND = auto()     # structural / semantic IDE command
    AMBIGUOUS = auto()   # not enough signal; escalate to LLM


# ---------------------------------------------------------------------------
# Explicit dictation trigger patterns
# ---------------------------------------------------------------------------

# When the transcript starts with one of these, it is unambiguously dictation.
_DICTATION_TRIGGERS: list[re.Pattern[str]] = [
    re.compile(r"^(type|insert|write|dictate|say)\b", re.I),
    re.compile(r"^(print|echo)\b", re.I),          # "print hello world" → dictate "print hello world"
]

# ---------------------------------------------------------------------------
# Structural command keywords / patterns
# ---------------------------------------------------------------------------

# High-confidence command prefixes — if the transcript starts with one of
# these, classify immediately as COMMAND.
_COMMAND_PREFIXES: list[re.Pattern[str]] = [
    re.compile(r"^(add|create|define|new)\s+(function|method|class|import|variable|constant|property|test)\b", re.I),
    re.compile(r"^(delete|remove|drop|chuck)\s+(function|method|class|variable|line|block|import)\b", re.I),
    re.compile(r"^(rename|refactor|extract|move|copy|wrap)\b", re.I),
    re.compile(r"^(go to|navigate to|jump to|find)\b", re.I),
    re.compile(r"^(run|execute|debug|step over|step into|step out|continue|pause|stop)\b", re.I),
    re.compile(r"^(comment|uncomment|format|lint|fix)\b", re.I),
    re.compile(r"^(undo|redo|save|open|close|split)\b", re.I),
    re.compile(r"^(show|hide|toggle)\s+(terminal|sidebar|panel|explorer)\b", re.I),
    re.compile(r"^(generate|write me|create me)\b", re.I),
]

# Scored keywords — each match adds to a score; if score ≥ threshold the
# transcript is classified as COMMAND.
_COMMAND_KEYWORDS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\b(function|method|class|module|import|variable|constant)\b", re.I), 2),
    (re.compile(r"\b(add|remove|delete|rename|extract|refactor|wrap|replace)\b", re.I), 2),
    (re.compile(r"\b(parameter|argument|return type|decorator|docstring)\b", re.I), 2),
    (re.compile(r"\b(line|block|scope|indent|dedent)\b", re.I), 1),
    (re.compile(r"\b(test|unit test|fixture|mock)\b", re.I), 1),
    (re.compile(r"\b(at (the |the )?top|at (the |the )?bottom|before|after|inside|outside)\b", re.I), 1),
]

_COMMAND_SCORE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    intent: IntentType
    confidence: float        # 0.0 – 1.0
    keyword_score: int       # raw keyword score (debug)


def classify(transcript: str) -> RoutingResult:
    """
    Classify *transcript* as DICTATION, COMMAND, or AMBIGUOUS.

    Args:
        transcript: Refined (post-ASR corrected) text to classify.

    Returns:
        :class:`RoutingResult` with the classification decision and confidence.
    """
    text = transcript.strip()

    if not text:
        return RoutingResult(IntentType.DICTATION, confidence=1.0, keyword_score=0)

    # 1. Explicit dictation triggers (highest priority).
    for pattern in _DICTATION_TRIGGERS:
        if pattern.match(text):
            logger.debug("IntentRouter: DICTATION (trigger match) — %r", text[:60])
            return RoutingResult(IntentType.DICTATION, confidence=0.95, keyword_score=0)

    # 2. High-confidence command prefixes.
    for pattern in _COMMAND_PREFIXES:
        if pattern.match(text):
            logger.debug("IntentRouter: COMMAND (prefix match) — %r", text[:60])
            return RoutingResult(IntentType.COMMAND, confidence=0.90, keyword_score=0)

    # 3. Keyword scoring.
    score = 0
    for pattern, weight in _COMMAND_KEYWORDS:
        if pattern.search(text):
            score += weight

    if score >= _COMMAND_SCORE_THRESHOLD:
        confidence = min(0.5 + score * 0.1, 0.85)
        logger.debug("IntentRouter: COMMAND (score=%d) — %r", score, text[:60])
        return RoutingResult(IntentType.COMMAND, confidence=confidence, keyword_score=score)

    # 4. Ambiguous — not enough signal.
    logger.debug("IntentRouter: AMBIGUOUS (score=%d) — %r", score, text[:60])
    return RoutingResult(IntentType.AMBIGUOUS, confidence=0.4, keyword_score=score)
