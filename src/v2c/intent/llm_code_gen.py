"""
LLM-backed code generator using a local Ollama model.

Uses ``qwen3:1.7b`` (default) with thinking disabled for ~500ms responses.
Falls back to rule-based pipeline if Ollama is unavailable.

The few-shot prompt teaches the model to:
  - Interpret natural language operators ("is equal to", "plus", "minus", etc.)
  - Treat the whole utterance as one code block, not multiple functions
  - Omit "remove pass" instead of creating a pass line
  - Use "next line" as a line separator inside a block
  - Output ONLY raw Python code, no markdown, no explanation
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from functools import lru_cache

import httpx

from v2c.config import settings

logger = logging.getLogger(__name__)

# ── Few-shot examples (teaches the model the exact mapping we need) ───────────

_FEW_SHOT = """\
voice: add function greet
code:
def greet():
    pass

voice: add function add, remove pass, x is equal to 10, y is equal to 20, return x plus y
code:
def add():
    x = 10
    y = 20
    return x + y

voice: add function calculate, remove pass, x is equal to 10, next line, y is equal to 10, next line, print x plus y
code:
def calculate():
    x = 10
    y = 10
    print(x + y)

voice: import numpy
code:
import numpy

voice: add class Animal, remove pass, name is equal to dog
code:
class Animal:
    name = "dog"

voice: x is equal to 5, y is equal to 3, print x times y
code:
x = 5
y = 3
print(x * y)

voice: for i in range 10, next line, print i
code:
for i in range(10):
    print(i)

"""

_PROMPT_TEMPLATE = (
    "Convert voice to Python. Output ONLY code.\n\n"
    + _FEW_SHOT
    + "voice: {command}\ncode:\n"
)

# ── Markdown fence stripper ───────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```", re.M)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


# ── Ollama client ─────────────────────────────────────────────────────────────


async def generate_code(
    voice_command: str,
    context_code: str = "",
    timeout: float | None = None,
) -> str:
    """
    Convert *voice_command* to Python code via a local Ollama model.

    Parameters
    ----------
    voice_command : str
        Raw (or lightly refined) ASR transcript.
    context_code : str
        Current file content (up to ~800 chars) sent as context so the
        model can match variable names, indentation style, etc.
    timeout : float | None
        HTTP timeout in seconds.  Defaults to ``settings.ollama_timeout``.

    Returns
    -------
    str
        Raw Python code string, ready to insert at the cursor.
        Returns empty string on failure.
    """
    timeout = timeout or settings.ollama_timeout
    prompt = _PROMPT_TEMPLATE.format(command=voice_command.strip())

    # Prepend a snippet of context so the model can infer indentation / names.
    if context_code.strip():
        snippet = context_code.strip()[-800:]
        prompt = f"# Current file context:\n{snippet}\n\n" + prompt

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,               # disable chain-of-thought for speed
        "options": {
            "temperature": 0,
            "num_predict": 400,
            "stop": ["\nvoice:", "\n\n\n"],
        },
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{settings.ollama_host}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data.get("response", "").strip()
        code = _strip_fences(raw)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Ollama %s: %.0fms → %d chars of code",
            settings.ollama_model,
            elapsed * 1000,
            len(code),
        )
        return code

    except httpx.ConnectError:
        logger.warning(
            "Ollama not reachable at %s — falling back to rules",
            settings.ollama_host,
        )
        return ""
    except Exception as exc:
        logger.warning("Ollama error (%s): %s", type(exc).__name__, exc)
        return ""


async def warmup() -> bool:
    """
    Send a tiny request to Ollama so the model is loaded and cached.
    Returns True if Ollama is available, False otherwise.
    """
    logger.info("Warming up Ollama model %s …", settings.ollama_model)
    code = await generate_code("add function test", timeout=30.0)
    if code:
        logger.info("Ollama warm-up complete.")
        return True
    logger.warning("Ollama not available — will use rule-based fallback.")
    return False
