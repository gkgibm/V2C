"""
LLM-backed code generator using a local Ollama model.

Default model: ``qwen2.5-coder:7b`` — a code-specialized 7B model,
~1s per call when warm (~4s cold start on first use).

Falls back to rule-based pipeline if Ollama is unreachable or returns an error.

Prompt strategy
---------------
Few-shot examples teach the model the exact natural-language → Python mapping:
  "is equal to"     → =
  "plus/minus/times/divided by" → +/-/*//
  "is greater/less than" → >/< 
  "remove pass"     → omit the pass statement
  "modulo"          → %
  "dot"             → .
  "next line"       → line separator (already handled by Whisper output)

The model is instructed to output ONLY raw Python code — no markdown fences,
no explanations, no trailing commentary.
"""

from __future__ import annotations

import logging
import re
import time

import httpx

from v2c.config import settings

logger = logging.getLogger(__name__)

# ── Few-shot prompt ───────────────────────────────────────────────────────────

_FEW_SHOT = """\
Convert voice command to Python code. Output ONLY the raw Python code, nothing else, no markdown fences.

voice: add function greet
code:
def greet():
    pass

voice: add function calc, remove pass, x is equal to 5, y is equal to 3, return x plus y
code:
def calc():
    x = 5
    y = 3
    return x + y

voice: import numpy
code:
import numpy

voice: x is equal to 10, y is equal to 20, print x plus y
code:
x = 10
y = 20
print(x + y)

voice: add class Animal, remove pass, name is equal to dog
code:
class Animal:
    name = "dog"

voice: for i in range 10, print i times 2
code:
for i in range(10):
    print(i * 2)

voice: if x is greater than 5, print x is big, else print x is small
code:
if x > 5:
    print("x is big")
else:
    print("x is small")

voice: for i in range 10, if i modulo 2 is equal to 0, print i
code:
for i in range(10):
    if i % 2 == 0:
        print(i)

voice: import json, import os, result is equal to json dot loads open config dot json dot read
code:
import json
import os
result = json.loads(open('config.json').read())

voice: add function multiply, remove pass, a is equal to 3, b is equal to 4, return a times b
code:
def multiply():
    a = 3
    b = 4
    return a * b

voice: {command}
code:"""

# ── Markdown / commentary stripper ───────────────────────────────────────────

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```", re.M)
# Strip any "voice:" continuation the model might hallucinate
_VOICE_RE = re.compile(r"\nvoice:.*", re.S)


def _clean(text: str) -> str:
    text = _FENCE_RE.sub("", text)
    text = _VOICE_RE.sub("", text)
    return text.strip()


# ── Ollama async client ───────────────────────────────────────────────────────


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
        Last ~800 chars of the active file — gives the model indentation
        style, variable names, and class context.
    timeout : float | None
        HTTP timeout seconds. Defaults to ``settings.ollama_timeout``.

    Returns
    -------
    str
        Raw Python code ready to insert at the cursor.
        Returns empty string on any failure (caller uses rule fallback).
    """
    timeout = timeout or settings.ollama_timeout
    prompt = _FEW_SHOT.format(command=voice_command.strip())

    if context_code.strip():
        snippet = context_code.strip()[-800:]
        prompt = f"# Current file context:\n{snippet}\n\n" + prompt

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        # keep_alive=-1 keeps the model loaded in RAM indefinitely between
        # calls so subsequent requests skip the ~3s load penalty.
        "keep_alive": -1,
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
        code = _clean(raw)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Ollama %s: %.0fms → %d chars",
            settings.ollama_model,
            elapsed * 1000,
            len(code),
        )
        return code

    except httpx.ConnectError:
        logger.warning(
            "Ollama not reachable at %s — using rule fallback",
            settings.ollama_host,
        )
        return ""
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Ollama HTTP %s: %s — using rule fallback",
            exc.response.status_code,
            exc.response.text[:120],
        )
        return ""
    except Exception as exc:
        logger.warning("Ollama error (%s): %s — using rule fallback", type(exc).__name__, exc)
        return ""


async def warmup() -> bool:
    """
    Pre-load the model into Ollama's RAM so the first real call is fast.
    Sets keep_alive=-1 so it stays loaded until Ollama is stopped.
    Returns True if available, False otherwise.
    """
    logger.info("Warming up Ollama model %s …", settings.ollama_model)
    code = await generate_code("add function warmup", timeout=30.0)
    if code:
        logger.info("Ollama %s ready (~1s per call).", settings.ollama_model)
        return True
    logger.warning(
        "Ollama unavailable — rule-based fallback active. "
        "Start Ollama and run: ollama pull %s",
        settings.ollama_model,
    )
    return False
