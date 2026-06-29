"""
LLM-based semantic command parser.

For transcripts that are classified as COMMAND but cannot be reliably
mapped by the deterministic rule engine (e.g. complex refactors, natural
language generation requests), this module calls an OpenAI-compatible API
with a carefully constructed system prompt that includes:

  1. The refined transcript.
  2. A JSON schema describing the available editor action types.
  3. The list of in-scope AST identifiers from the active file.

The LLM is instructed to return a single JSON object conforming to one of
the :mod:`v2c.ast_engine.editor_action` schemas.

Falls back to a :class:`~v2c.ast_engine.editor_action.DictationAction`
if the API call fails or returns malformed JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from v2c.ast_engine.editor_action import (
    DictationAction,
    EditorAction,
    StructuralAction,
    StructuralActionType,
    NavigateAction,
    NavigationTarget,
    GenerateAction,
)
from v2c.asr.refiner import RefinementContext
from v2c.config import settings

logger = logging.getLogger(__name__)


_PARSER_SYSTEM_PROMPT = """\
You are an intelligent voice-to-code command interpreter for a Python IDE.

Your input is a developer's spoken command (already cleaned by ASR post-processing).
Your output MUST be a single JSON object with an "action_type" field.

Supported action_types and their fields:

  DICTATION       – { "action_type": "DICTATION", "text": "<text to insert>" }
  ADD_FUNCTION    – { "action_type": "ADD_FUNCTION", "target_name": "<name>", "parameters": ["p1","p2"] }
  ADD_CLASS       – { "action_type": "ADD_CLASS", "target_name": "<name>" }
  DELETE_FUNCTION – { "action_type": "DELETE_FUNCTION", "target_name": "<name>" }
  DELETE_CLASS    – { "action_type": "DELETE_CLASS", "target_name": "<name>" }
  ADD_IMPORT      – { "action_type": "ADD_IMPORT", "target_name": "<module>" }
  NAVIGATE        – { "action_type": "NAVIGATE", "nav_target": "FUNCTION|CLASS|LINE|SYMBOL", "name": "<name>" }
  GENERATE        – { "action_type": "GENERATE", "description": "<full description>" }

Rules:
- Use GENERATE when the instruction is too complex for a single structural action.
- Snap target_name values to the known identifiers when they are close matches.
- Return ONLY the JSON object — no markdown, no explanation.
"""


def _build_user_message(transcript: str, context: RefinementContext) -> str:
    idents = ", ".join(context.identifiers[:60]) if context.identifiers else "(none)"
    return (
        f"File: {context.active_file or 'unknown'} ({context.language})\n"
        f"Known identifiers: {idents}\n\n"
        f"Command: {transcript}"
    )


def _parse_llm_response(raw: str) -> EditorAction:
    """
    Parse the raw LLM JSON string into an EditorAction.
    Raises ValueError if the JSON is malformed or the action_type is unknown.
    """
    data: dict = json.loads(raw.strip())
    action_type_str = data.get("action_type", "").upper()

    if action_type_str == "DICTATION":
        return DictationAction(text=data.get("text", ""))

    if action_type_str == "NAVIGATE":
        target_str = data.get("nav_target", "SYMBOL").upper()
        target = NavigationTarget[target_str] if target_str in NavigationTarget.__members__ else NavigationTarget.SYMBOL
        return NavigateAction(target=target, name=data.get("name", ""))

    if action_type_str == "GENERATE":
        return GenerateAction(description=data.get("description", ""))

    # Structural actions
    try:
        action_enum = StructuralActionType[action_type_str]
    except KeyError as exc:
        raise ValueError(f"Unknown action_type: {action_type_str!r}") from exc

    return StructuralAction(
        action_type=action_enum,
        target_name=data.get("target_name", ""),
        parameters=data.get("parameters", []),
        body_hint=data.get("body_hint"),
    )


async def parse(transcript: str, context: RefinementContext) -> EditorAction:
    """
    Parse *transcript* into an :class:`~v2c.ast_engine.editor_action.EditorAction`
    using an LLM.

    Falls back to a :class:`~v2c.ast_engine.editor_action.DictationAction` on
    any error.
    """
    if not settings.use_llm:
        logger.debug("LLMParser: no API key — falling back to DictationAction")
        return DictationAction(text=transcript)

    try:
        import openai  # type: ignore[import]

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(transcript, context)},
            ],
            temperature=settings.llm_temperature,
            max_tokens=512,
            timeout=settings.llm_timeout,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        action = _parse_llm_response(raw)
        logger.debug("LLMParser: %r → %s", transcript[:60], type(action).__name__)
        return action

    except Exception as exc:
        logger.warning("LLMParser failed (%s), falling back to DictationAction.", exc)
        return DictationAction(text=transcript)
