"""
Rule-based command mapper.

Converts high-confidence structural voice commands into
:class:`~v2c.ast_engine.editor_action.EditorAction` instances without
requiring any LLM call.

Each mapper is a simple function that accepts a transcript string and
returns an EditorAction if it matches, or None if it does not.

The dispatcher tries each mapper in registration order and returns the
first match.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from v2c.ast_engine.editor_action import (
    DictationAction,
    EditorAction,
    NavigateAction,
    NavigationTarget,
    StructuralAction,
    StructuralActionType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapper type alias
# ---------------------------------------------------------------------------

Mapper = Callable[[str], EditorAction | None]
_MAPPERS: list[Mapper] = []


def _register(fn: Mapper) -> Mapper:
    _MAPPERS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Navigation mappers
# ---------------------------------------------------------------------------


@_register
def _map_go_to_function(text: str) -> EditorAction | None:
    m = re.match(
        r"(?:go to|navigate to|jump to|find)\s+(?:function|method)\s+(\w+)",
        text,
        re.I,
    )
    if m:
        return NavigateAction(target=NavigationTarget.FUNCTION, name=m.group(1))
    return None


@_register
def _map_go_to_class(text: str) -> EditorAction | None:
    m = re.match(
        r"(?:go to|navigate to|jump to|find)\s+(?:class)\s+(\w+)",
        text,
        re.I,
    )
    if m:
        return NavigateAction(target=NavigationTarget.CLASS, name=m.group(1))
    return None


@_register
def _map_go_to_line(text: str) -> EditorAction | None:
    m = re.match(r"(?:go to|navigate to|jump to)\s+line\s+(\d+)", text, re.I)
    if m:
        return NavigateAction(target=NavigationTarget.LINE, name=m.group(1))
    return None


# ---------------------------------------------------------------------------
# Structural creation mappers
# ---------------------------------------------------------------------------


@_register
def _map_add_function(text: str) -> EditorAction | None:
    # "add function greet" | "define function greet" | "create function greet"
    m = re.match(
        r"(?:add|create|define|new)\s+(?:function|method|def)\s+(\w+)"
        r"(?:\s+(?:that\s+takes?|taking|with\s+params?|with\s+parameters?)\s+(.+))?",
        text,
        re.I,
    )
    if m:
        name = m.group(1)
        params_raw = m.group(2) or ""
        # Parse "a and b" or "a, b, c" into a list of names.
        params = [p.strip() for p in re.split(r"\s+and\s+|,\s*", params_raw) if p.strip()]
        return StructuralAction(
            action_type=StructuralActionType.ADD_FUNCTION,
            target_name=name,
            parameters=params,
        )
    return None


@_register
def _map_add_class(text: str) -> EditorAction | None:
    m = re.match(r"(?:add|create|define|new)\s+class\s+(\w+)", text, re.I)
    if m:
        return StructuralAction(
            action_type=StructuralActionType.ADD_CLASS,
            target_name=m.group(1),
        )
    return None


@_register
def _map_delete_function(text: str) -> EditorAction | None:
    m = re.match(
        r"(?:delete|remove|drop|chuck)\s+(?:function|method|def)\s+(\w+)",
        text,
        re.I,
    )
    if m:
        return StructuralAction(
            action_type=StructuralActionType.DELETE_FUNCTION,
            target_name=m.group(1),
        )
    return None


@_register
def _map_delete_class(text: str) -> EditorAction | None:
    m = re.match(
        r"(?:delete|remove|drop|chuck)\s+class\s+(\w+)",
        text,
        re.I,
    )
    if m:
        return StructuralAction(
            action_type=StructuralActionType.DELETE_CLASS,
            target_name=m.group(1),
        )
    return None


@_register
def _map_add_import(text: str) -> EditorAction | None:
    # "import numpy", "add import os.path"
    m = re.match(r"(?:add\s+)?import\s+(.+)", text, re.I)
    if m:
        return StructuralAction(
            action_type=StructuralActionType.ADD_IMPORT,
            target_name=m.group(1).strip(),
        )
    return None


# ---------------------------------------------------------------------------
# Dictation fallback (always matches last)
# ---------------------------------------------------------------------------


@_register
def _map_dictation_fallback(text: str) -> EditorAction | None:
    """Treat unrecognised commands as literal dictation (last-resort mapper)."""
    return DictationAction(text=text)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def dispatch(transcript: str) -> EditorAction:
    """
    Try each registered mapper in order and return the first non-None result.

    This function always returns an action (the fallback is dictation).
    """
    for mapper in _MAPPERS:
        result = mapper(transcript)
        if result is not None:
            logger.debug("Rules dispatcher matched %s for %r", type(result).__name__, transcript[:60])
            return result
    # Should never reach here due to the fallback mapper, but be explicit.
    return DictationAction(text=transcript)
