"""
Editor Action data models.

These Pydantic dataclasses represent the *intent* of a voice command after
it has been classified and parsed.  They are serialisable to JSON and sent
over the WebSocket bridge to the VS Code extension, which converts them into
concrete ``vscode.WorkspaceEdit`` operations.

Every action has:
  - A unique ``action_type`` discriminator field (string literal).
  - Optional metadata fields specific to the action kind.

The VS Code extension deserialises these using the same discriminator field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StructuralActionType(str, Enum):
    """Type of structural code mutation."""
    ADD_FUNCTION = "ADD_FUNCTION"
    ADD_CLASS = "ADD_CLASS"
    ADD_METHOD = "ADD_METHOD"
    DELETE_FUNCTION = "DELETE_FUNCTION"
    DELETE_CLASS = "DELETE_CLASS"
    DELETE_METHOD = "DELETE_METHOD"
    ADD_IMPORT = "ADD_IMPORT"
    RENAME = "RENAME"
    EXTRACT_FUNCTION = "EXTRACT_FUNCTION"
    WRAP_WITH_TRY = "WRAP_WITH_TRY"
    ADD_DECORATOR = "ADD_DECORATOR"
    ADD_DOCSTRING = "ADD_DOCSTRING"


class NavigationTarget(str, Enum):
    """Type of navigation target in the code."""
    FUNCTION = "FUNCTION"
    CLASS = "CLASS"
    LINE = "LINE"
    SYMBOL = "SYMBOL"
    DEFINITION = "DEFINITION"
    NEXT_ERROR = "NEXT_ERROR"
    PREVIOUS_ERROR = "PREVIOUS_ERROR"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class _BaseAction:
    """Base class for all editor actions."""

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON encoding."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Concrete action types
# ---------------------------------------------------------------------------


@dataclass
class DictationAction(_BaseAction):
    """
    Insert ``text`` verbatim at the current cursor position.

    This is the simplest action — used for raw code dictation.
    """
    action_type: str = field(default="DICTATION", init=False)
    text: str = ""


@dataclass
class StructuralAction(_BaseAction):
    """
    A structural mutation to the Python AST of the active file.

    Attributes:
        action_type:  One of :class:`StructuralActionType`.
        target_name:  Name of the symbol to create / delete / rename.
        parameters:   Parameter names (used for ADD_FUNCTION / ADD_METHOD).
        new_name:     New name (used for RENAME).
        body_hint:    Optional natural-language hint for the function body
                      (passed to LLM code generation).
        source_range: Optional ``[start_line, end_line]`` for EXTRACT_FUNCTION.
    """
    action_type: StructuralActionType = StructuralActionType.ADD_FUNCTION
    target_name: str = ""
    parameters: list[str] = field(default_factory=list)
    new_name: str = ""
    body_hint: str | None = None
    source_range: list[int] | None = None


@dataclass
class NavigateAction(_BaseAction):
    """
    Navigate the cursor to a specific location in the code.

    Attributes:
        target: What kind of node to navigate to.
        name:   Identifier name, or line number as a string.
    """
    action_type: str = field(default="NAVIGATE", init=False)
    target: NavigationTarget = NavigationTarget.SYMBOL
    name: str = ""


@dataclass
class GenerateAction(_BaseAction):
    """
    Delegate complex code generation to the LLM.

    The ``description`` is a cleaned natural-language specification that
    the VS Code extension (or a downstream LLM call) will use to generate
    the actual code.
    """
    action_type: str = field(default="GENERATE", init=False)
    description: str = ""


@dataclass
class NewlineAction(_BaseAction):
    """
    Insert one or more newlines at the cursor and optionally re-indent.

    Voice triggers: "new line", "next line", "enter", "line break",
                    "blank line", "newline".
    ``count`` controls how many newlines to insert (default 1).
    """
    action_type: str = field(default="NEWLINE", init=False)
    count: int = 1


# ---------------------------------------------------------------------------
# Union type for type annotations throughout the codebase
# ---------------------------------------------------------------------------

EditorAction = (
    DictationAction | StructuralAction | NavigateAction | GenerateAction | NewlineAction
)
