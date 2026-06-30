"""
Tests for v2c.intent.rules — rule-based command dispatcher.
"""

from __future__ import annotations

import pytest

from v2c.ast_engine.editor_action import (
    DictationAction,
    NavigateAction,
    NewlineAction,
    NavigationTarget,
    StructuralAction,
    StructuralActionType,
)
from v2c.intent.rules import dispatch


class TestNewlineDispatch:
    @pytest.mark.parametrize(
        "text, expected_count",
        [
            ("new line", 1),
            ("next line", 1),
            ("newline", 1),
            ("line break", 1),
            ("blank line", 1),
            ("enter", 1),
            ("new line twice", 2),
            ("two new lines", 2),
            ("three next lines", 3),
        ],
    )
    def test_newline_commands(self, text: str, expected_count: int) -> None:
        action = dispatch(text)
        assert isinstance(action, NewlineAction), (
            f"Expected NewlineAction for {text!r}, got {type(action).__name__}"
        )
        assert action.count == expected_count

    def test_newline_to_dict(self) -> None:
        action = NewlineAction(count=2)
        d = action.to_dict()
        assert d["action_type"] == "NEWLINE"
        assert d["count"] == 2


class TestStructuralDispatch:
    @pytest.mark.parametrize(
        "text, action_type, name",
        [
            ("add function calculate", StructuralActionType.ADD_FUNCTION, "calculate"),
            ("create function greet", StructuralActionType.ADD_FUNCTION, "greet"),
            ("define method save", StructuralActionType.ADD_FUNCTION, "save"),
            ("add class DataStore", StructuralActionType.ADD_CLASS, "DataStore"),
            ("create class UserModel", StructuralActionType.ADD_CLASS, "UserModel"),
            ("delete function old_fn", StructuralActionType.DELETE_FUNCTION, "old_fn"),
            ("remove class Legacy", StructuralActionType.DELETE_CLASS, "Legacy"),
        ],
    )
    def test_structural_commands(
        self, text: str, action_type: StructuralActionType, name: str
    ) -> None:
        action = dispatch(text)
        assert isinstance(action, StructuralAction)
        assert action.action_type == action_type
        assert action.target_name == name

    def test_function_with_params(self) -> None:
        action = dispatch("add function greet with params name and greeting")
        assert isinstance(action, StructuralAction)
        assert action.target_name == "greet"
        assert "name" in action.parameters
        assert "greeting" in action.parameters


class TestNavigationDispatch:
    @pytest.mark.parametrize(
        "text, target",
        [
            ("go to function main", NavigationTarget.FUNCTION),
            ("navigate to class UserManager", NavigationTarget.CLASS),
            ("jump to line 42", NavigationTarget.LINE),
        ],
    )
    def test_navigation(self, text: str, target: NavigationTarget) -> None:
        action = dispatch(text)
        assert isinstance(action, NavigateAction)
        assert action.target == target


class TestDictationFallback:
    def test_unknown_text_becomes_dictation(self) -> None:
        action = dispatch("something completely unrecognised xyz")
        assert isinstance(action, DictationAction)
        assert "unrecognised" in action.text
