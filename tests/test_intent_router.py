"""
Tests for v2c.intent.router — the transcript intent classifier.
"""

from __future__ import annotations

import pytest

from v2c.intent.router import IntentType, RoutingResult, classify


class TestIntentRouter:
    # ── Dictation ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "type hello world",
            "insert return None",
            "write x = 1",
            "dictate import os",
            "say print hello",
        ],
    )
    def test_dictation_trigger_words(self, text: str) -> None:
        result = classify(text)
        assert result.intent == IntentType.DICTATION
        assert result.confidence >= 0.9

    def test_empty_string_is_dictation(self) -> None:
        result = classify("")
        assert result.intent == IntentType.DICTATION

    # ── Command ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "add function greet",
            "create function calculate_tax",
            "define method save",
            "delete function old_helper",
            "remove class LegacyUser",
            "rename function foo to bar",
            "go to function main",
            "navigate to class UserManager",
            "run tests",
            "comment function",
            "import numpy",
        ],
    )
    def test_command_prefix_matches(self, text: str) -> None:
        result = classify(text)
        assert result.intent == IntentType.COMMAND

    @pytest.mark.parametrize(
        "text",
        [
            "add a class for managing users",
            "extract the loop into a separate function",
        ],
    )
    def test_command_keyword_scoring(self, text: str) -> None:
        result = classify(text)
        assert result.intent in (IntentType.COMMAND, IntentType.AMBIGUOUS)

    # ── Ambiguous ──────────────────────────────────────────────────────────

    def test_pure_prose_is_ambiguous(self) -> None:
        # Low-keyword density → ambiguous
        result = classify("the user wants a response")
        assert result.intent == IntentType.AMBIGUOUS

    # ── Return type ────────────────────────────────────────────────────────

    def test_returns_routing_result(self) -> None:
        result = classify("add function foo")
        assert isinstance(result, RoutingResult)
        assert isinstance(result.intent, IntentType)
        assert 0.0 <= result.confidence <= 1.0

    # ── Newline commands ───────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "new line",
            "next line",
            "newline",
            "line break",
            "blank line",
            "enter",
            "new line twice",
            "two new lines",
            "three next lines",
        ],
    )
    def test_newline_commands_are_command(self, text: str) -> None:
        result = classify(text)
        assert result.intent == IntentType.COMMAND, (
            f"Expected COMMAND for {text!r}, got {result.intent}"
        )
