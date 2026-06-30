"""
Tests for v2c.intent.splitter — multi-command transcript splitter.
"""

from __future__ import annotations

import pytest

from v2c.intent.splitter import split


class TestSplit:
    def test_no_newline_marker(self) -> None:
        assert split("add function greet") == [("add function greet", 0)]

    def test_single_next_line(self) -> None:
        assert split("add function first next line add function second") == [
            ("add function first", 1),
            ("add function second", 0),
        ]

    def test_two_consecutive_next_line(self) -> None:
        assert split("add function first next line next line add function second") == [
            ("add function first", 2),
            ("add function second", 0),
        ]

    def test_three_segments(self) -> None:
        result = split(
            "add function first next line add function second next line add function third"
        )
        assert result == [
            ("add function first", 1),
            ("add function second", 1),
            ("add function third", 0),
        ]

    def test_new_line_variant(self) -> None:
        assert split("import os new line add class Foo") == [
            ("import os", 1),
            ("add class Foo", 0),
        ]

    def test_enter_variant(self) -> None:
        result = split("add function greet enter add function farewell")
        assert result == [
            ("add function greet", 1),
            ("add function farewell", 0),
        ]

    def test_line_break_variant(self) -> None:
        result = split("add function greet line break add function farewell")
        assert result == [
            ("add function greet", 1),
            ("add function farewell", 0),
        ]

    def test_twice_modifier(self) -> None:
        result = split("add function first next line twice add function second")
        assert result[0] == ("add function first", 2)
        assert result[1] == ("add function second", 0)

    def test_two_new_lines_count_word(self) -> None:
        result = split("add function first two new lines add function second")
        assert result[0][1] == 2

    def test_three_count_word(self) -> None:
        result = split("add class Foo three new lines add class Bar")
        assert result[0][1] == 3

    def test_empty_string(self) -> None:
        assert split("") == []

    def test_whitespace_only(self) -> None:
        assert split("   ") == []

    def test_trailing_next_line(self) -> None:
        # Trailing newline marker after last command — the segment still has 0 after
        result = split("add function greet next line")
        # The trailing marker with no following segment: greet gets newlines_after=1
        assert result[0][0] == "add function greet"
        assert result[0][1] == 1

    def test_real_whisper_output(self) -> None:
        """Simulate the actual Whisper output from the logs."""
        text = (
            "add function first next line next line next line next line "
            "add function second next line next line next line next line "
            "add function third"
        )
        result = split(text)
        assert len(result) == 3
        assert result[0] == ("add function first", 4)
        assert result[1] == ("add function second", 4)
        assert result[2] == ("add function third", 0)

    def test_comma_punctuation_around_marker(self) -> None:
        result = split("add function first, next line, add function second")
        assert result == [
            ("add function first", 1),
            ("add function second", 0),
        ]
