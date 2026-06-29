"""
Tests for v2c.asr.refiner — both RuleRefiner and LLMRefiner.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from v2c.asr.refiner import (
    LLMRefiner,
    RefinementContext,
    RuleRefiner,
    _apply_symbol_subs,
    _double_metaphone_key,
    _snap_to_known_identifiers,
)


# ---------------------------------------------------------------------------
# Symbol substitution table tests
# ---------------------------------------------------------------------------

class TestSymbolSubstitutions:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("not equal", "!="),
            ("bang equals", "!="),
            ("equals equals", "=="),
            ("is equal to", "=="),
            ("greater than", ">"),
            ("less than", "<"),
            ("greater than or equals", ">="),
            ("plus equals", "+="),
            ("arrow", "->"),
            ("walrus", ":="),
            ("a sink request", "async request"),
            ("underscore init", "_init"),
            ("double underscore init", "__init"),
        ],
    )
    def test_symbol_sub(self, raw: str, expected: str) -> None:
        assert _apply_symbol_subs(raw) == expected


# ---------------------------------------------------------------------------
# Phonetic snapping
# ---------------------------------------------------------------------------

class TestPhoneticSnapping:
    def test_snaps_close_match(self) -> None:
        # "calculate text" is close to "calculate_tax" phonetically
        result = _snap_to_known_identifiers("calculatetax", ["calculate_tax", "format_currency"])
        assert result == "calculate_tax"

    def test_does_not_snap_far_match(self) -> None:
        result = _snap_to_known_identifiers("hello", ["UserManager", "calculate_tax"])
        assert result == "hello"

    def test_empty_identifiers_returns_word(self) -> None:
        assert _snap_to_known_identifiers("foo", []) == "foo"

    def test_empty_word_returns_empty(self) -> None:
        assert _snap_to_known_identifiers("", ["foo", "bar"]) == ""


# ---------------------------------------------------------------------------
# RuleRefiner
# ---------------------------------------------------------------------------

class TestRuleRefiner:
    @pytest.mark.asyncio
    async def test_basic_symbol_correction(self) -> None:
        refiner = RuleRefiner()
        ctx = RefinementContext.empty()
        result = await refiner.refine("x not equal y", ctx)
        assert "!=" in result

    @pytest.mark.asyncio
    async def test_identifier_snapping(self) -> None:
        """
        Token-level snapping: when a single spoken word is phonetically
        close to a known identifier, it should be snapped.
        "getuserinfo" (one word) ≈ "getUserInfo" via Double Metaphone.
        """
        refiner = RuleRefiner()
        ctx = RefinementContext(identifiers=["getUserInfo"])
        # "getuserinfo" is a single token — phonetically close to "getUserInfo"
        result = await refiner.refine("await getuserinfo paren", ctx)
        assert "getUserInfo" in result

    @pytest.mark.asyncio
    async def test_empty_transcript(self) -> None:
        refiner = RuleRefiner()
        result = await refiner.refine("", RefinementContext.empty())
        assert result == ""


# ---------------------------------------------------------------------------
# LLMRefiner
# ---------------------------------------------------------------------------

class TestLLMRefiner:
    @pytest.mark.asyncio
    async def test_falls_back_on_api_error(self) -> None:
        """If the OpenAI call raises, should fall back to rule-based."""
        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("network error")
            )
            mock_client_cls.return_value = mock_client

            refiner = LLMRefiner()
            ctx = RefinementContext.empty()
            result = await refiner.refine("a sink not equal", ctx)
            # Rule-based fallback should still apply symbol subs
            assert "async" in result or "!=" in result

    @pytest.mark.asyncio
    async def test_returns_llm_response_on_success(self) -> None:
        """When the API returns a valid response, use it."""
        with patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "async != "

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            refiner = LLMRefiner()
            ctx = RefinementContext.empty()
            result = await refiner.refine("a sink not equals", ctx)
            assert result == "async !="
