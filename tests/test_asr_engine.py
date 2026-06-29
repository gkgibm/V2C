"""
Tests for v2c.asr.engine.ASREngine.

These tests mock the underlying Whisper model to avoid requiring GPU / model
downloads in CI.  They verify the async wrapper, warmup, and error handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from v2c.asr.engine import ASREngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(transcript: str = "hello world") -> MagicMock:
    """Return a fake faster-whisper model that yields one segment."""
    segment = MagicMock()
    segment.text = transcript
    model = MagicMock()
    model.transcribe.return_value = ([segment], MagicMock())
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestASREngine:
    @patch("v2c.asr.engine.ASREngine._load_model")
    def test_transcribe_returns_string(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_mock_model("def hello()")
        engine = ASREngine()
        engine._backend = "faster_whisper"

        async def run() -> str:
            audio = np.zeros(16000, dtype=np.float32)
            return await engine.transcribe(audio)

        import asyncio
        result = asyncio.run(run())
        assert isinstance(result, str)
        assert "hello" in result

    @patch("v2c.asr.engine.ASREngine._load_model")
    def test_transcribe_strips_whitespace(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _make_mock_model("  calculate_tax  ")
        engine = ASREngine()
        engine._backend = "faster_whisper"

        import asyncio
        result = asyncio.run(engine.transcribe(np.zeros(8000, dtype=np.float32)))
        assert result == "calculate_tax"

    def test_transcribe_rejects_2d_audio(self) -> None:
        engine = ASREngine()
        import asyncio

        with pytest.raises(ValueError, match="1-D"):
            asyncio.run(engine.transcribe(np.zeros((100, 2), dtype=np.float32)))

    @patch("v2c.asr.engine.ASREngine._load_model")
    @patch("v2c.asr.engine.ASREngine.transcribe", new_callable=AsyncMock)
    def test_warmup_calls_transcribe(self, mock_transcribe: AsyncMock, mock_load: MagicMock) -> None:
        mock_transcribe.return_value = ""
        engine = ASREngine()

        import asyncio
        asyncio.run(engine.warmup())
        mock_transcribe.assert_called_once()
        # warmup passes a zero array of length sample_rate
        args = mock_transcribe.call_args[0]
        assert len(args[0]) == 16000
