"""
Whisper ASR engine wrapper.

Uses ``faster-whisper`` (CTranslate2 backend) by default for speed, but
falls back gracefully to the original ``openai-whisper`` if
faster-whisper is not installed.

Public API:

    engine = ASREngine()
    transcript: str = await engine.transcribe(audio_array)
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from v2c.config import settings

if TYPE_CHECKING:
    pass  # avoid circular import at runtime

logger = logging.getLogger(__name__)


class ASREngine:
    """
    Thin async wrapper around Whisper transcription.

    ``transcribe`` is made async by dispatching the synchronous
    inference call to a thread-pool executor so the event loop is
    never blocked.
    """

    # ------------------------------------------------------------------
    # Lazy model loading (only loads the model once, on first call)
    # ------------------------------------------------------------------

    @cached_property
    def _model(self) -> object:
        """Load and cache the Whisper model (runs once per process)."""
        return self._load_model()

    def _load_model(self) -> object:
        """
        Try faster-whisper first (faster CTranslate2 backend), fall back
        to the original openai-whisper package.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            logger.info(
                "Loading faster-whisper model '%s' on device '%s' …",
                settings.asr_model,
                settings.asr_device,
            )
            t0 = time.perf_counter()
            model = WhisperModel(
                settings.asr_model,
                device=settings.asr_device,
                compute_type="int8" if settings.asr_device == "cpu" else "float16",
            )
            logger.info(
                "faster-whisper model loaded in %.2f s",
                time.perf_counter() - t0,
            )
            self._backend = "faster_whisper"
            return model

        except ImportError:
            pass  # fall back to original openai-whisper

        try:
            import whisper  # type: ignore[import]

            logger.info(
                "Loading openai-whisper model '%s' …", settings.asr_model
            )
            t0 = time.perf_counter()
            model = whisper.load_model(
                settings.asr_model, device=settings.asr_device
            )
            logger.info(
                "openai-whisper model loaded in %.2f s",
                time.perf_counter() - t0,
            )
            self._backend = "openai_whisper"
            return model

        except ImportError as exc:
            raise RuntimeError(
                "Neither 'faster-whisper' nor 'openai-whisper' is installed. "
                "Run: pip install faster-whisper"
            ) from exc

    # ------------------------------------------------------------------
    # Transcription helpers (synchronous, called in executor)
    # ------------------------------------------------------------------

    def _transcribe_sync_faster(self, audio: np.ndarray) -> str:
        """Run transcription using faster-whisper (sync)."""
        segments, _info = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=settings.asr_language,
            beam_size=settings.asr_beam_size,
            vad_filter=False,  # we already ran VAD upstream
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _transcribe_sync_openai(self, audio: np.ndarray) -> str:
        """Run transcription using openai-whisper (sync)."""
        result = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=settings.asr_language,
            beam_size=settings.asr_beam_size,
        )
        return result["text"].strip()

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Dispatch to the loaded backend (sync)."""
        # Touch self._model to ensure it's loaded.
        backend = getattr(self, "_backend", None)
        if backend is None:
            _ = self._model  # triggers @cached_property
            backend = self._backend

        if backend == "faster_whisper":
            return self._transcribe_sync_faster(audio)
        return self._transcribe_sync_openai(audio)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a mono float32 audio array asynchronously.

        Args:
            audio: numpy float32 array, shape ``(N,)``, 16 kHz, mono.

        Returns:
            Raw transcript string (may contain ASR errors).
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")

        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()

        transcript = await loop.run_in_executor(
            None, self._transcribe_sync, audio.astype(np.float32)
        )

        elapsed = time.perf_counter() - t0
        logger.debug(
            "ASR completed in %.3f s: %r",
            elapsed,
            transcript[:80] + ("…" if len(transcript) > 80 else ""),
        )
        return transcript

    async def warmup(self) -> None:
        """
        Pre-load the model before the first real utterance arrives.

        Call once at server startup to avoid the first-utterance delay.
        """
        logger.info("Warming up ASR model …")
        silence = np.zeros(settings.sample_rate, dtype=np.float32)
        await self.transcribe(silence)
        logger.info("ASR model warm-up complete.")
