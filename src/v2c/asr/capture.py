"""
Audio capture with real-time Voice Activity Detection (VAD).

Responsibilities:
  1. Open a sounddevice input stream at the configured sample rate.
  2. Run each 30-ms frame through WebRTC VAD to detect speech.
  3. Buffer speech frames and yield complete utterances (numpy float32
     arrays, mono, 16 kHz) once a silence gap is detected.

The public interface is the async generator ``listen()``, which yields
one utterance at a time.  Everything else is implementation detail.

Design notes:
  - WebRTC VAD works only on 8 / 16 / 32 / 48 kHz PCM16 audio.
  - sounddevice records float32; we down-convert to int16 for VAD then
    keep float32 buffers for Whisper (which expects float32 in [-1, 1]).
  - All blocking I/O happens in a background thread (via asyncio
    run_in_executor) so the async event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncGenerator

import numpy as np
import sounddevice as sd
import webrtcvad

from v2c.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 numpy array ([-1, 1]) to little-endian PCM-16."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


class _SpeechBuffer:
    """Accumulates VAD-confirmed speech frames and detects end-of-utterance."""

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._silence_frames: int = 0
        self._speaking: bool = False

        # How many consecutive silent frames constitute end-of-utterance.
        silence_frames_needed = settings.vad_silence_threshold / (
            settings.vad_frame_ms / 1000
        )
        self._silence_limit: int = max(1, int(silence_frames_needed))

        # Minimum speech frames before we consider it a real utterance.
        min_speech_frames = settings.vad_min_speech_ms / settings.vad_frame_ms
        self._min_speech_frames: int = max(1, int(min_speech_frames))

        logger.debug(
            "SpeechBuffer: silence_limit=%d frames, min_speech=%d frames",
            self._silence_limit,
            self._min_speech_frames,
        )

    def feed(self, frame: np.ndarray, is_speech: bool) -> np.ndarray | None:
        """
        Feed one VAD frame.  Returns a complete utterance array when the
        speech segment ends, otherwise returns None.

        Args:
            frame:     Raw float32 audio frame (mono, ``frames_per_chunk`` samples).
            is_speech: Whether WebRTC VAD detected speech in this frame.

        Returns:
            Concatenated float32 audio of the full utterance, or ``None`` if
            the utterance is still ongoing.
        """
        if is_speech:
            if not self._speaking:
                logger.debug("VAD: speech start detected")
            self._speaking = True
            self._silence_frames = 0
            self._frames.append(frame)
        elif self._speaking:
            self._frames.append(frame)  # include trailing silence
            self._silence_frames += 1
            if self._silence_frames >= self._silence_limit:
                return self._flush()
        return None

    def _flush(self) -> np.ndarray | None:
        """Flush accumulated frames as a single utterance, reset state."""
        utterance = np.concatenate(self._frames) if self._frames else None
        speech_frame_count = len(self._frames) - self._silence_frames
        self._frames.clear()
        self._silence_frames = 0
        self._speaking = False

        if utterance is None or speech_frame_count < self._min_speech_frames:
            logger.debug("VAD: ignoring short segment (%d speech frames)", speech_frame_count)
            return None

        logger.debug(
            "VAD: utterance complete — %.2f s",
            len(utterance) / settings.sample_rate,
        )
        return utterance


# ---------------------------------------------------------------------------
# Public async generator
# ---------------------------------------------------------------------------

async def listen(
    device: int | str | None = None,
) -> AsyncGenerator[np.ndarray, None]:
    """
    Async generator that yields one float32 utterance (mono, 16 kHz) per
    detected speech segment.

    Args:
        device: sounddevice device index or name.  Defaults to system default.

    Yields:
        numpy float32 array, shape ``(N,)``, representing one utterance.

    Example::

        async for utterance in listen():
            transcript = await asr_engine.transcribe(utterance)
    """
    vad = webrtcvad.Vad(settings.vad_aggressiveness)
    buffer = _SpeechBuffer()
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _callback(
        indata: np.ndarray,
        _frames: int,
        _time: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — called in a background thread."""
        if status:
            logger.warning("sounddevice status: %s", status)

        # Work on a copy; sounddevice reuses the buffer.
        frame = indata[:, 0].copy()  # take channel 0 (mono)

        # VAD operates on PCM-16 bytes.
        pcm16 = _float32_to_pcm16(frame)
        try:
            is_speech = vad.is_speech(pcm16, settings.sample_rate)
        except Exception:
            is_speech = False

        result = buffer.feed(frame, is_speech)
        if result is not None:
            loop.call_soon_threadsafe(queue.put_nowait, result)

    logger.info(
        "Starting audio capture — sample_rate=%d Hz, frame=%d ms, device=%s",
        settings.sample_rate,
        settings.vad_frame_ms,
        device or "default",
    )

    with sd.InputStream(
        samplerate=settings.sample_rate,
        blocksize=settings.frames_per_chunk,
        device=device,
        channels=1,
        dtype="float32",
        callback=_callback,
    ):
        while True:
            utterance = await queue.get()
            if utterance is None:
                break  # sentinel for graceful shutdown
            yield utterance
