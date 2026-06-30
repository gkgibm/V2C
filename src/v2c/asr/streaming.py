"""
Streaming ASR engine — live transcription while the user is speaking.

Architecture
------------
                  sounddevice callback
                        │ (float32 PCM frames, ~30ms each)
                        ▼
                  _ring_buffer  (thread-safe deque)
                        │
              ┌─────────▼──────────┐
              │  _inference_loop   │  ← background thread
              │                    │
              │  every STRIDE_S:   │
              │    take last       │
              │    WINDOW_S audio  │
              │    → Whisper       │
              │    → emit partial  │
              └─────────┬──────────┘
                        │ partial text via asyncio.Queue
                        ▼
              StreamingASREngine.partials()  ← async generator
                        │
                        ▼
              SessionHandler → PARTIAL_TRANSCRIPT WebSocket message
                        │
                        ▼
              VS Code extension ghost text

On ``stop()``:
  - inference loop exits
  - full accumulated audio → single Whisper pass → final transcript
  - any partial state is discarded

Design decisions
----------------
- Whisper processes the last WINDOW_S seconds on every STRIDE_S tick.
  This gives the "growing context" effect: early words are re-confirmed
  as more audio arrives, reducing hallucinations at segment boundaries.
- WINDOW_S = 5s, STRIDE_S = 0.3s gives ~10 inferences/s on CPU tiny.
- ``condition_on_previous_text=False`` prevents the model from inventing
  continuations — critical for partial display accuracy.
- Word-level timestamps are disabled for speed (not needed for partials).
- The inference loop runs in a ThreadPoolExecutor so it never blocks the
  asyncio event loop.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue
import threading
import time
from typing import AsyncGenerator

import numpy as np

from v2c.config import settings

logger = logging.getLogger(__name__)

# ── Sliding-window parameters ─────────────────────────────────────────────────

# How much audio Whisper sees on each inference tick (seconds).
# Longer = more context = better accuracy, but more CPU per tick.
WINDOW_S: float = 5.0

# How often we run inference while recording (seconds).
# 0.3s gives responsive feel without saturating the CPU.
STRIDE_S: float = 0.3

# Minimum audio length before we bother running inference (seconds).
MIN_AUDIO_S: float = 0.4


class StreamingASREngine:
    """
    Captures microphone audio and emits partial transcripts in near-real-time
    while the user is speaking.

    Usage::

        engine = StreamingASREngine(whisper_model)
        engine.start()                           # opens mic, starts inference loop

        async for partial in engine.partials():  # yields text as it accumulates
            print(partial)                       # e.g. "add func" → "add function calc"

        final_audio = engine.stop()              # close mic, return full float32 array
        final_text  = await asr.transcribe(final_audio)   # one final clean pass
    """

    def __init__(self, model: object) -> None:
        self._model = model

        # Ring buffer: holds the last WINDOW_S * sample_rate samples.
        # Using a deque with maxlen automatically drops old frames.
        _maxlen = int((WINDOW_S + 1.0) * settings.sample_rate)
        self._ring: collections.deque[np.ndarray] = collections.deque(maxlen=_maxlen)

        # Full audio buffer — keeps everything for the final clean pass.
        self._full: list[np.ndarray] = []

        # Queue for partial text flowing from inference thread → async generator.
        self._partial_q: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Control flags.
        self._running = False
        self._inference_thread: threading.Thread | None = None
        self._stream: object = None  # sounddevice.InputStream

    # ── Mic stream ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone and start the inference loop."""
        import sounddevice as sd  # type: ignore[import]

        self._loop = asyncio.get_event_loop()
        self._running = True
        self._ring.clear()
        self._full.clear()

        def _callback(
            indata: np.ndarray, _frames: int, _time: object, status: object
        ) -> None:
            if status:
                logger.warning("sounddevice: %s", status)
            frame = indata[:, 0].copy()
            self._ring.append(frame)
            self._full.append(frame)

        self._stream = sd.InputStream(
            samplerate=settings.sample_rate,
            blocksize=settings.frames_per_chunk,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()  # type: ignore[union-attr]

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

        logger.info(
            "StreamingASR started — window=%.1fs stride=%.1fs", WINDOW_S, STRIDE_S
        )

    def stop(self) -> np.ndarray:
        """
        Stop recording.  Returns the full captured audio as a float32 array.
        """
        self._running = False

        # Stop mic first so no new frames arrive.
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None

        # Wait for inference thread to exit.
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None

        # Signal the async generator to stop.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._partial_q.put_nowait, None)

        if not self._full:
            logger.warning("StreamingASR stopped with no captured audio.")
            return np.zeros(settings.sample_rate, dtype=np.float32)

        audio = np.concatenate(self._full)
        duration = len(audio) / settings.sample_rate
        logger.info("StreamingASR stopped — captured %.2f s of audio", duration)
        return audio

    # ── Async partial generator ───────────────────────────────────────────────

    async def partials(self) -> AsyncGenerator[str, None]:
        """
        Async generator that yields partial transcript strings while recording.

        Yields each new non-empty partial text.  Stops when ``stop()`` is called
        (signalled by a ``None`` sentinel in the queue).
        """
        while True:
            text = await self._partial_q.get()
            if text is None:
                break  # sentinel — recording stopped
            if text.strip():
                yield text

    # ── Inference loop (background thread) ───────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Runs in a background thread.  Every STRIDE_S seconds, takes the last
        WINDOW_S seconds of audio from the ring buffer, runs Whisper, and pushes
        the result to the partial queue.
        """
        prev_text = ""

        while self._running:
            time.sleep(STRIDE_S)

            if not self._running:
                break

            # Snapshot the ring buffer (thread-safe: deque reads are atomic).
            frames = list(self._ring)
            if not frames:
                continue

            audio = np.concatenate(frames)
            duration = len(audio) / settings.sample_rate

            if duration < MIN_AUDIO_S:
                continue

            try:
                text = self._transcribe_chunk(audio)
            except Exception as exc:
                logger.debug("Partial inference error: %s", exc)
                continue

            # Only push if the text actually changed — avoids flooding the
            # extension with identical partial messages.
            if text and text != prev_text:
                prev_text = text
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._partial_q.put_nowait, text)

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Run a single Whisper inference pass on a float32 audio chunk."""
        try:
            # faster-whisper path
            segments, _ = self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=settings.asr_language,
                beam_size=1,                        # greedy — fastest for partials
                temperature=0.0,
                condition_on_previous_text=False,   # no hallucinated continuations
                without_timestamps=True,
                vad_filter=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        except AttributeError:
            # openai-whisper fallback
            result = self._model.transcribe(        # type: ignore[union-attr]
                audio,
                language=settings.asr_language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            return result["text"].strip()
