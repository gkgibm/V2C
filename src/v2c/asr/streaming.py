"""
Streaming ASR engine — live transcription while the user is speaking.

Architecture
------------
              sounddevice callback
                    │ (float32 PCM frames, ~30ms each)
                    ▼
              _ring_buffer  (thread-safe deque, last WINDOW_S seconds)
              _full_buf     (entire session audio for final clean pass)
                    │
        ┌───────────▼────────────┐
        │   _inference_loop      │  ← background thread
        │                        │
        │   every STRIDE_S:      │
        │     snapshot ring buf  │
        │     → own WhisperModel │  ← SEPARATE model instance, no contention
        │     → emit partial     │
        └───────────┬────────────┘
                    │ partial text via asyncio.Queue
                    ▼
          StreamingASREngine.partials()  ← async generator consumed by
                    │                       _stream_partials() asyncio task
                    ▼
          PARTIAL_TRANSCRIPT WebSocket msg → VS Code ghost text

On ``stop()``:
  - _running flag cleared → inference loop exits cleanly
  - mic stream stopped (no new frames)
  - None sentinel pushed to _partial_q → partials() generator returns
  - full accumulated audio returned for a single final clean Whisper pass

Design decisions
----------------
- **Dedicated model instance**: the streaming engine loads its OWN copy of
  WhisperModel so it never races with the main ASREngine (which handles the
  final transcription pass).  On CPU/tiny this adds ~100 MB RAM but eliminates
  all thread-safety issues.
- **asyncio.Queue re-created inside start()**: prevents stale None sentinels
  from a previous session immediately terminating the next partials() generator.
- **asyncio.get_running_loop()** (not get_event_loop): safe in async context,
  raises RuntimeError if called outside a running loop so bugs surface fast.
- WINDOW_S = 5s, STRIDE_S = 0.5s on CPU tiny: gives ~1-2 inference ticks/s
  which matches real perceived responsiveness on CPU (0.3s was optimistic).
- condition_on_previous_text=False: prevents hallucinated continuations.
- vad_filter=False on partials: VAD inside Whisper adds latency; we've already
  gated on MIN_AUDIO_S so silence-only chunks are skipped early.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

import numpy as np

from v2c.config import settings

logger = logging.getLogger(__name__)

# ── Sliding-window parameters ─────────────────────────────────────────────────

# How much audio Whisper sees on each inference tick (seconds).
WINDOW_S: float = 5.0

# How often we attempt an inference pass (seconds).
# On CPU/tiny a single 5s chunk takes ~0.5-1s, so 0.5s stride means one fresh
# result roughly every second — visibly responsive without CPU saturation.
STRIDE_S: float = 0.5

# Don't bother running Whisper until we have at least this much audio.
MIN_AUDIO_S: float = 0.4


def _load_streaming_model() -> object:
    """
    Load a WhisperModel dedicated to streaming partial inference.

    This is intentionally a separate instance from ASREngine._model so the
    two can run concurrently in different threads without data races.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]

        logger.info(
            "Loading streaming faster-whisper model '%s' on '%s' …",
            settings.asr_model,
            settings.asr_device,
        )
        t0 = time.perf_counter()
        model = WhisperModel(
            settings.asr_model,
            device=settings.asr_device,
            compute_type="int8" if settings.asr_device == "cpu" else "float16",
        )
        logger.info("Streaming model loaded in %.2f s", time.perf_counter() - t0)
        return model

    except ImportError:
        pass

    try:
        import whisper  # type: ignore[import]

        logger.info(
            "Loading streaming openai-whisper model '%s' …", settings.asr_model
        )
        model = whisper.load_model(settings.asr_model, device=settings.asr_device)
        return model

    except ImportError as exc:
        raise RuntimeError(
            "Neither 'faster-whisper' nor 'openai-whisper' is installed."
        ) from exc


class StreamingASREngine:
    """
    Captures microphone audio and emits partial transcripts in near-real-time
    while the user is speaking.

    Usage::

        engine = StreamingASREngine()    # loads its own Whisper model
        engine.start()                   # opens mic, starts inference loop

        async for partial in engine.partials():
            print(partial)               # "add func" → "add function calc…"

        full_audio = engine.stop()       # returns full float32 array
        # caller runs one final clean pass:
        final_text = await asr_engine.transcribe(full_audio)
    """

    def __init__(self) -> None:
        # Dedicated Whisper model — loaded lazily on first start().
        self._model: object = None
        self._model_loaded = False

        # Ring buffer: last WINDOW_S+1 seconds of audio (in individual frames).
        _maxlen = int((WINDOW_S + 1.0) * settings.sample_rate)
        self._ring: collections.deque[np.ndarray] = collections.deque(maxlen=_maxlen)

        # Full session audio (for the caller's final clean pass).
        self._full: list[np.ndarray] = []

        # Async queue: populated by inference thread, consumed by partials().
        # Re-created on every start() so stale sentinels never bleed across
        # sessions.
        self._partial_q: asyncio.Queue[str | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Thread control.
        self._running = False
        self._inference_thread: threading.Thread | None = None
        self._stream: object = None  # sounddevice.InputStream

        # Single-worker executor keeps inference calls serialised.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v2c-stream")

    # ── Mic stream ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone and start the inference loop."""
        import sounddevice as sd  # type: ignore[import]

        # Grab the running event loop (must be called from async context).
        self._loop = asyncio.get_running_loop()

        # Re-create queue fresh so no leftover sentinels from prior session.
        self._partial_q = asyncio.Queue()

        # Lazy-load the streaming model on first use.
        if not self._model_loaded:
            self._model = _load_streaming_model()
            self._model_loaded = True

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
            target=self._inference_loop, daemon=True, name="v2c-stream-infer"
        )
        self._inference_thread.start()

        logger.info(
            "StreamingASR started — window=%.1fs stride=%.1fs", WINDOW_S, STRIDE_S
        )

    def stop(self) -> np.ndarray:
        """
        Stop recording.  Returns the full captured audio as a float32 array.
        The caller should run one final clean Whisper pass on this array.
        """
        self._running = False

        # Stop mic first so no new frames arrive during cleanup.
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None

        # Wait for inference thread to drain its current tick.
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=3.0)
            self._inference_thread = None

        # Push sentinel so partials() generator exits cleanly.
        if self._loop is not None and self._partial_q is not None:
            self._loop.call_soon_threadsafe(self._partial_q.put_nowait, None)

        if not self._full:
            logger.warning("StreamingASR stopped with no captured audio.")
            return np.zeros(settings.sample_rate, dtype=np.float32)

        audio = np.concatenate(self._full)
        logger.info(
            "StreamingASR stopped — captured %.2f s of audio",
            len(audio) / settings.sample_rate,
        )
        return audio

    # ── Async partial generator ───────────────────────────────────────────────

    async def partials(self) -> AsyncGenerator[str, None]:
        """
        Async generator that yields partial transcript strings while recording.

        Each yielded string is the best current transcription of everything
        said so far in the session.  Stops automatically when stop() is called.
        """
        if self._partial_q is None:
            return  # start() not called yet
        while True:
            text = await self._partial_q.get()
            if text is None:
                break  # sentinel — recording stopped
            if text.strip():
                yield text

    # ── Inference loop (background thread) ───────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Background thread: every STRIDE_S seconds, snapshot the ring buffer,
        run Whisper, and push the result to the async queue.

        Uses call_soon_threadsafe so the queue put is always safe across
        the thread/asyncio boundary.
        """
        prev_text = ""

        while self._running:
            time.sleep(STRIDE_S)

            if not self._running:
                break

            # Snapshot ring buffer atomically (deque reads are GIL-safe).
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

            # Suppress duplicate or empty results.
            if not text or text == prev_text:
                continue

            prev_text = text
            logger.debug("Streaming partial: %r", text[:80])

            if self._loop is not None and self._partial_q is not None:
                self._loop.call_soon_threadsafe(self._partial_q.put_nowait, text)

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """
        Run a single Whisper inference pass on a float32 audio window.

        Handles both faster-whisper (lazy segment generator) and
        openai-whisper (dict return).
        """
        try:
            # faster-whisper returns a lazy generator — exhaust it immediately
            # so the model is not held open across the thread boundary.
            segments, _ = self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=settings.asr_language,
                beam_size=1,                         # greedy — fastest for partials
                temperature=0.0,
                condition_on_previous_text=False,    # no hallucinated continuations
                without_timestamps=True,
                vad_filter=False,                    # skip internal VAD for speed
            )
            # Consume the full generator NOW, inside this thread.
            texts = [seg.text.strip() for seg in segments]
            return " ".join(texts).strip()

        except (AttributeError, TypeError):
            # openai-whisper fallback (returns a dict, not a generator)
            result = self._model.transcribe(         # type: ignore[union-attr]
                audio,
                language=settings.asr_language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            return result["text"].strip()
