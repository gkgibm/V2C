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
        │   _inference_loop      │  ← background daemon thread
        │                        │
        │   every STRIDE_S:      │
        │     snapshot ring buf  │
        │     → own WhisperModel │  ← SEPARATE model, zero contention
        │     → emit partial     │
        └───────────┬────────────┘
                    │ str | None  via  call_soon_threadsafe → asyncio.Queue
                    ▼
          StreamingASREngine.partials()  ← async generator consumed by
                    │                       _stream_partials() asyncio task
                    ▼
          PARTIAL_TRANSCRIPT WebSocket msg → VS Code ghost text

On ``stop()``:
  - _running cleared → inference loop exits after current sleep tick
  - mic stream stopped (no new frames)
  - None sentinel pushed to queue → partials() generator returns
  - full accumulated audio returned for a single final clean Whisper pass

Design decisions
----------------
* Dedicated model instance: the streaming engine loads its OWN WhisperModel
  so it never races with ASREngine (which handles the final pass).
* start() is async: model loading is offloaded to run_in_executor so the
  event loop is never blocked — partials begin flowing immediately.
* stop() is synchronous (called from async context): thread.join() is
  replaced by a non-blocking flag + sentinel; the caller awaits a short
  asyncio.sleep() instead of joining the thread directly.
* asyncio.Queue re-created inside start(): prevents stale None sentinels
  from a previous session immediately killing the next partials() generator.
* asyncio.get_running_loop(): safe, raises if called outside loop context.
* STRIDE_S = 0.5s on CPU/tiny: inference takes ~0.5-1s per 5s window.
* condition_on_previous_text=False: no hallucinated continuations.
* vad_filter=False on partials: skip internal Whisper VAD for latency.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from typing import AsyncGenerator

import numpy as np

from v2c.config import settings

logger = logging.getLogger(__name__)

# ── Sliding-window parameters ─────────────────────────────────────────────────

# How much audio Whisper sees on each inference tick.
WINDOW_S: float = 5.0

# Sleep between inference attempts. On CPU/tiny a single pass takes ~0.5-1s,
# so 0.5s stride means one fresh result roughly every 1-1.5s of speech.
STRIDE_S: float = 0.5

# Don't bother running Whisper until we have this much audio.
MIN_AUDIO_S: float = 0.5


def _load_streaming_model() -> object:
    """
    Load a WhisperModel dedicated to streaming partial inference.

    Intentionally a separate instance from ASREngine._model so the two
    can run concurrently in their own threads without data races.
    Called via run_in_executor so it never blocks the event loop.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]

        logger.info(
            "Streaming model: loading faster-whisper '%s' on '%s' …",
            settings.asr_model,
            settings.asr_device,
        )
        t0 = time.perf_counter()
        model = WhisperModel(
            settings.asr_model,
            device=settings.asr_device,
            compute_type="int8" if settings.asr_device == "cpu" else "float16",
        )
        logger.info("Streaming model ready in %.2fs", time.perf_counter() - t0)
        return model

    except ImportError:
        pass

    try:
        import whisper  # type: ignore[import]

        logger.info(
            "Streaming model: loading openai-whisper '%s' …", settings.asr_model
        )
        model = whisper.load_model(settings.asr_model, device=settings.asr_device)
        return model

    except ImportError as exc:
        raise RuntimeError(
            "Neither 'faster-whisper' nor 'openai-whisper' is installed."
        ) from exc


class StreamingASREngine:
    """
    Captures microphone audio and emits partial transcripts in near-real-time.

    Call ``await engine.warmup()`` once at server startup to pre-load the
    model so the first recording session has zero cold-start latency.

    Usage per session::

        await engine.start()                   # non-blocking: opens mic
        async for partial in engine.partials():
            print(partial)                     # live text while speaking
        full_audio = engine.stop()             # mic closed, full PCM returned
        final = await asr.transcribe(full_audio)
    """

    def __init__(self) -> None:
        # Dedicated Whisper model (loaded async in warmup / first start).
        self._model: object = None
        self._model_loaded = False

        # Ring buffer: last WINDOW_S+1s of audio frames.
        _maxlen = int((WINDOW_S + 1.0) * settings.sample_rate)
        self._ring: collections.deque[np.ndarray] = collections.deque(maxlen=_maxlen)

        # Full session audio for the caller's final clean pass.
        self._full: list[np.ndarray] = []

        # Async queue re-created on every start() (no stale sentinels).
        self._partial_q: asyncio.Queue[str | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Thread control.
        self._running = False
        self._inference_thread: threading.Thread | None = None
        self._stream: object = None  # sounddevice.InputStream

    # ── Model pre-loading ─────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """
        Pre-load the streaming Whisper model in a background executor thread
        so the first START_RECORDING session has zero model-load latency.
        """
        if self._model_loaded:
            return
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, _load_streaming_model)
        self._model_loaded = True
        logger.info("StreamingASR model pre-loaded and ready.")

    # ── Session start/stop ────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Open the microphone and start the inference loop.

        async so that model loading (on first call) runs in an executor
        without blocking the event loop — WebSocket sends work while loading.
        """
        import sounddevice as sd  # type: ignore[import]

        self._loop = asyncio.get_running_loop()

        # Re-create queue fresh: no leftover sentinels from a prior session.
        self._partial_q = asyncio.Queue()

        # Load model if not pre-warmed (non-blocking via executor).
        if not self._model_loaded:
            logger.warning(
                "StreamingASR model not pre-loaded — loading now (adds latency)."
            )
            self._model = await self._loop.run_in_executor(None, _load_streaming_model)
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

        Non-blocking: sets the flag and closes the mic immediately.
        The inference thread exits on its own after the current sleep tick.
        A None sentinel is pushed to the queue so partials() exits cleanly.
        """
        self._running = False

        # Close mic immediately — no new frames.
        if self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None

        # Don't join the thread — it will exit on its own when _running=False.
        # The caller is an async coroutine; joining here would block the loop.
        self._inference_thread = None

        # Sentinel: causes partials() async generator to return.
        if self._loop is not None and self._partial_q is not None:
            self._loop.call_soon_threadsafe(self._partial_q.put_nowait, None)

        if not self._full:
            logger.warning("StreamingASR stopped with no captured audio.")
            return np.zeros(settings.sample_rate, dtype=np.float32)

        audio = np.concatenate(self._full)
        logger.info(
            "StreamingASR stopped — captured %.2fs of audio",
            len(audio) / settings.sample_rate,
        )
        return audio

    # ── Async partial generator ───────────────────────────────────────────────

    async def partials(self) -> AsyncGenerator[str, None]:
        """
        Async generator that yields partial transcript strings while recording.

        Stops automatically when stop() pushes the None sentinel.
        """
        if self._partial_q is None:
            return
        while True:
            text = await self._partial_q.get()
            if text is None:
                break
            if text.strip():
                yield text

    # ── Inference loop (background thread) ───────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Runs in a daemon thread.  Every STRIDE_S seconds:
          1. Snapshot the last WINDOW_S seconds from the ring buffer.
          2. Run Whisper (synchronous — blocks this thread only, not the loop).
          3. Push changed text to the async queue via call_soon_threadsafe.
        """
        prev_text = ""

        while self._running:
            time.sleep(STRIDE_S)

            if not self._running:
                break

            frames = list(self._ring)  # GIL-safe atomic snapshot
            if not frames:
                continue

            audio = np.concatenate(frames)
            if len(audio) / settings.sample_rate < MIN_AUDIO_S:
                continue

            try:
                text = self._transcribe_chunk(audio)
            except Exception as exc:
                logger.debug("Partial inference error: %s", exc)
                continue

            if not text or text == prev_text:
                continue

            prev_text = text
            # INFO level so it appears in default server logs without -v flag.
            logger.info("▶ Partial: %r", text[:100])

            if self._loop is not None and self._partial_q is not None:
                self._loop.call_soon_threadsafe(self._partial_q.put_nowait, text)

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """One Whisper pass on a float32 window.  Exhausts generator inline."""
        try:
            # faster-whisper — lazy generator must be exhausted here in the thread
            segments, _ = self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=settings.asr_language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
            )
            return " ".join(seg.text.strip() for seg in list(segments)).strip()

        except (AttributeError, TypeError):
            # openai-whisper fallback
            result = self._model.transcribe(  # type: ignore[union-attr]
                audio,
                language=settings.asr_language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            return result["text"].strip()
