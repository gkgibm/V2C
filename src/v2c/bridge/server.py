"""
Async WebSocket server — the core IPC bridge.

Audio capture model (revised):
  The VS Code extension has no reliable cross-platform access to the
  microphone (webview getUserMedia is blocked on non-https origins).
  Instead the extension sends START_RECORDING / STOP_RECORDING commands
  and the Python server captures audio locally via sounddevice.

Architecture::

    asyncio event loop
    │
    ├── websockets.serve() listening on 127.0.0.1:6789
    │     └── per-client coroutine → SessionHandler
    │           ├── START_RECORDING  → spawn _record_task() in background
    │           ├── STOP_RECORDING   → cancel _record_task(), run pipeline
    │           ├── CONTEXT          → update editor state
    │           └── pipeline:
    │                 1. ASREngine.transcribe(audio)
    │                 2. Refiner.refine(transcript, context)
    │                 3. IntentRouter.classify(refined)
    │                 4. RulesDispatcher or LLMParser
    │                 5. Send ActionMessage to extension
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any

import numpy as np

from v2c.ast_engine import parser as ast_parser
from v2c.asr.engine import ASREngine
from v2c.asr.refiner import RefinementContext, get_refiner
from v2c.bridge.protocol import (
    ActionMessage,
    AckMessage,
    AudioChunkMessage,
    AudioStopMessage,
    ClientErrorMessage,
    ContextMessage,
    EditorContext,
    ListeningStatus,
    ServerErrorMessage,
    StartRecordingMessage,
    StopRecordingMessage,
    StatusMessage,
    TranscriptMessage,
    parse_client_message,
)
from v2c.config import settings
from v2c.intent import llm_parser, router, rules
from v2c.intent.router import IntentType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local microphone recorder (runs in background asyncio task)
# ---------------------------------------------------------------------------

class _MicRecorder:
    """
    Captures audio from the default system microphone using sounddevice.

    Usage::

        recorder = _MicRecorder()
        recorder.start()          # opens mic stream
        ...
        audio = recorder.stop()   # closes stream, returns float32 array
    """

    def __init__(self) -> None:
        self._frames: list[np.ndarray] = []
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd  # type: ignore[import]

        def _callback(
            indata: np.ndarray,
            _frames: int,
            _time: object,
            status: Any,
        ) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
            self._frames.append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=settings.sample_rate,
            blocksize=settings.frames_per_chunk,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        logger.info("Microphone recording started (sample_rate=%d Hz)", settings.sample_rate)

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            logger.warning("No audio frames captured.")
            return np.zeros(settings.sample_rate, dtype=np.float32)

        audio = np.concatenate(self._frames)
        self._frames.clear()
        duration = len(audio) / settings.sample_rate
        logger.info("Microphone recording stopped — captured %.2f s of audio", duration)
        return audio


# ---------------------------------------------------------------------------
# Session handler — one per connected WebSocket client
# ---------------------------------------------------------------------------


class SessionHandler:
    """
    Manages the V2C processing pipeline for a single WebSocket connection.
    """

    def __init__(self, ws: Any, asr: ASREngine) -> None:
        self._ws = ws
        self._asr = asr
        self._refiner = get_refiner()
        self._context: EditorContext = EditorContext()
        self._recorder: _MicRecorder | None = None
        # For legacy AUDIO_CHUNK mode (future browser capture)
        self._pcm_buf: bytearray = bytearray()

    # ------------------------------------------------------------------ #
    # Sending helpers
    # ------------------------------------------------------------------ #

    async def _send(self, model: Any) -> None:
        await self._ws.send(model.model_dump_json())

    async def _send_status(self, status: ListeningStatus, detail: str = "") -> None:
        await self._send(StatusMessage(status=status, detail=detail))

    async def _send_error(self, message: str) -> None:
        await self._send(ServerErrorMessage(message=message))

    # ------------------------------------------------------------------ #
    # Message handlers
    # ------------------------------------------------------------------ #

    def _handle_context(self, msg: ContextMessage) -> None:
        self._context = msg.context
        logger.debug(
            "Context updated: file=%s language=%s cursor=(%d,%d)",
            self._context.active_file,
            self._context.language,
            self._context.cursor_line,
            self._context.cursor_char,
        )

    async def _handle_start_recording(self, _msg: StartRecordingMessage) -> None:
        if self._recorder is not None:
            logger.warning("START_RECORDING received while already recording — ignoring")
            return

        try:
            self._recorder = _MicRecorder()
            self._recorder.start()
            await self._send_status(ListeningStatus.LISTENING, "Microphone open")
        except Exception as exc:
            logger.error("Failed to open microphone: %s", exc)
            await self._send_error(
                f"Could not open microphone: {exc}. "
                "Check that no other app is using it and that sounddevice is installed."
            )
            self._recorder = None

    async def _handle_stop_recording(self, _msg: StopRecordingMessage) -> None:
        if self._recorder is None:
            logger.warning("STOP_RECORDING received but no active recording — ignoring")
            return

        await self._send_status(ListeningStatus.PROCESSING)
        audio = self._recorder.stop()
        self._recorder = None

        try:
            await self._run_pipeline(audio)
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            await self._send_error(f"Pipeline failed: {exc}")
        finally:
            await self._send_status(ListeningStatus.READY)

    def _handle_audio_chunk(self, msg: AudioChunkMessage) -> None:
        """Legacy: buffer PCM chunks sent from the client."""
        pcm_bytes = base64.b64decode(msg.data_b64)
        self._pcm_buf.extend(pcm_bytes)

    async def _handle_audio_stop(self, _msg: AudioStopMessage) -> None:
        """Legacy: process buffered PCM chunks sent from the client."""
        pcm_bytes = bytes(self._pcm_buf)
        self._pcm_buf.clear()

        if not pcm_bytes:
            await self._send_status(ListeningStatus.IDLE, "No audio received")
            return

        await self._send_status(ListeningStatus.PROCESSING)
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = audio_int16.astype(np.float32) / 32768.0

        try:
            await self._run_pipeline(audio)
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            await self._send_error(f"Pipeline failed: {exc}")
        finally:
            await self._send_status(ListeningStatus.READY)

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #

    async def _run_pipeline(self, audio: np.ndarray) -> None:
        # ── 1. ASR transcription ─────────────────────────────────────────
        raw_transcript = await self._asr.transcribe(audio.astype(np.float32))
        if not raw_transcript.strip():
            logger.debug("Empty transcript — ignoring utterance")
            await self._send_status(ListeningStatus.IDLE, "No speech detected")
            return

        # ── 2. Build refinement context from editor state ─────────────────
        identifiers: list[str] = []
        if self._context.source_code and self._context.language == "python":
            try:
                identifiers = ast_parser.extract_identifiers(self._context.source_code)
            except Exception as exc:
                logger.warning("AST identifier extraction failed: %s", exc)

        refine_ctx = RefinementContext(
            identifiers=identifiers,
            active_file=self._context.active_file,
            language=self._context.language,
        )

        # ── 3. Refinement ─────────────────────────────────────────────────
        refined = await self._refiner.refine(raw_transcript, refine_ctx)
        await self._send(TranscriptMessage(raw=raw_transcript, refined=refined))
        logger.info("Transcript — raw: %r  refined: %r", raw_transcript, refined)

        # ── 4. Intent routing ─────────────────────────────────────────────
        routing = router.classify(refined)
        logger.debug("Intent: %s (confidence=%.2f)", routing.intent.name, routing.confidence)

        if routing.intent == IntentType.DICTATION:
            from v2c.ast_engine.editor_action import DictationAction
            action = DictationAction(text=refined)

        elif routing.intent == IntentType.COMMAND:
            action = rules.dispatch(refined)
            from v2c.ast_engine.editor_action import DictationAction
            if isinstance(action, DictationAction) and settings.use_llm:
                action = await llm_parser.parse(refined, refine_ctx)

        else:  # AMBIGUOUS
            action = await llm_parser.parse(refined, refine_ctx)

        # ── 5. Send action to extension ───────────────────────────────────
        action_id = str(uuid.uuid4())
        await self._send(ActionMessage(action_id=action_id, action=action.to_dict()))
        logger.info("Action dispatched: id=%s type=%s", action_id[:8], type(action).__name__)

    # ------------------------------------------------------------------ #
    # Main message loop
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Process incoming messages until the connection closes."""
        await self._send_status(ListeningStatus.READY)
        logger.info("Client connected.")

        try:
            async for raw in self._ws:
                try:
                    msg = parse_client_message(raw)
                except (ValueError, json.JSONDecodeError) as exc:
                    logger.warning("Bad client message: %s", exc)
                    await self._send_error(str(exc))
                    continue

                if isinstance(msg, ContextMessage):
                    self._handle_context(msg)
                elif isinstance(msg, StartRecordingMessage):
                    await self._handle_start_recording(msg)
                elif isinstance(msg, StopRecordingMessage):
                    await self._handle_stop_recording(msg)
                elif isinstance(msg, AudioChunkMessage):
                    self._handle_audio_chunk(msg)
                elif isinstance(msg, AudioStopMessage):
                    await self._handle_audio_stop(msg)
                elif isinstance(msg, (AckMessage, ClientErrorMessage)):
                    pass  # nothing to do

        except Exception:
            pass
        finally:
            # Clean up mic if client disconnects mid-recording
            if self._recorder is not None:
                self._recorder.stop()
                self._recorder = None
            logger.info("Client disconnected.")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


async def start_server() -> None:
    """Start the WebSocket server and run indefinitely."""
    try:
        import websockets  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("Install websockets: pip install websockets") from exc

    asr = ASREngine()
    await asr.warmup()

    async def handler(ws: Any) -> None:
        session = SessionHandler(ws, asr)
        await session.run()

    logger.info(
        "V2C WebSocket server starting on ws://%s:%d",
        settings.ws_host,
        settings.ws_port,
    )

    async with websockets.serve(handler, settings.ws_host, settings.ws_port):
        await asyncio.Future()  # run forever


def run() -> None:
    """Synchronous wrapper — used as the ``v2c-server`` console script."""
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
    )
    logging.basicConfig(level=settings.log_level)
    asyncio.run(start_server())
