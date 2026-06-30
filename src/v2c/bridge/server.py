"""
Async WebSocket server — the core IPC bridge.

Audio flow (streaming):

  START_RECORDING received
    └─ StreamingASREngine.start()   ← opens mic + background inference loop
    └─ _stream_partials_task()      ← asyncio task: forwards partial text to
                                       extension as PARTIAL_TRANSCRIPT messages
                                       → extension shows live ghost text

  STOP_RECORDING received
    └─ StreamingASREngine.stop()    ← closes mic, returns full float32 audio
    └─ _stream_partials_task cancelled
    └─ final ASREngine.transcribe() ← one clean Whisper pass on full audio
    └─ Refiner + Router + Rules/LLM
    └─ ACTION message → extension   ← code inserted / command executed
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
from v2c.asr.streaming import StreamingASREngine
from v2c.bridge.protocol import (
    ActionMessage,
    AckMessage,
    AudioChunkMessage,
    AudioStopMessage,
    ClientErrorMessage,
    ContextMessage,
    EditorContext,
    ListeningStatus,
    PartialTranscriptMessage,
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
        # Streaming ASR — created fresh on each START_RECORDING
        self._streamer: StreamingASREngine | None = None
        self._partial_task: asyncio.Task | None = None
        # Legacy: buffer for AUDIO_CHUNK mode
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
        if self._streamer is not None:
            logger.warning("START_RECORDING received while already recording — ignoring")
            return

        try:
            self._streamer = StreamingASREngine()
            self._streamer.start()
            await self._send_status(ListeningStatus.LISTENING, "Microphone open")

            # Launch background task that forwards partial transcripts to extension
            self._partial_task = asyncio.create_task(self._stream_partials())

        except Exception as exc:
            logger.error("Failed to open microphone: %s", exc)
            await self._send_error(
                f"Could not open microphone: {exc}. "
                "Check System Settings → Privacy → Microphone and that "
                "sounddevice is installed."
            )
            self._streamer = None

    async def _handle_stop_recording(self, _msg: StopRecordingMessage) -> None:
        if self._streamer is None:
            logger.warning("STOP_RECORDING received but no active recording — ignoring")
            return

        await self._send_status(ListeningStatus.PROCESSING)

        # Stop streaming — returns the full captured audio
        audio = self._streamer.stop()
        self._streamer = None

        # Cancel the partial-forwarding task
        if self._partial_task is not None:
            self._partial_task.cancel()
            try:
                await self._partial_task
            except asyncio.CancelledError:
                pass
            self._partial_task = None

        # Clear the live ghost text from the extension
        await self._send(PartialTranscriptMessage(text="", is_final=True))

        try:
            await self._run_pipeline(audio)
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            await self._send_error(f"Pipeline failed: {exc}")
        finally:
            await self._send_status(ListeningStatus.READY)

    async def _stream_partials(self) -> None:
        """
        Forward partial transcripts from StreamingASREngine to the extension.

        Runs as an asyncio Task for the duration of a recording session.
        Each partial update replaces the previous ghost text in the editor.
        """
        if self._streamer is None:
            return
        try:
            async for text in self._streamer.partials():
                await self._send(PartialTranscriptMessage(text=text))
                logger.debug("Partial: %r", text[:60])
        except asyncio.CancelledError:
            pass  # normal — cancelled by _handle_stop_recording

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
            # Clean up if client disconnects mid-recording
            if self._streamer is not None:
                self._streamer.stop()
                self._streamer = None
            if self._partial_task is not None:
                self._partial_task.cancel()
                self._partial_task = None
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
