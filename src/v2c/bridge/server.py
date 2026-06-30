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
    LiveActionMessage,
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
from v2c.intent.splitter import split as split_transcript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segments_to_action_dicts(
    segments: list[tuple[str, int]],
) -> list[dict]:
    """
    Convert (command_text, newlines_after) pairs from the splitter into
    a list of action dicts ready for LiveActionMessage.actions.

    Uses rule-based router + dispatcher only (no LLM, no async).
    """
    from v2c.ast_engine.editor_action import DictationAction, NewlineAction

    result: list[dict] = []
    for seg_text, newlines_after in segments:
        seg_text = seg_text.strip()
        if not seg_text:
            continue
        routing = router.classify(seg_text)
        if routing.intent == IntentType.DICTATION:
            action = DictationAction(text=seg_text)
        else:
            action = rules.dispatch(seg_text)
            # If the rule dispatcher returned a NewlineAction for something
            # like "next line" that slipped through the splitter, absorb it.
            if isinstance(action, NewlineAction):
                newlines_after += action.count
                continue  # don't emit a separate action for it
        d = action.to_dict()
        d["newlines_after"] = newlines_after
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Session handler — one per connected WebSocket client
# ---------------------------------------------------------------------------


class SessionHandler:
    """
    Manages the V2C processing pipeline for a single WebSocket connection.
    """

    def __init__(self, ws: Any, asr: ASREngine, streamer: StreamingASREngine) -> None:
        self._ws = ws
        self._asr = asr
        self._streamer_engine = streamer   # shared pre-warmed engine (reused each session)
        self._refiner = get_refiner()
        self._context: EditorContext = EditorContext()
        # Active recording state
        self._recording = False
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
        if self._recording:
            logger.warning("START_RECORDING received while already recording — ignoring")
            return

        try:
            # start() is async: loads model via executor if not pre-warmed,
            # so the event loop stays unblocked.
            await self._streamer_engine.start()
            self._recording = True
            await self._send_status(ListeningStatus.LISTENING, "Microphone open")

            # Launch the task AFTER start() returns so the queue exists.
            self._partial_task = asyncio.create_task(self._stream_partials())
            logger.info("Recording started — partials task running.")

        except Exception as exc:
            logger.error("Failed to open microphone: %s", exc)
            self._recording = False
            await self._send_error(
                f"Could not open microphone: {exc}. "
                "Check System Settings → Privacy → Microphone and that "
                "sounddevice is installed."
            )

    async def _handle_stop_recording(self, _msg: StopRecordingMessage) -> None:
        if not self._recording:
            logger.warning("STOP_RECORDING received but no active recording — ignoring")
            return

        await self._send_status(ListeningStatus.PROCESSING)

        # Stop streaming engine — returns full audio, non-blocking
        audio = self._streamer_engine.stop()
        self._recording = False

        # Cancel the partial live-edit task
        if self._partial_task is not None:
            self._partial_task.cancel()
            try:
                await self._partial_task
            except asyncio.CancelledError:
                pass
            self._partial_task = None

        try:
            await self._run_pipeline(audio)
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            await self._send_error(f"Pipeline failed: {exc}")
        finally:
            await self._send_status(ListeningStatus.READY)

    async def _stream_partials(self) -> None:
        """
        For each partial transcript from the ASR engine:
          1. Run a fast rule-only pipeline (no ASR, no LLM, no refiner).
          2. Send a LIVE_ACTION(is_partial=True) to the extension.
          3. The extension applies it immediately into the document,
             undoing the previous partial edit first if there was one.

        When the user stops, _handle_stop_recording runs the full
        pipeline (with proper ASR on the complete audio) and sends
        LIVE_ACTION(is_partial=False), which the extension keeps permanently.
        """
        try:
            async for text in self._streamer_engine.partials():
                await self._run_partial_pipeline(text)
        except asyncio.CancelledError:
            pass  # normal — cancelled by _handle_stop_recording

    async def _run_partial_pipeline(self, text: str) -> None:
        """
        Fast synchronous-only pipeline for partial transcripts.

        Splits on "next line" markers, dispatches each segment through
        rule-based router+dispatcher (~5ms total), sends a single
        LIVE_ACTION(is_partial=True) carrying all resulting actions.
        """
        if not text.strip():
            return

        segments = split_transcript(text)
        logger.info("▶ Partial split: %d segment(s) from %r", len(segments), text[:80])

        action_dicts = _segments_to_action_dicts(segments)
        if not action_dicts:
            return

        action_id = str(uuid.uuid4())
        await self._send(LiveActionMessage(
            action_id=action_id,
            actions=action_dicts,
            is_partial=True,
        ))
        logger.info(
            "→ LIVE_ACTION(partial) id=%s %d action(s)", action_id[:8], len(action_dicts)
        )

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
    # Core pipeline (full, runs on stop)
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

        # ── 4. Split on "next line" markers, dispatch each segment ────────
        segments = split_transcript(refined)
        logger.info("Final split: %d segment(s) from %r", len(segments), refined[:80])

        action_dicts: list[dict] = []
        for seg_text, newlines_after in segments:
            seg_routing = router.classify(seg_text)
            if seg_routing.intent == IntentType.DICTATION:
                from v2c.ast_engine.editor_action import DictationAction
                action = DictationAction(text=seg_text)
            elif seg_routing.intent == IntentType.COMMAND:
                action = rules.dispatch(seg_text)
                from v2c.ast_engine.editor_action import DictationAction
                if isinstance(action, DictationAction) and settings.use_llm:
                    action = await llm_parser.parse(seg_text, refine_ctx)
            else:  # AMBIGUOUS
                if settings.use_llm:
                    action = await llm_parser.parse(seg_text, refine_ctx)
                else:
                    action = rules.dispatch(seg_text)
            d = action.to_dict()
            d["newlines_after"] = newlines_after
            action_dicts.append(d)

        # ── 5. Send final live action list (is_partial=False) ─────────────
        action_id = str(uuid.uuid4())
        await self._send(LiveActionMessage(
            action_id=action_id,
            actions=action_dicts,
            is_partial=False,
        ))
        logger.info(
            "→ LIVE_ACTION(final) id=%s %d action(s)", action_id[:8], len(action_dicts)
        )

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
            if self._recording:
                self._streamer_engine.stop()
                self._recording = False
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

    # Pre-warm the streaming engine at startup so the first recording session
    # has zero model-load latency (loads in executor, non-blocking).
    streamer = StreamingASREngine()
    await streamer.warmup()

    async def handler(ws: Any) -> None:
        session = SessionHandler(ws, asr, streamer)
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
