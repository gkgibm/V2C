"""
Async WebSocket server — the core IPC bridge.

This module implements the asyncio WebSocket server that the VS Code
extension connects to.  Each connected client gets its own
:class:`SessionHandler` which manages the full voice → ASR → refine →
classify → act pipeline for that connection.

Architecture::

    asyncio event loop
    │
    ├── websockets.serve() listening on 127.0.0.1:6789
    │     └── per-client coroutine → SessionHandler
    │           ├── _handle_context()   (update editor state)
    │           ├── _handle_audio_chunk() (accumulate PCM)
    │           ├── _handle_audio_stop()  (trigger pipeline)
    │           └── pipeline:
    │                 1. ASREngine.transcribe(audio)
    │                 2. Refiner.refine(transcript, context)
    │                 3. IntentRouter.classify(refined)
    │                 4. RulesDispatcher or LLMParser
    │                 5. Send ActionMessage to extension

The pipeline runs entirely in async coroutines; blocking Whisper inference
is dispatched to a thread-pool via :func:`asyncio.get_event_loop().run_in_executor`.
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
    AudioChunkMessage,
    AudioStopMessage,
    ClientErrorMessage,
    ContextMessage,
    EditorContext,
    ListeningStatus,
    ServerErrorMessage,
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

    Attributes:
        _ws:       The active WebSocket connection.
        _context:  Most recently received editor context.
        _pcm_buf:  Accumulated raw int16 PCM bytes from AUDIO_CHUNK messages.
        _asr:      Shared ASR engine instance.
        _refiner:  Refiner instance (LLM or rule-based).
    """

    def __init__(self, ws: Any, asr: ASREngine) -> None:
        self._ws = ws
        self._asr = asr
        self._refiner = get_refiner()
        self._context: EditorContext = EditorContext()
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
            "Context updated: file=%s, language=%s, cursor=(%d, %d)",
            self._context.active_file,
            self._context.language,
            self._context.cursor_line,
            self._context.cursor_char,
        )

    def _handle_audio_chunk(self, msg: AudioChunkMessage) -> None:
        pcm_bytes = base64.b64decode(msg.data_b64)
        self._pcm_buf.extend(pcm_bytes)

    async def _handle_audio_stop(self, _msg: AudioStopMessage) -> None:
        """
        Triggered when the user releases the recording button.
        Runs the full pipeline on the accumulated PCM buffer.
        """
        pcm_bytes = bytes(self._pcm_buf)
        self._pcm_buf.clear()

        if not pcm_bytes:
            await self._send_status(ListeningStatus.IDLE, "No audio received")
            return

        await self._send_status(ListeningStatus.PROCESSING)

        try:
            await self._run_pipeline(pcm_bytes)
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            await self._send_error(f"Pipeline failed: {exc}")
        finally:
            await self._send_status(ListeningStatus.READY)

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #

    async def _run_pipeline(self, pcm_bytes: bytes) -> None:
        # ── 1. Decode raw int16 PCM → float32 numpy array ───────────────
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # ── 2. ASR transcription ─────────────────────────────────────────
        raw_transcript = await self._asr.transcribe(audio_float32)
        if not raw_transcript.strip():
            logger.debug("Empty transcript — ignoring utterance")
            return

        # ── 3. Build refinement context from editor state ─────────────────
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

        # ── 4. Refinement ─────────────────────────────────────────────────
        refined = await self._refiner.refine(raw_transcript, refine_ctx)

        # Broadcast transcript so the extension can show it in the status bar.
        await self._send(TranscriptMessage(raw=raw_transcript, refined=refined))

        # ── 5. Intent routing ─────────────────────────────────────────────
        routing = router.classify(refined)

        if routing.intent == IntentType.DICTATION:
            from v2c.ast_engine.editor_action import DictationAction
            action = DictationAction(text=refined)

        elif routing.intent == IntentType.COMMAND:
            # Try fast rule-based dispatcher first.
            action = rules.dispatch(refined)
            # If the dispatcher fell back to dictation but we expected a
            # command, escalate to LLM.
            from v2c.ast_engine.editor_action import DictationAction
            if isinstance(action, DictationAction) and settings.use_llm:
                action = await llm_parser.parse(refined, refine_ctx)

        else:  # AMBIGUOUS
            # LLM is the tiebreaker; fall back to dictation if no key.
            action = await llm_parser.parse(refined, refine_ctx)

        # ── 6. Broadcast action to extension ─────────────────────────────
        action_id = str(uuid.uuid4())
        await self._send(
            ActionMessage(action_id=action_id, action=action.to_dict())
        )
        logger.info(
            "Action dispatched: id=%s type=%s",
            action_id[:8],
            type(action).__name__,
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
                elif isinstance(msg, AudioChunkMessage):
                    self._handle_audio_chunk(msg)
                elif isinstance(msg, AudioStopMessage):
                    await self._handle_audio_stop(msg)
                elif isinstance(msg, ClientErrorMessage):
                    logger.warning("Client reported error: %s", msg.message)
                # AckMessage: no action needed.

        except Exception:
            # Connection closed abruptly.
            pass
        finally:
            logger.info("Client disconnected.")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


async def start_server() -> None:
    """
    Start the WebSocket server and run indefinitely.

    Called by the ``v2c-server`` CLI entry-point and by unit tests.
    """
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
