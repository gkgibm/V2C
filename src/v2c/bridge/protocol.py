"""
WebSocket bridge protocol — message schemas.

All communication between the VS Code extension (client) and the V2C
Python backend (server) uses newline-delimited JSON messages over a
WebSocket connection on ``ws://127.0.0.1:6789``.

Message flow::

    VS Code extension                V2C Python server
    ──────────────────               ─────────────────
    CONTEXT (editor state)  ──────▶
                            ◀──────  STATUS (listening)
    AUDIO_CHUNK (raw PCM)   ──────▶
    AUDIO_STOP              ──────▶
                            ◀──────  TRANSCRIPT (raw ASR)
                            ◀──────  ACTION (EditorAction JSON)
    ACK / ERROR             ──────▶

All message types are discriminated by the ``type`` field.

Pydantic models are used for validation and serialisation.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message types (discriminator values)
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    # Client → Server
    CONTEXT = "CONTEXT"
    START_RECORDING = "START_RECORDING"   # ask server to open its mic
    STOP_RECORDING = "STOP_RECORDING"     # ask server to stop and transcribe
    AUDIO_CHUNK = "AUDIO_CHUNK"           # kept for future browser-side capture
    AUDIO_STOP = "AUDIO_STOP"             # kept for compatibility
    ACK = "ACK"
    ERROR = "ERROR"

    # Server → Client
    STATUS = "STATUS"
    PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"  # live text while still speaking
    TRANSCRIPT = "TRANSCRIPT"                  # final transcript after stop
    ACTION = "ACTION"
    LIVE_ACTION = "LIVE_ACTION"                # live code edit while speaking
    SERVER_ERROR = "SERVER_ERROR"


class ListeningStatus(str, Enum):
    READY = "READY"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    IDLE = "IDLE"


# ---------------------------------------------------------------------------
# Client → Server messages
# ---------------------------------------------------------------------------

class EditorContext(BaseModel):
    """Current state of the VS Code editor, sent before each recording."""

    active_file: str = ""
    language: str = "python"
    # Content of the active file (used for AST parsing and context injection).
    # May be truncated to the last 2000 lines to stay within token budgets.
    source_code: str = ""
    # Cursor position (0-based line and character).
    cursor_line: int = 0
    cursor_char: int = 0
    # Currently selected text (may be empty).
    selected_text: str = ""


class ContextMessage(BaseModel):
    type: Literal[MessageType.CONTEXT] = MessageType.CONTEXT
    context: EditorContext


class StartRecordingMessage(BaseModel):
    """Ask the Python server to open its microphone and start recording."""
    type: Literal[MessageType.START_RECORDING] = MessageType.START_RECORDING


class StopRecordingMessage(BaseModel):
    """Ask the Python server to stop recording and run the transcription pipeline."""
    type: Literal[MessageType.STOP_RECORDING] = MessageType.STOP_RECORDING


class AudioChunkMessage(BaseModel):
    """
    A chunk of raw PCM audio encoded as a base64 string.

    Format: mono, 16-bit little-endian PCM at the sample rate configured
    in the Python server (default 16 kHz).
    """
    type: Literal[MessageType.AUDIO_CHUNK] = MessageType.AUDIO_CHUNK
    # Base64-encoded raw PCM bytes.
    data_b64: str


class AudioStopMessage(BaseModel):
    """Signal that the user has released the mic button (browser-side capture)."""
    type: Literal[MessageType.AUDIO_STOP] = MessageType.AUDIO_STOP


class AckMessage(BaseModel):
    """Acknowledge a server action was successfully applied."""
    type: Literal[MessageType.ACK] = MessageType.ACK
    action_id: str = ""


class ClientErrorMessage(BaseModel):
    """Report an error from the extension side."""
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    message: str


# ---------------------------------------------------------------------------
# Server → Client messages
# ---------------------------------------------------------------------------

class StatusMessage(BaseModel):
    """Inform the extension about the server's current listening state."""
    type: Literal[MessageType.STATUS] = MessageType.STATUS
    status: ListeningStatus
    detail: str = ""


class PartialTranscriptMessage(BaseModel):
    """
    Live partial transcript — sent while the user is still speaking.

    The extension shows this as ghost/preview text at the cursor so the
    user gets immediate visual feedback. It is NOT acted upon — only the
    final TranscriptMessage triggers the command pipeline.
    """
    type: Literal[MessageType.PARTIAL_TRANSCRIPT] = MessageType.PARTIAL_TRANSCRIPT
    text: str          # accumulated partial text so far
    is_final: bool = False


class TranscriptMessage(BaseModel):
    """Deliver the final ASR transcript after the user stops speaking."""
    type: Literal[MessageType.TRANSCRIPT] = MessageType.TRANSCRIPT
    raw: str
    refined: str


class LiveActionMessage(BaseModel):
    """
    A live code edit produced from a partial transcript while the user is
    still speaking.

    ``is_partial=True``: the extension applies this edit immediately, but
    records it for undo if the next partial produces a different result.

    ``is_partial=False``: this is the final clean result from the full
    pipeline after stop — the extension keeps it permanently.

    ``action`` carries the same dict shape as ActionMessage.action, so the
    same _applyAction() handler is reused.
    """
    type: Literal[MessageType.LIVE_ACTION] = MessageType.LIVE_ACTION
    action_id: str
    action: dict[str, Any]
    is_partial: bool = True


class ActionMessage(BaseModel):
    """
    Deliver the computed editor action to the VS Code extension.

    ``action`` is the raw dict from :meth:`EditorAction.to_dict()`.
    The extension deserialises it based on the ``action_type`` field.
    """
    type: Literal[MessageType.ACTION] = MessageType.ACTION
    action_id: str
    action: dict[str, Any]


class ServerErrorMessage(BaseModel):
    """Report a server-side processing error to the extension."""
    type: Literal[MessageType.SERVER_ERROR] = MessageType.SERVER_ERROR
    message: str


# ---------------------------------------------------------------------------
# Discriminated union helper
# ---------------------------------------------------------------------------

ClientMessage = Union[
    ContextMessage,
    StartRecordingMessage,
    StopRecordingMessage,
    AudioChunkMessage,
    AudioStopMessage,
    AckMessage,
    ClientErrorMessage,
]

ServerMessage = Union[
    StatusMessage,
    PartialTranscriptMessage,
    LiveActionMessage,
    TranscriptMessage,
    ActionMessage,
    ServerErrorMessage,
]


def parse_client_message(raw: str | bytes) -> ClientMessage:
    """
    Deserialise a raw JSON string from the extension into a typed message.

    Raises:
        ValueError: if the message type is unknown or the schema is invalid.
    """
    data = json.loads(raw)
    msg_type = data.get("type", "")

    _MAP: dict[str, type[BaseModel]] = {
        MessageType.CONTEXT: ContextMessage,
        MessageType.START_RECORDING: StartRecordingMessage,
        MessageType.STOP_RECORDING: StopRecordingMessage,
        MessageType.AUDIO_CHUNK: AudioChunkMessage,
        MessageType.AUDIO_STOP: AudioStopMessage,
        MessageType.ACK: AckMessage,
        MessageType.ERROR: ClientErrorMessage,
    }
    cls = _MAP.get(msg_type)
    if cls is None:
        raise ValueError(f"Unknown client message type: {msg_type!r}")
    return cls.model_validate(data)
