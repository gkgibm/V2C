"""
Tests for v2c.bridge.protocol — message parsing and serialisation.
"""

from __future__ import annotations

import json

import pytest

from v2c.bridge.protocol import (
    ActionMessage,
    AudioChunkMessage,
    AudioStopMessage,
    ContextMessage,
    EditorContext,
    ListeningStatus,
    MessageType,
    ServerErrorMessage,
    StatusMessage,
    TranscriptMessage,
    parse_client_message,
)


# ---------------------------------------------------------------------------
# Serialisation / deserialisation round-trips
# ---------------------------------------------------------------------------

class TestContextMessage:
    def test_round_trip(self) -> None:
        ctx = EditorContext(
            active_file="main.py",
            language="python",
            source_code="x = 1\n",
            cursor_line=0,
            cursor_char=0,
        )
        msg = ContextMessage(context=ctx)
        raw = msg.model_dump_json()
        parsed = parse_client_message(raw)
        assert isinstance(parsed, ContextMessage)
        assert parsed.context.active_file == "main.py"

    def test_type_field(self) -> None:
        msg = ContextMessage(context=EditorContext())
        data = json.loads(msg.model_dump_json())
        assert data["type"] == MessageType.CONTEXT


class TestAudioChunkMessage:
    def test_round_trip(self) -> None:
        import base64
        msg = AudioChunkMessage(data_b64=base64.b64encode(b"\x00\x01\x02").decode())
        raw = msg.model_dump_json()
        parsed = parse_client_message(raw)
        assert isinstance(parsed, AudioChunkMessage)

    def test_type_field(self) -> None:
        msg = AudioChunkMessage(data_b64="AAEC")
        data = json.loads(msg.model_dump_json())
        assert data["type"] == MessageType.AUDIO_CHUNK


class TestAudioStopMessage:
    def test_round_trip(self) -> None:
        msg = AudioStopMessage()
        raw = msg.model_dump_json()
        parsed = parse_client_message(raw)
        assert isinstance(parsed, AudioStopMessage)


class TestServerMessages:
    def test_status_message(self) -> None:
        msg = StatusMessage(status=ListeningStatus.LISTENING)
        data = json.loads(msg.model_dump_json())
        assert data["type"] == MessageType.STATUS
        assert data["status"] == ListeningStatus.LISTENING

    def test_transcript_message(self) -> None:
        msg = TranscriptMessage(raw="a sink request", refined="async request")
        data = json.loads(msg.model_dump_json())
        assert data["type"] == MessageType.TRANSCRIPT
        assert data["raw"] == "a sink request"
        assert data["refined"] == "async request"

    def test_action_message(self) -> None:
        msg = ActionMessage(
            action_id="abc-123",
            action={"action_type": "DICTATION", "text": "hello"},
        )
        data = json.loads(msg.model_dump_json())
        assert data["type"] == MessageType.ACTION
        assert data["action"]["text"] == "hello"

    def test_server_error_message(self) -> None:
        msg = ServerErrorMessage(message="something went wrong")
        data = json.loads(msg.model_dump_json())
        assert data["type"] == MessageType.SERVER_ERROR


class TestParseClientMessage:
    def test_rejects_unknown_type(self) -> None:
        raw = json.dumps({"type": "UNKNOWN_MSG_TYPE"})
        with pytest.raises(ValueError, match="Unknown client message type"):
            parse_client_message(raw)

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_client_message("not json at all{{{")
