"""
Centralised configuration loaded from environment variables / .env file.

All settings are optional — sensible defaults let the engine run fully
offline with no API keys required (uses local Whisper tiny model and
rule-based refinement).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal  # still used for str Literal fields (asr_model, log_level, etc.)

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the project root (two levels up from this file when
# installed as a package, or from cwd when running in-repo).
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # will look in cwd


class Settings(BaseSettings):
    """Runtime configuration for the V2C engine."""

    model_config = SettingsConfigDict(
        env_prefix="V2C_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── ASR ──────────────────────────────────────────────────────────────────
    asr_model: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
        default="tiny",
        description="Whisper model size.  'tiny' is recommended for latency.",
    )
    asr_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="PyTorch device for Whisper inference.",
    )
    asr_language: str = Field(
        default="en",
        description="BCP-47 language code passed to Whisper.",
    )
    asr_beam_size: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Beam size for Whisper decoding. 1 = greedy (fastest).",
    )

    # ── Voice Activity Detection ─────────────────────────────────────────────
    # Note: these are plain int fields (not Literal[int]) so that pydantic-settings
    # can coerce the env-var string "2" → int before validation.
    vad_aggressiveness: int = Field(
        default=2,
        ge=0,
        le=3,
        description="WebRTC VAD aggressiveness (0–3). Higher = more aggressive.",
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz. Must be 8000, 16000, 32000, or 48000.",
    )
    vad_frame_ms: int = Field(
        default=30,
        description="VAD frame duration in ms. Must be 10, 20, or 30.",
    )
    vad_silence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=5.0,
        description=(
            "Seconds of silence (post speech) before an utterance is "
            "considered complete."
        ),
    )
    vad_min_speech_ms: int = Field(
        default=300,
        ge=100,
        description="Minimum speech segment length in ms to trigger ASR.",
    )

    # ── Bridge / WebSocket ───────────────────────────────────────────────────
    ws_host: str = Field(default="127.0.0.1", description="WebSocket server bind host.")
    ws_port: int = Field(default=6789, ge=1024, le=65535, description="WebSocket port.")

    # ── LLM Refinement ───────────────────────────────────────────────────────
    refine_enabled: bool = Field(
        default=True,
        description="Enable post-ASR LLM-based transcription refinement.",
    )
    openai_api_key: str = Field(
        default="",
        description=(
            "OpenAI API key for cloud refinement and command parsing. "
            "If empty, falls back to rule-based processing."
        ),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model used for refinement and semantic parsing.",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM calls. Low = deterministic.",
    )
    llm_timeout: float = Field(
        default=10.0,
        ge=1.0,
        description="Timeout in seconds for a single LLM API call.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity.",
    )

    # ── Derived helpers ──────────────────────────────────────────────────────
    @model_validator(mode="after")
    def _resolve_openai_key(self) -> "Settings":
        """Also accept the bare OPENAI_API_KEY env var (no V2C_ prefix)."""
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        return self

    @property
    def use_llm(self) -> bool:
        """True when a valid OpenAI key is available."""
        return bool(self.openai_api_key)

    @property
    def frames_per_chunk(self) -> int:
        """Number of audio samples per VAD frame."""
        return int(self.sample_rate * self.vad_frame_ms / 1000)


# Module-level singleton — import from here everywhere else.
settings = Settings()
