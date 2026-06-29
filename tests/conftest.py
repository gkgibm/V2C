"""
Pytest fixtures and shared test utilities.
"""

from __future__ import annotations

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Simple Python source snippets used across tests
# ---------------------------------------------------------------------------

SIMPLE_PYTHON_SOURCE = """
import os
import sys
from pathlib import Path

MAX_RETRIES = 3


class UserManager:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def get_user(self, user_id: int) -> dict:
        pass

    def create_user(self, name: str, email: str) -> dict:
        pass


def calculate_tax(price: float, rate: float = 0.1) -> float:
    return price * rate


def format_currency(amount: float) -> str:
    return f"${amount:.2f}"
""".strip()


@pytest.fixture
def sample_source() -> str:
    return SIMPLE_PYTHON_SOURCE


@pytest.fixture
def silent_audio() -> np.ndarray:
    """16 kHz, 1-second silence (float32)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """16 kHz, 0.5-second sine wave at 440 Hz (not real speech, but non-zero)."""
    t = np.linspace(0, 0.5, 8000, dtype=np.float32)
    return 0.1 * np.sin(2 * np.pi * 440 * t)
