"""
CLI entry-point for V2C.

Commands:

    v2c server   — start the WebSocket bridge server
    v2c status   — print version and configuration summary
"""

from __future__ import annotations

import logging

import click
from rich.console import Console
from rich.table import Table

from v2c import __version__
from v2c.config import settings

console = Console()


@click.group()
@click.version_option(__version__, prog_name="v2c")
def main() -> None:
    """V2C – Voice-to-Code engine."""
    logging.basicConfig(level=settings.log_level)


@main.command()
def server() -> None:
    """Start the V2C WebSocket bridge server."""
    from v2c.bridge.server import run

    console.print(f"[bold green]V2C {__version__}[/bold green] — starting server …")
    run()


@main.command()
def status() -> None:
    """Print current configuration and feature availability."""
    table = Table(title=f"V2C {__version__} — Configuration", show_lines=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Status", style="green")

    rows = [
        ("ASR model", settings.asr_model, "✓"),
        ("ASR device", settings.asr_device, "✓"),
        ("Sample rate", f"{settings.sample_rate} Hz", "✓"),
        ("VAD aggressiveness", str(settings.vad_aggressiveness), "✓"),
        ("WS port", str(settings.ws_port), "✓"),
        (
            "LLM refinement",
            settings.llm_model if settings.use_llm else "disabled",
            "✓" if settings.use_llm else "⚠ (no OPENAI_API_KEY)",
        ),
    ]

    for name, value, status_icon in rows:
        table.add_row(name, value, status_icon)

    console.print(table)

    # Check optional dependencies
    _check_dep("faster_whisper", "faster-whisper (recommended ASR backend)")
    _check_dep("webrtcvad", "webrtcvad (Voice Activity Detection)")
    _check_dep("tree_sitter", "tree-sitter (AST parsing)")
    _check_dep("tree_sitter_python", "tree-sitter-python (Python grammar)")
    _check_dep("sounddevice", "sounddevice (microphone capture)")
    _check_dep("websockets", "websockets (bridge IPC)")
    _check_dep("openai", "openai (LLM refinement — optional)")


def _check_dep(module: str, label: str) -> None:
    try:
        __import__(module)
        console.print(f"  [green]✓[/green] {label}")
    except ImportError:
        console.print(f"  [red]✗[/red] {label} [dim](not installed)[/dim]")
