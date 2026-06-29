"""
V2C – Voice-to-Code

A privacy-first, AST-aware voice coding engine for Python.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("v2c")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
