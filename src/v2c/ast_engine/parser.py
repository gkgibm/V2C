"""
Tree-sitter Python parser.

Provides incremental AST parsing of Python source files and a clean
interface for querying nodes by type, name, and range.

Key public functions:

  parse(source: str) -> PythonAST
      Parse source code and return a queryable wrapper.

  extract_identifiers(source: str) -> list[str]
      Fast extraction of all identifiers in the current scope.

Design notes:
  - Tree-sitter grammar is loaded from the ``tree-sitter-python`` package.
  - The underlying parser is re-used across calls (thread-safe for reads).
  - All positions are 0-based (byte offsets / row-col); they are converted
    to 1-based line numbers when producing editor actions for VS Code.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parser loading
# ---------------------------------------------------------------------------

_parser_lock = threading.Lock()
_parser: object | None = None
_PY_LANGUAGE: object | None = None


def _get_parser() -> tuple[object, object]:
    """
    Return ``(Parser, Language)`` for Python, loading on first call.

    Uses tree-sitter 0.21+ API.
    """
    global _parser, _PY_LANGUAGE

    with _parser_lock:
        if _parser is not None and _PY_LANGUAGE is not None:
            return _parser, _PY_LANGUAGE

        try:
            import tree_sitter_python as tspython  # type: ignore[import]
            from tree_sitter import Language, Parser  # type: ignore[import]

            lang = Language(tspython.language())
            parser = Parser(lang)
            _parser = parser
            _PY_LANGUAGE = lang
            logger.info("tree-sitter Python parser loaded.")
            return parser, lang

        except Exception as exc:
            raise RuntimeError(
                "Failed to load tree-sitter Python parser. "
                "Run: pip install tree-sitter tree-sitter-python"
            ) from exc


# ---------------------------------------------------------------------------
# Node dataclass (thin wrapper for easier access)
# ---------------------------------------------------------------------------


@dataclass
class ASTNode:
    """A simplified, serialisable representation of a tree-sitter Node."""

    node_type: str
    name: str             # text of the *name* child, or empty string
    start_line: int       # 1-based
    end_line: int         # 1-based (inclusive)
    start_byte: int
    end_byte: int
    text: str             # full source text of this node


# ---------------------------------------------------------------------------
# PythonAST wrapper
# ---------------------------------------------------------------------------


class PythonAST:
    """
    Wrapper around a tree-sitter parse result for a Python file.

    Provides helper methods for common voice-to-code queries.
    """

    def __init__(self, tree: object, source: str) -> None:
        self._tree = tree
        self._source = source
        self._source_bytes = source.encode("utf-8")

    # ------------------------------------------------------------------ #
    # Generic traversal helpers
    # ------------------------------------------------------------------ #

    def _walk(self, node: object) -> Generator[object, None, None]:
        """Depth-first walk of the tree."""
        yield node
        for child in node.children:  # type: ignore[union-attr]
            yield from self._walk(child)

    def _node_text(self, node: object) -> str:
        return self._source_bytes[node.start_byte : node.end_byte].decode("utf-8")  # type: ignore[index]

    def _name_of(self, node: object) -> str:
        """Return the text of the first ``identifier`` child of *node*."""
        for child in node.children:  # type: ignore[union-attr]
            if child.type == "identifier":  # type: ignore[union-attr]
                return self._node_text(child)
        return ""

    def _to_ast_node(self, node: object) -> ASTNode:
        return ASTNode(
            node_type=node.type,  # type: ignore[union-attr]
            name=self._name_of(node),
            start_line=node.start_point[0] + 1,  # type: ignore[index]
            end_line=node.end_point[0] + 1,  # type: ignore[index]
            start_byte=node.start_byte,  # type: ignore[union-attr]
            end_byte=node.end_byte,  # type: ignore[union-attr]
            text=self._node_text(node),
        )

    # ------------------------------------------------------------------ #
    # Public query methods
    # ------------------------------------------------------------------ #

    def functions(self) -> list[ASTNode]:
        """Return all top-level and nested function definitions."""
        return [
            self._to_ast_node(n)
            for n in self._walk(self._tree.root_node)  # type: ignore[union-attr]
            if n.type in ("function_definition", "decorated_definition")  # type: ignore[union-attr]
            and (
                n.type != "decorated_definition"
                or any(
                    c.type == "function_definition"
                    for c in n.children  # type: ignore[union-attr]
                )
            )
        ]

    def classes(self) -> list[ASTNode]:
        """Return all class definitions."""
        return [
            self._to_ast_node(n)
            for n in self._walk(self._tree.root_node)  # type: ignore[union-attr]
            if n.type == "class_definition"  # type: ignore[union-attr]
        ]

    def imports(self) -> list[ASTNode]:
        """Return all import statements."""
        return [
            self._to_ast_node(n)
            for n in self._walk(self._tree.root_node)  # type: ignore[union-attr]
            if n.type in ("import_statement", "import_from_statement")  # type: ignore[union-attr]
        ]

    def identifiers(self) -> list[str]:
        """
        Return a de-duplicated list of all identifiers visible in this file.

        Used to seed the refinement context for phonetic snapping.
        """
        seen: set[str] = set()
        result: list[str] = []
        for node in self._walk(self._tree.root_node):  # type: ignore[union-attr]
            if node.type == "identifier":  # type: ignore[union-attr]
                name = self._node_text(node)
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        return result

    def find_function(self, name: str) -> ASTNode | None:
        """Find the first function definition with the given name."""
        for node in self._walk(self._tree.root_node):  # type: ignore[union-attr]
            if node.type == "function_definition":  # type: ignore[union-attr]
                if self._name_of(node) == name:
                    return self._to_ast_node(node)
        return None

    def find_class(self, name: str) -> ASTNode | None:
        """Find the first class definition with the given name."""
        for node in self._walk(self._tree.root_node):  # type: ignore[union-attr]
            if node.type == "class_definition":  # type: ignore[union-attr]
                if self._name_of(node) == name:
                    return self._to_ast_node(node)
        return None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def parse(source: str) -> PythonAST:
    """
    Parse *source* (Python code) and return a :class:`PythonAST`.

    This is the main entry point for AST operations.

    Args:
        source: UTF-8 Python source code string.

    Returns:
        A :class:`PythonAST` wrapping the parse result.
    """
    parser, _ = _get_parser()
    tree = parser.parse(source.encode("utf-8"))  # type: ignore[union-attr]
    return PythonAST(tree, source)


def extract_identifiers(source: str) -> list[str]:
    """
    Fast extraction of all identifiers in *source*.

    Convenience wrapper used to populate :class:`~v2c.asr.refiner.RefinementContext`.
    """
    return parse(source).identifiers()
