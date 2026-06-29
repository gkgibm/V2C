"""
Tree-sitter query builders for common Python patterns.

These queries are expressed in tree-sitter's S-expression syntax and
capture specific AST node types.  The results can be used both for
navigation (jump to function) and for structural edits (delete class).

Usage::

    from v2c.ast_engine.queries import FUNCTION_QUERY, run_query
    matches = run_query(FUNCTION_QUERY, source)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-defined Tree-sitter queries (S-expression syntax)
# ---------------------------------------------------------------------------

# Capture all top-level function definitions and their names.
FUNCTION_QUERY = """
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body) @function.def
"""

# Capture all class definitions and their names.
CLASS_QUERY = """
(class_definition
  name: (identifier) @class.name
  body: (block) @class.body) @class.def
"""

# Capture all import statements.
IMPORT_QUERY = """
[
  (import_statement) @import.stmt
  (import_from_statement) @import.from_stmt
]
"""

# Capture method definitions inside a class body.
METHOD_QUERY = """
(class_definition
  name: (identifier) @class.name
  body: (block
    (function_definition
      name: (identifier) @method.name) @method.def))
"""

# Capture all identifiers (for context-aware refinement).
IDENTIFIER_QUERY = """
(identifier) @id
"""


# ---------------------------------------------------------------------------
# Query result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryMatch:
    """A single capture result from a tree-sitter query."""
    capture_name: str   # e.g. "function.name"
    text: str           # the captured text
    start_line: int     # 1-based
    end_line: int       # 1-based


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def run_query(query_str: str, source: str) -> list[QueryMatch]:
    """
    Execute a tree-sitter query against *source* and return a list of
    :class:`QueryMatch` results.

    Compatible with tree-sitter >= 0.22 (uses QueryCursor API).

    Args:
        query_str: Tree-sitter S-expression query string.
        source:    Python source code.

    Returns:
        List of captures sorted by line number.
    """
    try:
        import tree_sitter_python as tspython  # type: ignore[import]
        from tree_sitter import Language, Parser, Query, QueryCursor  # type: ignore[import]

        lang = Language(tspython.language())
        parser = Parser(lang)
        tree = parser.parse(source.encode("utf-8"))

        query = Query(lang, query_str)
        cursor = QueryCursor(query)
        # captures() returns dict[str, list[Node]] in tree-sitter 0.25+
        captures_dict: dict = cursor.captures(tree.root_node)

        source_bytes = source.encode("utf-8")
        results: list[QueryMatch] = []

        for capture_name, nodes in captures_dict.items():
            for node in nodes:
                text = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
                results.append(
                    QueryMatch(
                        capture_name=capture_name,
                        text=text,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    )
                )

        results.sort(key=lambda m: m.start_line)
        return results

    except Exception as exc:
        logger.error("Tree-sitter query failed: %s", exc)
        return []
