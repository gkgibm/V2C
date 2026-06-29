"""
Tests for the AST engine — v2c.ast_engine.parser and v2c.ast_engine.queries.
"""

from __future__ import annotations

import pytest

from v2c.ast_engine.parser import PythonAST, extract_identifiers, parse
from v2c.ast_engine.queries import FUNCTION_QUERY, CLASS_QUERY, run_query


SAMPLE_CODE = """
import os
from pathlib import Path


MAX_ITEMS = 100


class Repository:
    def __init__(self, path: str) -> None:
        self.path = path

    def read(self) -> list[str]:
        return []

    def write(self, data: list[str]) -> None:
        pass


def load_config(filepath: str) -> dict:
    return {}


def save_config(filepath: str, config: dict) -> None:
    with open(filepath, "w") as f:
        pass
""".strip()


class TestPythonASTParser:
    def test_parse_returns_python_ast(self) -> None:
        ast = parse(SAMPLE_CODE)
        assert isinstance(ast, PythonAST)

    def test_functions_detected(self) -> None:
        ast = parse(SAMPLE_CODE)
        fns = ast.functions()
        names = [f.name for f in fns]
        assert "load_config" in names
        assert "save_config" in names

    def test_classes_detected(self) -> None:
        ast = parse(SAMPLE_CODE)
        classes = ast.classes()
        names = [c.name for c in classes]
        assert "Repository" in names

    def test_imports_detected(self) -> None:
        ast = parse(SAMPLE_CODE)
        imports = ast.imports()
        assert len(imports) >= 2

    def test_identifiers_non_empty(self) -> None:
        ast = parse(SAMPLE_CODE)
        ids = ast.identifiers()
        assert "Repository" in ids
        assert "load_config" in ids
        assert "MAX_ITEMS" in ids

    def test_find_function_existing(self) -> None:
        ast = parse(SAMPLE_CODE)
        node = ast.find_function("load_config")
        assert node is not None
        assert node.name == "load_config"
        assert node.start_line >= 1

    def test_find_function_missing_returns_none(self) -> None:
        ast = parse(SAMPLE_CODE)
        node = ast.find_function("nonexistent_function")
        assert node is None

    def test_find_class_existing(self) -> None:
        ast = parse(SAMPLE_CODE)
        node = ast.find_class("Repository")
        assert node is not None
        assert node.name == "Repository"

    def test_find_class_missing_returns_none(self) -> None:
        ast = parse(SAMPLE_CODE)
        node = ast.find_class("Ghost")
        assert node is None


class TestExtractIdentifiers:
    def test_returns_list_of_strings(self) -> None:
        ids = extract_identifiers(SAMPLE_CODE)
        assert isinstance(ids, list)
        assert all(isinstance(i, str) for i in ids)

    def test_known_names_present(self) -> None:
        ids = extract_identifiers(SAMPLE_CODE)
        assert "Repository" in ids
        assert "load_config" in ids

    def test_empty_source(self) -> None:
        ids = extract_identifiers("")
        assert isinstance(ids, list)


class TestQueryRunner:
    def test_function_query_returns_names(self) -> None:
        matches = run_query(FUNCTION_QUERY, SAMPLE_CODE)
        names = [m.text for m in matches if m.capture_name == "function.name"]
        assert "load_config" in names
        assert "save_config" in names

    def test_class_query_returns_names(self) -> None:
        matches = run_query(CLASS_QUERY, SAMPLE_CODE)
        names = [m.text for m in matches if m.capture_name == "class.name"]
        assert "Repository" in names

    def test_invalid_query_returns_empty_list(self) -> None:
        # An intentionally bad query should not raise, just return []
        result = run_query("(this_is_not_valid @@@@)", SAMPLE_CODE)
        assert result == []
