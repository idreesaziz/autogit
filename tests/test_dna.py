"""Tests for agent/dna.py — AST indexer, diff engine, context renderer."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent.dna import (
    _index_python_file,
    _format_arguments,
    _diff_python_file,
    index_repo,
    diff_dna,
    load_dna,
    save_dna,
    render_dna_context,
    _carry_over_descriptions,
)


# ═══════════════════════════════════════════════════════════════════════
# AST Indexer
# ═══════════════════════════════════════════════════════════════════════

class TestIndexPythonFile:
    """Tests for _index_python_file."""

    def test_extracts_functions(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            def hello(name: str, count: int = 1) -> str:
                return name * count
        """), encoding="utf-8")
        result = _index_python_file(f)
        assert "hello" in result["functions"]
        sig = result["functions"]["hello"]["signature"]
        assert "name: str" in sig
        assert "count: int=1" in sig
        assert "-> str" in sig

    def test_extracts_async_functions(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            async def fetch(url: str, *, timeout: int = 30) -> dict:
                pass
        """), encoding="utf-8")
        result = _index_python_file(f)
        sig = result["functions"]["fetch"]["signature"]
        assert sig.startswith("async ")
        assert "timeout: int=30" in sig
        assert "-> dict" in sig

    def test_extracts_classes_and_methods(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            class MyClass(Base):
                size: int

                def __init__(self, name: str):
                    self.name = name

                def process(self) -> bool:
                    return True
        """), encoding="utf-8")
        result = _index_python_file(f)
        cls = result["classes"]["MyClass"]
        assert cls["bases"] == ["Base"]
        assert "__init__" in cls["methods"]
        assert "process" in cls["methods"]
        assert "-> bool" in cls["methods"]["process"]["signature"]
        assert cls["class_attributes"]["size"] == "int"

    def test_extracts_decorators(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            import functools

            @functools.lru_cache(maxsize=128)
            def expensive(n: int) -> int:
                return n * 2
        """), encoding="utf-8")
        result = _index_python_file(f)
        assert "functools.lru_cache(maxsize=128)" in result["functions"]["expensive"]["decorators"]

    def test_extracts_constants(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            MAX_SIZE = 100
            DEFAULT_NAME = "world"
            _private = "not a constant"
            lowercase = "also not"
        """), encoding="utf-8")
        result = _index_python_file(f)
        assert "MAX_SIZE" in result["constants"]
        assert "DEFAULT_NAME" in result["constants"]
        # lowercase names should not be captured
        assert "_private" not in result["constants"]
        assert "lowercase" not in result["constants"]

    def test_extracts_imports(self, tmp_path: Path):
        # Create a package so "mypkg" is detected as internal
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")

        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            import os
            import json
            from pathlib import Path
            from mypkg import utils
        """), encoding="utf-8")
        # We need to set _LIKELY_INTERNAL for this to work
        from agent.dna import _LIKELY_INTERNAL
        _LIKELY_INTERNAL.clear()
        _LIKELY_INTERNAL.add("mypkg")

        result = _index_python_file(f)
        assert "os" in result["external_imports"]
        assert "json" in result["external_imports"]
        assert "pathlib" in result["external_imports"]
        assert "mypkg" in result["internal_imports"]

    def test_handles_syntax_error(self, tmp_path: Path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n", encoding="utf-8")
        result = _index_python_file(f)
        assert result.get("parse_error") is True

    def test_handles_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        result = _index_python_file(f)
        assert result["functions"] == {}
        assert result["classes"] == {}
        assert result["constants"] == {}

    def test_posonly_and_kwonly_args(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            def mixed(a, b, /, c, *, d, e=5):
                pass
        """), encoding="utf-8")
        result = _index_python_file(f)
        sig = result["functions"]["mixed"]["signature"]
        assert "a, b, /, c, *, d, e=5" in sig

    def test_varargs_and_kwargs(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text(textwrap.dedent("""\
            def flex(*args: str, **kwargs: int) -> None:
                pass
        """), encoding="utf-8")
        result = _index_python_file(f)
        sig = result["functions"]["flex"]["signature"]
        assert "*args: str" in sig
        assert "**kwargs: int" in sig
        assert "-> None" in sig


# ═══════════════════════════════════════════════════════════════════════
# Full Repo Indexer
# ═══════════════════════════════════════════════════════════════════════

class TestIndexRepo:
    """Tests for index_repo."""

    def test_indexes_python_and_text_files(self, tmp_repo: Path):
        result = index_repo(tmp_repo)
        # Should find the Python file
        assert "mypackage/core.py" in result
        assert "functions" in result["mypackage/core.py"]
        # Should find the JSON config
        assert "mypackage/config.json" in result
        assert result["mypackage/config.json"]["type"] == "non-python"
        # Should find README
        assert "README.md" in result

    def test_skips_git_directory(self, tmp_repo: Path):
        git_dir = tmp_repo / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        result = index_repo(tmp_repo)
        assert not any(k.startswith(".git/") for k in result)

    def test_skips_dna_file(self, tmp_repo: Path):
        (tmp_repo / ".dna").write_text("{}", encoding="utf-8")
        result = index_repo(tmp_repo)
        assert ".dna" not in result

    def test_skips_pycache(self, tmp_repo: Path):
        cache = tmp_repo / "__pycache__"
        cache.mkdir()
        (cache / "mod.cpython-312.pyc").write_text("", encoding="utf-8")
        result = index_repo(tmp_repo)
        assert not any("__pycache__" in k for k in result)

    def test_indexes_functions_correctly(self, tmp_repo: Path):
        result = index_repo(tmp_repo)
        core = result["mypackage/core.py"]
        assert "greet" in core["functions"]
        assert "fetch_data" in core["functions"]
        assert core["functions"]["greet"]["signature"].startswith("greet(")

    def test_indexes_classes_correctly(self, tmp_repo: Path):
        result = index_repo(tmp_repo)
        core = result["mypackage/core.py"]
        assert "Processor" in core["classes"]
        proc = core["classes"]["Processor"]
        assert "run" in proc["methods"]
        assert "shutdown" in proc["methods"]


# ═══════════════════════════════════════════════════════════════════════
# Diff Engine
# ═══════════════════════════════════════════════════════════════════════

class TestDiffDna:
    """Tests for diff_dna and _diff_python_file."""

    def test_detects_added_files(self):
        old = {"a.py": {"functions": {}, "classes": {}, "constants": {}}}
        new = {
            "a.py": {"functions": {}, "classes": {}, "constants": {}},
            "b.py": {"functions": {}, "classes": {}, "constants": {}},
        }
        diff = diff_dna(old, new)
        assert "b.py" in diff["added_files"]
        assert diff["removed_files"] == []

    def test_detects_removed_files(self):
        old = {
            "a.py": {"functions": {}, "classes": {}, "constants": {}},
            "b.py": {"functions": {}, "classes": {}, "constants": {}},
        }
        new = {"a.py": {"functions": {}, "classes": {}, "constants": {}}}
        diff = diff_dna(old, new)
        assert "b.py" in diff["removed_files"]
        assert diff["added_files"] == []

    def test_detects_added_function(self):
        old = {"a.py": {"functions": {}, "classes": {}, "constants": {}}}
        new = {"a.py": {"functions": {"foo": {"signature": "foo()", "description": ""}}, "classes": {}, "constants": {}}}
        diff = diff_dna(old, new)
        assert "a.py" in diff["modified_files"]
        assert "functions.foo" in diff["modified_files"]["a.py"]["added_symbols"]

    def test_detects_removed_function(self):
        old = {"a.py": {"functions": {"foo": {"signature": "foo()", "description": ""}}, "classes": {}, "constants": {}}}
        new = {"a.py": {"functions": {}, "classes": {}, "constants": {}}}
        diff = diff_dna(old, new)
        assert "functions.foo" in diff["modified_files"]["a.py"]["removed_symbols"]

    def test_detects_changed_signature(self):
        old = {"a.py": {"functions": {"foo": {"signature": "foo(x)", "description": ""}}, "classes": {}, "constants": {}}}
        new = {"a.py": {"functions": {"foo": {"signature": "foo(x, y)", "description": ""}}, "classes": {}, "constants": {}}}
        diff = diff_dna(old, new)
        assert "functions.foo" in diff["modified_files"]["a.py"]["changed_symbols"]

    def test_unchanged_file(self):
        entry = {"functions": {"foo": {"signature": "foo()", "description": "does stuff"}}, "classes": {}, "constants": {}}
        old = {"a.py": entry}
        new = {"a.py": entry}
        diff = diff_dna(old, new)
        assert "a.py" in diff["unchanged_files"]
        assert diff["modified_files"] == {}

    def test_non_python_size_change(self):
        old = {"config.json": {"type": "non-python", "size_bytes": 10, "description": ""}}
        new = {"config.json": {"type": "non-python", "size_bytes": 20, "description": ""}}
        diff = diff_dna(old, new)
        assert "config.json" in diff["modified_files"]

    def test_non_python_unchanged(self):
        entry = {"type": "non-python", "size_bytes": 10, "description": "config"}
        diff = diff_dna({"f.json": entry}, {"f.json": entry})
        assert "f.json" in diff["unchanged_files"]

    def test_detects_new_class_method(self):
        old_cls = {
            "MyClass": {
                "signature": "",
                "bases": [],
                "methods": {"run": {"signature": "run(self)", "description": ""}},
                "description": "",
            }
        }
        new_cls = {
            "MyClass": {
                "signature": "",
                "bases": [],
                "methods": {
                    "run": {"signature": "run(self)", "description": ""},
                    "stop": {"signature": "stop(self)", "description": ""},
                },
                "description": "",
            }
        }
        old = {"a.py": {"functions": {}, "classes": old_cls, "constants": {}}}
        new = {"a.py": {"functions": {}, "classes": new_cls, "constants": {}}}
        diff = diff_dna(old, new)
        assert "classes.MyClass" in diff["modified_files"]["a.py"]["changed_symbols"]


# ═══════════════════════════════════════════════════════════════════════
# DNA I/O
# ═══════════════════════════════════════════════════════════════════════

class TestDnaIO:
    """Tests for load_dna / save_dna."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        dna = {"project": {"name": "test"}, "files": {}}
        save_dna(tmp_path, dna)
        loaded = load_dna(tmp_path)
        assert loaded == dna

    def test_load_missing_returns_empty(self, tmp_path: Path):
        assert load_dna(tmp_path) == {}

    def test_load_corrupt_returns_empty(self, tmp_path: Path):
        (tmp_path / ".dna").write_text("not json{{{", encoding="utf-8")
        assert load_dna(tmp_path) == {}

    def test_save_creates_file(self, tmp_path: Path):
        save_dna(tmp_path, {"test": True})
        assert (tmp_path / ".dna").exists()
        data = json.loads((tmp_path / ".dna").read_text(encoding="utf-8"))
        assert data["test"] is True


# ═══════════════════════════════════════════════════════════════════════
# Context Renderer
# ═══════════════════════════════════════════════════════════════════════

class TestRenderDnaContext:
    """Tests for render_dna_context."""

    def test_includes_project_name(self, sample_dna):
        text = render_dna_context(sample_dna)
        assert "test-project" in text

    def test_includes_roadmap_phases(self, sample_dna):
        text = render_dna_context(sample_dna)
        assert "Phase 1" in text
        assert "Core" in text
        assert "Phase 2" in text
        assert "Tests" in text

    def test_includes_current_phase_marker(self, sample_dna):
        text = render_dna_context(sample_dna)
        # Phase 1 should have the → marker
        lines = text.splitlines()
        phase1_lines = [l for l in lines if "Phase 1" in l]
        assert any("→" in l for l in phase1_lines)

    def test_includes_function_signatures(self, sample_dna):
        text = render_dna_context(sample_dna)
        assert "main()" in text

    def test_includes_file_purpose(self, sample_dna):
        text = render_dna_context(sample_dna)
        assert "Application entry point" in text

    def test_includes_imports(self, sample_dna):
        text = render_dna_context(sample_dna)
        assert "sys" in text

    def test_empty_dna_returns_empty_string(self):
        text = render_dna_context({})
        assert text == ""

    def test_includes_non_python_description(self, sample_dna):
        text = render_dna_context(sample_dna)
        assert "Project readme" in text


# ═══════════════════════════════════════════════════════════════════════
# Carry-Over Descriptions
# ═══════════════════════════════════════════════════════════════════════

class TestCarryOverDescriptions:
    """Tests for _carry_over_descriptions."""

    def test_carries_over_function_description(self):
        old = {
            "functions": {"foo": {"signature": "foo()", "description": "Does foo"}},
            "classes": {},
            "constants": {},
        }
        new = {
            "functions": {"foo": {"signature": "foo()", "description": ""}},
            "classes": {},
            "constants": {},
        }
        _carry_over_descriptions(old, new)
        assert new["functions"]["foo"]["description"] == "Does foo"

    def test_does_not_carry_over_if_signature_changed(self):
        old = {
            "functions": {"foo": {"signature": "foo(x)", "description": "Does foo"}},
            "classes": {},
            "constants": {},
        }
        new = {
            "functions": {"foo": {"signature": "foo(x, y)", "description": ""}},
            "classes": {},
            "constants": {},
        }
        _carry_over_descriptions(old, new)
        assert new["functions"]["foo"]["description"] == ""

    def test_carries_over_class_and_method_descriptions(self):
        old = {
            "functions": {},
            "classes": {
                "Bar": {
                    "description": "A bar",
                    "methods": {"run": {"signature": "run(self)", "description": "Runs it"}},
                }
            },
            "constants": {},
        }
        new = {
            "functions": {},
            "classes": {
                "Bar": {
                    "description": "",
                    "methods": {"run": {"signature": "run(self)", "description": ""}},
                }
            },
            "constants": {},
        }
        _carry_over_descriptions(old, new)
        assert new["classes"]["Bar"]["description"] == "A bar"
        assert new["classes"]["Bar"]["methods"]["run"]["description"] == "Runs it"

    def test_carries_over_purpose(self):
        old = {"purpose": "Entry point", "functions": {}, "classes": {}, "constants": {}}
        new = {"functions": {}, "classes": {}, "constants": {}}
        _carry_over_descriptions(old, new)
        assert new["purpose"] == "Entry point"
