"""Tests for agent/validator.py — Python, JSON, YAML validation."""

from __future__ import annotations

import pytest

from agent.validator import (
    validate_python,
    validate_json,
    validate_yaml,
    validate_source,
    format_errors_for_retry,
)


# ═══════════════════════════════════════════════════════════════════════
# Python Validation
# ═══════════════════════════════════════════════════════════════════════

class TestValidatePython:
    """Tests for validate_python."""

    def test_valid_code(self):
        assert validate_python("x = 1\nprint(x)\n") == []

    def test_valid_function(self):
        code = "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n"
        assert validate_python(code) == []

    def test_valid_class(self):
        code = "class Foo:\n    def bar(self):\n        pass\n"
        assert validate_python(code) == []

    def test_syntax_error_missing_colon(self):
        errors = validate_python("def broken()\n    pass\n")
        assert len(errors) == 1
        assert "SyntaxError" in errors[0]

    def test_syntax_error_unmatched_paren(self):
        errors = validate_python("x = (1 + 2\n")
        assert len(errors) == 1
        assert "SyntaxError" in errors[0]

    def test_syntax_error_includes_line_number(self):
        errors = validate_python("x = 1\ny = 2\ndef bad(\n")
        assert len(errors) == 1
        assert "line" in errors[0].lower()

    def test_empty_string_valid(self):
        assert validate_python("") == []

    def test_comment_only_valid(self):
        assert validate_python("# just a comment\n") == []

    def test_indentation_error(self):
        errors = validate_python("def foo():\npass\n")
        assert len(errors) == 1

    def test_valid_async_code(self):
        code = "async def fetch():\n    await something()\n"
        assert validate_python(code) == []


# ═══════════════════════════════════════════════════════════════════════
# JSON Validation
# ═══════════════════════════════════════════════════════════════════════

class TestValidateJson:
    """Tests for validate_json."""

    def test_valid_object(self):
        assert validate_json('{"key": "value"}') == []

    def test_valid_array(self):
        assert validate_json('[1, 2, 3]') == []

    def test_valid_nested(self):
        assert validate_json('{"a": {"b": [1, 2]}}') == []

    def test_trailing_comma(self):
        errors = validate_json('{"a": 1,}')
        assert len(errors) == 1
        assert "JSONDecodeError" in errors[0]

    def test_missing_quotes(self):
        errors = validate_json('{key: "value"}')
        assert len(errors) == 1

    def test_empty_string_invalid(self):
        errors = validate_json("")
        assert len(errors) == 1

    def test_valid_number(self):
        assert validate_json("42") == []

    def test_valid_null(self):
        assert validate_json("null") == []


# ═══════════════════════════════════════════════════════════════════════
# YAML Validation
# ═══════════════════════════════════════════════════════════════════════

class TestValidateYaml:
    """Tests for validate_yaml."""

    def test_valid_yaml(self):
        assert validate_yaml("key: value\nlist:\n  - item1\n  - item2\n") == []

    def test_tab_indentation(self):
        errors = validate_yaml("key:\n\t- value\n")
        assert len(errors) == 1
        assert "tab" in errors[0].lower()

    def test_empty_string_valid(self):
        assert validate_yaml("") == []


# ═══════════════════════════════════════════════════════════════════════
# validate_source (extension routing)
# ═══════════════════════════════════════════════════════════════════════

class TestValidateSource:
    """Tests for validate_source."""

    def test_routes_python(self):
        errors = validate_source("test.py", "def bad(\n")
        assert len(errors) == 1

    def test_routes_json(self):
        errors = validate_source("config.json", "{broken}")
        assert len(errors) == 1

    def test_routes_yaml(self):
        errors = validate_source("config.yml", "key:\n\tbad\n")
        assert len(errors) == 1

    def test_unknown_extension_passes(self):
        assert validate_source("readme.md", "anything goes") == []

    def test_valid_python_passes(self):
        assert validate_source("main.py", "print('hi')\n") == []


# ═══════════════════════════════════════════════════════════════════════
# format_errors_for_retry
# ═══════════════════════════════════════════════════════════════════════

class TestFormatErrorsForRetry:
    """Tests for format_errors_for_retry."""

    def test_includes_filename(self):
        result = format_errors_for_retry("main.py", ["SyntaxError: bad"])
        assert "main.py" in result

    def test_includes_error_messages(self):
        errors = ["SyntaxError (line 5): unexpected EOF", "SyntaxError (line 10): invalid"]
        result = format_errors_for_retry("app.py", errors)
        assert "line 5" in result
        assert "line 10" in result

    def test_includes_fix_instruction(self):
        result = format_errors_for_retry("x.py", ["error"])
        assert "Fix" in result
