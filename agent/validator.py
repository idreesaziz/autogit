"""Code validation — syntax-check generated files before committing.

Validates Python (ast.parse), JSON (json.loads), YAML (basic structure).
Returns a list of error strings. Empty list = valid.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def validate_source(file_path: str, content: str) -> list[str]:
    """Validate generated source code based on file extension.

    Args:
        file_path: Relative path (used to determine extension).
        content: The file content to validate.

    Returns:
        List of error messages. Empty means valid.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        return validate_python(content, file_path)
    elif ext == ".json":
        return validate_json(content, file_path)
    elif ext in (".yaml", ".yml"):
        return validate_yaml(content, file_path)

    return []  # no validation for other file types


def validate_python(source: str, filename: str = "<string>") -> list[str]:
    """Check Python source for syntax errors using ast.parse.

    Returns:
        List of error messages (empty if valid).
    """
    try:
        ast.parse(source, filename=filename)
        return []
    except SyntaxError as exc:
        line_info = f" (line {exc.lineno})" if exc.lineno else ""
        return [f"SyntaxError{line_info}: {exc.msg}"]


def validate_json(source: str, filename: str = "<string>") -> list[str]:
    """Check JSON source for parse errors.

    Returns:
        List of error messages (empty if valid).
    """
    try:
        json.loads(source)
        return []
    except json.JSONDecodeError as exc:
        return [f"JSONDecodeError (line {exc.lineno}, col {exc.colno}): {exc.msg}"]


def validate_yaml(source: str, filename: str = "<string>") -> list[str]:
    """Basic YAML validation — check for tab indentation and obvious issues.

    Does NOT require PyYAML. Just catches the most common mistakes.
    """
    errors: list[str] = []

    for i, line in enumerate(source.splitlines(), 1):
        if "\t" in line:
            errors.append(f"YAML error (line {i}): tabs are not allowed in YAML")
            break  # one error is enough

    return errors


def format_errors_for_retry(file_path: str, errors: list[str]) -> str:
    """Format validation errors into a prompt section for Gemini retry.

    Returns a string that can be appended to the generation prompt.
    """
    error_text = "\n".join(f"  - {e}" for e in errors)
    return (
        f"\n\nCRITICAL: The previous version of `{file_path}` had these errors:\n"
        f"{error_text}\n\n"
        f"Fix these errors. Return the COMPLETE corrected file content."
    )
