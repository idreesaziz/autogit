"""Tests for agent/coder.py — parse_task_plan, _strip_fences."""

from __future__ import annotations

import pytest

from agent.coder import parse_task_plan, _strip_fences


# ═══════════════════════════════════════════════════════════════════════
# _strip_fences
# ═══════════════════════════════════════════════════════════════════════

class TestStripFences:
    """Tests for _strip_fences."""

    def test_strips_python_fences(self):
        text = '```python\nprint("hello")\n```'
        assert _strip_fences(text) == 'print("hello")'

    def test_strips_json_fences(self):
        text = '```json\n{"key": "val"}\n```'
        assert _strip_fences(text) == '{"key": "val"}'

    def test_strips_bare_fences(self):
        text = '```\nsome content\n```'
        assert _strip_fences(text) == "some content"

    def test_no_fences_unchanged(self):
        text = 'no fences here'
        assert _strip_fences(text) == "no fences here"

    def test_strips_whitespace(self):
        text = '  \n```\ncontent\n```\n  '
        assert _strip_fences(text) == "content"

    def test_empty_string(self):
        assert _strip_fences("") == ""

    def test_only_opening_fence(self):
        text = '```python\ncontent here'
        result = _strip_fences(text)
        assert "```" not in result
        assert "content here" in result

    def test_only_closing_fence(self):
        text = 'content here\n```'
        result = _strip_fences(text)
        assert result == "content here"


# ═══════════════════════════════════════════════════════════════════════
# parse_task_plan
# ═══════════════════════════════════════════════════════════════════════

class TestParseTaskPlan:
    """Tests for parse_task_plan."""

    def test_parses_clean_json(self):
        raw = '{"task": "Add tests", "rationale": "Testing is good", "files_to_create": ["tests/test.py"], "files_to_modify": []}'
        result = parse_task_plan(raw)
        assert result["task"] == "Add tests"
        assert result["files_to_create"] == ["tests/test.py"]

    def test_parses_fenced_json(self):
        raw = '```json\n{"task": "Fix bug", "rationale": "It was broken", "files_to_create": [], "files_to_modify": ["main.py"]}\n```'
        result = parse_task_plan(raw)
        assert result["task"] == "Fix bug"
        assert result["files_to_modify"] == ["main.py"]

    def test_extracts_json_from_surrounding_text(self):
        raw = 'Here is my plan:\n\n{"task": "Refactor", "rationale": "Clean up", "files_to_create": [], "files_to_modify": ["app.py"]}\n\nDone!'
        result = parse_task_plan(raw)
        assert result["task"] == "Refactor"

    def test_fallback_on_garbage(self):
        raw = "this is not json at all!!!"
        result = parse_task_plan(raw)
        # Should return fallback
        assert "task" in result
        assert result["rationale"].startswith("Fallback")

    def test_fallback_on_empty(self):
        result = parse_task_plan("")
        assert "task" in result
        assert result["rationale"].startswith("Fallback")

    def test_handles_multiline_json(self):
        raw = """{
  "task": "Add logging",
  "rationale": "Better debugging",
  "files_to_create": [],
  "files_to_modify": ["service.py"]
}"""
        result = parse_task_plan(raw)
        assert result["task"] == "Add logging"

    def test_handles_nested_braces(self):
        raw = '{"task": "Config", "rationale": "Need config", "files_to_create": [], "files_to_modify": [], "meta": {"nested": true}}'
        result = parse_task_plan(raw)
        assert result["task"] == "Config"

    def test_parses_with_extra_keys(self):
        raw = '{"task": "Optimize", "rationale": "Speed", "files_to_create": [], "files_to_modify": [], "priority": "high"}'
        result = parse_task_plan(raw)
        assert result["task"] == "Optimize"
        assert result["priority"] == "high"
