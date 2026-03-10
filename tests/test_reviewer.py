"""Tests for agent/reviewer.py — self-review and retry formatting."""

from __future__ import annotations

import json

import pytest

from agent.reviewer import self_review, format_review_for_retry


class TestSelfReview:
    """Tests for the self_review function."""

    def test_approved_when_lgtm(self):
        def mock_call(prompt, system=""):
            return json.dumps({"approved": True, "issues": []})

        result = self_review("Add tests", {"test.py": "pass"}, "DNA context", mock_call)
        assert result["approved"] is True
        assert result["issues"] == []

    def test_not_approved_with_issues(self):
        def mock_call(prompt, system=""):
            return json.dumps({
                "approved": False,
                "issues": [
                    {"file": "main.py", "problem": "broken import", "fix": "remove import foo"}
                ],
            })

        result = self_review("Add feature", {"main.py": "import foo"}, "DNA", mock_call)
        assert result["approved"] is False
        assert len(result["issues"]) == 1
        assert result["issues"][0]["file"] == "main.py"

    def test_handles_fenced_json(self):
        def mock_call(prompt, system=""):
            return '```json\n{"approved": true, "issues": []}\n```'

        result = self_review("Task", {"a.py": "x"}, "DNA", mock_call)
        assert result["approved"] is True

    def test_approves_on_parse_error(self):
        def mock_call(prompt, system=""):
            return "This is not JSON at all"

        result = self_review("Task", {"a.py": "x"}, "DNA", mock_call)
        # Should approve by default on parse failure
        assert result["approved"] is True

    def test_approves_on_exception(self):
        def mock_call(prompt, system=""):
            raise RuntimeError("API error")

        # Should not raise, should approve by default
        result = self_review("Task", {"a.py": "x"}, "DNA", mock_call)
        assert result["approved"] is True

    def test_multiple_issues(self):
        def mock_call(prompt, system=""):
            return json.dumps({
                "approved": False,
                "issues": [
                    {"file": "a.py", "problem": "p1", "fix": "f1"},
                    {"file": "b.py", "problem": "p2", "fix": "f2"},
                    {"file": "c.py", "problem": "p3", "fix": "f3"},
                ],
            })

        result = self_review("Task", {"a.py": "x"}, "DNA", mock_call)
        assert len(result["issues"]) == 3


class TestFormatReviewForRetry:
    """Tests for format_review_for_retry."""

    def test_includes_problem(self):
        issues = [{"file": "main.py", "problem": "broken import", "fix": "fix it"}]
        text = format_review_for_retry(issues)
        assert "broken import" in text

    def test_includes_file_name(self):
        issues = [{"file": "utils.py", "problem": "bug", "fix": "fix"}]
        text = format_review_for_retry(issues)
        assert "utils.py" in text

    def test_includes_fix_instruction(self):
        issues = [{"file": "a.py", "problem": "p", "fix": "do this"}]
        text = format_review_for_retry(issues)
        assert "do this" in text

    def test_multiple_issues_numbered(self):
        issues = [
            {"file": "a.py", "problem": "p1", "fix": "f1"},
            {"file": "b.py", "problem": "p2", "fix": "f2"},
        ]
        text = format_review_for_retry(issues)
        assert "1." in text
        assert "2." in text

    def test_empty_issues(self):
        text = format_review_for_retry([])
        assert isinstance(text, str)
