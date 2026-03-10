"""Tests for agent/test_runner.py — runner detection and test execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.test_runner import detect_runner, run_tests, format_test_errors_for_retry, TestResult


class TestDetectRunner:
    """Auto-detection of test runners."""

    def test_detects_pytest_from_tests_dir(self, tmp_path: Path):
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_core.py").write_text("def test_x(): pass", encoding="utf-8")
        assert detect_runner(tmp_path) == "pytest"

    def test_detects_pytest_from_pytest_ini(self, tmp_path: Path):
        (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")
        assert detect_runner(tmp_path) == "pytest"

    def test_detects_pytest_from_pyproject_toml(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['tests']", encoding="utf-8"
        )
        assert detect_runner(tmp_path) == "pytest"

    def test_detects_pytest_from_root_test_file(self, tmp_path: Path):
        (tmp_path / "test_app.py").write_text("def test_y(): pass", encoding="utf-8")
        assert detect_runner(tmp_path) == "pytest"

    def test_detects_npm_from_package_json(self, tmp_path: Path):
        pkg = {"scripts": {"test": "jest"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        assert detect_runner(tmp_path) == "npm"

    def test_ignores_npm_default_test_script(self, tmp_path: Path):
        pkg = {"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}
        (tmp_path / "package.json").write_text(json.dumps(pkg), encoding="utf-8")
        assert detect_runner(tmp_path) is None

    def test_detects_cargo(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'", encoding="utf-8")
        assert detect_runner(tmp_path) == "cargo"

    def test_detects_go(self, tmp_path: Path):
        (tmp_path / "main_test.go").write_text("package main", encoding="utf-8")
        assert detect_runner(tmp_path) == "go"

    def test_returns_none_for_empty_repo(self, tmp_path: Path):
        assert detect_runner(tmp_path) is None


class TestRunTests:
    """Running test suites."""

    def test_no_runner_skips(self, tmp_path: Path):
        result = run_tests(tmp_path)
        assert result.passed is True
        assert result.runner == "none"
        assert result.tests_found is False

    def test_runs_pytest_on_real_tests(self, tmp_path: Path):
        # Create a simple passing test
        (tmp_path / "test_simple.py").write_text(
            "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
        )
        result = run_tests(tmp_path, runner="pytest")
        assert result.passed is True
        assert result.runner == "pytest"
        assert result.tests_found is True

    def test_failing_test_detected(self, tmp_path: Path):
        (tmp_path / "test_fail.py").write_text(
            "def test_bad():\n    assert False\n", encoding="utf-8"
        )
        result = run_tests(tmp_path, runner="pytest")
        assert result.passed is False
        assert result.exit_code != 0

    def test_syntax_error_in_test_fails(self, tmp_path: Path):
        (tmp_path / "test_broken.py").write_text(
            "def test_x(\n    pass\n", encoding="utf-8"
        )
        result = run_tests(tmp_path, runner="pytest")
        assert result.passed is False

    def test_missing_runner_executable(self, tmp_path: Path):
        # Use a runner that doesn't exist
        result = run_tests(tmp_path, runner="nonexistent_runner_xyz")
        assert result.passed is True  # shouldn't block
        assert result.tests_found is False


class TestFormatTestErrors:
    """Formatting test failures for Gemini retry."""

    def test_includes_runner_name(self):
        result = TestResult(passed=False, runner="pytest", output="FAILED test_x", exit_code=1)
        text = format_test_errors_for_retry(result)
        assert "pytest" in text

    def test_includes_output(self):
        result = TestResult(passed=False, runner="npm", output="TypeError: x is undefined", exit_code=1)
        text = format_test_errors_for_retry(result)
        assert "TypeError: x is undefined" in text

    def test_includes_fix_instruction(self):
        result = TestResult(passed=False, runner="pytest", output="FAILED", exit_code=1)
        text = format_test_errors_for_retry(result)
        assert "fix" in text.lower() or "Fix" in text
