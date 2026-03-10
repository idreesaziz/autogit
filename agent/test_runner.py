"""Test runner — detect and execute test suites in managed repos.

Supports auto-detection of:
  - pytest (Python)
  - npm test (Node.js)
  - cargo test (Rust)
  - go test (Go)
Falls back gracefully if no test suite is found.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()

# Maximum output to capture (prevent runaway test suites from filling memory)
_MAX_OUTPUT_CHARS = 8_000


@dataclass
class TestResult:
    """Outcome of a test suite run."""
    __test__ = False  # prevent pytest from collecting this as a test class

    passed: bool
    runner: str          # e.g. "pytest", "npm", "none"
    output: str = ""     # stdout + stderr (truncated)
    exit_code: int = 0
    tests_found: bool = True


def detect_runner(repo_path: Path) -> str | None:
    """Auto-detect which test runner to use for a repo.

    Returns the runner name or None if no test suite is detected.
    """
    # Python — pytest / unittest
    if (repo_path / "pytest.ini").exists():
        return "pytest"
    if (repo_path / "pyproject.toml").exists():
        try:
            text = (repo_path / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
            if "pytest" in text or "[tool.pytest" in text:
                return "pytest"
        except OSError:
            pass
    if (repo_path / "setup.cfg").exists():
        try:
            text = (repo_path / "setup.cfg").read_text(encoding="utf-8", errors="replace")
            if "[tool:pytest]" in text:
                return "pytest"
        except OSError:
            pass
    # Check for tests/ directory with Python files
    tests_dir = repo_path / "tests"
    if tests_dir.is_dir():
        py_tests = list(tests_dir.glob("test_*.py")) + list(tests_dir.glob("*_test.py"))
        if py_tests:
            return "pytest"
    # Check for test_*.py at repo root
    if list(repo_path.glob("test_*.py")):
        return "pytest"

    # Node.js — npm test
    pkg_json = repo_path / "package.json"
    if pkg_json.exists():
        try:
            import json
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            scripts = pkg.get("scripts", {})
            if "test" in scripts and scripts["test"] != 'echo "Error: no test specified" && exit 1':
                return "npm"
        except (OSError, ValueError):
            pass

    # Rust — cargo test
    if (repo_path / "Cargo.toml").exists():
        return "cargo"

    # Go — go test
    if list(repo_path.glob("*_test.go")) or list(repo_path.glob("**/*_test.go")):
        return "go"

    return None


def run_tests(repo_path: Path, runner: str | None = None) -> TestResult:
    """Run the test suite for a repo and return the result.

    Args:
        repo_path: Absolute path to the repo root.
        runner: Override the auto-detected runner. Pass "auto" or None to detect.

    Returns:
        TestResult with pass/fail status and output.
    """
    if runner is None or runner == "auto":
        runner = detect_runner(repo_path)

    if runner is None:
        return TestResult(
            passed=True,
            runner="none",
            output="No test suite detected — skipping.",
            tests_found=False,
        )

    cmd = _build_command(runner)
    console.print(f"  [dim]Running {runner} tests…[/dim]")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=120,  # 2-minute timeout for test suites
            shell=(runner == "npm"),  # npm needs shell on Windows
        )

        output = (result.stdout + "\n" + result.stderr).strip()
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n… (truncated)"

        passed = result.returncode == 0
        return TestResult(
            passed=passed,
            runner=runner,
            output=output,
            exit_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            runner=runner,
            output="Tests timed out after 120 seconds.",
            exit_code=-1,
        )
    except FileNotFoundError:
        return TestResult(
            passed=True,  # don't block on missing runner executable
            runner=runner,
            output=f"Runner '{runner}' not found on PATH — skipping tests.",
            tests_found=False,
        )
    except OSError as exc:
        return TestResult(
            passed=True,  # don't block on OS errors
            runner=runner,
            output=f"Could not run tests: {exc}",
            tests_found=False,
        )


def format_test_errors_for_retry(result: TestResult) -> str:
    """Format test failures into a prompt section for Gemini to fix.

    Args:
        result: A failed TestResult.

    Returns:
        Markdown section describing the failures.
    """
    return (
        f"\n\n## Test Failures ({result.runner})\n"
        f"The test suite failed (exit code {result.exit_code}). "
        f"Fix the code so all tests pass.\n\n"
        f"```\n{result.output}\n```\n\n"
        f"IMPORTANT: Only fix the SOURCE code, not the tests. "
        f"The tests define the expected behavior.\n"
    )


def _build_command(runner: str) -> list[str]:
    """Build the subprocess command for a given runner."""
    if runner == "pytest":
        return ["python", "-m", "pytest", "--tb=short", "-q"]
    if runner == "npm":
        return ["npm", "test"]
    if runner == "cargo":
        return ["cargo", "test"]
    if runner == "go":
        return ["go", "test", "./..."]
    return [runner]
