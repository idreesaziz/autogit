"""Shared pytest fixtures for autogit tests."""

from __future__ import annotations

import os
import tempfile

# Set dummy env vars BEFORE any project module is imported,
# so config.py doesn't call sys.exit(1).
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("GITHUB_USERNAME", "test-user")
os.environ.setdefault("LOCAL_REPOS_DIR", tempfile.mkdtemp(prefix="autogit_test_"))

import json
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo directory with a Python file and __init__.py."""
    # Create a package
    pkg = tmp_path / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # A sample Python source file
    (pkg / "core.py").write_text(textwrap.dedent("""\
        \"\"\"Core module.\"\"\"

        import os
        import json
        from pathlib import Path

        from mypackage import utils

        MAX_RETRIES: int = 3
        DEFAULT_NAME = "world"

        def greet(name: str, loud: bool = False) -> str:
            \"\"\"Return a greeting.\"\"\"
            msg = f"Hello, {name}!"
            if loud:
                return msg.upper()
            return msg

        async def fetch_data(url: str, *, timeout: int = 30) -> dict:
            \"\"\"Fetch JSON from a URL.\"\"\"
            return {}

        class Processor:
            \"\"\"Processes items in a queue.\"\"\"
            queue_size: int

            def __init__(self, name: str, workers: int = 4):
                self.name = name
                self.workers = workers

            def run(self) -> None:
                pass

            async def shutdown(self, force: bool = False) -> bool:
                return True
    """), encoding="utf-8")

    # A non-Python file
    (pkg / "config.json").write_text('{"key": "value"}', encoding="utf-8")

    # A README at root
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def sample_dna() -> dict:
    """Return a sample DNA dict for testing the renderer and diff engine."""
    return {
        "project": {
            "name": "test-project",
            "purpose": "A test project for unit tests",
            "tech_stack": ["Python"],
            "roadmap": [
                {"phase": 1, "title": "Core", "description": "Build basics", "status": "in-progress"},
                {"phase": 2, "title": "Tests", "description": "Add tests", "status": "not-started"},
            ],
            "current_phase": 1,
        },
        "files": {
            "main.py": {
                "functions": {
                    "main": {
                        "signature": "main()",
                        "decorators": [],
                        "line": 5,
                        "description": "Entry point",
                    }
                },
                "classes": {},
                "constants": {},
                "internal_imports": [],
                "external_imports": ["sys"],
                "purpose": "Application entry point",
            },
            "README.md": {
                "type": "non-python",
                "extension": ".md",
                "size_bytes": 50,
                "description": "Project readme",
            },
        },
    }


@pytest.fixture
def mock_gemini_call():
    """Return a mock gemini_call that returns canned JSON responses."""
    responses = []

    def _call(prompt: str, system: str = "") -> str:
        if responses:
            return responses.pop(0)
        return '{"description": "A test file"}'

    _call.responses = responses
    return _call
