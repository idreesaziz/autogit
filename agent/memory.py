"""Persistent agent memory — read/write .agent_state.json per repo."""

import json
from datetime import date
from pathlib import Path
from typing import Any

_STATE_FILENAME = ".agent_state.json"


def _state_path(repo_local_path: str | Path) -> Path:
    return Path(repo_local_path) / _STATE_FILENAME


def load_state(repo_local_path: str | Path) -> dict[str, Any]:
    """Read and parse .agent_state.json from a repo directory.

    Raises FileNotFoundError if the state file doesn't exist.
    """
    path = _state_path(repo_local_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No agent state file found at {path}. "
            "Run initialize_state() first or choose 'Create a new repo'."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(repo_local_path: str | Path, state: dict[str, Any]) -> None:
    """Write state back to .agent_state.json, pretty-printed."""
    path = _state_path(repo_local_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def initialize_state(
    repo_name: str,
    repo_url: str,
    description: str,
    tech_stack: list[str],
) -> dict[str, Any]:
    """Create a fresh agent state dict for a brand-new repo."""
    return {
        "repo_name": repo_name,
        "repo_url": repo_url,
        "created_date": date.today().isoformat(),
        "project_description": description,
        "tech_stack": tech_stack,
        "current_phase": "initial scaffold",
        "maturity_level": "early",
        "completed_milestones": [],
        "planned_next_steps": [
            "flesh out core functionality",
            "add README with usage instructions",
            "add basic tests",
        ],
        "session_log": [],
        "open_questions": [],
        "decisions_log": [],
    }


def append_session_log(
    state: dict[str, Any],
    summary: str,
    files_changed: list[str],
    requests_used: int,
) -> dict[str, Any]:
    """Append a new session entry and return the updated state."""
    entry = {
        "date": date.today().isoformat(),
        "summary": summary,
        "files_changed": files_changed,
        "requests_used": requests_used,
    }
    state.setdefault("session_log", []).append(entry)
    return state
