"""Code generation — produce / modify files and commit messages via Gemini."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from config import MAX_FILE_CHARS

console = Console()


def generate_file_content(
    task: str,
    file_path: str,
    current_content: str | None,
    project_context: str,
    gemini_call,
) -> str:
    """Ask Gemini to generate or update a single file.

    Args:
        task: Description of the improvement to make.
        file_path: Relative path of the file within the repo.
        current_content: Existing file content (None if creating new).
        project_context: Summary of the project state + tree.
        gemini_call: Rate-limited Gemini call function.

    Returns:
        The complete file content as a string.
    """
    if current_content:
        file_section = (
            f"Current content of `{file_path}` (may be truncated):\n"
            f"```\n{current_content[:MAX_FILE_CHARS]}\n```"
        )
        action = "Modify"
    else:
        file_section = f"`{file_path}` does not exist yet."
        action = "Create"

    prompt = f"""{action} the file `{file_path}` for this task:
{task}

{file_section}

Project context:
{project_context}

Rules:
- Return ONLY the complete file content
- No markdown fences, no explanation, no commentary
- The code must be production-quality and idiomatic
- Include appropriate imports and error handling
- If modifying, preserve existing functionality unless the task requires changing it"""

    raw = gemini_call(
        prompt,
        system="You are an expert software engineer. Output raw file content only.",
    )
    # Strip any accidental fences Gemini might add
    return _strip_fences(raw)


def generate_commit_message(
    task: str,
    files_changed: list[str],
    gemini_call,
) -> str:
    """Ask Gemini for a conventional commit message.

    Returns:
        A single-line conventional commit message (e.g. 'feat: add config parser').
    """
    prompt = f"""Generate a single-line git commit message for this change.

Task: {task}
Files changed: {', '.join(files_changed)}

Rules:
- Use conventional commit format: <type>: <short description>
- Types: feat, fix, docs, refactor, test, chore, style
- Write like a real developer — vary your wording naturally
- Sometimes be terse ("fix: handle null case"), sometimes slightly more descriptive
- NO emojis, NO unicode symbols, NO special characters — plain ASCII only
- Lowercase, no period at the end
- Do NOT start with generic filler like "implement" or "add support for" every time
- Vary verbs: add, wire up, hook in, set up, drop, rework, clean up, flesh out, etc.

Return ONLY the commit message line. No quotes, no explanation."""

    raw = gemini_call(
        prompt,
        system="You write terse, natural-sounding git commit messages like a real developer. Never use emojis.",
    )
    # Take just the first non-empty line and strip any emojis/non-ASCII
    for line in raw.strip().splitlines():
        line = line.strip().strip('"').strip("'")
        # Remove any non-ASCII characters (emojis, unicode symbols)
        line = line.encode("ascii", errors="ignore").decode("ascii").strip()
        if line:
            return line
    return "chore: update project files"


def parse_task_plan(raw: str) -> dict[str, Any]:
    """Parse a Gemini task-plan response into a dict.

    Expected keys: task, rationale, files_to_create, files_to_modify
    """
    cleaned = _strip_fences(raw)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Try to extract JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    console.print("[yellow]Warning: could not parse task plan JSON, using fallback.[/yellow]")
    return {
        "task": "Improve project documentation",
        "rationale": "Fallback task after JSON parse failure",
        "files_to_create": [],
        "files_to_modify": ["README.md"],
    }


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from Gemini output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
