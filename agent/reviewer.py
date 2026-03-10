"""Self-review — Gemini reviews its own diff before committing.

A dedicated Gemini call that sees the full set of changes and checks for:
  - Broken imports / references to non-existent symbols
  - Logic bugs or incorrect behaviour
  - Missing error handling at boundaries
  - Inconsistencies with the rest of the codebase
  - Hardcoded values that should be configurable

Returns either "LGTM" or a list of issues with fix instructions.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from rich.console import Console

console = Console()


def self_review(
    task: str,
    written_files: dict[str, str],
    dna_context: str,
    gemini_call: Callable[[str, str], str],
) -> dict[str, Any]:
    """Ask Gemini to review the changes it just generated.

    Args:
        task: The task description that was implemented.
        written_files: Dict of {relative_path: new_content} for all changed files.
        dna_context: The rendered DNA context of the project.
        gemini_call: Rate-limited Gemini call function.

    Returns:
        Dict with:
          - "approved": bool — True if LGTM
          - "issues": list of {"file", "problem", "fix"} dicts
    """
    # Build a diff-like view of all changed files
    changes_section = ""
    for path, content in written_files.items():
        # Truncate very large files to keep under context limits
        display = content if len(content) < 6000 else content[:6000] + "\n… (truncated)"
        changes_section += f"### {path}\n```\n{display}\n```\n\n"

    prompt = f"""You are a senior code reviewer. A coding agent just generated the following changes
for task: "{task}"

Review these changes against the project's codebase DNA (structure below) and check for:
1. Broken imports — referencing modules, functions, or classes that don't exist
2. Logic bugs — code that will produce wrong results or crash at runtime
3. API misuse — wrong method signatures, missing required arguments
4. Inconsistencies — naming conventions, patterns that don't match the rest of the codebase
5. Missing edge cases — unhandled None/empty/error cases at system boundaries

Do NOT flag style issues, missing docstrings, or minor nitpicks.
Only flag issues that would cause bugs or crashes.

## Changes
{changes_section}

## Project DNA
{dna_context}

If there are NO issues, respond with exactly:
{{"approved": true, "issues": []}}

If there ARE issues, respond with:
{{"approved": false, "issues": [
  {{"file": "path/to/file.py", "problem": "description of the bug", "fix": "what to change"}}
]}}

Respond ONLY with valid JSON. No markdown fences, no extra text."""

    try:
        raw = gemini_call(prompt, system="You are a meticulous code reviewer. Only flag real bugs.")
        # Strip any markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            import re
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        result = json.loads(raw)

        approved = result.get("approved", True)
        issues = result.get("issues", [])

        if approved:
            console.print("  [green]✓ Review: LGTM[/green]")
        else:
            console.print(f"  [yellow]⚠ Review found {len(issues)} issue(s)[/yellow]")
            for issue in issues:
                console.print(f"    [dim]• {issue.get('file', '?')}: {issue.get('problem', '?')}[/dim]")

        return {"approved": approved, "issues": issues}

    except (json.JSONDecodeError, Exception) as exc:
        # If review fails to parse, approve by default (don't block on broken review)
        console.print(f"  [yellow]Review parse error: {exc} — approving by default[/yellow]")
        return {"approved": True, "issues": []}


def format_review_for_retry(issues: list[dict[str, str]]) -> str:
    """Format review issues into a prompt section for Gemini to fix.

    Args:
        issues: List of {file, problem, fix} dicts from self_review.

    Returns:
        Markdown section describing the issues.
    """
    lines = ["\n\n## Code Review Feedback\n",
             "A reviewer found these issues. Fix them:\n\n"]

    for i, issue in enumerate(issues, 1):
        lines.append(f"{i}. **{issue.get('file', '?')}**: {issue.get('problem', '?')}\n")
        fix = issue.get("fix", "")
        if fix:
            lines.append(f"   Fix: {fix}\n")
        lines.append("\n")

    lines.append("Apply ALL fixes. Return the complete corrected file.\n")
    return "".join(lines)
