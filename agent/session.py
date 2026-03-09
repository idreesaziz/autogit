"""Core agent loop — read state → decide → act → commit → push."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel

from config import (
    GEMINI_API_KEY,
    GEMINI_PRIMARY_MODEL,
    GEMINI_FALLBACK_MODEL,
    MAX_REQUESTS_PER_SESSION,
    REQUEST_DELAY_SECONDS,
    MAX_CONTEXT_CHARS,
    MAX_FILE_CHARS,
    MAX_CONTEXT_FILES,
)
from agent.memory import load_state, save_state, append_session_log
from agent.coder import generate_file_content, generate_commit_message, parse_task_plan
from github_ops.git_ops import get_recent_files, get_file_tree, commit_and_push
from github_ops.api import update_last_session

console = Console()

# Initialise Gemini client once
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Centralised Gemini call with rate limiting ───────────────────────

def _make_gemini_call(
    prompt: str,
    system: str = "",
    session_tracker: dict[str, int] | None = None,
) -> str:
    """Rate-limited Gemini request with retry and fallback.

    - Sleeps REQUEST_DELAY_SECONDS before each call
    - Tracks requests_used in session_tracker
    - Retries on 429 (rate limit) up to 3 times with 60s wait
    - Falls back to GEMINI_FALLBACK_MODEL on quota errors
    - Retries once on JSON parse failures with explicit instruction
    """
    tracker = session_tracker or {"requests_used": 0}

    if tracker["requests_used"] >= MAX_REQUESTS_PER_SESSION - 3:
        console.print("[bold yellow]⚠ Request budget nearly exhausted — aborting Gemini call.[/bold yellow]")
        raise BudgetExhaustedError("Fewer than 3 requests remaining in session budget.")

    time.sleep(REQUEST_DELAY_SECONDS)
    tracker["requests_used"] = tracker.get("requests_used", 0) + 1
    console.print(
        f"[dim]Gemini request {tracker['requests_used']}/{MAX_REQUESTS_PER_SESSION}…[/dim]"
    )

    model_name = GEMINI_PRIMARY_MODEL
    for attempt in range(4):  # initial + 3 retries
        try:
            config = types.GenerateContentConfig(
                system_instruction=system or None,
            )
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                if model_name == GEMINI_PRIMARY_MODEL and attempt == 0:
                    console.print(
                        f"[yellow]Primary model quota hit — falling back to {GEMINI_FALLBACK_MODEL}[/yellow]"
                    )
                    model_name = GEMINI_FALLBACK_MODEL
                    continue
                if attempt < 3:
                    console.print(f"[yellow]Rate limited — waiting 60s (attempt {attempt + 1}/3)…[/yellow]")
                    time.sleep(60)
                    continue
            raise GeminiCallError(f"Gemini API error: {exc}") from exc

    raise GeminiCallError("Gemini call failed after all retries.")


class BudgetExhaustedError(Exception):
    """Raised when the session request budget is nearly exhausted."""


class GeminiCallError(Exception):
    """Raised when a Gemini API call fails irrecoverably."""


# ── Main session runner ──────────────────────────────────────────────

def run_session(
    repo_local_path: str | Path,
    mode: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Run a single improvement session on a repo.

    Args:
        repo_local_path: Absolute path to the local clone.
        mode: "auto" or "manual" (both run the same loop currently).
        force: If True, skip the "already ran today" check.

    Returns:
        Session summary dict with task, files_changed, commit_message, requests_used.
    """
    repo_local_path = Path(repo_local_path)
    tracker: dict[str, int] = {"requests_used": 0}

    def gemini_call(prompt: str, system: str = "") -> str:
        return _make_gemini_call(prompt, system, tracker)

    # ── Step 1: Load context ─────────────────────────────────────────
    console.print("[bold]Step 1/6:[/bold] Loading project context…")

    try:
        state = load_state(repo_local_path)
    except FileNotFoundError:
        console.print("[yellow]No .agent_state.json found — cannot run session.[/yellow]")
        raise

    # Idempotency: skip if already ran today (unless forced)
    if not force:
        last_log = state.get("session_log", [])
        if last_log and last_log[-1].get("date") == date.today().isoformat():
            console.print("[yellow]Session already ran today — skipping.[/yellow]")
            return {
                "task": "(skipped — already ran today)",
                "files_changed": [],
                "commit_message": "",
                "requests_used": 0,
            }

    recent_files = get_recent_files(repo_local_path)
    file_tree = get_file_tree(repo_local_path)

    # Build context string: state + tree + file contents
    context_parts: list[str] = [
        "## Agent State\n```json\n" + _safe_json(state) + "\n```\n",
        "## File Tree\n```\n" + file_tree + "\n```\n",
    ]
    chars_used = sum(len(p) for p in context_parts)

    context_parts.append("## Recent File Contents\n")
    files_included = 0
    for rel_path in recent_files:
        if files_included >= MAX_CONTEXT_FILES:
            break
        full_path = repo_local_path / rel_path
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        truncated = content[:MAX_FILE_CHARS]
        chunk = f"### {rel_path}\n```\n{truncated}\n```\n"
        if chars_used + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk)
        chars_used += len(chunk)
        files_included += 1

    project_context = "".join(context_parts)

    # ── Step 2: Decide what to do ────────────────────────────────────
    console.print("[bold]Step 2/6:[/bold] Deciding next improvement…")

    decide_prompt = f"""You are an autonomous software development agent maintaining a GitHub repository.
You will be given the current project state and recent code. Your job is to decide
the single most valuable incremental improvement to make today.

Rules:
- Make ONE focused improvement (not multiple unrelated changes)
- The change must be implementable in this session
- Prefer improvements that make the project more useful to real developers
- Consider: adding tests, improving docs, fixing edge cases, adding a feature, improving error messages
- Output a JSON plan with: {{ "task": "...", "rationale": "...", "files_to_create": [...], "files_to_modify": [...] }}

Return ONLY valid JSON, no markdown fences, no extra text.

{project_context}"""

    raw_plan = gemini_call(decide_prompt, system="You are a senior software architect.")
    plan = parse_task_plan(raw_plan)

    # Retry once with explicit JSON instruction if parse fell back
    if plan.get("rationale", "").startswith("Fallback"):
        retry_prompt = decide_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation."
        raw_plan = gemini_call(retry_prompt, system="You are a senior software architect.")
        plan = parse_task_plan(raw_plan)

    task_desc = plan.get("task", "General improvement")
    console.print(f"  [cyan]Task:[/cyan] {task_desc}")

    # ── Step 3: Implement ────────────────────────────────────────────
    console.print("[bold]Step 3/6:[/bold] Generating code changes…")

    all_files = list(plan.get("files_to_create", [])) + list(plan.get("files_to_modify", []))
    if not all_files:
        all_files = ["README.md"]  # sensible fallback

    written_files: list[str] = []
    for rel_path in all_files:
        full_path = repo_local_path / rel_path
        current = None
        if full_path.exists():
            try:
                current = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

        try:
            new_content = generate_file_content(
                task=task_desc,
                file_path=rel_path,
                current_content=current,
                project_context=project_context,
                gemini_call=gemini_call,
            )
        except (BudgetExhaustedError, GeminiCallError) as exc:
            console.print(f"[yellow]Stopping file generation: {exc}[/yellow]")
            break

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(new_content, encoding="utf-8")
        written_files.append(rel_path)
        console.print(f"  [green]✓[/green] {rel_path}")

    if not written_files:
        console.print("[yellow]No files were written — aborting session.[/yellow]")
        return {
            "task": task_desc,
            "files_changed": [],
            "commit_message": "",
            "requests_used": tracker["requests_used"],
        }

    # ── Step 4: Generate commit message ──────────────────────────────
    console.print("[bold]Step 4/6:[/bold] Generating commit message…")

    try:
        commit_msg = generate_commit_message(task_desc, written_files, gemini_call)
    except (BudgetExhaustedError, GeminiCallError):
        commit_msg = "chore: automated improvement"

    console.print(f"  [dim]{commit_msg}[/dim]")

    # ── Step 5: Update memory ────────────────────────────────────────
    console.print("[bold]Step 5/6:[/bold] Updating agent state…")

    state["current_phase"] = task_desc
    state = append_session_log(state, task_desc, written_files, tracker["requests_used"])
    save_state(repo_local_path, state)
    written_files.append(".agent_state.json")

    # ── Step 6: Commit and push ──────────────────────────────────────
    console.print("[bold]Step 6/6:[/bold] Committing and pushing…")

    try:
        commit_and_push(repo_local_path, written_files, commit_msg)
        console.print("[green]Pushed successfully.[/green]")
    except Exception as exc:
        console.print(f"[red]Git push failed: {exc}[/red]")
        console.print("[yellow]Changes are committed locally. Re-run or push manually.[/yellow]")

    # Update managed_repos.json
    repo_name = state.get("repo_name", repo_local_path.name)
    update_last_session(repo_name)

    # ── Session summary ──────────────────────────────────────────────
    summary = {
        "task": task_desc,
        "files_changed": written_files,
        "commit_message": commit_msg,
        "requests_used": tracker["requests_used"],
    }

    console.print(Panel(
        f"[bold]Repo:[/bold]     {repo_name}\n"
        f"[bold]Task:[/bold]     {task_desc}\n"
        f"[bold]Files:[/bold]    {', '.join(written_files)}\n"
        f"[bold]Commit:[/bold]   {commit_msg}\n"
        f"[bold]Requests:[/bold] {tracker['requests_used']} used",
        title="Session Complete",
        border_style="green",
    ))

    return summary


def _safe_json(obj: Any) -> str:
    """JSON-serialise with truncation for large states."""
    import json
    text = json.dumps(obj, indent=2, default=str)
    if len(text) > 4000:
        return text[:4000] + "\n... (truncated)"
    return text
