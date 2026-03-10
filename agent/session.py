"""Core agent loop — DNA-driven two-call architecture with oneshot/agentic modes.

Modes:
  oneshot — Plan → Generate → Validate → Push  (fast, 2-3 Gemini calls)
  agentic — Plan → Generate → Validate → Test → Fix loop → Review → Fix loop → Push

Flow (shared steps 1-4, agentic adds 4a-4b):
  1. Load state + update DNA (AST scan → diff → annotate changed symbols)
  2. Call 1: Send DNA + state → Gemini decides task + requests specific files
  3. Read requested files from disk
  4. Call 2: Send task + files → Gemini generates implementation + validate
  4a. [agentic] Run tests → if fail, feed errors to Gemini, retry up to 3x
  4b. [agentic] Self-review → if issues, feed to Gemini, retry up to 3x
  5. Generate commit message
  6. Update DNA for changed files (AST diff → describe new symbols)
  7. Update memory / state, commit + push (including .dna)
"""

from __future__ import annotations

import json
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
    MAX_FILE_CHARS,
    SESSION_MODE,
    AGENTIC_MAX_FIX_ATTEMPTS,
)
from agent.memory import load_state, save_state, append_session_log
from agent.coder import generate_file_content, generate_commit_message, parse_task_plan
from agent.dna import (
    load_dna,
    update_dna,
    render_dna_context,
    generate_initial_dna,
)
from agent.validator import validate_source, format_errors_for_retry
from agent.test_runner import run_tests, format_test_errors_for_retry
from agent.reviewer import self_review, format_review_for_retry
from github_ops.git_ops import commit_and_push
from github_ops.api import update_last_session, create_issue, close_issue

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
    session_mode: str | None = None,
) -> dict[str, Any]:
    """Run a single DNA-driven improvement session on a repo.

    Two-call architecture:
      Call 1  — DNA + state → plan (task + files to read)
      Call 2  — task + file contents → implementation
      [agentic] — test → fix loop → review → fix loop
      Post    — update DNA with any new/changed symbols

    Args:
        repo_local_path: Absolute path to the local clone.
        mode: "auto" or "manual".
        force: If True, skip the "already ran today" check.
        session_mode: "oneshot" or "agentic". Overrides SESSION_MODE env var.

    Returns:
        Session summary dict.
    """
    repo_local_path = Path(repo_local_path)
    tracker: dict[str, int] = {"requests_used": 0}
    effective_mode = (session_mode or SESSION_MODE).lower()
    is_agentic = effective_mode == "agentic"

    def gemini_call(prompt: str, system: str = "") -> str:
        return _make_gemini_call(prompt, system, tracker)

    # ── Step 1: Load state + update DNA ──────────────────────────────
    console.print("[bold]Step 1/7:[/bold] Loading state & updating DNA…")

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

    # Update DNA — AST scan, diff against stored, annotate new/changed symbols
    dna = load_dna(repo_local_path)
    if not dna:
        # First session or missing DNA — generate from scratch
        console.print("[dim]No .dna found — generating initial DNA…[/dim]")
        project_info = {
            "name": state.get("repo_name", repo_local_path.name),
            "description": state.get("project_description", ""),
            "tech_stack": state.get("tech_stack", []),
        }
        dna = generate_initial_dna(repo_local_path, project_info, gemini_call)
    else:
        dna = update_dna(repo_local_path, gemini_call)

    dna_context = render_dna_context(dna)
    state_context = _safe_json(state)

    # ── Step 2: Call 1 — Decide task + request files ─────────────────
    console.print("[bold]Step 2/7:[/bold] Planning next improvement (Call 1)…")

    decide_prompt = f"""You are an autonomous software development agent maintaining a GitHub repository.
Below is the complete DNA map of the codebase (every file, every function/class signature,
and a description of each symbol), plus the agent's memory/state.

Your job: decide the single most valuable incremental improvement to make today.
Then specify which files you need to READ to implement it.

Rules:
- Make ONE focused improvement (not multiple unrelated changes)
- Prefer improvements that advance the current roadmap phase
- Consider: adding tests, improving docs, fixing edge cases, adding features, refactoring
- You already know every declaration and its purpose from the DNA — choose wisely
- Request the MINIMUM files needed (you'll see their full source in the next step)

## DNA
{dna_context}

## Agent State
```json
{state_context}
```

Return ONLY valid JSON (no markdown fences, no extra text):
{{
  "task": "one-line description of the improvement",
  "rationale": "why this is the most valuable next step",
  "files_to_read": ["path/to/file1.py", "path/to/file2.py"],
  "files_to_create": ["new/file.py"],
  "files_to_modify": ["existing/file.py"]
}}"""

    raw_plan = gemini_call(decide_prompt, system="You are a senior software architect.")
    plan = parse_task_plan(raw_plan)

    # Retry once if parse failed
    if plan.get("rationale", "").startswith("Fallback"):
        retry_prompt = decide_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation."
        raw_plan = gemini_call(retry_prompt, system="You are a senior software architect.")
        plan = parse_task_plan(raw_plan)

    task_desc = plan.get("task", "General improvement")
    console.print(f"  [cyan]Task:[/cyan] {task_desc}")

    # ── Step 2.5: Create GitHub Issue for this task ──────────────────
    repo_name = state.get("repo_name", repo_local_path.name)
    issue_number = None
    try:
        issue_number = create_issue(repo_name, task_desc, plan.get("rationale", ""))
        if issue_number:
            console.print(f"  [dim]Opened issue #{issue_number}[/dim]")
    except Exception:
        pass  # non-critical

    # ── Step 3: Read requested files ─────────────────────────────────
    console.print("[bold]Step 3/7:[/bold] Reading requested files…")

    files_to_read = list(plan.get("files_to_read", []))
    # Also include files_to_modify (agent needs to see current content)
    for f in plan.get("files_to_modify", []):
        if f not in files_to_read:
            files_to_read.append(f)

    file_contents: dict[str, str] = {}
    for rel_path in files_to_read:
        full_path = repo_local_path / rel_path
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            file_contents[rel_path] = content[:MAX_FILE_CHARS * 3]  # generous limit for targeted reads
            console.print(f"  [dim]Read {rel_path} ({len(content)} chars)[/dim]")
        except OSError:
            continue

    # ── Step 4: Call 2 — Generate implementation + validate ──────────
    console.print("[bold]Step 4/7:[/bold] Generating implementation (Call 2)…")

    all_target_files = list(plan.get("files_to_create", [])) + list(plan.get("files_to_modify", []))
    if not all_target_files:
        all_target_files = ["README.md"]

    written_files: list[str] = []
    written_contents: dict[str, str] = {}  # track content for agentic review
    for rel_path in all_target_files:
        full_path = repo_local_path / rel_path
        current = file_contents.get(rel_path)
        if current is None and full_path.exists():
            try:
                current = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

        # Build file-specific context from DNA + read files
        file_context = _build_implementation_context(rel_path, task_desc, file_contents, dna_context)

        try:
            new_content = generate_file_content(
                task=task_desc,
                file_path=rel_path,
                current_content=current,
                project_context=file_context,
                gemini_call=gemini_call,
            )
        except (BudgetExhaustedError, GeminiCallError) as exc:
            console.print(f"[yellow]Stopping file generation: {exc}[/yellow]")
            break

        # Validate generated content
        errors = validate_source(rel_path, new_content)
        if errors:
            console.print(f"  [yellow]⚠ Validation errors in {rel_path}:[/yellow]")
            for err in errors:
                console.print(f"    [dim]{err}[/dim]")

            # Retry once with error feedback
            retry_context = file_context + format_errors_for_retry(rel_path, errors)
            try:
                new_content = generate_file_content(
                    task=task_desc,
                    file_path=rel_path,
                    current_content=current,
                    project_context=retry_context,
                    gemini_call=gemini_call,
                )
                errors = validate_source(rel_path, new_content)
                if errors:
                    console.print(f"  [red]✗ {rel_path} still invalid after retry — skipping[/red]")
                    continue
                console.print(f"  [green]✓ {rel_path} fixed on retry[/green]")
            except (BudgetExhaustedError, GeminiCallError) as exc:
                console.print(f"  [red]✗ Retry failed for {rel_path}: {exc} — skipping[/red]")
                continue

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(new_content, encoding="utf-8")
        written_files.append(rel_path)
        written_contents[rel_path] = new_content
        console.print(f"  [green]✓[/green] {rel_path}")

    if not written_files:
        console.print("[yellow]No files were written — aborting session.[/yellow]")
        return {
            "task": task_desc,
            "files_changed": [],
            "commit_message": "",
            "requests_used": tracker["requests_used"],
        }

    # ── Step 4a: [Agentic] Run tests + fix loop ─────────────────────
    if is_agentic:
        console.print("[bold]Step 4a/9:[/bold] Running test suite…")
        test_result = run_tests(repo_local_path)

        if test_result.tests_found and not test_result.passed:
            console.print(f"  [yellow]⚠ Tests failed ({test_result.runner}, exit {test_result.exit_code})[/yellow]")

            for attempt in range(1, AGENTIC_MAX_FIX_ATTEMPTS + 1):
                console.print(f"  [dim]Fix attempt {attempt}/{AGENTIC_MAX_FIX_ATTEMPTS}…[/dim]")
                error_context = format_test_errors_for_retry(test_result)

                any_fixed = False
                for rel_path in list(written_files):
                    full_path = repo_local_path / rel_path
                    if not full_path.exists():
                        continue
                    current = full_path.read_text(encoding="utf-8", errors="replace")
                    file_context = _build_implementation_context(
                        rel_path, task_desc, file_contents, dna_context
                    ) + error_context
                    try:
                        fixed_content = generate_file_content(
                            task=task_desc,
                            file_path=rel_path,
                            current_content=current,
                            project_context=file_context,
                            gemini_call=gemini_call,
                        )
                        full_path.write_text(fixed_content, encoding="utf-8")
                        written_contents[rel_path] = fixed_content
                        any_fixed = True
                    except (BudgetExhaustedError, GeminiCallError):
                        console.print("  [yellow]Budget exhausted during test fix — stopping.[/yellow]")
                        break

                if not any_fixed:
                    break

                # Re-run tests after fix
                test_result = run_tests(repo_local_path)
                if test_result.passed:
                    console.print(f"  [green]✓ Tests pass after fix attempt {attempt}[/green]")
                    break
                console.print(f"  [yellow]Tests still failing after attempt {attempt}[/yellow]")

            if not test_result.passed:
                console.print("[red]✗ Tests still failing after all fix attempts — reverting.[/red]")
                _revert_changes(repo_local_path, written_files)
                return {
                    "task": task_desc,
                    "files_changed": [],
                    "commit_message": "",
                    "requests_used": tracker["requests_used"],
                    "reverted": True,
                    "reason": "tests_failed",
                }
        elif test_result.tests_found:
            console.print(f"  [green]✓ All tests pass ({test_result.runner})[/green]")
        else:
            console.print(f"  [dim]{test_result.output}[/dim]")

    # ── Step 4b: [Agentic] Self-review + fix loop ────────────────────
    if is_agentic:
        console.print("[bold]Step 4b/9:[/bold] Self-review…")
        try:
            review = self_review(task_desc, written_contents, dna_context, gemini_call)
        except (BudgetExhaustedError, GeminiCallError):
            review = {"approved": True, "issues": []}

        if not review["approved"]:
            for attempt in range(1, AGENTIC_MAX_FIX_ATTEMPTS + 1):
                console.print(f"  [dim]Review fix attempt {attempt}/{AGENTIC_MAX_FIX_ATTEMPTS}…[/dim]")
                review_context = format_review_for_retry(review["issues"])

                # Fix only the files flagged by the reviewer
                flagged_files = {issue.get("file") for issue in review["issues"] if issue.get("file")}
                if not flagged_files:
                    flagged_files = set(written_files)

                for rel_path in flagged_files:
                    if rel_path not in written_files:
                        continue
                    full_path = repo_local_path / rel_path
                    if not full_path.exists():
                        continue
                    current = full_path.read_text(encoding="utf-8", errors="replace")
                    file_context = _build_implementation_context(
                        rel_path, task_desc, file_contents, dna_context
                    ) + review_context
                    try:
                        fixed_content = generate_file_content(
                            task=task_desc,
                            file_path=rel_path,
                            current_content=current,
                            project_context=file_context,
                            gemini_call=gemini_call,
                        )
                        errors = validate_source(rel_path, fixed_content)
                        if errors:
                            console.print(f"  [yellow]Review fix introduced syntax errors in {rel_path} — keeping original[/yellow]")
                            continue
                        full_path.write_text(fixed_content, encoding="utf-8")
                        written_contents[rel_path] = fixed_content
                    except (BudgetExhaustedError, GeminiCallError):
                        console.print("  [yellow]Budget exhausted during review fix — stopping.[/yellow]")
                        break

                # Re-review
                try:
                    review = self_review(task_desc, written_contents, dna_context, gemini_call)
                except (BudgetExhaustedError, GeminiCallError):
                    review = {"approved": True, "issues": []}
                    break

                if review["approved"]:
                    console.print(f"  [green]✓ Review passed after fix attempt {attempt}[/green]")
                    break

            # If review still not approved after all attempts, log but proceed
            # (review issues are softer than test failures — don't revert)
            if not review["approved"]:
                console.print("[yellow]Review issues remain — proceeding anyway.[/yellow]")

    step_label = "5/9" if is_agentic else "5/7"

    # ── Step 5: Generate commit message ──────────────────────────────
    console.print(f"[bold]Step {step_label}:[/bold] Generating commit message…")

    try:
        commit_msg = generate_commit_message(task_desc, written_files, gemini_call)
    except (BudgetExhaustedError, GeminiCallError):
        commit_msg = "chore: automated improvement"

    # Append issue reference to close it via commit
    if issue_number:
        commit_msg += f" (closes #{issue_number})"

    console.print(f"  [dim]{commit_msg}[/dim]")

    # ── Step 6: Update DNA for changed files ─────────────────────────
    step6_label = "6/9" if is_agentic else "6/7"
    console.print(f"[bold]Step {step6_label}:[/bold] Updating DNA for changed files…")

    try:
        dna = update_dna(repo_local_path, gemini_call)
        _advance_roadmap_phase(dna, state, gemini_call)
    except (BudgetExhaustedError, GeminiCallError):
        console.print("[yellow]DNA post-update skipped (budget).[/yellow]")

    written_files.append(".dna")

    # ── Step 7: Update memory, commit & push ─────────────────────────
    step7_label = "7/9" if is_agentic else "7/7"
    console.print(f"[bold]Step {step7_label}:[/bold] Saving state, committing & pushing…")

    state["current_phase"] = task_desc
    state = append_session_log(state, task_desc, written_files, tracker["requests_used"])
    save_state(repo_local_path, state)
    written_files.append(".agent_state.json")

    try:
        commit_and_push(repo_local_path, written_files, commit_msg)
        console.print("[green]Pushed successfully.[/green]")
    except Exception as exc:
        console.print(f"[red]Git push failed: {exc}[/red]")
        console.print("[yellow]Changes are committed locally. Re-run or push manually.[/yellow]")

    # Update managed_repos.json
    update_last_session(repo_name)

    # ── Session summary ──────────────────────────────────────────────
    summary = {
        "task": task_desc,
        "files_changed": written_files,
        "commit_message": commit_msg,
        "requests_used": tracker["requests_used"],
        "session_mode": effective_mode,
    }

    mode_tag = "[cyan]agentic[/cyan]" if is_agentic else "[dim]oneshot[/dim]"
    console.print(Panel(
        f"[bold]Repo:[/bold]     {repo_name}\n"
        f"[bold]Mode:[/bold]     {effective_mode}\n"
        f"[bold]Task:[/bold]     {task_desc}\n"
        f"[bold]Files:[/bold]    {', '.join(written_files)}\n"
        f"[bold]Commit:[/bold]   {commit_msg}\n"
        f"[bold]Requests:[/bold] {tracker['requests_used']} used",
        title="Session Complete",
        border_style="green",
    ))

    return summary


# ── Helpers ──────────────────────────────────────────────────────────

def _build_implementation_context(
    target_file: str,
    task: str,
    file_contents: dict[str, str],
    dna_context: str,
) -> str:
    """Build context for Call 2 — targeted file contents + condensed DNA."""
    parts: list[str] = [f"## DNA Overview\n{dna_context}\n"]

    if file_contents:
        parts.append("## Referenced Files\n")
        for path, content in file_contents.items():
            if path == target_file:
                continue  # coder.py already injects current_content
            parts.append(f"### {path}\n```\n{content}\n```\n")

    return "".join(parts)


def _revert_changes(repo_path: Path, written_files: list[str]) -> None:
    """Revert written files using git checkout (agentic safety net)."""
    import subprocess
    try:
        subprocess.run(
            ["git", "checkout", "."],
            cwd=str(repo_path),
            capture_output=True,
            timeout=30,
        )
        console.print("[dim]Reverted all changes via git checkout.[/dim]")
    except Exception as exc:
        # Fallback: delete the files we created
        console.print(f"[yellow]git checkout failed ({exc}) — deleting written files.[/yellow]")
        for rel_path in written_files:
            full = repo_path / rel_path
            try:
                if full.exists():
                    full.unlink()
            except OSError:
                pass


def _advance_roadmap_phase(dna: dict, state: dict, gemini_call) -> None:
    """Ask Gemini if the current roadmap phase is complete and advance if so."""
    project = dna.get("project", {})
    roadmap = project.get("roadmap", [])
    current = project.get("current_phase", 1)

    if not roadmap:
        return

    current_phase = None
    for phase in roadmap:
        if phase.get("phase") == current:
            current_phase = phase
            break

    if not current_phase or current_phase.get("status") == "complete":
        return

    # Check with Gemini
    phase_check_prompt = (
        f"The project '{project.get('name', '?')}' is on phase {current}: "
        f"'{current_phase.get('title', '?')}'.\n"
        f"Description: {current_phase.get('description', '?')}\n\n"
        f"Recent completed tasks from session log:\n"
    )
    for entry in state.get("session_log", [])[-5:]:
        phase_check_prompt += f"  - {entry.get('task', '?')}\n"

    phase_check_prompt += (
        f"\nIs phase {current} complete? Reply ONLY with JSON: "
        f'{{ "complete": true/false, "reason": "..." }}'
    )

    try:
        raw = gemini_call(phase_check_prompt, system="You are a project manager.")
        result = json.loads(_strip_json_fences(raw))
        if result.get("complete"):
            current_phase["status"] = "complete"
            # Advance to next phase
            for phase in roadmap:
                if phase.get("phase") == current + 1:
                    phase["status"] = "in-progress"
                    project["current_phase"] = current + 1
                    console.print(
                        f"[bold green]🎯 Phase {current} complete! "
                        f"Advancing to Phase {current + 1}: {phase.get('title', '?')}[/bold green]"
                    )
                    break
            from agent.dna import save_dna
            save_dna(Path(state.get("repo_url", "")).parent, dna)
    except (json.JSONDecodeError, KeyError, Exception):
        pass  # non-critical


def _strip_json_fences(text: str) -> str:
    """Remove markdown fences from JSON response."""
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _safe_json(obj: Any) -> str:
    """JSON-serialise with truncation for large states."""
    text = json.dumps(obj, indent=2, default=str)
    if len(text) > 4000:
        return text[:4000] + "\n... (truncated)"
    return text
