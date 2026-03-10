"""Core agent loop — DNA-driven architecture with oneshot/agentic modes.

Modes:
  oneshot — Plan (1 task) → Generate → Validate → Push  (fast, 2-3 Gemini calls)
  agentic — Multi-step plan → per-subtask: Generate → Validate → Test → Fix → Review → Push

Flow:
  1. Load state + update DNA
  2. Plan: oneshot gets 1 task; agentic gets 2-5 ordered subtasks
  3. [per subtask] Read files → Generate → Validate → [agentic: test + review + fix]
  4. Generate commit message
  5. Update DNA
  6. Save state, commit & push
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
    console.print(f"[bold]Step 1:[/bold] Loading state & updating DNA… [dim]({effective_mode} mode)[/dim]")

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

    # Update DNA
    dna = load_dna(repo_local_path)
    if not dna:
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
    repo_name = state.get("repo_name", repo_local_path.name)

    # ── Step 2: Plan ─────────────────────────────────────────────────
    if is_agentic:
        console.print("[bold]Step 2:[/bold] Building multi-step plan…")
        subtasks = _plan_agentic(dna_context, state_context, gemini_call)
        task_desc = subtasks[0].get("goal", "Multi-step improvement")
        console.print(f"  [cyan]Goal:[/cyan] {task_desc}")
        for i, st in enumerate(subtasks, 1):
            console.print(f"  [dim]  {i}. {st.get('task', '?')}[/dim]")
    else:
        console.print("[bold]Step 2:[/bold] Planning next improvement…")
        plan = _plan_oneshot(dna_context, state_context, gemini_call)
        task_desc = plan.get("task", "General improvement")
        console.print(f"  [cyan]Task:[/cyan] {task_desc}")
        # Wrap single plan as one subtask so the execution path is uniform
        subtasks = [plan]

    # ── Step 2.5: Create GitHub Issue ────────────────────────────────
    issue_number = None
    try:
        rationale = subtasks[0].get("rationale", "")
        issue_body = rationale
        if is_agentic and len(subtasks) > 1:
            issue_body += "\n\n## Subtasks\n"
            for i, st in enumerate(subtasks, 1):
                issue_body += f"- [ ] {st.get('task', '?')}\n"
        issue_number = create_issue(repo_name, task_desc, issue_body)
        if issue_number:
            console.print(f"  [dim]Opened issue #{issue_number}[/dim]")
    except Exception:
        pass  # non-critical

    # ── Step 3: Execute subtasks ─────────────────────────────────────
    all_written_files: list[str] = []
    all_written_contents: dict[str, str] = {}
    cumulative_file_contents: dict[str, str] = {}  # grows across subtasks

    for idx, subtask in enumerate(subtasks):
        step_num = idx + 1
        step_total = len(subtasks)
        sub_task_desc = subtask.get("task", task_desc)
        console.print(
            f"\n[bold]Step 3.{step_num}/{step_total}:[/bold] "
            f"{'Generating implementation' if step_total == 1 else sub_task_desc}…"
        )

        try:
            written, contents, file_ctx = _execute_subtask(
                repo_local_path=repo_local_path,
                subtask=subtask,
                task_desc=sub_task_desc,
                dna_context=dna_context,
                cumulative_file_contents=cumulative_file_contents,
                gemini_call=gemini_call,
                is_agentic=is_agentic,
                tracker=tracker,
            )
        except (BudgetExhaustedError, GeminiCallError) as exc:
            console.print(f"[yellow]Budget exhausted at subtask {step_num} — stopping.[/yellow]")
            break

        if not written:
            console.print(f"  [yellow]No files produced for subtask {step_num} — skipping.[/yellow]")
            continue

        all_written_files.extend(written)
        all_written_contents.update(contents)
        cumulative_file_contents.update(file_ctx)
        # Also update cumulative with what was just written so next subtask sees it
        for fp, c in contents.items():
            cumulative_file_contents[fp] = c

    if not all_written_files:
        console.print("[yellow]No files were written — aborting session.[/yellow]")
        return {
            "task": task_desc,
            "files_changed": [],
            "commit_message": "",
            "requests_used": tracker["requests_used"],
        }

    # ── Step 4: Generate commit message ──────────────────────────────
    console.print("[bold]Step 4:[/bold] Generating commit message…")

    try:
        commit_msg = generate_commit_message(task_desc, all_written_files, gemini_call)
    except (BudgetExhaustedError, GeminiCallError):
        commit_msg = "chore: automated improvement"

    if issue_number:
        commit_msg += f" (closes #{issue_number})"

    console.print(f"  [dim]{commit_msg}[/dim]")

    # ── Step 5: Update DNA ───────────────────────────────────────────
    console.print("[bold]Step 5:[/bold] Updating DNA for changed files…")

    try:
        dna = update_dna(repo_local_path, gemini_call)
        _advance_roadmap_phase(dna, state, gemini_call)
    except (BudgetExhaustedError, GeminiCallError):
        console.print("[yellow]DNA post-update skipped (budget).[/yellow]")

    all_written_files.append(".dna")

    # ── Step 6: Save state, commit & push ────────────────────────────
    console.print("[bold]Step 6:[/bold] Saving state, committing & pushing…")

    state["current_phase"] = task_desc
    state = append_session_log(state, task_desc, all_written_files, tracker["requests_used"])
    save_state(repo_local_path, state)
    all_written_files.append(".agent_state.json")

    try:
        commit_and_push(repo_local_path, all_written_files, commit_msg)
        console.print("[green]Pushed successfully.[/green]")
    except Exception as exc:
        console.print(f"[red]Git push failed: {exc}[/red]")
        console.print("[yellow]Changes are committed locally. Re-run or push manually.[/yellow]")

    update_last_session(repo_name)

    # ── Session summary ──────────────────────────────────────────────
    subtask_count = len(subtasks)
    summary = {
        "task": task_desc,
        "files_changed": all_written_files,
        "commit_message": commit_msg,
        "requests_used": tracker["requests_used"],
        "session_mode": effective_mode,
        "subtasks": subtask_count,
    }

    subtask_line = f" ({subtask_count} subtasks)" if subtask_count > 1 else ""
    console.print(Panel(
        f"[bold]Repo:[/bold]     {repo_name}\n"
        f"[bold]Mode:[/bold]     {effective_mode}{subtask_line}\n"
        f"[bold]Task:[/bold]     {task_desc}\n"
        f"[bold]Files:[/bold]    {', '.join(all_written_files)}\n"
        f"[bold]Commit:[/bold]   {commit_msg}\n"
        f"[bold]Requests:[/bold] {tracker['requests_used']} used",
        title="Session Complete",
        border_style="green",
    ))

    return summary


# ── Planning ─────────────────────────────────────────────────────────

def _plan_oneshot(dna_context: str, state_context: str, gemini_call) -> dict:
    """Oneshot: single task plan (original behaviour)."""
    decide_prompt = f"""You are an autonomous software development agent maintaining a GitHub repository.
Below is the complete DNA map of the codebase and the agent's memory/state.

Your job: decide the single most valuable incremental improvement to make today.
Then specify which files you need to READ to implement it.

Rules:
- Make ONE focused improvement (not multiple unrelated changes)
- Prefer improvements that advance the current roadmap phase
- You already know every declaration from the DNA — choose wisely
- Request the MINIMUM files needed

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
  "files_to_read": ["path/to/file1.py"],
  "files_to_create": ["new/file.py"],
  "files_to_modify": ["existing/file.py"]
}}"""

    raw = gemini_call(decide_prompt, system="You are a senior software architect.")
    plan = parse_task_plan(raw)

    if plan.get("rationale", "").startswith("Fallback"):
        retry_prompt = decide_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON."
        raw = gemini_call(retry_prompt, system="You are a senior software architect.")
        plan = parse_task_plan(raw)

    return plan


def _plan_agentic(dna_context: str, state_context: str, gemini_call) -> list[dict]:
    """Agentic: multi-step plan with 2-5 ordered subtasks."""
    decide_prompt = f"""You are an autonomous software development agent maintaining a GitHub repository.
Below is the complete DNA map of the codebase and the agent's memory/state.

Your job: design a multi-step plan for the single most valuable improvement to make today.
Break it into 2-5 ordered subtasks that build on each other.

Rules:
- Pick ONE coherent improvement (e.g. "add user auth", "add CLI commands", "improve test coverage")
- Break it into 2-5 subtasks executed in order — each subtask should produce working code
- Earlier subtasks should create foundations that later subtasks build on
- Each subtask specifies which files to READ, CREATE, and MODIFY
- Prefer improvements that advance the current roadmap phase
- Keep subtasks small and focused — one concern per subtask

## DNA
{dna_context}

## Agent State
```json
{state_context}
```

Return ONLY valid JSON (no markdown fences, no extra text):
{{
  "goal": "high-level description of the improvement",
  "rationale": "why this is the most valuable next step",
  "subtasks": [
    {{
      "task": "subtask 1 description",
      "files_to_read": ["path/to/file.py"],
      "files_to_create": ["new/file.py"],
      "files_to_modify": ["existing/file.py"]
    }},
    {{
      "task": "subtask 2 description",
      "files_to_read": ["new/file.py"],
      "files_to_create": [],
      "files_to_modify": ["another/file.py"]
    }}
  ]
}}"""

    raw = gemini_call(decide_prompt, system="You are a senior software architect.")
    plan = _parse_multi_step_plan(raw)

    # Retry once if parse failed
    if not plan:
        retry_prompt = decide_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown."
        raw = gemini_call(retry_prompt, system="You are a senior software architect.")
        plan = _parse_multi_step_plan(raw)

    if not plan:
        # Final fallback — treat as oneshot
        console.print("[yellow]Multi-step plan parse failed — falling back to single task.[/yellow]")
        single = _plan_oneshot(dna_context, state_context, gemini_call)
        return [single]

    return plan


def _parse_multi_step_plan(raw: str) -> list[dict] | None:
    """Parse a multi-step plan JSON into a list of subtask dicts.

    Returns None if parsing fails entirely.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        import re
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object
        import re
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(result, dict):
        return None

    subtasks = result.get("subtasks", [])
    if not subtasks or not isinstance(subtasks, list):
        return None

    # Inject the top-level goal/rationale into each subtask for context
    goal = result.get("goal", "")
    rationale = result.get("rationale", "")
    for st in subtasks:
        st.setdefault("goal", goal)
        st.setdefault("rationale", rationale)

    return subtasks


# ── Subtask execution ────────────────────────────────────────────────

def _execute_subtask(
    repo_local_path: Path,
    subtask: dict,
    task_desc: str,
    dna_context: str,
    cumulative_file_contents: dict[str, str],
    gemini_call,
    is_agentic: bool,
    tracker: dict[str, int],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Execute a single subtask: read → generate → validate → [test → review].

    Returns:
        (written_files, written_contents, file_contents_read)
    """
    # ── Read requested files ─────────────────────────────────────────
    files_to_read = list(subtask.get("files_to_read", []))
    for f in subtask.get("files_to_modify", []):
        if f not in files_to_read:
            files_to_read.append(f)

    file_contents: dict[str, str] = dict(cumulative_file_contents)  # start with prior context
    for rel_path in files_to_read:
        if rel_path in file_contents:
            continue  # already have it from a prior subtask
        full_path = repo_local_path / rel_path
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            file_contents[rel_path] = content[:MAX_FILE_CHARS * 3]
            console.print(f"  [dim]Read {rel_path} ({len(content)} chars)[/dim]")
        except OSError:
            continue

    # ── Generate files ───────────────────────────────────────────────
    all_target_files = list(subtask.get("files_to_create", [])) + list(subtask.get("files_to_modify", []))
    if not all_target_files:
        all_target_files = ["README.md"]

    written_files: list[str] = []
    written_contents: dict[str, str] = {}

    for rel_path in all_target_files:
        full_path = repo_local_path / rel_path
        current = file_contents.get(rel_path)
        if current is None and full_path.exists():
            try:
                current = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

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

        # Validate
        errors = validate_source(rel_path, new_content)
        if errors:
            console.print(f"  [yellow]⚠ Validation errors in {rel_path}:[/yellow]")
            for err in errors:
                console.print(f"    [dim]{err}[/dim]")

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
        return [], {}, file_contents

    # ── [Agentic] Tests ──────────────────────────────────────────────
    if is_agentic:
        test_result = run_tests(repo_local_path)

        if test_result.tests_found and not test_result.passed:
            console.print(f"  [yellow]⚠ Tests failed ({test_result.runner})[/yellow]")

            for attempt in range(1, AGENTIC_MAX_FIX_ATTEMPTS + 1):
                console.print(f"  [dim]Fix attempt {attempt}/{AGENTIC_MAX_FIX_ATTEMPTS}…[/dim]")
                error_context = format_test_errors_for_retry(test_result)

                any_fixed = False
                for rel_path in list(written_files):
                    full_path = repo_local_path / rel_path
                    if not full_path.exists():
                        continue
                    current = full_path.read_text(encoding="utf-8", errors="replace")
                    fix_ctx = _build_implementation_context(
                        rel_path, task_desc, file_contents, dna_context
                    ) + error_context
                    try:
                        fixed = generate_file_content(
                            task=task_desc, file_path=rel_path,
                            current_content=current, project_context=fix_ctx,
                            gemini_call=gemini_call,
                        )
                        full_path.write_text(fixed, encoding="utf-8")
                        written_contents[rel_path] = fixed
                        any_fixed = True
                    except (BudgetExhaustedError, GeminiCallError):
                        console.print("  [yellow]Budget exhausted during test fix.[/yellow]")
                        break

                if not any_fixed:
                    break

                test_result = run_tests(repo_local_path)
                if test_result.passed:
                    console.print(f"  [green]✓ Tests pass (attempt {attempt})[/green]")
                    break

            if not test_result.passed:
                console.print("[red]✗ Tests failing — reverting subtask.[/red]")
                _revert_changes(repo_local_path, written_files)
                return [], {}, file_contents

        elif test_result.tests_found:
            console.print(f"  [green]✓ Tests pass ({test_result.runner})[/green]")
        else:
            console.print(f"  [dim]{test_result.output}[/dim]")

    # ── [Agentic] Self-review ────────────────────────────────────────
    if is_agentic:
        try:
            review = self_review(task_desc, written_contents, dna_context, gemini_call)
        except (BudgetExhaustedError, GeminiCallError):
            review = {"approved": True, "issues": []}

        if not review["approved"]:
            for attempt in range(1, AGENTIC_MAX_FIX_ATTEMPTS + 1):
                console.print(f"  [dim]Review fix {attempt}/{AGENTIC_MAX_FIX_ATTEMPTS}…[/dim]")
                review_context = format_review_for_retry(review["issues"])

                flagged = {i.get("file") for i in review["issues"] if i.get("file")}
                if not flagged:
                    flagged = set(written_files)

                for rel_path in flagged:
                    if rel_path not in written_files:
                        continue
                    full_path = repo_local_path / rel_path
                    if not full_path.exists():
                        continue
                    current = full_path.read_text(encoding="utf-8", errors="replace")
                    fix_ctx = _build_implementation_context(
                        rel_path, task_desc, file_contents, dna_context
                    ) + review_context
                    try:
                        fixed = generate_file_content(
                            task=task_desc, file_path=rel_path,
                            current_content=current, project_context=fix_ctx,
                            gemini_call=gemini_call,
                        )
                        if validate_source(rel_path, fixed):
                            console.print(f"  [yellow]Review fix broke syntax in {rel_path} — keeping original[/yellow]")
                            continue
                        full_path.write_text(fixed, encoding="utf-8")
                        written_contents[rel_path] = fixed
                    except (BudgetExhaustedError, GeminiCallError):
                        break

                try:
                    review = self_review(task_desc, written_contents, dna_context, gemini_call)
                except (BudgetExhaustedError, GeminiCallError):
                    review = {"approved": True, "issues": []}
                    break

                if review["approved"]:
                    console.print(f"  [green]✓ Review passed (attempt {attempt})[/green]")
                    break

            if not review["approved"]:
                console.print("[yellow]Review issues remain — proceeding.[/yellow]")

    return written_files, written_contents, file_contents


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
