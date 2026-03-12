"""External repo contributor — find easy issues, fix them, open draft PRs.

Pipeline:
  1. Discover: fetch open issues from target repo
  2. Context: fetch README + annotated directory tree
  3. Assess: LLM ranks issues by complexity, picks the easiest
  4. Fix: clone/fork, create branch, implement fix with review loop
  5. PR: push branch to fork, open draft PR upstream
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import LOCAL_REPOS_DIR, MAX_FILE_CHARS, AGENTIC_MAX_FIX_ATTEMPTS
from agent.coder import generate_file_content, generate_commit_message, _strip_fences
from agent.validator import validate_source
from agent.test_runner import run_tests
from agent.reviewer import self_review, format_review_for_retry
from github_ops.api import (
    fetch_open_issues,
    fetch_repo_readme,
    fetch_repo_tree,
    fetch_file_content,
    fork_repo,
    create_draft_pr,
)
from github_ops.git_ops import clone_repo, commit_files, push_to_remote, _authenticated_url

console = Console()

# Minimum confidence to proceed with a fix (0–1)
MIN_CONFIDENCE = 0.5


def contribute_to_repo(
    full_repo_name: str,
    gemini_call,
) -> dict[str, Any]:
    """Full pipeline: discover → assess → fix → PR.

    Args:
        full_repo_name: "owner/repo" format.
        gemini_call: Rate-limited Gemini call function.

    Returns:
        Summary dict with issue, files_changed, pr_number, etc.
    """
    owner, repo_name = full_repo_name.split("/", 1)

    # ── Step 1: Fetch issues ─────────────────────────────────────────
    console.print(f"[bold]Step 1:[/bold] Fetching open issues from {full_repo_name}…")
    issues = fetch_open_issues(full_repo_name, limit=20)

    if not issues:
        console.print("[yellow]No open issues found on this repo.[/yellow]")
        return {"status": "no_issues"}

    console.print(f"  Found {len(issues)} open issue(s)")

    # ── Step 2: Fetch repo context ───────────────────────────────────
    console.print("[bold]Step 2:[/bold] Fetching repo context…")

    readme = fetch_repo_readme(full_repo_name)
    tree = fetch_repo_tree(full_repo_name)

    tree_text = _format_tree(tree)
    console.print(f"  README: {len(readme)} chars, tree: {len(tree)} entries")

    # ── Step 3: Complexity assessment ────────────────────────────────
    console.print("[bold]Step 3:[/bold] Assessing issue complexity…")

    ranked = _assess_issues(issues, readme, tree_text, gemini_call)

    if not ranked:
        console.print("[yellow]Could not assess any issues.[/yellow]")
        return {"status": "assessment_failed"}

    # Display ranked issues
    _display_ranked_issues(ranked)

    # Pick the winner
    winner = ranked[0]
    if winner["confidence"] < MIN_CONFIDENCE:
        console.print(
            f"[yellow]Best candidate confidence ({winner['confidence']:.0%}) "
            f"is below threshold ({MIN_CONFIDENCE:.0%}) — aborting.[/yellow]"
        )
        return {"status": "low_confidence", "best_issue": winner}

    issue = winner
    console.print(
        f"\n[bold cyan]Selected issue #{issue['number']}:[/bold cyan] {issue['title']}"
    )

    # ── Step 4: Clone + fix ──────────────────────────────────────────
    console.print("[bold]Step 4:[/bold] Cloning repo and implementing fix…")

    # Fork first
    fork_name = fork_repo(full_repo_name)
    fork_url = f"https://github.com/{fork_name}.git"

    local_path = LOCAL_REPOS_DIR / f"_contrib_{repo_name}"
    repo = clone_repo(fork_url, local_path)

    # Sync fork with upstream
    try:
        _sync_fork_with_upstream(local_path, full_repo_name)
    except Exception as exc:
        console.print(f"  [yellow]Could not sync fork: {exc}[/yellow]")

    # Create branch
    branch_name = f"fix/issue-{issue['number']}"
    _create_branch(local_path, branch_name)

    # Implement the fix
    fix_result = _implement_fix(
        local_path=local_path,
        full_repo_name=full_repo_name,
        issue=issue,
        readme=readme,
        tree_text=tree_text,
        gemini_call=gemini_call,
    )

    if not fix_result["files_changed"]:
        console.print("[yellow]No files were changed — aborting.[/yellow]")
        return {"status": "no_changes", "issue": issue}

    # ── Step 5: Push + PR ────────────────────────────────────────────
    console.print("[bold]Step 5:[/bold] Pushing and opening draft PR…")

    try:
        _push_branch(local_path, branch_name)
    except Exception as exc:
        console.print(f"[red]Push failed: {exc}[/red]")
        return {"status": "push_failed", "issue": issue}

    pr_body = (
        f"Closes #{issue['number']}\n\n"
        f"## Summary\n{fix_result.get('summary', 'Automated fix')}\n\n"
        f"## Changes\n"
        + "\n".join(f"- `{f}`" for f in fix_result["files_changed"])
    )

    pr_number = create_draft_pr(
        upstream_full_name=full_repo_name,
        fork_full_name=fork_name,
        branch=branch_name,
        title=fix_result.get("pr_title", f"Fix #{issue['number']}: {issue['title']}"),
        body=pr_body,
    )

    summary = {
        "status": "success",
        "repo": full_repo_name,
        "issue_number": issue["number"],
        "issue_title": issue["title"],
        "files_changed": fix_result["files_changed"],
        "pr_number": pr_number,
        "branch": branch_name,
    }

    console.print(Panel(
        f"[bold]Repo:[/bold]    {full_repo_name}\n"
        f"[bold]Issue:[/bold]   #{issue['number']} — {issue['title']}\n"
        f"[bold]Files:[/bold]   {', '.join(fix_result['files_changed'])}\n"
        f"[bold]PR:[/bold]      #{pr_number or '(failed)'}",
        title="Contribution Complete",
        border_style="green",
    ))

    return summary


# ── Step 3 helpers: complexity assessment ────────────────────────────

def _assess_issues(
    issues: list[dict],
    readme: str,
    tree_text: str,
    gemini_call,
) -> list[dict]:
    """Feed issues + context to LLM, get complexity rankings."""
    issues_text = ""
    for i, iss in enumerate(issues):
        labels = ", ".join(iss.get("labels", [])) or "none"
        issues_text += (
            f"\n### Issue #{iss['number']}: {iss['title']}\n"
            f"Labels: {labels}\n"
            f"Created: {iss['created_at']}\n"
            f"Comments: {iss['comments']}\n"
            f"Body:\n{iss['body'][:1500]}\n"
        )

    prompt = f"""You are evaluating open GitHub issues to find the easiest one to fix.

## Repository README (truncated)
{readme[:4000]}

## Directory Tree (with file sizes in bytes)
{tree_text[:4000]}

## Open Issues
{issues_text}

For EACH issue, assess:
1. files_touched: estimated number of files that need changes (1-2 is easy)
2. complexity: "trivial" | "easy" | "medium" | "hard" | "very_hard"
3. is_bug: true if it's a bug fix (usually more localized), false if feature
4. confidence: 0.0-1.0 how confident you are in your assessment
5. reasoning: one-line explanation

Return ONLY valid JSON (no markdown fences):
{{
  "assessments": [
    {{
      "number": <issue number>,
      "title": "<issue title>",
      "files_touched": <int>,
      "complexity": "<level>",
      "is_bug": <bool>,
      "confidence": <float>,
      "reasoning": "<why>"
    }}
  ]
}}

Sort by easiest first (bugs > features, fewer files > more files, trivial > easy > medium).
Only include issues you think are feasible to fix programmatically."""

    raw = gemini_call(prompt, system="You are an expert at triaging open-source issues.")
    cleaned = _strip_fences(raw)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    assessments = result.get("assessments", [])
    if not assessments:
        return []

    # Merge back the full issue data
    issue_map = {iss["number"]: iss for iss in issues}
    ranked: list[dict] = []
    for a in assessments:
        num = a.get("number")
        if num in issue_map:
            merged = {**issue_map[num], **a}
            ranked.append(merged)

    return ranked


def _display_ranked_issues(ranked: list[dict]) -> None:
    """Show a table of ranked issues."""
    table = Table(title="Issue Complexity Ranking", show_lines=True)
    table.add_column("Rank", style="bold", width=5)
    table.add_column("#", width=6)
    table.add_column("Title", max_width=40)
    table.add_column("Complexity", width=10)
    table.add_column("Files", justify="right", width=6)
    table.add_column("Bug?", width=5)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Reasoning", max_width=35, style="dim")

    complexity_colors = {
        "trivial": "green",
        "easy": "green",
        "medium": "yellow",
        "hard": "red",
        "very_hard": "red",
    }

    for i, issue in enumerate(ranked[:10], 1):
        cx = issue.get("complexity", "?")
        color = complexity_colors.get(cx, "white")
        table.add_row(
            str(i),
            str(issue.get("number", "?")),
            issue.get("title", "?")[:40],
            f"[{color}]{cx}[/{color}]",
            str(issue.get("files_touched", "?")),
            "Y" if issue.get("is_bug") else "N",
            f"{issue.get('confidence', 0):.0%}",
            issue.get("reasoning", "")[:35],
        )

    console.print(table)


# ── Step 4 helpers: fix implementation ───────────────────────────────

def _implement_fix(
    local_path: Path,
    full_repo_name: str,
    issue: dict,
    readme: str,
    tree_text: str,
    gemini_call,
) -> dict[str, Any]:
    """Plan and implement the fix for the selected issue."""
    # Plan: ask LLM which files to read and modify
    plan_prompt = f"""You need to fix this GitHub issue:

## Issue #{issue['number']}: {issue['title']}
{issue.get('body', '')[:2000]}

## Repository README
{readme[:3000]}

## Directory Tree
{tree_text[:3000]}

Plan the fix:
1. Which files do you need to READ to understand the context?
2. Which files do you need to MODIFY or CREATE?
3. What's the fix strategy?

Return ONLY valid JSON:
{{
  "strategy": "one-line description of the fix",
  "files_to_read": ["path/to/file.py"],
  "files_to_modify": ["path/to/file.py"],
  "files_to_create": [],
  "pr_title": "short PR title"
}}"""

    raw = gemini_call(plan_prompt, system="You are a senior open-source contributor.")
    plan = _parse_plan(raw)

    strategy = plan.get("strategy", "Fix the issue")
    files_to_read = plan.get("files_to_read", [])
    files_to_modify = plan.get("files_to_modify", [])
    files_to_create = plan.get("files_to_create", [])
    pr_title = plan.get("pr_title", f"Fix #{issue['number']}: {issue['title']}")

    console.print(f"  [cyan]Strategy:[/cyan] {strategy}")
    console.print(f"  [dim]Read: {', '.join(files_to_read)}[/dim]")
    console.print(f"  [dim]Modify: {', '.join(files_to_modify)}[/dim]")
    if files_to_create:
        console.print(f"  [dim]Create: {', '.join(files_to_create)}[/dim]")

    # Read the files
    file_contents: dict[str, str] = {}
    for rel_path in files_to_read:
        # Try local first (already cloned), then fall back to API
        full = local_path / rel_path
        if full.is_file():
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                file_contents[rel_path] = content[:MAX_FILE_CHARS * 3]
                console.print(f"  [dim]Read {rel_path} ({len(content)} chars)[/dim]")
            except OSError:
                pass
        else:
            content = fetch_file_content(full_repo_name, rel_path)
            if content:
                file_contents[rel_path] = content[:MAX_FILE_CHARS * 3]
                console.print(f"  [dim]Fetched {rel_path} ({len(content)} chars)[/dim]")

    # Also read files_to_modify that aren't in files_to_read
    for rel_path in files_to_modify:
        if rel_path in file_contents:
            continue
        full = local_path / rel_path
        if full.is_file():
            try:
                content = full.read_text(encoding="utf-8", errors="replace")
                file_contents[rel_path] = content[:MAX_FILE_CHARS * 3]
            except OSError:
                pass

    # Generate fixes
    all_target_files = files_to_modify + files_to_create
    written_files: list[str] = []
    written_contents: dict[str, str] = {}

    context_parts = [
        f"## Issue #{issue['number']}: {issue['title']}\n{issue.get('body', '')[:2000]}\n",
        f"## Fix Strategy\n{strategy}\n",
    ]
    for path, content in file_contents.items():
        if path not in all_target_files:
            context_parts.append(f"### {path}\n```\n{content}\n```\n")
    project_context = "\n".join(context_parts)

    task = f"Fix issue #{issue['number']}: {issue['title']}. Strategy: {strategy}"

    for rel_path in all_target_files:
        current = file_contents.get(rel_path)
        full = local_path / rel_path
        if current is None and full.exists():
            try:
                current = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

        try:
            new_content = generate_file_content(
                task=task,
                file_path=rel_path,
                current_content=current,
                project_context=project_context,
                gemini_call=gemini_call,
            )
        except Exception as exc:
            console.print(f"  [yellow]Generation failed for {rel_path}: {exc}[/yellow]")
            continue

        # Validate
        errors = validate_source(rel_path, new_content)
        if errors:
            console.print(f"  [yellow]Validation errors in {rel_path} — retrying…[/yellow]")
            try:
                retry_ctx = project_context + f"\n\nValidation errors:\n" + "\n".join(errors)
                new_content = generate_file_content(
                    task=task, file_path=rel_path,
                    current_content=current, project_context=retry_ctx,
                    gemini_call=gemini_call,
                )
                errors = validate_source(rel_path, new_content)
                if errors:
                    console.print(f"  [red]Still invalid after retry — skipping {rel_path}[/red]")
                    continue
            except Exception:
                continue

        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(new_content, encoding="utf-8")
        written_files.append(rel_path)
        written_contents[rel_path] = new_content
        console.print(f"  [green]✓[/green] {rel_path}")

    if not written_files:
        return {"files_changed": [], "summary": strategy, "pr_title": pr_title}

    # Run tests if available
    test_result = run_tests(local_path)
    if test_result.tests_found and not test_result.passed:
        console.print(f"  [yellow]Tests failed ({test_result.runner}) — attempting fix…[/yellow]")
        # One round of test fixes
        from agent.test_runner import format_test_errors_for_retry
        error_ctx = format_test_errors_for_retry(test_result)
        for rel_path in list(written_files):
            full = local_path / rel_path
            if not full.exists():
                continue
            current = full.read_text(encoding="utf-8", errors="replace")
            try:
                fixed = generate_file_content(
                    task=task, file_path=rel_path,
                    current_content=current,
                    project_context=project_context + error_ctx,
                    gemini_call=gemini_call,
                )
                if not validate_source(rel_path, fixed):
                    full.write_text(fixed, encoding="utf-8")
                    written_contents[rel_path] = fixed
            except Exception:
                pass

        test_result = run_tests(local_path)
        if test_result.passed:
            console.print(f"  [green]✓ Tests pass after fix[/green]")
        else:
            console.print(f"  [yellow]Tests still failing — proceeding anyway[/yellow]")
    elif test_result.tests_found:
        console.print(f"  [green]✓ Tests pass ({test_result.runner})[/green]")

    # Self-review
    try:
        review = self_review(task, written_contents, project_context, gemini_call)
        if not review["approved"]:
            console.print("  [yellow]Review flagged issues — attempting fix…[/yellow]")
            review_ctx = format_review_for_retry(review["issues"])
            for rel_path in written_files:
                full = local_path / rel_path
                if not full.exists():
                    continue
                current = full.read_text(encoding="utf-8", errors="replace")
                try:
                    fixed = generate_file_content(
                        task=task, file_path=rel_path,
                        current_content=current,
                        project_context=project_context + review_ctx,
                        gemini_call=gemini_call,
                    )
                    if not validate_source(rel_path, fixed):
                        full.write_text(fixed, encoding="utf-8")
                        written_contents[rel_path] = fixed
                except Exception:
                    pass
            console.print("  [green]✓ Review fixes applied[/green]")
    except Exception:
        pass  # non-critical

    # Commit
    commit_msg = f"fix: {strategy.lower()[:70]}"
    try:
        commit_msg = generate_commit_message(task, written_files, gemini_call)
    except Exception:
        pass
    commit_files(local_path, written_files, commit_msg)
    console.print(f"  [dim]Committed: {commit_msg}[/dim]")

    return {
        "files_changed": written_files,
        "summary": strategy,
        "pr_title": pr_title,
        "commit_message": commit_msg,
    }


# ── Git helpers ──────────────────────────────────────────────────────

def _sync_fork_with_upstream(local_path: Path, upstream_full_name: str) -> None:
    """Add upstream remote and pull latest."""
    from git import Repo as GitRepo
    repo = GitRepo(str(local_path))
    upstream_url = _authenticated_url(f"https://github.com/{upstream_full_name}.git")

    if "upstream" not in [r.name for r in repo.remotes]:
        repo.create_remote("upstream", upstream_url)
    else:
        repo.remotes.upstream.set_url(upstream_url)

    repo.remotes.upstream.fetch()
    # Reset to upstream default branch
    default = "main"
    try:
        repo.git.checkout(default)
        repo.git.reset("--hard", f"upstream/{default}")
    except Exception:
        try:
            default = "master"
            repo.git.checkout(default)
            repo.git.reset("--hard", f"upstream/{default}")
        except Exception:
            pass  # best effort


def _create_branch(local_path: Path, branch_name: str) -> None:
    """Create and checkout a new branch."""
    from git import Repo as GitRepo
    repo = GitRepo(str(local_path))
    try:
        repo.git.checkout("-b", branch_name)
    except Exception:
        # Branch might already exist
        repo.git.checkout(branch_name)
    console.print(f"  [dim]On branch: {branch_name}[/dim]")


def _push_branch(local_path: Path, branch_name: str) -> None:
    """Push a branch to origin."""
    from git import Repo as GitRepo
    repo = GitRepo(str(local_path))
    repo.remotes.origin.push(refspec=f"{branch_name}:{branch_name}")
    console.print(f"  [green]Pushed branch {branch_name}[/green]")


# ── Formatting helpers ───────────────────────────────────────────────

def _format_tree(tree: list[dict]) -> str:
    """Format the tree API response into a readable annotated listing."""
    lines: list[str] = []
    for entry in tree:
        if entry["type"] == "blob":
            size = entry.get("size", 0)
            # Estimate line count (rough: 40 chars/line)
            est_lines = max(1, size // 40) if size else 0
            lines.append(f"  {entry['path']}  ({size}B, ~{est_lines} lines)")
        elif entry["type"] == "tree":
            lines.append(f"  {entry['path']}/")
    return "\n".join(lines[:500])  # cap for context window


def _parse_plan(raw: str) -> dict:
    """Parse a fix plan JSON from Gemini."""
    cleaned = _strip_fences(raw)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "strategy": "Analyze and fix the reported issue",
        "files_to_read": [],
        "files_to_modify": [],
        "files_to_create": [],
        "pr_title": "Fix reported issue",
    }
