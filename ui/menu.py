"""Main menu — Rich interactive menu for autogit."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.status import Status

from agent.session import run_session, _make_gemini_call, BudgetExhaustedError, GeminiCallError
from agent.researcher import research_trending_ideas, create_repo_from_idea
from github_ops.api import list_managed_repos
from agent.memory import load_state
from config import LOCAL_REPOS_DIR
from ui.repo_selector import select_repo

console = Console()

ROLLS_FILE = LOCAL_REPOS_DIR / "autogit_rolls.json"
SCHEDULE_FILE = LOCAL_REPOS_DIR / "autogit_schedule.json"


def show_menu() -> None:
    """Render the main menu and handle user choices in a loop."""
    while True:
        try:
            console.print()
            console.print(Panel(
                "[bold]\\[1][/bold]  Create a new repo\n"
                "[bold]\\[2][/bold]  Work on existing repo\n"
                "[bold]\\[3][/bold]  Run all repos (auto)\n"
                "[bold]\\[4][/bold]  View session logs\n"
                "[bold]\\[5][/bold]  View dice rolls & schedule\n"
                "[bold]\\[6][/bold]  Roll & schedule now\n"
                "[bold]\\[Q][/bold]  Quit",
                title="[bold cyan]autogit[/bold cyan]",
                subtitle="Autonomous GitHub Agent",
                border_style="cyan",
            ))

            choice = Prompt.ask(
                "Choose an option",
                choices=["1", "2", "3", "4", "5", "6", "q", "Q"],
                default="q",
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if choice in ("q", "Q"):
            console.print("[dim]Goodbye![/dim]")
            break
        elif choice == "1":
            _handle_create_repo()
        elif choice == "2":
            _handle_work_on_repo()
        elif choice == "3":
            _handle_run_all()
        elif choice == "4":
            _handle_view_logs()
        elif choice == "5":
            _handle_view_rolls()
        elif choice == "6":
            _handle_run_dice()


# ── Option 1: Create a new repo ─────────────────────────────────────

def _handle_create_repo() -> None:
    """Research trending ideas, let user pick, create repo."""
    tracker: dict[str, int] = {"requests_used": 0}

    def gemini_call(prompt: str, system: str = "") -> str:
        return _make_gemini_call(prompt, system, tracker)

    with Status("[bold cyan]Researching trending ideas…[/bold cyan]", console=console):
        try:
            ideas = research_trending_ideas(gemini_call)
        except (BudgetExhaustedError, GeminiCallError) as exc:
            console.print(f"[red]Research failed: {exc}[/red]")
            return

    if not ideas:
        console.print("[yellow]No ideas generated. Try again later.[/yellow]")
        return

    # Display ideas in a table
    table = Table(title="Project Ideas", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Name", style="cyan", max_width=25)
    table.add_column("Tagline", max_width=40)
    table.add_column("Tech", style="green", max_width=15)
    table.add_column("Why Now", style="dim", max_width=35)

    for i, idea in enumerate(ideas, 1):
        table.add_row(
            str(i),
            idea.get("name", "?"),
            idea.get("tagline", ""),
            ", ".join(idea.get("tech", [])),
            idea.get("why_now", ""),
        )

    console.print(table)

    choice = IntPrompt.ask(
        "Pick an idea to create (number, 0 to cancel)",
        default=0,
    )
    if choice < 1 or choice > len(ideas):
        console.print("[dim]Cancelled.[/dim]")
        return

    idea = ideas[choice - 1]
    console.print(f"\n[bold]Creating repo:[/bold] {idea['name']}")

    with Status("[bold cyan]Setting up repository…[/bold cyan]", console=console):
        try:
            repo_url, local_path = create_repo_from_idea(idea, gemini_call)
        except Exception as exc:
            console.print(f"[red]Repo creation failed: {exc}[/red]")
            return

    console.print(Panel(
        f"[bold]Repo:[/bold]  {idea['name']}\n"
        f"[bold]URL:[/bold]   {repo_url}\n"
        f"[bold]Local:[/bold] {local_path}",
        title="Repo Created",
        border_style="green",
    ))


# ── Option 2: Work on an existing repo ──────────────────────────────

def _handle_work_on_repo() -> None:
    """Let user pick a repo, then run a session on it."""
    repo = select_repo()
    if not repo:
        return

    console.print(f"\n[bold]Running session on:[/bold] {repo['name']}")
    try:
        run_session(repo["local_path"], mode="manual")
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
    except (BudgetExhaustedError, GeminiCallError) as exc:
        console.print(f"[yellow]Session stopped early: {exc}[/yellow]")
    except Exception as exc:
        console.print(f"[red]Session error: {exc}[/red]")


# ── Option 3: Run all repos automatically ───────────────────────────

def run_all_repos(silent: bool = False, force: bool = False) -> None:
    """Run a session on every managed repo. Used by menu and deadline watchdog."""
    repos = list_managed_repos()
    if not repos:
        if not silent:
            console.print("[yellow]No managed repos to process.[/yellow]")
        return

    results: list[dict] = []
    for repo in repos:
        if not silent:
            console.print(f"\n[bold cyan]── {repo['name']} ──[/bold cyan]")
        try:
            summary = run_session(repo["local_path"], mode="auto", force=force)
            results.append({"repo": repo["name"], **summary})
        except FileNotFoundError:
            if not silent:
                console.print(f"[yellow]Skipped {repo['name']}: no agent state.[/yellow]")
            results.append({"repo": repo["name"], "task": "(skipped)", "requests_used": 0})
        except (BudgetExhaustedError, GeminiCallError) as exc:
            if not silent:
                console.print(f"[yellow]{repo['name']} stopped: {exc}[/yellow]")
            results.append({"repo": repo["name"], "task": f"(error: {exc})", "requests_used": 0})
        except Exception as exc:
            if not silent:
                console.print(f"[red]{repo['name']} error: {exc}[/red]")
            results.append({"repo": repo["name"], "task": f"(error: {exc})", "requests_used": 0})

    # Summary table
    if not silent and results:
        table = Table(title="Session Summary", show_lines=True)
        table.add_column("Repo", style="cyan")
        table.add_column("Task")
        table.add_column("Requests", justify="right")

        for r in results:
            table.add_row(
                r.get("repo", "?"),
                r.get("task", "?"),
                str(r.get("requests_used", 0)),
            )
        console.print(table)


def _handle_run_all() -> None:
    run_all_repos(silent=False)


# ── Option 4: View session logs ──────────────────────────────────────

def _handle_view_logs() -> None:
    """Display the last 5 session log entries for each managed repo."""
    repos = list_managed_repos()
    if not repos:
        console.print("[yellow]No managed repos.[/yellow]")
        return

    for repo in repos:
        try:
            state = load_state(repo["local_path"])
        except (FileNotFoundError, Exception):
            console.print(f"[yellow]{repo['name']}: no state file found.[/yellow]")
            continue

        logs = state.get("session_log", [])[-5:]
        if not logs:
            console.print(f"[dim]{repo['name']}: no session logs yet.[/dim]")
            continue

        table = Table(title=repo["name"], show_lines=True)
        table.add_column("Date", style="green", width=12)
        table.add_column("Summary")
        table.add_column("Files Changed", style="dim")
        table.add_column("Reqs", justify="right", width=5)

        for entry in logs:
            table.add_row(
                entry.get("date", "?"),
                entry.get("summary", "?"),
                ", ".join(entry.get("files_changed", [])),
                str(entry.get("requests_used", 0)),
            )

        console.print(table)


# ── Option 5: View dice rolls ───────────────────────────────────────

def _handle_view_rolls() -> None:
    """Display roll history and today's schedule."""
    # ── Today's schedule ─────────────────────────────────────────────
    if SCHEDULE_FILE.exists():
        try:
            schedule = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            schedule = {}

        if schedule and schedule.get("sessions"):
            sched_table = Table(
                title=f"Today's Schedule  ({schedule.get('date', '?')}  |  prob={schedule.get('probability', 0):.0%}  |  {schedule.get('session_count', 0)} session(s) from {schedule.get('total_rolls', 0)} roll(s))",
                show_lines=True,
            )
            sched_table.add_column("#", style="bold", width=4, justify="right")
            sched_table.add_column("Scheduled Time", width=20)
            sched_table.add_column("Status", width=10)

            for s in schedule["sessions"]:
                ts = s.get("scheduled_time", "?")
                if "T" in ts:
                    ts = ts.replace("T", " ")[:19]
                status = s.get("status", "?")
                status_map = {
                    "pending": "[yellow]pending[/yellow]",
                    "running": "[cyan]running[/cyan]",
                    "done": "[green]done[/green]",
                    "error": "[red]error[/red]",
                }
                sched_table.add_row(
                    str(s.get("session_num", "?")),
                    ts,
                    status_map.get(status, status),
                )
            console.print(sched_table)
        elif schedule:
            console.print(
                f"[dim]Today ({schedule.get('date', '?')}): "
                f"0 sessions scheduled (all rolls failed).[/dim]"
            )

    # ── Roll history ─────────────────────────────────────────────────
    if not ROLLS_FILE.exists():
        console.print("[yellow]No roll history yet. Start the service to begin rolling.[/yellow]")
        return

    try:
        rolls = json.loads(ROLLS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        console.print("[red]Could not read roll history.[/red]")
        return

    if not rolls:
        console.print("[yellow]No rolls recorded yet.[/yellow]")
        return

    # Show last 20 rolls
    recent = rolls[-20:]

    table = Table(title=f"Dice Roll History  ({len(rolls)} total rolls)", show_lines=True)
    table.add_column("#", style="bold", width=5, justify="right")
    table.add_column("Timestamp", style="dim", width=20)
    table.add_column("Day", width=10)
    table.add_column("Sess", justify="right", width=5)
    table.add_column("Prob", justify="right", width=6)
    table.add_column("Roll", justify="right", width=7)
    table.add_column("Result", width=8)

    start_idx = len(rolls) - len(recent) + 1
    for i, entry in enumerate(recent, start_idx):
        result = entry.get("result", "?")
        result_style = "[green]COMMIT[/green]" if result == "COMMIT" else "[red]STOP[/red]"
        ts = entry.get("timestamp", "?")
        if "T" in ts:
            ts = ts.replace("T", " ")[:19]
        table.add_row(
            str(i),
            ts,
            entry.get("weekday", "?"),
            str(entry.get("session_num", "")),
            f"{entry.get('probability', 0):.0%}",
            f"{entry.get('roll', 0):.4f}",
            result_style,
        )

    console.print(table)

    # Stats summary
    commits = sum(1 for r in rolls if r.get("result") == "COMMIT")
    stops = len(rolls) - commits
    console.print(
        f"\n  [bold]Total:[/bold] {len(rolls)} rolls  |  "
        f"[green]{commits} COMMIT[/green]  |  "
        f"[red]{stops} STOP[/red]  |  "
        f"Win rate: [bold]{commits / len(rolls):.0%}[/bold]"
    )


# ── Option 6: Run dice sequence now (debug) ─────────────────────────

def _handle_run_dice() -> None:
    """Manually plan the day (roll dice + schedule times), then run due sessions."""
    from service import _plan_day, _run_pending_sessions, _get_today_probability

    prob = _get_today_probability()
    console.print(
        f"\n[bold cyan]Planning today's sessions[/bold cyan]  "
        f"(probability: [bold]{prob:.0%}[/bold])\n"
    )

    schedule = _plan_day(live=True)

    if schedule.get("session_count", 0) > 0:
        console.print("\n[bold cyan]Running all scheduled sessions now…[/bold cyan]")
        _run_pending_sessions(live=True, run_now=True)
    else:
        console.print("\n[dim]No sessions to run (all rolls failed).[/dim]")
