"""autogit — Autonomous GitHub Agent.

Entry point: UI router and deadline watchdog.
Run with: python main.py
"""

from __future__ import annotations

import random
import threading
from datetime import date, datetime

import schedule
import time as _time
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.panel import Panel

from config import DAILY_DEADLINE_HOUR, TIMEZONE, WEEKDAY_BASE, MOMENTUM_BOOST, MOMENTUM_DECAY
from github_ops.api import list_managed_repos
from ui.menu import show_menu, run_all_repos

console = Console()


def _now() -> datetime:
    """Current time in the configured timezone."""
    return datetime.now(ZoneInfo(TIMEZONE))


def _all_repos_ran_today() -> bool:
    """Check if every managed repo already had a session today."""
    repos = list_managed_repos()
    if not repos:
        return False
    today = date.today().isoformat()
    return all(r.get("last_session") == today for r in repos)


def _any_repo_ran_yesterday() -> bool:
    """Check if at least one repo committed yesterday (momentum check)."""
    repos = list_managed_repos()
    yesterday = date.today().toordinal() - 1
    for r in repos:
        last = r.get("last_session", "")
        try:
            if date.fromisoformat(last).toordinal() == yesterday:
                return True
        except ValueError:
            continue
    return False


def _should_commit_today() -> bool:
    """Use weekday-based probability + momentum to decide if today is a commit day."""
    weekday = _now().weekday()
    base = WEEKDAY_BASE.get(weekday, 0.5)
    momentum = MOMENTUM_BOOST if _any_repo_ran_yesterday() else MOMENTUM_DECAY
    probability = max(0.0, min(1.0, base + momentum))
    roll = random.random()
    console.print(
        f"[dim]Commit probability today ({_now().strftime('%A')}): "
        f"{probability:.0%} — roll: {roll:.2f}[/dim]"
    )
    return roll < probability


def _deadline_auto_run() -> None:
    """Called at the deadline hour if no session ran today — runs all repos."""
    if _all_repos_ran_today():
        return
    if not _should_commit_today():
        console.print("[dim]Skipping today based on commit cadence.[/dim]")
        return
    console.print("[bold yellow]Deadline reached — running autonomous sessions…[/bold yellow]")
    run_all_repos(silent=True)


def _start_watchdog() -> None:
    """Schedule a deadline check and run it in a background thread."""
    schedule.every().day.at(f"{DAILY_DEADLINE_HOUR:02d}:00").do(_deadline_auto_run)

    def _loop() -> None:
        while True:
            schedule.run_pending()
            _time.sleep(30)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()


def main() -> None:
    """Application entry point."""
    # ── Header ───────────────────────────────────────────────────────
    console.print(Panel(
        "[bold]Autonomous GitHub Agent[/bold]\n"
        "[dim]Maintains your repos with daily incremental improvements[/dim]",
        title="[bold cyan]autogit[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Check if all repos already ran today ─────────────────────────
    repos = list_managed_repos()
    if repos and _all_repos_ran_today():
        console.print(Panel(
            "[green]All managed repos have already been updated today.[/green]\n"
            f"[dim]Repos: {', '.join(r['name'] for r in repos)}[/dim]",
            title="Status",
            border_style="green",
        ))
        return

    # ── Deadline watchdog ────────────────────────────────────────────
    now = _now()
    if repos and now.hour >= DAILY_DEADLINE_HOUR and not _all_repos_ran_today():
        console.print(
            f"[bold yellow]Past deadline ({DAILY_DEADLINE_HOUR}:00) — "
            "checking commit cadence…[/bold yellow]"
        )
        if _should_commit_today():
            run_all_repos(silent=False)
            return
        else:
            console.print("[dim]Skipping today based on natural commit cadence.[/dim]")
            return

    # ── Start background watchdog ────────────────────────────────────
    _start_watchdog()

    # ── Interactive menu ─────────────────────────────────────────────
    show_menu()


if __name__ == "__main__":
    main()
