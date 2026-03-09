"""autogit — Autonomous GitHub Agent.

Commands:
    python main.py              Interactive menu (opens in a new window if called from service)
    python main.py service      Start the background service (runs 24/7, no window)
    python main.py stop         Stop the background service
    python main.py status       Check if the service is running
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from config import DAILY_DEADLINE_HOUR, TIMEZONE
from github_ops.api import list_managed_repos
from service import PID_FILE, LOG_FILE, is_running

console = Console()
_APP_DIR = Path(__file__).resolve().parent


def _start_service() -> None:
    """Launch the background service using pythonw (no console window)."""
    pid = is_running()
    if pid:
        console.print(f"[yellow]Service already running (PID {pid}).[/yellow]")
        return

    service_script = str(_APP_DIR / "service.py")

    # Try pythonw.exe first (windowless), fall back to python in background
    pythonw = Path(sys.executable).parent / "pythonw.exe"
    if pythonw.exists():
        proc = subprocess.Popen(
            [str(pythonw), service_script],
            cwd=str(_APP_DIR),
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
        )
    else:
        proc = subprocess.Popen(
            [sys.executable, service_script],
            cwd=str(_APP_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
        )

    console.print(f"[green]Service started (PID {proc.pid}).[/green]")
    console.print(f"[dim]Log file: {LOG_FILE}[/dim]")
    console.print(f"[dim]Deadline: {DAILY_DEADLINE_HOUR}:00 {TIMEZONE}[/dim]")


def _stop_service() -> None:
    """Stop the background service."""
    pid = is_running()
    if not pid:
        console.print("[yellow]Service is not running.[/yellow]")
        return

    try:
        os.kill(pid, 15)  # SIGTERM
        console.print(f"[green]Service stopped (PID {pid}).[/green]")
    except OSError as exc:
        console.print(f"[red]Could not stop service: {exc}[/red]")

    # Clean up PID file
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _show_status() -> None:
    """Show service and repo status."""
    pid = is_running()
    if pid:
        status_line = f"[green]Running[/green] (PID {pid})"
    else:
        status_line = "[red]Stopped[/red]"

    repos = list_managed_repos()
    repo_lines = ""
    if repos:
        for r in repos:
            repo_lines += f"\n  • {r['name']} — last session: {r.get('last_session', 'never')}"
    else:
        repo_lines = "\n  (no managed repos)"

    console.print(Panel(
        f"[bold]Service:[/bold]  {status_line}\n"
        f"[bold]Deadline:[/bold] {DAILY_DEADLINE_HOUR}:00 {TIMEZONE}\n"
        f"[bold]Log:[/bold]      {LOG_FILE}\n"
        f"[bold]Repos:[/bold]{repo_lines}",
        title="[bold cyan]autogit status[/bold cyan]",
        border_style="cyan",
    ))


def _open_cli() -> None:
    """Open the interactive CLI menu in a new console window."""
    cli_script = str(_APP_DIR / "cli.py")
    subprocess.Popen(
        ["cmd", "/c", "start", "autogit", sys.executable, cli_script],
        cwd=str(_APP_DIR),
    )
    console.print("[dim]CLI opened in a new window.[/dim]")


def main() -> None:
    """Application entry point — route to the right command."""
    console.print(Panel(
        "[bold]Autonomous GitHub Agent[/bold]\n"
        "[dim]Maintains your repos with daily incremental improvements[/dim]",
        title="[bold cyan]autogit[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    args = sys.argv[1:]
    command = args[0].lower() if args else ""

    if command == "service":
        _start_service()
    elif command == "stop":
        _stop_service()
    elif command == "status":
        _show_status()
    elif command == "cli":
        _open_cli()
    else:
        # Default: start service if not running, then open CLI in new window
        pid = is_running()
        if not pid:
            _start_service()
        else:
            console.print(f"[dim]Service already running (PID {pid}).[/dim]")
        _open_cli()


if __name__ == "__main__":
    main()
