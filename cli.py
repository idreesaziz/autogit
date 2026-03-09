"""autogit CLI — interactive menu in its own console window.

This is launched by `main.py` via `start` so it opens in a separate
terminal window. Closing this window does NOT stop the background service.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from service import is_running
from config import DAILY_DEADLINE_HOUR, TIMEZONE
from ui.menu import show_menu

console = Console()


def main() -> None:
    console.print(Panel(
        "[bold]Autonomous GitHub Agent[/bold]\n"
        "[dim]Maintains your repos with daily incremental improvements[/dim]",
        title="[bold cyan]autogit[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    pid = is_running()
    if pid:
        console.print(
            f"[green]Background service running (PID {pid}) — "
            f"deadline {DAILY_DEADLINE_HOUR}:00 {TIMEZONE}[/green]"
        )
    else:
        console.print(
            "[yellow]Background service not running. "
            "Run `python main.py service` to start it.[/yellow]"
        )

    console.print("[dim]Closing this window will NOT stop the background service.[/dim]\n")
    show_menu()


if __name__ == "__main__":
    main()
