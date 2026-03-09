"""Repo selector — Rich table for picking a managed repo."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from github_ops.api import list_managed_repos
from agent.memory import load_state

console = Console()


def select_repo() -> dict | None:
    """Display managed repos in a table and let the user pick one.

    Returns:
        The selected repo dict from managed_repos.json, or None if cancelled.
    """
    repos = list_managed_repos()
    if not repos:
        console.print("[yellow]No managed repos yet. Create one first![/yellow]")
        return None

    table = Table(title="Managed Repositories", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Repo Name", style="cyan")
    table.add_column("Last Session", style="green")
    table.add_column("Sessions", justify="right")
    table.add_column("Phase", style="dim")

    for i, repo in enumerate(repos, 1):
        phase = ""
        try:
            state = load_state(repo["local_path"])
            phase = state.get("current_phase", "")
        except (FileNotFoundError, Exception):
            phase = "(unknown)"

        table.add_row(
            str(i),
            repo["name"],
            repo.get("last_session", "never"),
            str(repo.get("total_sessions", 0)),
            phase,
        )

    console.print(table)

    choice = IntPrompt.ask(
        "\nSelect a repo (number)",
        default=1,
        choices=[str(i) for i in range(1, len(repos) + 1)],
    )
    return repos[choice - 1]
