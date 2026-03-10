"""PyGithub wrapper — create repos, track managed repos."""

import json
from datetime import date
from pathlib import Path
from typing import Any

from github import Github, GithubException, Repository
from rich.console import Console

from config import GITHUB_TOKEN, GITHUB_USERNAME, LOCAL_REPOS_DIR

console = Console()

_MANAGED_REPOS_FILE = LOCAL_REPOS_DIR / "managed_repos.json"


def get_github_client() -> Github:
    """Return an authenticated PyGithub instance."""
    return Github(GITHUB_TOKEN)


def create_repo(name: str, description: str) -> Repository.Repository:
    """Create a new public GitHub repo under the authenticated user."""
    gh = get_github_client()
    user = gh.get_user()
    try:
        repo = user.create_repo(
            name=name,
            description=description,
            private=False,
            auto_init=False,
        )
        console.print(f"  [green]Created GitHub repo:[/green] {repo.html_url}")
        return repo
    except GithubException as exc:
        console.print(f"[red]GitHub API error creating repo: {exc}[/red]")
        raise


def list_managed_repos() -> list[dict[str, Any]]:
    """Read managed_repos.json and return the repos list."""
    if not _MANAGED_REPOS_FILE.exists():
        return []
    with open(_MANAGED_REPOS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("repos", [])


def register_repo(name: str, url: str, local_path: str) -> None:
    """Add a repo to managed_repos.json."""
    repos = list_managed_repos()
    repos.append({
        "name": name,
        "url": url,
        "local_path": str(local_path),
        "created_date": date.today().isoformat(),
        "last_session": date.today().isoformat(),
        "total_sessions": 0,
    })
    _save_managed(repos)


def update_last_session(repo_name: str, session_date: str | None = None) -> None:
    """Update the last_session date and bump total_sessions for a repo."""
    repos = list_managed_repos()
    session_date = session_date or date.today().isoformat()
    for repo in repos:
        if repo["name"] == repo_name:
            repo["last_session"] = session_date
            repo["total_sessions"] = repo.get("total_sessions", 0) + 1
            break
    _save_managed(repos)


def get_repo_info(repo_name: str) -> dict[str, Any] | None:
    """Look up a single repo by name from managed_repos.json."""
    for repo in list_managed_repos():
        if repo["name"] == repo_name:
            return repo
    return None


def _save_managed(repos: list[dict[str, Any]]) -> None:
    """Persist the repos list to managed_repos.json."""
    _MANAGED_REPOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_MANAGED_REPOS_FILE, "w", encoding="utf-8") as f:
        json.dump({"repos": repos}, f, indent=2, ensure_ascii=False)


# ── GitHub Issues ────────────────────────────────────────────────────

def create_issue(repo_name: str, title: str, body: str = "") -> int | None:
    """Create a GitHub issue on a managed repo.

    Args:
        repo_name: Short repo name (not full owner/name).
        title: Issue title.
        body: Issue body/description.

    Returns:
        Issue number, or None if creation failed.
    """
    try:
        gh = get_github_client()
        repo = gh.get_repo(f"{GITHUB_USERNAME}/{repo_name}")
        issue = repo.create_issue(title=title, body=body)
        console.print(f"  [green]Created issue #{issue.number}:[/green] {title}")
        return issue.number
    except GithubException as exc:
        console.print(f"[yellow]Could not create issue: {exc}[/yellow]")
        return None


def close_issue(repo_name: str, issue_number: int) -> bool:
    """Close a GitHub issue by number.

    Args:
        repo_name: Short repo name.
        issue_number: The issue number to close.

    Returns:
        True if closed successfully.
    """
    try:
        gh = get_github_client()
        repo = gh.get_repo(f"{GITHUB_USERNAME}/{repo_name}")
        issue = repo.get_issue(number=issue_number)
        issue.edit(state="closed")
        return True
    except GithubException as exc:
        console.print(f"[yellow]Could not close issue #{issue_number}: {exc}[/yellow]")
        return False
