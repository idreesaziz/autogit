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
