"""PyGithub wrapper — create repos, track managed repos, external repo ops."""

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


# ── External repo operations ────────────────────────────────────────

def fetch_open_issues(
    full_repo_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch open issues from an external repo, oldest first.

    Args:
        full_repo_name: "owner/repo" format.
        limit: Max issues to return.

    Returns:
        List of dicts with number, title, body, labels, created_at, comments.
    """
    gh = get_github_client()
    repo = gh.get_repo(full_repo_name)
    issues = repo.get_issues(state="open", sort="created", direction="asc")

    results: list[dict[str, Any]] = []
    for issue in issues:
        if issue.pull_request:
            continue  # skip PRs
        results.append({
            "number": issue.number,
            "title": issue.title,
            "body": (issue.body or "")[:3000],
            "labels": [l.name for l in issue.labels],
            "created_at": issue.created_at.isoformat(),
            "comments": issue.comments,
        })
        if len(results) >= limit:
            break

    return results


def fetch_repo_readme(full_repo_name: str) -> str:
    """Fetch a repo's README content."""
    gh = get_github_client()
    repo = gh.get_repo(full_repo_name)
    try:
        readme = repo.get_readme()
        return readme.decoded_content.decode("utf-8", errors="replace")[:8000]
    except GithubException:
        return "(no README found)"


def fetch_repo_tree(full_repo_name: str) -> list[dict[str, Any]]:
    """Fetch the full directory tree via the Git tree API.

    Returns list of dicts with path, type (blob/tree), and size.
    """
    gh = get_github_client()
    repo = gh.get_repo(full_repo_name)
    try:
        default_branch = repo.default_branch
        tree = repo.get_git_tree(default_branch, recursive=True)
        entries: list[dict[str, Any]] = []
        for item in tree.tree:
            entries.append({
                "path": item.path,
                "type": item.type,  # "blob" or "tree"
                "size": item.size or 0,
            })
        return entries
    except GithubException as exc:
        console.print(f"[yellow]Could not fetch tree: {exc}[/yellow]")
        return []


def fetch_file_content(full_repo_name: str, file_path: str) -> str | None:
    """Fetch a single file's content from a repo."""
    gh = get_github_client()
    repo = gh.get_repo(full_repo_name)
    try:
        content = repo.get_contents(file_path)
        if isinstance(content, list):
            return None  # it's a directory
        return content.decoded_content.decode("utf-8", errors="replace")
    except GithubException:
        return None


def fork_repo(full_repo_name: str) -> str:
    """Fork a repo under the authenticated user. Returns the fork's full name.

    If already forked, returns the existing fork.
    """
    gh = get_github_client()
    repo = gh.get_repo(full_repo_name)
    user = gh.get_user()

    # Check if already forked
    try:
        existing = gh.get_repo(f"{GITHUB_USERNAME}/{repo.name}")
        if existing.fork:
            console.print(f"  [dim]Fork already exists: {existing.full_name}[/dim]")
            return existing.full_name
    except GithubException:
        pass  # not forked yet

    try:
        fork = user.create_fork(repo)
        console.print(f"  [green]Forked:[/green] {fork.full_name}")
        return fork.full_name
    except GithubException as exc:
        console.print(f"[red]Fork failed: {exc}[/red]")
        raise


def create_draft_pr(
    upstream_full_name: str,
    fork_full_name: str,
    branch: str,
    title: str,
    body: str,
) -> int | None:
    """Open a draft PR from fork branch to upstream default branch.

    Returns the PR number or None on failure.
    """
    gh = get_github_client()
    upstream = gh.get_repo(upstream_full_name)
    fork_owner = fork_full_name.split("/")[0]

    try:
        pr = upstream.create_pull(
            title=title,
            body=body,
            head=f"{fork_owner}:{branch}",
            base=upstream.default_branch,
            draft=True,
        )
        console.print(f"  [green]Draft PR #{pr.number}:[/green] {pr.html_url}")
        return pr.number
    except GithubException as exc:
        console.print(f"[red]PR creation failed: {exc}[/red]")
        return None
