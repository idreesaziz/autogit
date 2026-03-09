"""GitPython wrapper — clone, stage, commit, push, file tree."""

import os
from pathlib import Path

from git import Repo, InvalidGitRepositoryError, GitCommandError
from rich.console import Console

from config import GITHUB_TOKEN, GITHUB_USERNAME

console = Console()

# Directories / patterns to skip when scanning files
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}


def _authenticated_url(url: str) -> str:
    """Embed the GitHub token into an HTTPS remote URL."""
    # https://github.com/user/repo.git → https://TOKEN@github.com/user/repo.git
    if url.startswith("https://"):
        return url.replace("https://", f"https://{GITHUB_TOKEN}@", 1)
    return url


def clone_repo(url: str, local_path: str | Path) -> Repo:
    """Clone a repo if it doesn't exist locally; pull if it does."""
    local_path = Path(local_path)
    auth_url = _authenticated_url(url)

    if local_path.exists():
        try:
            repo = Repo(local_path)
            # Update remote URL in case token changed
            repo.remotes.origin.set_url(auth_url)
            console.print(f"  [dim]Pulling latest for {local_path.name}…[/dim]")
            repo.remotes.origin.pull()
            return repo
        except (InvalidGitRepositoryError, GitCommandError) as exc:
            console.print(f"  [yellow]Warning: could not pull — {exc}[/yellow]")
            return Repo(local_path)

    console.print(f"  [dim]Cloning {url} → {local_path}…[/dim]")
    local_path.mkdir(parents=True, exist_ok=True)
    repo = Repo.clone_from(auth_url, str(local_path))
    _configure_identity(repo)
    return repo


def init_repo(local_path: str | Path, remote_url: str) -> Repo:
    """Initialise a new local repo, set remote origin, configure identity."""
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    repo = Repo.init(local_path)
    _configure_identity(repo)
    auth_url = _authenticated_url(remote_url)
    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", auth_url)
    else:
        repo.remotes.origin.set_url(auth_url)
    return repo


def commit_and_push(
    local_path: str | Path,
    files: list[str],
    message: str,
) -> None:
    """Stage specific files, commit, and push to origin main."""
    repo = Repo(str(local_path))
    _configure_identity(repo)

    # Stage the listed files
    repo.index.add(files)
    repo.index.commit(message)

    # Push — create remote branch on first push
    try:
        repo.remotes.origin.push(refspec="HEAD:refs/heads/main")
    except GitCommandError as exc:
        console.print(f"[red]Push failed: {exc}[/red]")
        console.print("[yellow]Tip: check your GITHUB_TOKEN permissions or resolve conflicts manually.[/yellow]")
        raise


def get_recent_files(local_path: str | Path, n: int = 10) -> list[str]:
    """Return the N most recently modified files (relative paths), excluding ignored dirs."""
    local_path = Path(local_path)
    files: list[tuple[float, str]] = []

    for root, dirs, filenames in os.walk(local_path):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in filenames:
            full = Path(root) / fname
            rel = str(full.relative_to(local_path)).replace("\\", "/")
            try:
                mtime = full.stat().st_mtime
            except OSError:
                continue
            files.append((mtime, rel))

    files.sort(key=lambda t: t[0], reverse=True)
    return [f[1] for f in files[:n]]


def get_file_tree(local_path: str | Path, max_depth: int = 3) -> str:
    """Return a readable directory-tree string for the repo."""
    local_path = Path(local_path)
    lines: list[str] = [local_path.name + "/"]

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        # Filter out ignored dirs
        entries = [e for e in entries if e.name not in _SKIP_DIRS]
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    _walk(local_path, "", 1)
    return "\n".join(lines)


def _configure_identity(repo: Repo) -> None:
    """Set git user.name and user.email on the repo config."""
    with repo.config_writer() as cw:
        cw.set_value("user", "name", GITHUB_USERNAME)
        cw.set_value("user", "email", f"{GITHUB_USERNAME}@users.noreply.github.com")
