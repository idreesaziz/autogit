"""Tests for github_ops/api.py — managed repos, issues."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import github_ops.api as api_module
from github_ops.api import (
    list_managed_repos,
    register_repo,
    update_last_session,
    get_repo_info,
    create_issue,
    close_issue,
    _save_managed,
)


# ═══════════════════════════════════════════════════════════════════════
# Managed Repos JSON
# ═══════════════════════════════════════════════════════════════════════

class TestManagedRepos:
    """Tests for managed_repos.json CRUD."""

    def test_list_empty_when_no_file(self, tmp_path: Path):
        with patch.object(api_module, "_MANAGED_REPOS_FILE", tmp_path / "managed_repos.json"):
            result = list_managed_repos()
        assert result == []

    def test_register_and_list(self, tmp_path: Path):
        managed_file = tmp_path / "managed_repos.json"
        with patch.object(api_module, "_MANAGED_REPOS_FILE", managed_file):
            register_repo("test-repo", "https://github.com/user/test-repo", "/path/to/repo")
            repos = list_managed_repos()
        assert len(repos) == 1
        assert repos[0]["name"] == "test-repo"
        assert repos[0]["url"] == "https://github.com/user/test-repo"

    def test_register_multiple(self, tmp_path: Path):
        managed_file = tmp_path / "managed_repos.json"
        with patch.object(api_module, "_MANAGED_REPOS_FILE", managed_file):
            register_repo("repo-a", "https://github.com/u/a", "/a")
            register_repo("repo-b", "https://github.com/u/b", "/b")
            repos = list_managed_repos()
        assert len(repos) == 2

    def test_update_last_session(self, tmp_path: Path):
        managed_file = tmp_path / "managed_repos.json"
        with patch.object(api_module, "_MANAGED_REPOS_FILE", managed_file):
            register_repo("myrepo", "https://github.com/u/myrepo", "/myrepo")
            update_last_session("myrepo", "2026-03-10")
            repos = list_managed_repos()
        assert repos[0]["last_session"] == "2026-03-10"
        assert repos[0]["total_sessions"] == 1

    def test_update_increments_total(self, tmp_path: Path):
        managed_file = tmp_path / "managed_repos.json"
        with patch.object(api_module, "_MANAGED_REPOS_FILE", managed_file):
            register_repo("myrepo", "https://github.com/u/myrepo", "/myrepo")
            update_last_session("myrepo", "2026-03-09")
            update_last_session("myrepo", "2026-03-10")
            repos = list_managed_repos()
        assert repos[0]["total_sessions"] == 2

    def test_get_repo_info_found(self, tmp_path: Path):
        managed_file = tmp_path / "managed_repos.json"
        with patch.object(api_module, "_MANAGED_REPOS_FILE", managed_file):
            register_repo("target", "https://github.com/u/target", "/target")
            info = get_repo_info("target")
        assert info is not None
        assert info["name"] == "target"

    def test_get_repo_info_not_found(self, tmp_path: Path):
        managed_file = tmp_path / "managed_repos.json"
        with patch.object(api_module, "_MANAGED_REPOS_FILE", managed_file):
            info = get_repo_info("nonexistent")
        assert info is None


# ═══════════════════════════════════════════════════════════════════════
# GitHub Issues
# ═══════════════════════════════════════════════════════════════════════

class TestCreateIssue:
    """Tests for create_issue."""

    @patch.object(api_module, "get_github_client")
    def test_creates_issue_and_returns_number(self, mock_gh):
        mock_repo = MagicMock()
        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_repo.create_issue.return_value = mock_issue
        mock_gh.return_value.get_repo.return_value = mock_repo

        result = create_issue("my-repo", "Add tests", "We need tests")
        assert result == 42
        mock_repo.create_issue.assert_called_once_with(title="Add tests", body="We need tests")

    @patch.object(api_module, "get_github_client")
    def test_returns_none_on_failure(self, mock_gh):
        from github import GithubException
        mock_gh.return_value.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)
        result = create_issue("bad-repo", "Title")
        assert result is None


class TestCloseIssue:
    """Tests for close_issue."""

    @patch.object(api_module, "get_github_client")
    def test_closes_issue(self, mock_gh):
        mock_repo = MagicMock()
        mock_issue = MagicMock()
        mock_repo.get_issue.return_value = mock_issue
        mock_gh.return_value.get_repo.return_value = mock_repo

        result = close_issue("my-repo", 42)
        assert result is True
        mock_issue.edit.assert_called_once_with(state="closed")

    @patch.object(api_module, "get_github_client")
    def test_returns_false_on_failure(self, mock_gh):
        from github import GithubException
        mock_gh.return_value.get_repo.side_effect = GithubException(404, {"message": "Not Found"}, None)
        result = close_issue("bad-repo", 99)
        assert result is False
