"""Tests for service.py — probability math, scheduling, dice logic, repo picker."""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# We need to mock config before importing service, since service.py
# imports config at module level.  We'll import the functions we need
# directly from the module after patching.

import service
from service import (
    _get_today_probability,
    _sample_session_time,
    _pick_repo,
    _roll_dice,
    _load_rolls,
    _save_roll,
    _load_schedule,
    _save_schedule,
)


# ═══════════════════════════════════════════════════════════════════════
# Probability Calculation
# ═══════════════════════════════════════════════════════════════════════

class TestGetTodayProbability:
    """Tests for _get_today_probability."""

    @patch.object(service, "_now")
    @patch.object(service, "_any_repo_ran_yesterday", return_value=True)
    def test_weekday_with_momentum(self, mock_yesterday, mock_now):
        # Monday with momentum
        mock_now.return_value = datetime(2026, 3, 9, 14, 0)  # Monday
        prob = _get_today_probability()
        # Monday base=0.70, momentum_boost=0.10 → 0.80
        assert prob == pytest.approx(0.80, abs=0.01)

    @patch.object(service, "_now")
    @patch.object(service, "_any_repo_ran_yesterday", return_value=False)
    def test_weekday_with_decay(self, mock_yesterday, mock_now):
        mock_now.return_value = datetime(2026, 3, 9, 14, 0)  # Monday
        prob = _get_today_probability()
        # Monday base=0.70, momentum_decay=-0.05 → 0.65
        assert prob == pytest.approx(0.65, abs=0.01)

    @patch.object(service, "_now")
    @patch.object(service, "_any_repo_ran_yesterday", return_value=False)
    def test_weekend_lower(self, mock_yesterday, mock_now):
        mock_now.return_value = datetime(2026, 3, 8, 14, 0)  # Sunday
        prob = _get_today_probability()
        # Sunday base=0.25, decay=-0.05 → 0.20
        assert prob == pytest.approx(0.20, abs=0.01)

    @patch.object(service, "_now")
    @patch.object(service, "_any_repo_ran_yesterday", return_value=True)
    def test_clamped_to_0_1(self, mock_yesterday, mock_now):
        mock_now.return_value = datetime(2026, 3, 9, 14, 0)
        prob = _get_today_probability()
        assert 0.0 <= prob <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Gaussian Time Sampling
# ═══════════════════════════════════════════════════════════════════════

class TestSampleSessionTime:
    """Tests for _sample_session_time."""

    def test_returns_datetime(self):
        deadline = datetime(2026, 3, 10, 11, 0)
        result = _sample_session_time(deadline)
        assert isinstance(result, datetime)

    def test_result_after_deadline(self):
        deadline = datetime(2026, 3, 10, 11, 0)
        for _ in range(50):
            result = _sample_session_time(deadline)
            assert result >= deadline

    def test_result_within_window(self):
        from config import GAUSS_WINDOW
        deadline = datetime(2026, 3, 10, 11, 0)
        max_time = deadline + timedelta(hours=GAUSS_WINDOW)
        for _ in range(50):
            result = _sample_session_time(deadline)
            assert result <= max_time

    def test_distribution_roughly_centered(self):
        """Most samples should be near the mean offset."""
        from config import GAUSS_MEAN_OFFSET
        deadline = datetime(2026, 3, 10, 11, 0)
        offsets = []
        for _ in range(200):
            result = _sample_session_time(deadline)
            offset_hours = (result - deadline).total_seconds() / 3600
            offsets.append(offset_hours)
        mean_offset = sum(offsets) / len(offsets)
        # Should be roughly within 1 hour of the configured mean
        assert abs(mean_offset - GAUSS_MEAN_OFFSET) < 1.5


# ═══════════════════════════════════════════════════════════════════════
# Dice Rolling
# ═══════════════════════════════════════════════════════════════════════

class TestRollDice:
    """Tests for _roll_dice."""

    @patch.object(service, "_save_roll")
    @patch.object(service, "_load_rolls", return_value=[])
    @patch.object(service, "_now")
    def test_pass_when_roll_below_probability(self, mock_now, mock_load, mock_save):
        mock_now.return_value = datetime(2026, 3, 10, 14, 0)
        with patch("random.random", return_value=0.3):
            result = _roll_dice(0.7, 1)
        assert result is True

    @patch.object(service, "_save_roll")
    @patch.object(service, "_load_rolls", return_value=[])
    @patch.object(service, "_now")
    def test_fail_when_roll_above_probability(self, mock_now, mock_load, mock_save):
        mock_now.return_value = datetime(2026, 3, 10, 14, 0)
        with patch("random.random", return_value=0.9):
            result = _roll_dice(0.7, 1)
        assert result is False

    @patch.object(service, "_save_roll")
    @patch.object(service, "_load_rolls", return_value=[])
    @patch.object(service, "_now")
    def test_logs_roll_entry(self, mock_now, mock_load, mock_save):
        mock_now.return_value = datetime(2026, 3, 10, 14, 0)
        with patch("random.random", return_value=0.5):
            _roll_dice(0.7, 2)
        mock_save.assert_called_once()
        entry = mock_save.call_args[0][0]
        assert entry["session_num"] == 2
        assert entry["result"] == "COMMIT"
        assert entry["probability"] == 0.7


# ═══════════════════════════════════════════════════════════════════════
# Roll / Schedule Persistence
# ═══════════════════════════════════════════════════════════════════════

class TestRollPersistence:
    """Tests for _load_rolls / _save_roll."""

    def test_load_empty(self, tmp_path: Path):
        with patch.object(service, "ROLLS_FILE", tmp_path / "rolls.json"):
            result = _load_rolls()
        assert result == []

    def test_save_and_load(self, tmp_path: Path):
        rolls_file = tmp_path / "rolls.json"
        with patch.object(service, "ROLLS_FILE", rolls_file):
            _save_roll({"test": 1})
            _save_roll({"test": 2})
            result = _load_rolls()
        assert len(result) == 2
        assert result[0]["test"] == 1

    def test_load_corrupt_returns_empty(self, tmp_path: Path):
        rolls_file = tmp_path / "rolls.json"
        rolls_file.write_text("not json!!!", encoding="utf-8")
        with patch.object(service, "ROLLS_FILE", rolls_file):
            result = _load_rolls()
        assert result == []


class TestSchedulePersistence:
    """Tests for _load_schedule / _save_schedule."""

    def test_load_empty(self, tmp_path: Path):
        with patch.object(service, "SCHEDULE_FILE", tmp_path / "schedule.json"):
            result = _load_schedule()
        assert result == {}

    def test_save_and_load(self, tmp_path: Path):
        schedule_file = tmp_path / "schedule.json"
        with patch.object(service, "SCHEDULE_FILE", schedule_file):
            _save_schedule({"date": "2026-03-10", "sessions": []})
            result = _load_schedule()
        assert result["date"] == "2026-03-10"


# ═══════════════════════════════════════════════════════════════════════
# Repo Picker
# ═══════════════════════════════════════════════════════════════════════

class TestPickRepo:
    """Tests for _pick_repo."""

    @patch.object(service, "list_managed_repos")
    def test_returns_none_with_no_repos(self, mock_list):
        mock_list.return_value = []
        assert _pick_repo() is None

    @patch.object(service, "list_managed_repos")
    def test_returns_single_repo(self, mock_list):
        mock_list.return_value = [{"name": "solo", "last_session": "2026-03-08"}]
        result = _pick_repo()
        assert result["name"] == "solo"

    @patch.object(service, "list_managed_repos")
    def test_picks_from_multiple(self, mock_list):
        mock_list.return_value = [
            {"name": "repo-a", "last_session": "2026-03-01"},
            {"name": "repo-b", "last_session": "2026-03-01"},
        ]
        result = _pick_repo()
        assert result["name"] in ("repo-a", "repo-b")

    @patch.object(service, "list_managed_repos")
    def test_recency_boost_favors_yesterday(self, mock_list):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        mock_list.return_value = [
            {"name": "old-repo", "last_session": "2020-01-01"},
            {"name": "recent-repo", "last_session": yesterday},
        ]
        # Run many times — the recent repo should win more often
        picks = {"old-repo": 0, "recent-repo": 0}
        random.seed(42)
        for _ in range(200):
            result = _pick_repo()
            picks[result["name"]] += 1
        # With RECENCY_BOOST=2.0, recent-repo should win ~67% of the time
        assert picks["recent-repo"] > picks["old-repo"]
