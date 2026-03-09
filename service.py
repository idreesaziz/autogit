"""autogit background service — runs 24/7 as a persistent process.

At the deadline hour each day, rolls the commit-dice in a loop:
  roll → pass? → do work → roll again → pass? → do work → …
  …until a roll fails, then stop for the day.

With probability p the bot does at least 1 session, p² for 2, p³ for 3,
etc.  (geometric distribution — expected sessions ≈ p / (1 − p)).

Every roll is logged to autogit_rolls.json for CLI visibility.

Designed to be launched via `pythonw.exe` so it runs without a console
window. Communicates status through a PID file and log files.

Usage (managed by main.py):
    pythonw.exe service.py          # start silently (no window)
    python service.py               # start with console (for debugging)
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config import (
    DAILY_DEADLINE_HOUR,
    LOCAL_REPOS_DIR,
    TIMEZONE,
    WEEKDAY_BASE,
    MOMENTUM_BOOST,
    MOMENTUM_DECAY,
)
from github_ops.api import list_managed_repos
from ui.menu import run_all_repos

# ── Paths ────────────────────────────────────────────────────────────
PID_FILE = LOCAL_REPOS_DIR / "autogit_service.pid"
LOG_FILE = LOCAL_REPOS_DIR / "autogit_service.log"
ROLLS_FILE = LOCAL_REPOS_DIR / "autogit_rolls.json"

# ── Delay between consecutive rolls in a session (seconds) ──────────
INTER_ROLL_DELAY = 5  # small pause between successive rolls in one sequence

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("autogit-service")


def _now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def _all_repos_ran_today() -> bool:
    repos = list_managed_repos()
    if not repos:
        return False
    today = date.today().isoformat()
    return all(r.get("last_session") == today for r in repos)


def _any_repo_ran_yesterday() -> bool:
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


def _load_rolls() -> list[dict]:
    """Load the roll history from disk."""
    if not ROLLS_FILE.exists():
        return []
    try:
        return json.loads(ROLLS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_roll(entry: dict) -> None:
    """Append a single roll entry to the rolls log file."""
    rolls = _load_rolls()
    rolls.append(entry)
    ROLLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROLLS_FILE.write_text(json.dumps(rolls, indent=2), encoding="utf-8")


def _get_today_probability() -> float:
    """Compute today's commit probability (base + momentum)."""
    weekday = _now().weekday()
    base = WEEKDAY_BASE.get(weekday, 0.5)
    momentum = MOMENTUM_BOOST if _any_repo_ran_yesterday() else MOMENTUM_DECAY
    return max(0.0, min(1.0, base + momentum))


def _roll_dice(probability: float, session_num: int) -> bool:
    """Do a single dice roll, log it, and return whether it passed."""
    now = _now()
    weekday = now.weekday()
    base = WEEKDAY_BASE.get(weekday, 0.5)
    momentum = probability - base  # recover the momentum that was applied
    roll = random.random()
    passed = roll < probability

    entry = {
        "timestamp": now.isoformat(),
        "weekday": now.strftime("%A"),
        "session_num": session_num,
        "base": round(base, 2),
        "momentum": round(momentum, 2),
        "probability": round(probability, 2),
        "roll": round(roll, 4),
        "result": "COMMIT" if passed else "STOP",
    }
    _save_roll(entry)

    total = len(_load_rolls())
    log.info(
        "Roll #%d (session %d): %s, prob=%.0f%%, roll=%.4f → %s",
        total, session_num, now.strftime("%A"),
        probability * 100, roll, entry["result"],
    )
    return passed


def _run_rolling_sequence() -> None:
    """Roll → work → roll → work → … until a roll fails.

    Each passing roll triggers an autonomous session on all repos.
    The sequence stops on the first failing roll (geometric distribution).
    """
    probability = _get_today_probability()
    session_num = 0

    while True:
        session_num += 1
        passed = _roll_dice(probability, session_num)

        if not passed:
            log.info(
                "Stopping after %d roll(s) today (last roll failed).",
                session_num,
            )
            return

        log.info("Roll %d passed — running autonomous sessions.", session_num)
        try:
            run_all_repos(silent=True)
            log.info("Session %d completed successfully.", session_num)
        except Exception as exc:
            log.error("Session %d error: %s", session_num, exc)

        # Small pause before the next roll
        time.sleep(INTER_ROLL_DELAY)


def _write_pid() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _handle_stop(signum, frame) -> None:
    log.info("Received stop signal (%s). Shutting down.", signum)
    _remove_pid()
    sys.exit(0)


def is_running() -> int | None:
    """Return the service PID if it's running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None

    # Check if process is actually alive
    try:
        os.kill(pid, 0)  # signal 0 = just check existence
        return pid
    except OSError:
        # Stale PID file — process is dead
        _remove_pid()
        return None


def run_service() -> None:
    """Main service loop — runs forever, checks deadline every 30 seconds."""
    existing = is_running()
    if existing:
        log.warning("Service already running (PID %d). Exiting.", existing)
        print(f"Service already running (PID {existing}).")
        return

    _write_pid()
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    log.info("Service started (PID %d). Deadline hour: %d:00 %s",
             os.getpid(), DAILY_DEADLINE_HOUR, TIMEZONE)

    last_roll_date: str = ""

    try:
        while True:
            now = _now()
            today = now.date().isoformat()

            # Once per day, after the deadline hour, run the full rolling sequence
            if now.hour >= DAILY_DEADLINE_HOUR and today != last_roll_date:
                last_roll_date = today
                log.info("Deadline hour reached — starting dice roll sequence.")
                _run_rolling_sequence()

            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Interrupted. Shutting down.")
    finally:
        _remove_pid()


if __name__ == "__main__":
    run_service()
