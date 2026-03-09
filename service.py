"""autogit background service — runs 24/7 as a persistent process.

Checks the deadline every 30 seconds. When the deadline hour arrives
and no session has run today, it rolls the commit-cadence dice and
(if the roll passes) runs autonomous sessions on all managed repos.

Designed to be launched via `pythonw.exe` so it runs without a console
window. Communicates status through a PID file and a log file.

Usage (managed by main.py):
    pythonw.exe service.py          # start silently (no window)
    python service.py               # start with console (for debugging)
"""

from __future__ import annotations

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


def _should_commit_today() -> bool:
    weekday = _now().weekday()
    base = WEEKDAY_BASE.get(weekday, 0.5)
    momentum = MOMENTUM_BOOST if _any_repo_ran_yesterday() else MOMENTUM_DECAY
    probability = max(0.0, min(1.0, base + momentum))
    roll = random.random()
    log.info(
        "Commit check: %s, base=%.2f, momentum=%.2f, prob=%.0f%%, roll=%.2f → %s",
        _now().strftime("%A"), base, momentum, probability * 100, roll,
        "COMMIT" if roll < probability else "SKIP",
    )
    return roll < probability


def _check_deadline() -> None:
    """Core check — called every loop iteration."""
    now = _now()

    if now.hour < DAILY_DEADLINE_HOUR:
        return  # not time yet

    if _all_repos_ran_today():
        return  # already done

    if not _should_commit_today():
        log.info("Skipping today based on commit cadence.")
        return

    log.info("Deadline reached — running autonomous sessions on all repos.")
    try:
        run_all_repos(silent=True)
        log.info("Autonomous sessions completed successfully.")
    except Exception as exc:
        log.error("Session error: %s", exc)


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

    last_check_date: str = ""

    try:
        while True:
            today = date.today().isoformat()

            # Only run the deadline check once per day
            if today != last_check_date:
                now = _now()
                if now.hour >= DAILY_DEADLINE_HOUR:
                    _check_deadline()
                    last_check_date = today

            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Interrupted. Shutting down.")
    finally:
        _remove_pid()


if __name__ == "__main__":
    run_service()
