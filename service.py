"""autogit background service — runs 24/7 as a persistent process.

Two-phase daily cycle:
  1. PLAN  — at the deadline hour, roll the geometric dice to decide
     how many sessions today.  For each session, sample a time from a
     Gaussian distribution (peak ≈ late afternoon, spread across the
     full window until 20 min before tomorrow's deadline).
  2. EXECUTE — the service sleeps / polls and fires each session when
     its scheduled time arrives.

Every roll and every scheduled session is logged to JSON files so the
CLI can display full history.

Designed to be launched via `pythonw.exe` so it runs without a console
window.  Communicates status through a PID file and log files.

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
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from config import (
    DAILY_DEADLINE_HOUR,
    LOCAL_REPOS_DIR,
    TIMEZONE,
    WEEKDAY_BASE,
    MOMENTUM_BOOST,
    MOMENTUM_DECAY,
    GAUSS_MEAN_OFFSET,
    GAUSS_STDDEV,
    GAUSS_WINDOW,
)
from github_ops.api import list_managed_repos
from ui.menu import run_all_repos

# ── Paths ────────────────────────────────────────────────────────────
PID_FILE = LOCAL_REPOS_DIR / "autogit_service.pid"
LOG_FILE = LOCAL_REPOS_DIR / "autogit_service.log"
ROLLS_FILE = LOCAL_REPOS_DIR / "autogit_rolls.json"
SCHEDULE_FILE = LOCAL_REPOS_DIR / "autogit_schedule.json"

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
    momentum = probability - base
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


# ── Schedule persistence ─────────────────────────────────────────────

def _load_schedule() -> dict:
    """Load today's schedule from disk."""
    if not SCHEDULE_FILE.exists():
        return {}
    try:
        return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_schedule(schedule: dict) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(schedule, indent=2), encoding="utf-8")


# ── Gaussian time sampling ───────────────────────────────────────────

def _sample_session_time(deadline: datetime) -> datetime:
    """Sample a session time from a Gaussian centred after the deadline.

    The mean is GAUSS_MEAN_OFFSET hours after the deadline, with
    GAUSS_STDDEV hours of spread.  Results are clamped to the window
    [deadline, deadline + GAUSS_WINDOW hours].
    """
    mean_offset = GAUSS_MEAN_OFFSET * 3600        # seconds
    stddev = GAUSS_STDDEV * 3600                   # seconds
    max_offset = GAUSS_WINDOW * 3600               # seconds

    offset = random.gauss(mean_offset, stddev)
    offset = max(0.0, min(max_offset, offset))    # clamp to window
    return deadline + timedelta(seconds=offset)


# ── Phase 1: Plan the day ────────────────────────────────────────────

def _plan_day(live: bool = False) -> dict:
    """Roll the geometric dice, sample Gaussian times, save schedule.

    Returns the schedule dict with keys: date, probability, rolls,
    sessions (list of {session_num, scheduled_time, status}).
    """
    from rich.console import Console
    con = Console() if live else None

    probability = _get_today_probability()
    now = _now()
    deadline = now.replace(hour=DAILY_DEADLINE_HOUR, minute=0, second=0, microsecond=0)
    if now < deadline:
        deadline = deadline  # shouldn't happen but guard
    session_count = 0
    roll_num = 0

    # Roll the geometric dice
    while True:
        roll_num += 1
        passed = _roll_dice(probability, roll_num)

        if live:
            rolls = _load_rolls()
            entry = rolls[-1] if rolls else {}
            roll_val = entry.get("roll", 0)
            result = entry.get("result", "?")
            style = "green" if result == "COMMIT" else "red"
            con.print(
                f"  Roll {roll_num}: "
                f"prob=[bold]{probability:.0%}[/bold]  "
                f"roll=[bold]{roll_val:.4f}[/bold]  "
                f"→ [{style}]{result}[/{style}]"
            )

        if not passed:
            break
        session_count += 1

    # Sample Gaussian times for each session
    sessions = []
    for i in range(1, session_count + 1):
        scheduled = _sample_session_time(deadline)
        sessions.append({
            "session_num": i,
            "scheduled_time": scheduled.isoformat(),
            "status": "pending",
        })

    # Sort by scheduled time
    sessions.sort(key=lambda s: s["scheduled_time"])

    schedule = {
        "date": now.date().isoformat(),
        "probability": round(probability, 2),
        "total_rolls": roll_num,
        "session_count": session_count,
        "sessions": sessions,
    }
    _save_schedule(schedule)

    log.info(
        "Day planned: %d session(s) from %d roll(s) (prob=%.0f%%).",
        session_count, roll_num, probability * 100,
    )
    for s in sessions:
        log.info("  Session %d scheduled at %s", s["session_num"], s["scheduled_time"])

    if live:
        con.print(
            f"\n  [dim]Sequence done: {session_count} session(s) scheduled, "
            f"stopped on roll {roll_num}.[/dim]"
        )
        if sessions:
            con.print()
            for s in sessions:
                ts = s["scheduled_time"].replace("T", " ")[:19]
                con.print(f"  📅 Session {s['session_num']} → [bold]{ts}[/bold]")

    return schedule


# ── Phase 2: Execute scheduled sessions ──────────────────────────────

def _run_pending_sessions(live: bool = False, run_now: bool = False) -> None:
    """Check the schedule and run any sessions whose time has arrived.

    Args:
        live: Print output to console.
        run_now: If True, run all pending sessions immediately (for debug).
    """
    from rich.console import Console
    con = Console() if live else None

    schedule = _load_schedule()
    if not schedule:
        return

    now = _now()
    today = now.date().isoformat()

    # Only process today's schedule
    if schedule.get("date") != today:
        return

    changed = False
    for session in schedule.get("sessions", []):
        if session["status"] != "pending":
            continue

        if not run_now:
            scheduled_time = datetime.fromisoformat(session["scheduled_time"])
            if now < scheduled_time:
                continue

        # Time has arrived — run the session
        session_num = session["session_num"]
        log.info("Session %d: scheduled time reached (%s). Running.", session_num, session["scheduled_time"])

        if live:
            ts = session["scheduled_time"].replace("T", " ")[:19]
            con.print(f"\n  [bold cyan]Running session {session_num}[/bold cyan] (scheduled {ts})")

        session["status"] = "running"
        _save_schedule(schedule)

        try:
            run_all_repos(silent=not live, force=True)
            session["status"] = "done"
            log.info("Session %d completed successfully.", session_num)
        except Exception as exc:
            session["status"] = "error"
            log.error("Session %d error: %s", session_num, exc)
            if live:
                con.print(f"  [red]Session error: {exc}[/red]")

        changed = True
        _save_schedule(schedule)

    if changed and live:
        done = sum(1 for s in schedule["sessions"] if s["status"] == "done")
        total = len(schedule["sessions"])
        con.print(f"\n  [dim]{done}/{total} sessions completed.[/dim]")


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
    """Main service loop — plans the day at deadline, runs sessions at scheduled times."""
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

    last_plan_date: str = ""

    try:
        while True:
            now = _now()
            today = now.date().isoformat()

            # Phase 1: Plan the day (once, at deadline hour)
            if now.hour >= DAILY_DEADLINE_HOUR and today != last_plan_date:
                last_plan_date = today
                log.info("Deadline hour reached — planning today's sessions.")
                _plan_day()

            # Phase 2: Run any sessions whose scheduled time has arrived
            _run_pending_sessions()

            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Interrupted. Shutting down.")
    finally:
        _remove_pid()


if __name__ == "__main__":
    run_service()
