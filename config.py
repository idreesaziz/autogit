"""Configuration module — loads .env and exposes typed constants."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

_REQUIRED_VARS = [
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_USERNAME",
    "LOCAL_REPOS_DIR",
]


def _require(var: str) -> str:
    """Return the env var value or exit with a clear error."""
    value = os.getenv(var)
    if not value:
        print(f"[ERROR] Missing required environment variable: {var}")
        print(f"        Copy .env.example → .env and fill in your values.")
        sys.exit(1)
    return value


# ── Required secrets / paths ─────────────────────────────────────────
GEMINI_API_KEY: str = _require("GEMINI_API_KEY")
GITHUB_TOKEN: str = _require("GITHUB_TOKEN")
GITHUB_USERNAME: str = _require("GITHUB_USERNAME")
LOCAL_REPOS_DIR: Path = Path(_require("LOCAL_REPOS_DIR"))

# ── Optional settings with defaults ─────────────────────────────────
DAILY_DEADLINE_HOUR: int = int(os.getenv("DAILY_DEADLINE_HOUR", "11"))
TIMEZONE: str = os.getenv("TIMEZONE", "Europe/London")

# ── Rate-limit / budget constants ────────────────────────────────────
MAX_REQUESTS_PER_SESSION: int = 40
REQUEST_DELAY_SECONDS: int = 13  # 5 RPM → 12s min; 13s gives buffer

# ── Gemini model names ───────────────────────────────────────────────
GEMINI_PRIMARY_MODEL: str = "gemini-3.1-pro-preview"
GEMINI_FALLBACK_MODEL: str = "gemini-3-flash-preview"

# ── Context caps ─────────────────────────────────────────────────────
MAX_CONTEXT_CHARS: int = 15_000
MAX_FILE_CHARS: int = 3_000

# ── Commit cadence weights (weekday → base probability) ─────────────
WEEKDAY_BASE: dict[int, float] = {
    0: 0.70,  # Mon  → avg 2.3 commits
    1: 0.70,  # Tue  → avg 2.3 commits
    2: 0.65,  # Wed  → avg 1.9 commits
    3: 0.65,  # Thu  → avg 1.9 commits
    4: 0.55,  # Fri  → avg 1.2 commits
    5: 0.35,  # Sat  → avg 0.5 commits
    6: 0.25,  # Sun  → avg 0.3 commits
}
MOMENTUM_BOOST: float = 0.10   # bonus if committed yesterday
MOMENTUM_DECAY: float = -0.05  # penalty if skipped yesterday

# ── Gaussian time distribution ───────────────────────────────────────
# Sessions are scheduled with a Gaussian centered GAUSS_MEAN_OFFSET hours
# after the deadline hour, with GAUSS_STDDEV hours of spread.
# ~68% of sessions fall within ±1σ of the mean.
GAUSS_MEAN_OFFSET: float = 6.5   # hours after deadline (e.g. 11+6.5 = 17:30)
GAUSS_STDDEV: float = 1.5        # hours (1σ covers ~4 PM – 7 PM)
GAUSS_WINDOW: float = 23.667     # max hours after deadline (23h40m = next day minus 20min)

# ── Session mode ─────────────────────────────────────────────────────
# "oneshot" = plan → generate → validate → push (fast, 2-3 Gemini calls)
# "agentic" = plan → generate → validate → test → review → fix loops → push
SESSION_MODE: str = os.getenv("SESSION_MODE", "agentic").lower()
AGENTIC_MAX_FIX_ATTEMPTS: int = 3  # max retry loops for test/review failures

# ── Repo selection ───────────────────────────────────────────────────
RECENCY_BOOST: float = 2.0  # weight multiplier for a repo that was picked yesterday

# ── Ensure repos directory exists ────────────────────────────────────
LOCAL_REPOS_DIR.mkdir(parents=True, exist_ok=True)
