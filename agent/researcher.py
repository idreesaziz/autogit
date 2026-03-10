"""Trending-topic research and new-repo creation via Gemini."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from rich.console import Console

from config import (
    GITHUB_USERNAME,
    LOCAL_REPOS_DIR,
    MAX_CONTEXT_CHARS,
)
from agent.memory import initialize_state, save_state
from agent.dna import generate_initial_dna
from github_ops.api import create_repo, register_repo
from github_ops.git_ops import init_repo, commit_and_push

console = Console()


# ── Trending scraper ─────────────────────────────────────────────────

def _scrape_trending(since: str = "daily") -> list[dict[str, str]]:
    """Scrape GitHub trending page and return repo metadata."""
    url = "https://github.com/trending"
    params = {"since": since} if since != "daily" else {}
    try:
        resp = requests.get(url, params=params, timeout=15, headers={
            "User-Agent": "autogit/1.0"
        })
        resp.raise_for_status()
    except requests.RequestException as exc:
        console.print(f"[yellow]Could not fetch trending: {exc}[/yellow]")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    repos: list[dict[str, str]] = []

    for article in soup.select("article.Box-row")[:20]:
        h2 = article.select_one("h2")
        if not h2:
            continue
        a_tag = h2.select_one("a")
        if not a_tag:
            continue
        full_name = a_tag.get("href", "").strip("/")
        desc_tag = article.select_one("p")
        description = desc_tag.get_text(strip=True) if desc_tag else ""
        lang_tag = article.select_one("[itemprop='programmingLanguage']")
        language = lang_tag.get_text(strip=True) if lang_tag else "Unknown"
        stars_tag = article.select("span.d-inline-block.float-sm-right")
        stars_today = stars_tag[0].get_text(strip=True) if stars_tag else ""

        repos.append({
            "name": full_name,
            "description": description,
            "language": language,
            "stars_today": stars_today,
        })

    return repos


def research_trending_ideas(gemini_call) -> list[dict[str, Any]]:
    """Scrape trending repos and ask Gemini for project ideas.

    Args:
        gemini_call: The rate-limited Gemini call function from session.py.
    """
    console.print("[dim]Scraping GitHub trending (daily)…[/dim]")
    daily = _scrape_trending("daily")
    console.print("[dim]Scraping GitHub trending (weekly)…[/dim]")
    weekly = _scrape_trending("weekly")

    trending_text = "## Daily Trending Repos\n"
    for r in daily:
        trending_text += f"- {r['name']} ({r['language']}): {r['description']} [{r['stars_today']}]\n"
    trending_text += "\n## Weekly Trending Repos\n"
    for r in weekly:
        trending_text += f"- {r['name']} ({r['language']}): {r['description']} [{r['stars_today']}]\n"

    # Cap context to avoid oversized prompt
    trending_text = trending_text[:MAX_CONTEXT_CHARS]

    prompt = f"""You are analyzing GitHub trending repositories to find gaps and opportunities.
Given these trending repos, identify 5-6 project ideas that:
- Complement or improve on what's trending (not copy it)
- Are small enough for one developer to build incrementally over weeks
- Would genuinely be useful to other developers
- Can be built in Python, JavaScript, or TypeScript
- Have a clear, searchable name and purpose

Trending data:
{trending_text}

Return ONLY a JSON array (no markdown fences, no explanation):
[{{"name": "repo-name", "tagline": "one line", "description": "2-3 sentences", "tech": ["Python"], "why_now": "why this is timely"}}]"""

    raw = gemini_call(prompt, system="You are a developer-tool analyst.")
    return _parse_json_array(raw)


def generate_ideas_from_hint(
    hint: str,
    gemini_call,
) -> list[dict[str, Any]]:
    """Generate project ideas based on a user-provided suggestion/hint.

    Args:
        hint: Free-text description of what the user wants.
        gemini_call: Rate-limited Gemini call function.
    """
    prompt = f"""A developer wants to create a new open-source project.  They gave this hint:

\"{hint}\"

Generate 5-6 concrete project ideas inspired by this hint.  Each idea should:
- Be small enough for one developer to build incrementally over weeks
- Be genuinely useful to other developers or end-users
- Can be built in Python, JavaScript, or TypeScript
- Have a clear, searchable name and purpose

Return ONLY a JSON array (no markdown fences, no explanation):
[{{"name": "repo-name", "tagline": "one line", "description": "2-3 sentences", "tech": ["Python"], "why_now": "why this idea is valuable"}}]"""

    raw = gemini_call(prompt, system="You are a developer-tool analyst.")
    return _parse_json_array(raw)


# ── New repo creation ────────────────────────────────────────────────

def create_repo_from_idea(
    idea: dict[str, Any],
    gemini_call,
) -> tuple[str, str]:
    """Create a GitHub repo from an idea dict, scaffold it, and push.

    Returns:
        (repo_url, local_path) tuple.
    """
    name = idea["name"]
    tagline = idea["tagline"]
    description = idea.get("description", tagline)
    tech = idea.get("tech", ["Python"])

    # 1. Create remote repo
    gh_repo = create_repo(name, tagline)
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{name}"

    # 2. Init local repo
    local_path = str(LOCAL_REPOS_DIR / name)
    repo = init_repo(local_path, repo_url)

    # 3. Generate scaffold via Gemini
    scaffold_prompt = f"""Generate the initial project scaffold for a new open-source project.

Project name: {name}
Tagline: {tagline}
Description: {description}
Tech stack: {', '.join(tech)}

Create these files (return ONLY valid JSON, no markdown fences):
{{
  "files": [
    {{"path": "README.md", "content": "...full content..."}},
    {{"path": ".gitignore", "content": "..."}},
    ...additional files appropriate for the tech stack...
  ]
}}

README should include: project name, description, features (planned), installation, usage placeholder, license (MIT).
Include a requirements.txt (Python) or package.json (JS/TS) as appropriate.
Include a main source file with a minimal working skeleton."""

    raw = gemini_call(scaffold_prompt, system="You are a senior open-source developer.")
    scaffold = _parse_json_object(raw)

    # 4. Write files to disk
    written_files: list[str] = []
    for entry in scaffold.get("files", []):
        fpath = Path(local_path) / entry["path"]
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(entry["content"], encoding="utf-8")
        written_files.append(entry["path"])

    # 5. Initialize agent state
    state = initialize_state(name, repo_url, description, tech)
    save_state(local_path, state)
    written_files.append(".agent_state.json")

    # 6. Generate initial DNA
    project_info = {
        "name": name,
        "description": description,
        "tech_stack": tech,
        "tagline": tagline,
    }
    generate_initial_dna(local_path, project_info, gemini_call)
    written_files.append(".dna")

    # 7. Commit and push
    commit_and_push(local_path, written_files, "feat: initial project scaffold")

    # 8. Register in managed_repos.json
    register_repo(name, repo_url, local_path)

    return repo_url, local_path


# ── JSON parsing helpers ─────────────────────────────────────────────

def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array from a Gemini response, tolerating markdown fences."""
    cleaned = _strip_fences(raw)
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Try to find array in the text
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    console.print("[yellow]Warning: could not parse Gemini response as JSON array.[/yellow]")
    return []


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a Gemini response."""
    cleaned = _strip_fences(raw)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    console.print("[yellow]Warning: could not parse Gemini response as JSON object.[/yellow]")
    return {}


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from Gemini output."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
