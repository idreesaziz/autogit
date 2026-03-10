"""DNA system — hybrid AST + Gemini codebase manifest.

The .dna file is a JSON manifest that gives the agent a complete map of
every file in the repo:  declarations, signatures, imports, and a short
AI-written description of each symbol.

Architecture:
  Layer 1 (AST)    — extract structure mechanically from source code.
  Layer 2 (Gemini) — annotate each declaration with a one-line description.
  Diff engine      — detect what changed so only new/modified symbols need AI.

Public API:
  load_dna(repo_path)                → dict
  save_dna(repo_path, dna)           → None
  index_repo(repo_path)              → dict   (Layer 1 — pure AST)
  diff_dna(old, new)                 → dict   (changed/added/removed entries)
  update_dna(repo_path, gemini_call) → dict   (full pipeline: index → diff → annotate → save)
  generate_initial_dna(repo_path, project_info, gemini_call) → dict
  render_dna_context(dna)            → str    (formatted for prompt injection)
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

DNA_FILENAME = ".dna"

# Directories to skip when scanning repo files
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
              ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
              ".eggs", "*.egg-info"}

# File extensions we can AST-parse
_PYTHON_EXTS = {".py"}

# File extensions we index (but can't AST-parse)
_TEXT_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg",
              ".ini", ".html", ".css", ".js", ".ts", ".tsx", ".jsx",
              ".sh", ".bat", ".ps1", ".env", ".gitignore", ".dockerfile"}

# Binary / irrelevant files to skip entirely
_SKIP_FILES = {".dna", ".agent_state.json", ".DS_Store", "Thumbs.db"}


# ═══════════════════════════════════════════════════════════════════════
# Layer 1 — AST Indexer
# ═══════════════════════════════════════════════════════════════════════

def _format_arg(arg: ast.arg) -> str:
    """Format a single function argument with optional type annotation."""
    name = arg.arg
    if arg.annotation:
        return f"{name}: {ast.unparse(arg.annotation)}"
    return name


def _format_arguments(args: ast.arguments) -> str:
    """Format a full arguments node into a signature string."""
    parts: list[str] = []

    # Positional-only args
    for a in args.posonlyargs:
        parts.append(_format_arg(a))
    if args.posonlyargs:
        parts.append("/")

    # Regular args (with defaults aligned from the right)
    n_regular = len(args.args)
    n_defaults = len(args.defaults)
    for i, a in enumerate(args.args):
        formatted = _format_arg(a)
        default_idx = i - (n_regular - n_defaults)
        if default_idx >= 0:
            default_val = ast.unparse(args.defaults[default_idx])
            formatted += f"={default_val}"
        parts.append(formatted)

    # *args
    if args.vararg:
        parts.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only args
    for i, a in enumerate(args.kwonlyargs):
        formatted = _format_arg(a)
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            formatted += f"={ast.unparse(args.kw_defaults[i])}"
        parts.append(formatted)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{_format_arg(args.kwarg)}")

    return ", ".join(parts)


def _format_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Get the return type annotation string, or empty."""
    if node.returns:
        return ast.unparse(node.returns)
    return ""


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names."""
    return [ast.unparse(d) for d in node.decorator_list]


def _index_python_file(file_path: Path) -> dict[str, Any]:
    """Parse a single Python file and extract all declarations."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return {"parse_error": True}

    result: dict[str, Any] = {
        "functions": {},
        "classes": {},
        "constants": {},
        "internal_imports": [],
        "external_imports": [],
    }

    for node in ast.iter_child_nodes(tree):
        # ── Functions ────────────────────────────────────────────
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig_args = _format_arguments(node.args)
            ret = _format_return(node)
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            sig = f"{prefix}{node.name}({sig_args})"
            if ret:
                sig += f" -> {ret}"
            decorators = _get_decorators(node)
            result["functions"][node.name] = {
                "signature": sig,
                "decorators": decorators,
                "line": node.lineno,
                "description": "",  # filled by Layer 2
            }

        # ── Classes ──────────────────────────────────────────────
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            decorators = _get_decorators(node)
            methods: dict[str, Any] = {}
            class_attrs: dict[str, str] = {}

            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig_args = _format_arguments(item.args)
                    ret = _format_return(item)
                    prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                    sig = f"{prefix}{item.name}({sig_args})"
                    if ret:
                        sig += f" -> {ret}"
                    methods[item.name] = {
                        "signature": sig,
                        "decorators": _get_decorators(item),
                        "line": item.lineno,
                        "description": "",
                    }
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    ann = ast.unparse(item.annotation) if item.annotation else ""
                    class_attrs[item.target.id] = ann

            result["classes"][node.name] = {
                "bases": bases,
                "decorators": decorators,
                "methods": methods,
                "class_attributes": class_attrs,
                "line": node.lineno,
                "description": "",
            }

        # ── Module-level assignments (constants) ─────────────────
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        val_repr = ast.unparse(node.value)
                        if len(val_repr) > 80:
                            val_repr = val_repr[:77] + "..."
                    except Exception:
                        val_repr = "..."
                    result["constants"][target.id] = {
                        "value_preview": val_repr,
                        "line": node.lineno,
                        "description": "",
                    }

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            ann = ast.unparse(node.annotation) if node.annotation else ""
            if name.isupper() or name.startswith("_") and name[1:].isupper():
                try:
                    val_repr = ast.unparse(node.value) if node.value else ""
                    if len(val_repr) > 80:
                        val_repr = val_repr[:77] + "..."
                except Exception:
                    val_repr = ""
                result["constants"][name] = {
                    "type": ann,
                    "value_preview": val_repr,
                    "line": node.lineno,
                    "description": "",
                }

        # ── Imports ──────────────────────────────────────────────
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["external_imports"].append(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            # Heuristic: if it matches a known project package or is relative
            if node.level > 0 or top in _LIKELY_INTERNAL:
                result["internal_imports"].append(module)
            else:
                result["external_imports"].append(top)

    # Deduplicate imports
    result["internal_imports"] = sorted(set(result["internal_imports"]))
    result["external_imports"] = sorted(set(result["external_imports"]))

    return result


# Internally-import detection: names that look like project packages
# (populated dynamically per-repo during index_repo)
_LIKELY_INTERNAL: set[str] = set()


def _discover_internal_packages(repo_path: Path) -> set[str]:
    """Find top-level Python package names in the repo."""
    packages: set[str] = set()
    for item in repo_path.iterdir():
        if item.name.startswith(".") or item.name in _SKIP_DIRS:
            continue
        if item.is_dir() and (item / "__init__.py").exists():
            packages.add(item.name)
        elif item.is_file() and item.suffix == ".py" and item.name != "setup.py":
            packages.add(item.stem)
    return packages


def _should_skip_dir(name: str) -> bool:
    """Check if a directory should be skipped during indexing."""
    if name.startswith("."):
        return True
    if name in _SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def index_repo(repo_path: str | Path) -> dict[str, dict]:
    """Walk the repo and AST-index every source file.

    Returns a dict mapping relative file paths to their extracted structure.
    """
    global _LIKELY_INTERNAL
    repo_path = Path(repo_path)
    _LIKELY_INTERNAL = _discover_internal_packages(repo_path)

    files_index: dict[str, dict] = {}

    for root, dirs, filenames in os.walk(repo_path):
        # Prune skipped directories in-place
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]

        for fname in filenames:
            if fname in _SKIP_FILES:
                continue

            full_path = Path(root) / fname
            rel_path = str(full_path.relative_to(repo_path)).replace("\\", "/")
            ext = full_path.suffix.lower()

            if ext in _PYTHON_EXTS:
                files_index[rel_path] = _index_python_file(full_path)
            elif ext in _TEXT_EXTS or fname in _TEXT_EXTS:
                # Non-Python text file — record existence, Gemini describes
                files_index[rel_path] = {
                    "type": "non-python",
                    "extension": ext,
                    "size_bytes": full_path.stat().st_size,
                    "description": "",  # filled by Layer 2
                }

    return files_index


# ═══════════════════════════════════════════════════════════════════════
# Diff Engine
# ═══════════════════════════════════════════════════════════════════════

def diff_dna(old_files: dict, new_files: dict) -> dict[str, Any]:
    """Compare old DNA file entries against a fresh AST index.

    Returns:
        {
          "added_files": [rel_path, ...],
          "removed_files": [rel_path, ...],
          "modified_files": {
              rel_path: {
                  "added_symbols": [...],
                  "removed_symbols": [...],
                  "changed_symbols": [...],
              }
          },
          "unchanged_files": [rel_path, ...],
        }
    """
    old_keys = set(old_files.keys())
    new_keys = set(new_files.keys())

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    common = sorted(old_keys & new_keys)

    modified: dict[str, dict] = {}
    unchanged: list[str] = []

    for path in common:
        old_entry = old_files[path]
        new_entry = new_files[path]

        # Non-Python files: just check if size changed
        if new_entry.get("type") == "non-python":
            old_size = old_entry.get("size_bytes", 0)
            new_size = new_entry.get("size_bytes", 0)
            if old_size != new_size:
                modified[path] = {"size_changed": True}
            else:
                unchanged.append(path)
            continue

        # Python files: compare symbols
        file_diff = _diff_python_file(old_entry, new_entry)
        if file_diff:
            modified[path] = file_diff
        else:
            unchanged.append(path)

    return {
        "added_files": added,
        "removed_files": removed,
        "modified_files": modified,
        "unchanged_files": unchanged,
    }


def _diff_python_file(old: dict, new: dict) -> dict | None:
    """Compare two Python file indexes. Returns diff dict or None if identical."""
    changes: dict[str, list[str]] = {
        "added_symbols": [],
        "removed_symbols": [],
        "changed_symbols": [],
    }

    for category in ("functions", "classes", "constants"):
        old_syms = old.get(category, {})
        new_syms = new.get(category, {})

        old_names = set(old_syms.keys())
        new_names = set(new_syms.keys())

        for name in sorted(new_names - old_names):
            changes["added_symbols"].append(f"{category}.{name}")

        for name in sorted(old_names - new_names):
            changes["removed_symbols"].append(f"{category}.{name}")

        for name in sorted(old_names & new_names):
            # Compare signature (for functions/methods)
            old_sig = old_syms[name].get("signature", "")
            new_sig = new_syms[name].get("signature", "")
            if old_sig != new_sig:
                changes["changed_symbols"].append(f"{category}.{name}")
                continue

            # For classes, also compare methods
            if category == "classes":
                old_methods = set(old_syms[name].get("methods", {}).keys())
                new_methods = set(new_syms[name].get("methods", {}).keys())
                if old_methods != new_methods:
                    changes["changed_symbols"].append(f"{category}.{name}")
                    continue
                # Check if any method signatures changed
                for m in old_methods & new_methods:
                    o_sig = old_syms[name]["methods"][m].get("signature", "")
                    n_sig = new_syms[name]["methods"][m].get("signature", "")
                    if o_sig != n_sig:
                        changes["changed_symbols"].append(f"{category}.{name}")
                        break

    if any(changes.values()):
        return changes
    return None


# ═══════════════════════════════════════════════════════════════════════
# DNA I/O
# ═══════════════════════════════════════════════════════════════════════

def load_dna(repo_path: str | Path) -> dict[str, Any]:
    """Load the .dna file from a repo. Returns empty dict if missing."""
    path = Path(repo_path) / DNA_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_dna(repo_path: str | Path, dna: dict[str, Any]) -> None:
    """Write the .dna file to the repo root."""
    path = Path(repo_path) / DNA_FILENAME
    path.write_text(json.dumps(dna, indent=2, ensure_ascii=False), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# Layer 2 — Gemini Annotator
# ═══════════════════════════════════════════════════════════════════════

def _build_annotation_prompt(file_path: str, file_entry: dict, source: str) -> str:
    """Build a prompt asking Gemini to describe declarations in one file."""
    symbols_to_describe: list[str] = []

    if file_entry.get("type") == "non-python":
        return (
            f"Describe what the file `{file_path}` does in one sentence.\n"
            f"File contents (may be truncated):\n```\n{source[:3000]}\n```\n\n"
            f"Return ONLY a JSON object: {{\"description\": \"...\"}}"
        )

    # Collect all symbols that need descriptions
    for func_name, info in file_entry.get("functions", {}).items():
        if not info.get("description"):
            symbols_to_describe.append(f"function `{info['signature']}`")

    for cls_name, info in file_entry.get("classes", {}).items():
        if not info.get("description"):
            symbols_to_describe.append(f"class `{cls_name}`")
        for method_name, m_info in info.get("methods", {}).items():
            if not m_info.get("description"):
                symbols_to_describe.append(f"method `{cls_name}.{m_info['signature']}`")

    for const_name, info in file_entry.get("constants", {}).items():
        if not info.get("description"):
            symbols_to_describe.append(f"constant `{const_name}`")

    if not symbols_to_describe:
        return ""  # nothing to annotate

    symbols_list = "\n".join(f"- {s}" for s in symbols_to_describe)

    return (
        f"For the file `{file_path}`, write a one-sentence description for each symbol below.\n"
        f"Also write a one-sentence purpose for the file itself.\n\n"
        f"Symbols:\n{symbols_list}\n\n"
        f"Source code (may be truncated):\n```python\n{source[:5000]}\n```\n\n"
        f"Return ONLY valid JSON (no markdown fences):\n"
        f"{{\n"
        f'  "file_purpose": "...",\n'
        f'  "symbols": {{\n'
        f'    "<symbol_name>": "one-line description",\n'
        f"    ...\n"
        f"  }}\n"
        f"}}"
    )


def _apply_annotations(file_entry: dict, annotations: dict) -> dict:
    """Merge Gemini-generated descriptions into a file's index entry."""
    symbols = annotations.get("symbols", {})

    # Apply file-level purpose
    if "file_purpose" in annotations:
        file_entry["purpose"] = annotations["file_purpose"]

    # Apply to functions
    for name, info in file_entry.get("functions", {}).items():
        if name in symbols:
            info["description"] = symbols[name]

    # Apply to classes and their methods
    for cls_name, cls_info in file_entry.get("classes", {}).items():
        if cls_name in symbols:
            cls_info["description"] = symbols[cls_name]
        for method_name, m_info in cls_info.get("methods", {}).items():
            # Try "ClassName.method_name" first, then just "method_name"
            key = f"{cls_name}.{method_name}"
            if key in symbols:
                m_info["description"] = symbols[key]
            elif method_name in symbols:
                m_info["description"] = symbols[method_name]

    # Apply to constants
    for name, info in file_entry.get("constants", {}).items():
        if name in symbols:
            info["description"] = symbols[name]

    return file_entry


def annotate_files(
    repo_path: str | Path,
    files_to_annotate: list[str],
    file_entries: dict[str, dict],
    gemini_call,
) -> dict[str, dict]:
    """Call Gemini to describe symbols in the given files.

    Batches all files into a single Gemini call to minimise API usage.
    Returns updated file entries.
    """
    repo_path = Path(repo_path)

    # Collect per-file prompts, then merge into one batched call
    file_sections: list[str] = []
    annotatable_files: list[str] = []

    for rel_path in files_to_annotate:
        entry = file_entries.get(rel_path)
        if not entry:
            continue

        full_path = repo_path / rel_path
        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        section = _build_annotation_section(rel_path, entry, source)
        if not section:
            continue

        file_sections.append(section)
        annotatable_files.append(rel_path)

    if not file_sections:
        return file_entries

    batched_prompt = (
        "Describe the purpose and symbols for each file below.\n"
        "Return ONLY valid JSON (no markdown fences) with this structure:\n"
        '{"files": {"<relative_path>": {"file_purpose": "...", "symbols": {"<name>": "one-line desc", ...}}, ...}}\n\n'
        + "\n---\n".join(file_sections)
    )

    try:
        raw = gemini_call(
            batched_prompt,
            system="You are a code documentation expert. Be concise — one sentence per symbol.",
        )
        parsed = _parse_annotation_response(raw)
        files_data = parsed.get("files", parsed)  # tolerate flat or nested

        for rel_path in annotatable_files:
            annotations = files_data.get(rel_path, {})
            if annotations:
                file_entries[rel_path] = _apply_annotations(file_entries[rel_path], annotations)
    except Exception as exc:
        console.print(f"[yellow]DNA batch annotation failed: {exc}[/yellow]")

    return file_entries


def _build_annotation_section(file_path: str, file_entry: dict, source: str) -> str:
    """Build a section for the batched annotation prompt for one file."""
    if file_entry.get("type") == "non-python":
        return (
            f"### File: `{file_path}`\n"
            f"```\n{source[:2000]}\n```\n"
            f"Describe what this file does."
        )

    symbols_to_describe: list[str] = []
    for func_name, info in file_entry.get("functions", {}).items():
        if not info.get("description"):
            symbols_to_describe.append(f"function `{info['signature']}`")
    for cls_name, info in file_entry.get("classes", {}).items():
        if not info.get("description"):
            symbols_to_describe.append(f"class `{cls_name}`")
        for method_name, m_info in info.get("methods", {}).items():
            if not m_info.get("description"):
                symbols_to_describe.append(f"method `{cls_name}.{m_info['signature']}`")
    for const_name, info in file_entry.get("constants", {}).items():
        if not info.get("description"):
            symbols_to_describe.append(f"constant `{const_name}`")

    if not symbols_to_describe:
        return ""

    symbols_list = "\n".join(f"- {s}" for s in symbols_to_describe)
    return (
        f"### File: `{file_path}`\n"
        f"Symbols:\n{symbols_list}\n\n"
        f"```python\n{source[:3000]}\n```\n"
        f"Write a one-sentence purpose for the file and describe each symbol."
    )


def _parse_annotation_response(raw: str) -> dict:
    """Parse Gemini's annotation response, tolerating fences."""
    cleaned = raw.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ═══════════════════════════════════════════════════════════════════════
# High-Level API
# ═══════════════════════════════════════════════════════════════════════

def update_dna(
    repo_path: str | Path,
    gemini_call,
) -> dict[str, Any]:
    """Full DNA update pipeline: index → diff → annotate → save.

    Efficient: only calls Gemini for files with new/changed symbols.
    Returns the updated DNA dict.
    """
    repo_path = Path(repo_path)
    console.print("[dim]Scanning codebase (AST)…[/dim]")

    # Load existing DNA
    dna = load_dna(repo_path)
    old_files = dna.get("files", {})

    # Fresh AST index
    new_files = index_repo(repo_path)

    # Diff
    diff = diff_dna(old_files, new_files)
    added = diff["added_files"]
    modified = list(diff["modified_files"].keys())
    removed = diff["removed_files"]

    # Carry over existing descriptions for unchanged symbols
    for path in diff["unchanged_files"]:
        if path in old_files:
            new_files[path] = old_files[path]

    # For modified files, carry over descriptions of unchanged symbols
    for path in modified:
        if path in old_files and path in new_files:
            _carry_over_descriptions(old_files[path], new_files[path])

    # Files that need Gemini annotation
    needs_annotation = added + modified
    if needs_annotation:
        n_total = len(needs_annotation)
        console.print(f"[dim]Annotating {n_total} file(s) with Gemini…[/dim]")
        new_files = annotate_files(repo_path, needs_annotation, new_files, gemini_call)
    else:
        console.print("[dim]DNA is up to date — no annotation needed.[/dim]")

    if removed:
        console.print(f"[dim]Removed {len(removed)} deleted file(s) from DNA.[/dim]")

    # Update DNA
    dna["files"] = new_files
    save_dna(repo_path, dna)

    return dna


def _carry_over_descriptions(old_entry: dict, new_entry: dict) -> None:
    """Copy descriptions from old entry to new for symbols that haven't changed."""
    # File purpose
    if old_entry.get("purpose") and not new_entry.get("purpose"):
        new_entry["purpose"] = old_entry["purpose"]

    for category in ("functions", "constants"):
        old_syms = old_entry.get(category, {})
        new_syms = new_entry.get(category, {})
        for name in new_syms:
            if name in old_syms:
                old_sig = old_syms[name].get("signature", "")
                new_sig = new_syms[name].get("signature", "")
                if old_sig == new_sig and old_syms[name].get("description"):
                    new_syms[name]["description"] = old_syms[name]["description"]

    # Classes — carry over class desc + method descs
    old_classes = old_entry.get("classes", {})
    new_classes = new_entry.get("classes", {})
    for cls_name in new_classes:
        if cls_name not in old_classes:
            continue
        old_cls = old_classes[cls_name]
        new_cls = new_classes[cls_name]
        if old_cls.get("description") and not new_cls.get("description"):
            new_cls["description"] = old_cls["description"]
        old_methods = old_cls.get("methods", {})
        new_methods = new_cls.get("methods", {})
        for m_name in new_methods:
            if m_name in old_methods:
                if (old_methods[m_name].get("signature") == new_methods[m_name].get("signature")
                        and old_methods[m_name].get("description")):
                    new_methods[m_name]["description"] = old_methods[m_name]["description"]


def generate_initial_dna(
    repo_path: str | Path,
    project_info: dict[str, Any],
    gemini_call,
) -> dict[str, Any]:
    """Create the .dna file for a brand-new repo.

    Args:
        repo_path: Path to the local repo.
        project_info: Dict with name, description, tech_stack, tagline.
        gemini_call: Rate-limited Gemini call function.

    Returns:
        The complete DNA dict.
    """
    repo_path = Path(repo_path)

    # Generate the project-level section via Gemini
    roadmap_prompt = (
        f"You are planning the development roadmap for a new open-source project.\n\n"
        f"Project: {project_info.get('name', '?')}\n"
        f"Description: {project_info.get('description', '?')}\n"
        f"Tech stack: {', '.join(project_info.get('tech_stack', []))}\n\n"
        f"Create 5-8 development phases, ordered from foundational to polished.\n"
        f"Each phase should be achievable in 3-7 incremental sessions.\n\n"
        f"Return ONLY valid JSON (no markdown fences):\n"
        f"{{\n"
        f'  "roadmap": [\n'
        f'    {{"phase": 1, "title": "...", "description": "...", "status": "in-progress"}},\n'
        f'    {{"phase": 2, "title": "...", "description": "...", "status": "not-started"}},\n'
        f"    ...\n"
        f"  ]\n"
        f"}}"
    )

    try:
        raw = gemini_call(roadmap_prompt, system="You are a senior software architect.")
        roadmap_data = _parse_annotation_response(raw)
    except Exception:
        roadmap_data = {"roadmap": [{"phase": 1, "title": "Core functionality", "description": "Build the basic features", "status": "in-progress"}]}

    # AST-index whatever files exist
    files_index = index_repo(repo_path)

    # Annotate all files
    all_files = list(files_index.keys())
    if all_files:
        files_index = annotate_files(repo_path, all_files, files_index, gemini_call)

    dna: dict[str, Any] = {
        "project": {
            "name": project_info.get("name", ""),
            "purpose": project_info.get("description", ""),
            "tech_stack": project_info.get("tech_stack", []),
            "roadmap": roadmap_data.get("roadmap", []),
            "current_phase": 1,
        },
        "files": files_index,
    }

    save_dna(repo_path, dna)
    return dna


# ═══════════════════════════════════════════════════════════════════════
# Context Rendering — format DNA for prompt injection
# ═══════════════════════════════════════════════════════════════════════

def render_dna_context(dna: dict[str, Any]) -> str:
    """Render the DNA into a text format suitable for Gemini prompts."""
    parts: list[str] = []

    # Project section
    project = dna.get("project", {})
    if project:
        parts.append("## Project DNA\n")
        parts.append(f"**Name:** {project.get('name', '?')}\n")
        parts.append(f"**Purpose:** {project.get('purpose', '?')}\n")
        parts.append(f"**Tech:** {', '.join(project.get('tech_stack', []))}\n")

        roadmap = project.get("roadmap", [])
        if roadmap:
            current = project.get("current_phase", 1)
            parts.append(f"\n### Roadmap (current phase: {current})\n")
            for phase in roadmap:
                marker = "→" if phase.get("phase") == current else " "
                status = phase.get("status", "not-started")
                parts.append(
                    f"  {marker} Phase {phase.get('phase', '?')}: "
                    f"{phase.get('title', '?')} [{status}]\n"
                    f"    {phase.get('description', '')}\n"
                )

    # Files section
    files = dna.get("files", {})
    if files:
        parts.append("\n## Codebase Map\n")
        for rel_path in sorted(files.keys()):
            entry = files[rel_path]
            purpose = entry.get("purpose", "")
            parts.append(f"\n### {rel_path}")
            if purpose:
                parts.append(f"\n{purpose}\n")
            else:
                parts.append("\n")

            # Non-Python file
            if entry.get("type") == "non-python":
                desc = entry.get("description", "")
                if desc:
                    parts.append(f"  {desc}\n")
                continue

            # Functions
            for name, info in entry.get("functions", {}).items():
                sig = info.get("signature", name)
                desc = info.get("description", "")
                line = f"  `{sig}`"
                if desc:
                    line += f" — {desc}"
                parts.append(line + "\n")

            # Classes
            for cls_name, cls_info in entry.get("classes", {}).items():
                bases = ", ".join(cls_info.get("bases", []))
                desc = cls_info.get("description", "")
                header = f"  class `{cls_name}`"
                if bases:
                    header += f"({bases})"
                if desc:
                    header += f" — {desc}"
                parts.append(header + "\n")

                for m_name, m_info in cls_info.get("methods", {}).items():
                    sig = m_info.get("signature", m_name)
                    desc = m_info.get("description", "")
                    line = f"    `{sig}`"
                    if desc:
                        line += f" — {desc}"
                    parts.append(line + "\n")

            # Constants
            for const_name, const_info in entry.get("constants", {}).items():
                desc = const_info.get("description", "")
                line = f"  `{const_name}`"
                if const_info.get("type"):
                    line += f": {const_info['type']}"
                if desc:
                    line += f" — {desc}"
                parts.append(line + "\n")

            # Imports
            internal = entry.get("internal_imports", [])
            external = entry.get("external_imports", [])
            if internal:
                parts.append(f"  Imports (internal): {', '.join(internal)}\n")
            if external:
                parts.append(f"  Imports (external): {', '.join(external)}\n")

    return "".join(parts)
