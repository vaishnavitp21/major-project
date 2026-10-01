"""Resolve the project root for notebooks launched from root or notebooks/."""

from pathlib import Path
import sys


def ensure_project_root():
    """Find the project root, add it to sys.path once, and return its Path."""
    current = Path.cwd().resolve()
    candidates = [current, *current.parents]
    project_root = next((p for p in candidates if (p / "src" / "__init__.py").is_file()), None)
    if project_root is None:
        raise RuntimeError("Could not locate project root containing src/__init__.py")
    root_string = str(project_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    return project_root
