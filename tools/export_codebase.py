"""
tools/export_codebase.py
------------------------
Interactive codebase exporter.

- Asks user permission
- Anchors to true project root (not cwd)
- Recursively scans ALL project files safely
- Collects:
    - all .py files
    - Makefile
- Excludes venv, caches, git, build artifacts
- Writes timestamped snapshot into tools/snapshots/
"""

from pathlib import Path
from datetime import datetime
import sys


# ------------------------------------------------------------
# ROOT DETECTION
# ------------------------------------------------------------
def find_project_root(current_path: Path) -> Path:
    """
    Walk upward until we find project markers.
    We assume root contains core folders like:
        - models/
        - pipeline/
        - api/
    """
    for parent in [current_path] + list(current_path.parents):
        if (parent / "models").exists() and (parent / "pipeline").exists():
            return parent
    return current_path.parent


# ------------------------------------------------------------
# FILE COLLECTION (FIXED)
# ------------------------------------------------------------
def collect_project_files(root: Path):
    """
    Collect ALL project-relevant files safely.

    Strategy:
    - Walk entire tree
    - Exclude noise directories (venv, cache, etc.)
    - Include ALL valid project modules regardless of depth
    - Explicitly include Makefile
    """

    ignored_dirs = {
        "venv",
        ".venv",
        "__pycache__",
        ".git",
        ".idea",
        ".pytest_cache",
        ".mypy_cache",
        "build",
        "dist",
        "snapshots",
        "site-packages",
    }

    files = []

    for path in root.rglob("*"):

        # only Python files + Makefile
        if not (path.name.endswith(".py") or path.name == "Makefile"):
            continue

        # skip non-files
        if not path.is_file():
            continue

        # skip hidden paths anywhere (.git, .env, etc.)
        if any(part.startswith(".") for part in path.parts):
            continue

        # skip ignored dirs anywhere in path
        if any(part in ignored_dirs for part in path.parts):
            continue

        files.append(path)

    return sorted(set(files))


# ------------------------------------------------------------
# SNAPSHOT BUILDER
# ------------------------------------------------------------
def build_snapshot(root: Path, files: list[Path]) -> str:
    """
    Combine all code into one structured snapshot.
    """

    snapshot = []
    snapshot.append("# CODEBASE SNAPSHOT\n")
    snapshot.append(f"# Root: {root}\n")
    snapshot.append(f"# Generated: {datetime.now().isoformat()}\n\n")
    snapshot.append("=" * 100 + "\n")

    for f in files:
        rel = f.relative_to(root)

        snapshot.append("\n\n")
        snapshot.append("#" * 100 + "\n")
        snapshot.append(f"# FILE: {rel}\n")
        snapshot.append("#" * 100 + "\n\n")

        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            content = f"# ERROR READING FILE: {e}"

        snapshot.append(content)
        snapshot.append("\n")

    return "".join(snapshot)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("\nCODEBASE EXPORT TOOL")
    print("====================\n")

    answer = input(
        "Would you like to export the full codebase snapshot for 'credit_model_project'? (y/n): "
    ).strip().lower()

    if answer != "y":
        if answer == "n":
            print("Okay thanks — snapshot not created.")
            sys.exit(0)
        else:
            print("Please enter 'y' for yes or 'n' for no.")
            sys.exit(1)

    # Anchor to script location
    script_path = Path(__file__).resolve()
    tools_dir = script_path.parent
    project_root = find_project_root(tools_dir)

    print(f"\nDetected project root: {project_root}")

    files = collect_project_files(project_root)

    print(f"Found {len(files)} project files")

    snapshot = build_snapshot(project_root, files)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    snapshot_dir = project_root / "tools" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    output_file = snapshot_dir / f"codebase_snapshot_{timestamp}.txt"

    output_file.write_text(snapshot, encoding="utf-8")

    print("\nDONE")
    print(f"Snapshot saved to:\n{output_file}\n")


if __name__ == "__main__":
    main()