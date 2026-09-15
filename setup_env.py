"""
setup_env.py
------------
Creates a virtual environment, installs dependencies, and runs the pipeline.
Safe to run every time — skips setup steps that are already done.

Usage:
    python setup_env.py
"""
import os
import subprocess
import sys
import venv
from pathlib import Path

VENV_NAME = "venv"
REQUIREMENTS = "requirements.txt"
MIN_PYTHON = (3, 10)


def check_python_version():
    if sys.version_info < MIN_PYTHON:
        print(f"Error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.")
        print(f"  Current version: {sys.version}")
        sys.exit(1)
    print(f"Python {sys.version.split()[0]} — OK")


def run_command(command: str):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"\nError running: {command}")
        sys.exit(1)


def get_pip_path() -> str:
    if os.name == "nt":
        return str(Path(VENV_NAME) / "Scripts" / "pip.exe")
    return str(Path(VENV_NAME) / "bin" / "pip")


def get_python_path() -> str:
    if os.name == "nt":
        return str(Path(VENV_NAME) / "Scripts" / "python.exe")
    return str(Path(VENV_NAME) / "bin" / "python")


def is_venv_ready() -> bool:
    """Check if the venv exists and all requirements are installed."""
    python = get_python_path()
    if not Path(python).exists():
        return False

    # Check every package in requirements.txt is importable
    if not Path(REQUIREMENTS).exists():
        return False

    # Use pip to verify installed packages match requirements
    pip = get_pip_path()
    result = subprocess.run(
        f'"{pip}" check',
        shell=True,
        capture_output=True,
        text=True,
    )
    # Also verify the key packages are actually present
    check = subprocess.run(
        f'"{python}" -c "import pandas, numpy, statsmodels, fastapi, uvicorn"',
        shell=True,
        capture_output=True,
    )
    return check.returncode == 0


def create_venv():
    if Path(VENV_NAME).exists():
        print(f"Virtual environment '{VENV_NAME}' already exists — skipping creation.")
    else:
        print(f"Creating virtual environment '{VENV_NAME}'...")
        venv.create(VENV_NAME, with_pip=True)
        print("  Done.")


def install_requirements():
    pip = get_pip_path()

    if not Path(REQUIREMENTS).exists():
        print(f"Error: {REQUIREMENTS} not found.")
        sys.exit(1)

    print("Upgrading pip...")
    run_command(f'"{pip}" install --upgrade pip')

    print(f"Installing packages from {REQUIREMENTS}...")
    run_command(f'"{pip}" install -r {REQUIREMENTS}')
    print("  All packages installed.")


def run_pipeline():
    python = get_python_path()
    print("\n" + "=" * 50)
    print("Running pipeline...")
    print("=" * 50 + "\n")
    run_command(f'"{python}" main.py')


def main():
    check_python_version()

    if is_venv_ready():
        print("Environment already set up — skipping install.")
    else:
        create_venv()
        install_requirements()

    run_pipeline()


if __name__ == "__main__":
    main()