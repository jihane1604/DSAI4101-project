from pathlib import Path

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root.

    This always works, because:
    - this file lives in: project_root/src/utils_paths.py
    - __file__ gives its actual path on ANY computer
    """
    return Path(__file__).resolve().parents[1]

