"""Repository and data locations.

Global context: small files (configs, results tables, track files for the game) live in the
repository; large outputs (checkpoints, TensorBoard logs, track pools) live under
``RACINGMANIA_DATA_DIR`` from ``.env`` (see ``.env.example``), which may be a Windows path even when
the code runs in WSL.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/racingmania/utils/paths.py -> repository root
load_dotenv(REPO_ROOT / ".env")  # does not override variables already set in the environment


def _translate_windows_path(p: str) -> str:
    """Turn ``D:/AI/x`` or ``D:\\AI\\x`` into ``/mnt/d/AI/x`` when running under WSL/Linux."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m and os.name != "nt":
        return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"
    return p


def data_dir() -> Path:
    """Folder for large outputs; created if missing."""
    p = Path(_translate_windows_path(os.environ.get("RACINGMANIA_DATA_DIR", str(REPO_ROOT / "runs"))))
    p.mkdir(parents=True, exist_ok=True)
    return p


RESULTS_DIR = REPO_ROOT / "results"
WEB_ASSETS = REPO_ROOT / "web" / "assets"
