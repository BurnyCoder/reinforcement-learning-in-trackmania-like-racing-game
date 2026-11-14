"""Timestamped logging to the terminal and to a log file.

Global context: every script calls ``setup_logging(run_dir)`` once; modules use
``get_logger(__name__)``. Lines look like ``2026-11-14 12:00:00.123 INFO racingmania.rl.ppo: ...``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

FMT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: Path | None = None, level=logging.INFO):
    """Log to stderr and, if given, to ``log_file`` (appending, flushed line by line)."""
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(logging.Formatter(FMT, DATEFMT))
        root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
