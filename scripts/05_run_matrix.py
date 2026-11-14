"""Phase 5: the resumable run queue (train → evaluate → export → publish for every run in order).

Usage (on Windows; GPU work is sent to WSL): uv run python scripts/05_run_matrix.py [--phases 2-single-seed ...] [--no-publish] [--limit N]
"""
import argparse

import _common  # noqa: F401

from racingmania.matrix import run_queue

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--phases", nargs="*", default=None)
ap.add_argument("--no-publish", action="store_true")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
run_queue(a.phases, publish=not a.no_publish, limit=a.limit)
