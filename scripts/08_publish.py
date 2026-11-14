"""Phase 8: regenerate all reports and push everything to GitHub (via pull request) and Hugging Face.

Usage (on Windows, where gh and hf are logged in): uv run python scripts/08_publish.py --message "..." [--run generalist/seed1 --status evaluated] [--all-models]
"""
import argparse

import _common  # noqa: F401

from racingmania.publish import publish_all

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--message", required=True)
ap.add_argument("--run", default=None)
ap.add_argument("--status", default=None)
ap.add_argument("--all-models", action="store_true",
                help="also refresh the model repository (card, videos, files) of every evaluated trained run")
a = ap.parse_args()
publish_all(a.message, a.run, a.status, all_models=a.all_models)
