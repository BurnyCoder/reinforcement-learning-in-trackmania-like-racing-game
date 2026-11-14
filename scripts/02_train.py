"""Phase 2: train one run (experiment config + seed), resuming from its last checkpoint if there is one.

Usage: uv run python scripts/02_train.py --experiment generalist --seed 0 [--track campaign-01] [--stop-after STEPS]
"""
import argparse

import _common  # noqa: F401

from racingmania.runs import train_run

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--experiment", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--track", default=None, help="campaign track for specialist runs")
ap.add_argument("--stop-after", type=int, default=None, help="stop after this many total steps (resumable)")
ap.add_argument("--total-steps", type=int, default=None, help="override the step budget")
ap.add_argument("--no-resume", action="store_true")
a = ap.parse_args()
over = {"total_steps": a.total_steps} if a.total_steps else None
train_run(a.experiment, a.seed, track=a.track, resume=not a.no_resume, stop_after_steps=a.stop_after, overrides=over)
