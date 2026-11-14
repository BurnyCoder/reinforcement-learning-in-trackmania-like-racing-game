"""Phase 3: evaluate a trained run on track sets with the common evaluator (preregistration, section 4).

Usage: uv run python scripts/03_evaluate.py --experiment generalist --seed 1 --sets val train_sample campaign test oor
Reading the test and out-of-range sets is logged in results/test_access_log.jsonl.
"""
import argparse

import _common  # noqa: F401

from racingmania.runs import evaluate_run

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--experiment", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--track", default=None)
ap.add_argument("--sets", nargs="+", default=["val", "campaign"])
a = ap.parse_args()
evaluate_run(a.experiment, a.seed, a.sets, track=a.track)
