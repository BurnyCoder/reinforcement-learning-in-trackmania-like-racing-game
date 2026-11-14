"""Name a run's failure mode: where along the track its failed runs leave the road.

Usage: uv run python scripts/09_failure_analysis.py --run generalist/seed0 --sets val [train_sample]
       → results/<run>/failures_<set>.json

Reads the run's evaluation CSV and the set's track pool (geometry only; no policy is run, so a
protected set's tracks are not driven again). Not allowed on test sets before the confirmatory
analysis, to keep failure analysis on development data.
"""
import argparse
import json

import numpy as np

import _common  # noqa: F401

from racingmania.eval.failures import failure_positions
from racingmania.tracks.store import load_pool, pool_path, subset
from racingmania.utils.paths import RESULTS_DIR

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", required=True, help="run id, e.g. generalist/seed0")
ap.add_argument("--sets", nargs="+", default=["val"], choices=["val", "train_sample"])
a = ap.parse_args()
for s in a.sets:
    pool = load_pool(pool_path("train" if s == "train_sample" else s))
    if s == "train_sample":  # the fixed 1,000-track sample is the first 1,000 tracks of the pool (runs.eval_tracks)
        pool = subset(pool, np.arange(min(1000, len(pool["n"]))))
    res = failure_positions(RESULTS_DIR / a.run / f"eval_{s}.csv", pool)
    out = RESULTS_DIR / a.run / f"failures_{s}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(s, json.dumps(res))
