"""Choose the CEM planner's horizon on validation tracks only (never campaign or test tracks).

Usage: uv run python scripts/tune_cem.py 30 60 90  → results/baselines/cem_horizon_tuning.txt

Global context: the CEM planner (src/racingmania/drivers/cem.py) is the planning reference of the
run table (reports/plan.md). With a 1.5 s horizon it fell off most tracks, because it could not
see a tight corner early enough to slow down for it. This script drives the first 10 validation
tracks with each given horizon (other settings as in ``CEMParams``) and records, per horizon, how
many it finishes and how fast it is on those, relative to the scripted driver's reference time.
The chosen horizon is the shortest one that finishes all 10; it is written into ``CEMParams``.
"""
import dataclasses
import sys
import time

import numpy as np

import _common  # noqa: F401

from racingmania.drivers.cem import CEMParams, plan_tracks
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.store import batch_from_pool, load_pool, pool_path
from racingmania.utils.paths import RESULTS_DIR

cfg = load_vehicle_config()
pool = load_pool(pool_path("val"))
tb = batch_from_pool(pool, np.arange(10), device="cuda")  # the first 10 validation tracks
ref = tb.ref_ticks.cpu().numpy()  # scripted driver's reference time in ticks
out = RESULTS_DIR / "baselines" / "cem_horizon_tuning.txt"
out.parent.mkdir(parents=True, exist_ok=True)
for H in map(int, sys.argv[1:]):
    t0 = time.time()
    fin, ticks = plan_tracks(cfg, tb, dataclasses.replace(CEMParams(), horizon=H), device="cuda")
    fin, ticks = np.array(fin, bool), np.array(ticks)
    ratio = ref[fin] / ticks[fin]  # > 1 means faster than the scripted driver
    line = (f"H={H}: finished {fin.sum()}/10, mean ref/agent on finished {ratio.mean() if fin.any() else float('nan'):.3f}, "
            f"ticks {ticks.tolist()}, {time.time() - t0:.0f} s")
    print(line, flush=True)
    with open(out, "a", encoding="utf-8") as fh:  # one line per horizon, appended so a run can be split up
        fh.write(line + "\n")
