"""Phase 4: training-free baselines (random, untrained network, scripted driver) on track sets.

Usage: uv run python scripts/04_baselines.py --sets val campaign [test oor]
"""
import argparse

import _common  # noqa: F401

from racingmania.baselines import run_baselines, run_cem

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--sets", nargs="+", default=["val", "campaign"])
ap.add_argument("--untrained-seeds", type=int, default=10)
ap.add_argument("--random-seeds", type=int, default=3)
ap.add_argument("--cem", action="store_true", help="also run the CEM planner on 5 campaign + 25 test tracks")
ap.add_argument("--only-cem", action="store_true")
a = ap.parse_args()
if not a.only_cem:
    run_baselines(a.sets, untrained_seeds=range(a.untrained_seeds), random_seeds=range(a.random_seeds))
if a.cem or a.only_cem:
    run_cem()
