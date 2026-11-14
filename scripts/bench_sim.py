"""Benchmark gate: agent steps per second of the batched simulator (reports/plan.md, stage B gate ≥ 60,000).

Usage: uv run python scripts/bench_sim.py [--compile] [--envs 4096 16384] ; writes results/bench/<platform>.json
"""
import argparse
import json
import platform
import time

import torch

import _common  # noqa: F401

from racingmania.env.vec_env import RacingVecEnv
from racingmania.runs import campaign_pool
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.campaign import CAMPAIGN_PLANS
from racingmania.tracks.generator import generate_track
from racingmania.tracks.store import batch_from_pool, pack
from racingmania.utils.paths import RESULTS_DIR

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--compile", action="store_true")
ap.add_argument("--envs", type=int, nargs="+", default=[4096, 16384])
ap.add_argument("--steps", type=int, default=30)
a = ap.parse_args()
cfg = load_vehicle_config()
tracks = [t for t in (generate_track(s) for s in range(30)) if t]
tb = batch_from_pool(pack(tracks), device="cuda")
out = {"platform": platform.platform(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "compile": a.compile, "runs": []}
for n in a.envs:
    env = RacingVecEnv(cfg, tb, n, device="cuda", compile=a.compile)
    t0 = time.time()
    for _ in range(3):
        env.step(torch.randint(0, 9, (n,), device="cuda"))
    torch.cuda.synchronize()
    warm = time.time() - t0
    t0 = time.time()
    for _ in range(a.steps):
        env.step(torch.full((n,), 3, device="cuda"))
    torch.cuda.synchronize()
    dt = time.time() - t0
    r = {"envs": n, "agent_steps_per_s": a.steps * n / dt, "ms_per_step": 1000 * dt / a.steps, "warmup_s": warm}
    out["runs"].append(r)
    print(r, flush=True)
name = ("linux" if platform.system() == "Linux" else "windows") + ("-compiled" if a.compile else "-eager")
(RESULTS_DIR / "bench").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "bench" / f"{name}.json").write_text(json.dumps(out, indent=2))
