"""Random actions against the untrained network with sampled actions, over many runs on one track.

Usage (in WSL, GPU): uv run python scripts/14_random_vs_sampled.py [--track campaign-01] [--runs 200]
                     → results/demos/random-vs-sampled/<track>.json

Global context: the README shows one run of each (clips of the random-actions control and of the untrained
network with sampled actions, docs/videos/clips.yaml). Both press each key combination about one time in
nine, so any difference between two single runs is expected to be luck. This script drives ``--runs``
independent runs of each through the common evaluator (one car per copy of the track, standing start), and
records the distributions of the time until the run is out and of the progress, with two-sample tests
(Mann-Whitney U and Kolmogorov-Smirnov, scipy.stats). It is a demonstration, not part of the scored
evaluation (results/demos/ holds no run.json, so the report does not list it).
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import ks_2samp, mannwhitneyu

import _common  # noqa: F401

from racingmania.baselines import untrained_policy
from racingmania.drivers.simple import RandomDriver
from racingmania.env.vec_env import EnvConfig
from racingmania.eval.evaluator import evaluate
from racingmania.runs import campaign_pool
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.store import batch_from_pool
from racingmania.utils.paths import RESULTS_DIR

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--track", default="campaign-01")
ap.add_argument("--runs", type=int, default=200)
ap.add_argument("--random-seed", type=int, default=7, help="seed of the random driver's generator")
ap.add_argument("--sample-seed", type=int, default=123, help="seed of torch's generator for the sampled actions")
a = ap.parse_args()

cfg = load_vehicle_config()
tb = batch_from_pool(campaign_pool([a.track] * a.runs), device="cuda")  # one car per copy: independent runs
drivers = {"random_actions": lambda: RandomDriver(seed=a.random_seed, device="cuda"),
           "untrained_sampled": lambda: untrained_policy(0, sample=True)}
res = {"track": a.track, "runs": a.runs, "random_seed": a.random_seed, "sample_seed": a.sample_seed, "drivers": {}}
samples = {}
for name, make in drivers.items():
    policy = make()
    torch.manual_seed(a.sample_seed)  # used by the sampled network; the random driver has its own generator
    rows = evaluate(policy, cfg, tb, device="cuda", env_cfg=EnvConfig(), compile=True)
    t = np.array([r["time"] for r in rows]); s = np.array([r["progress"] for r in rows])
    samples[name] = (t, s)
    pct = lambda x: [float(v) for v in np.percentile(x, [10, 50, 90])]
    res["drivers"][name] = {"time_mean_s": float(t.mean()), "time_p10_p50_p90_s": pct(t), "time_max_s": float(t.max()),
                            "progress_mean_m": float(s.mean()), "progress_p10_p50_p90_m": pct(s),
                            "share_out_at_2s": float(np.mean(t <= 2.0)), "share_finished": float(np.mean([r["finished"] for r in rows]))}
    print(name, res["drivers"][name], flush=True)
(ta, sa), (tb_, sb) = samples["random_actions"], samples["untrained_sampled"]
res["tests"] = {"time_mannwhitney_p": float(mannwhitneyu(ta, tb_).pvalue), "time_ks_p": float(ks_2samp(ta, tb_).pvalue),
                "progress_mannwhitney_p": float(mannwhitneyu(sa, sb).pvalue), "progress_ks_p": float(ks_2samp(sa, sb).pvalue)}
print(res["tests"])
out = RESULTS_DIR / "demos" / "random-vs-sampled" / f"{a.track}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
