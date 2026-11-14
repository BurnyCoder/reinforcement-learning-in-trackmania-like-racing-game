"""Demonstration: the untrained network with sampled instead of most-probable actions, as a ghost for the game.

Usage (in WSL, GPU): uv run python scripts/13_sampled_untrained_demo.py [--seed 0] [--sample-seed 1000]

Global context: the untrained control (baselines.untrained_agent) is scored, like every driver, with its most
probable action, which for near-equal scores is an almost fixed choice. Its action probabilities are close to
uniform because the policy's output layer starts near zero (rl/networks.py, out_std=0.01, as in CleanRL), so
sampling from them presses keys almost uniformly at random, which is how PPO starts exploring. This script
drives the same network on the five campaign tracks with sampled actions (torch's generator seeded, so the
run is repeatable) through the common evaluator, records the runs, and exports them as ghosts. It is not part
of the scored evaluation: the files go to ``results/demos/`` (no run.json, so the report does not list it).
"""
import argparse
import json

import torch

import _common  # noqa: F401

from racingmania.baselines import untrained_policy
from racingmania.env.vec_env import EnvConfig
from racingmania.eval.evaluator import evaluate, write_eval_csv
from racingmania.export.ghosts import export_ghosts
from racingmania.runs import eval_tracks, save_replays
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.utils.paths import RESULTS_DIR, WEB_ASSETS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--seed", type=int, default=0, help="initialisation seed of the untrained network (the control's)")
ap.add_argument("--sample-seed", type=int, default=1000, help="seed of torch's generator for the sampled actions")
a = ap.parse_args()

cfg = load_vehicle_config()
tb, _ = eval_tracks("campaign")
policy = untrained_policy(a.seed, sample=True)  # builds the network first, then the sampling generator is seeded
torch.manual_seed(a.sample_seed)
rows = evaluate(policy, cfg, tb, device="cuda", env_cfg=EnvConfig(), compile=True, record_replays=True)
out = RESULTS_DIR / "demos" / "untrained-sampled" / f"seed{a.seed}"
write_eval_csv(rows, out / "eval_campaign.csv")
save_replays(rows, out / "replays", cfg.dt)
(out / "demo.json").write_text(json.dumps({"what": "untrained network, actions sampled from its probabilities", "network_seed": a.seed,
                                          "sample_seed": a.sample_seed, "scored": False}, indent=2), encoding="utf-8")
index_path = WEB_ASSETS / "index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
written = export_ghosts(index, out, f"untrained-seed{a.seed}-sampled", keep_unfinished=True)
index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
for r in rows:
    print(f"{r['track']}: {r['end']} after {r['time']:.2f} s, {r['progress']:.0f} m", flush=True)
print("ghosts:", written)
