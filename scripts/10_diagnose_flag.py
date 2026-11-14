"""Replay one exploit-flagged track step by step: translational, gravitational and rotational energy.

Usage (in WSL, GPU): uv run python scripts/10_diagnose_flag.py --experiment generalist --seed 0 --set train_sample --track train-198

Global context: the evaluator flags "energy gained while coasting" (preregistration, section 8) and
every flag is confirmed by hand. A replay must reproduce the official evaluation exactly, so this
drives the run's policy on the whole set in one batch with a fresh compile, as ``evaluate`` does
(reports/plan.md, amendments on evaluation reproducibility), and prints the steps around the
largest coasting gain for the chosen track: action, wheel contacts, height, speed and the change in
½v² + g·y, in ½ωᵀIω/m and in their sum (J/kg per agent step). Replaying a protected set's track is
logged in the test-access log like any other read.
"""
import argparse

import numpy as np
import torch

import _common  # noqa: F401

from racingmania.env.vec_env import EnvConfig, RacingVecEnv
from racingmania.eval.evaluator import _log_access, mechanical_energy
from racingmania.runs import eval_tracks, load_policy, run_id
from racingmania.sim.vehicle_config import load_vehicle_config

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--experiment", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--run-track", default=None, help="campaign track of a specialist run")
ap.add_argument("--set", required=True)
ap.add_argument("--track", required=True)
a = ap.parse_args()
cfg = load_vehicle_config()
tb, protected = eval_tracks(a.set)
if protected:
    _log_access(a.set, f"diagnose:{run_id(a.experiment, a.seed, a.run_track)}", tb.num_tracks)
idx = tb.names.index(a.track)
policy, _, _, env_cfg = load_policy(a.experiment, a.seed, a.run_track)
ecfg = EnvConfig(**{**env_cfg.__dict__, "random_start_prob": 0.0, "max_steps": 10**9})
torch.compiler.reset()  # same compiled kernels as a fresh official evaluation
env = RacingVecEnv(cfg, tb, tb.num_tracks, env_cfg=ecfg, assign=torch.arange(tb.num_tracks), auto_reset=False, compile=True)
g = -cfg.gravity[1]


def parts(car):
    total = mechanical_energy(car, cfg, g)[idx]
    lin = (0.5 * (car.vel[idx] ** 2).sum() + g * car.pos[idx, 1])
    return float(lin), float(total - lin)


rows = []
with torch.no_grad():
    while not bool(env.rules.finished[idx] | env.rules.out[idx]):
        lin0, rot0 = parts(env.car)
        action = policy(env)
        env.step(action)
        lin1, rot1 = parts(env.car)
        rows.append((len(rows), int(action[idx]), int(env.contacts[idx].sum()), float(env.car.pos[idx, 1]),
                     float(torch.linalg.norm(env.car.vel[idx])), lin1 - lin0, rot1 - rot0, lin1 + rot1 - lin0 - rot0))
arr = np.array(rows)
coast = arr[:, 1] % 3 == 1  # coast pedal (action = steer·3 + pedal)
j = int(np.argmax(np.where(coast, arr[:, 7], -np.inf)))
print(f"{a.track}: {len(rows)} steps, {'finished' if bool(env.rules.finished[idx]) else 'out'}; "
      f"largest coasting gain including rotation {arr[j, 7]:.3f} J/kg at step {j}")
print("step action contacts height speed d(lin+pot) d(rot) d(total)")
for r in rows[max(0, j - 8): j + 6]:
    print(f"{r[0]:5d} {r[1]:6d} {r[2]:8d} {r[3]:7.2f} {r[4]:6.2f} {r[5]:10.3f} {r[6]:7.3f} {r[7]:8.3f}")
