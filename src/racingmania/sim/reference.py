"""Compare the PyTorch port with the original cannon-es code running in Node.js.

Global context: the browser game is the tested original; ``tests/js/parity_harness.mjs`` runs its
physics headless. This module builds random test inputs on a track, runs them through both
implementations, and returns both results so ``tests/parity/`` can assert agreement
(plan: "Physics, against the original (the main physics test)").
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..utils.paths import REPO_ROOT
from .rules import reset_rules, update_rules
from .step import CarState, agent_step, physics_tick
from .track_tensors import TrackBatch
from .vehicle_config import load_vehicle_config

HARNESS = REPO_ROOT / "tests" / "js" / "parity_harness.mjs"


def run_harness(request: dict) -> dict:
    """Run the Node.js harness on ``request`` and return its parsed response."""
    with tempfile.TemporaryDirectory() as d:
        req, res = Path(d) / "req.json", Path(d) / "res.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        subprocess.run(["node", str(HARNESS), str(req), str(res)], check=True, cwd=REPO_ROOT)
        return json.loads(res.read_text(encoding="utf-8"))


def _quat_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [ax * bw + aw * bx + ay * bz - az * by, ay * bw + aw * by + az * bx - ax * bz,
            az * bw + aw * bz + ax * by - ay * bx, aw * bw - ax * bx - ay * by - az * bz]


def random_cases(track: dict, n: int, seed: int = 0, cfg=None) -> list[dict]:
    """Random single-tick inputs near the road: chassis 0.5–1.5 m above a random road point, with
    random yaw, small roll and pitch, random velocities, steering angle and action."""
    from ..tracks.ribbon import quat_from_frame
    rng = np.random.default_rng(seed)
    c = np.asarray(track["center"]); left = np.asarray(track["left"]); gap = np.asarray(track["gap"], bool)
    hw = float(track["half_width"])
    road = [i for i in range(2, len(c) - 3) if not gap[i - 1] and not gap[i]]
    cases = []
    for _ in range(n):
        i = int(rng.choice(road))
        t = c[i + 1] - c[i - 1]; t /= np.linalg.norm(t)
        up = np.cross(t, left[i]); up /= np.linalg.norm(up)
        p = c[i] + left[i] * rng.uniform(-0.9, 0.9) * hw + up * rng.uniform(0.5, 1.5)
        q = quat_from_frame(t, up)
        for axis, lim in (((0, 1, 0), 0.8), ((1, 0, 0), 0.15), ((0, 0, 1), 0.15)):  # yaw, roll, pitch (body axes)
            ang = rng.uniform(-lim, lim)
            dq = [axis[0] * math.sin(ang / 2), axis[1] * math.sin(ang / 2), axis[2] * math.sin(ang / 2), math.cos(ang / 2)]
            q = _quat_mul(q, dq)
        qn = np.asarray(q) / np.linalg.norm(q)
        v = t * rng.uniform(-5, 45) + left[i] * rng.uniform(-5, 5) + up * rng.uniform(-3, 3)
        w = rng.uniform(-1.5, 1.5, size=3)
        cases.append({"state": p.tolist() + qn.tolist() + v.tolist() + w.tolist(),
                      "steer": float(rng.uniform(-0.3, 0.3)), "index": i, "action": int(rng.integers(0, 9))})
    return cases


def port_cases(track: dict, cases: list[dict], device="cpu", dtype=torch.float64, cfg=None) -> list[dict]:
    """Run the same single ticks through the port (all cases as one batch)."""
    cfg = cfg or load_vehicle_config()
    tb = TrackBatch.from_tracks([track], device=device, dtype=dtype)
    n = len(cases)
    tid = torch.zeros(n, dtype=torch.long, device=device)
    s = torch.tensor([c["state"] for c in cases], dtype=dtype, device=device)
    car = CarState.from_flat(s, torch.tensor([c["steer"] for c in cases], dtype=dtype, device=device))
    rules = reset_rules(tb, tid, car.pos, torch.tensor([c["index"] for c in cases], device=device))
    a = torch.tensor([c["action"] for c in cases], device=device)
    from .controls import decode_action
    steer_cmd, pedal = decode_action(a)
    active = torch.ones(n, dtype=torch.bool, device=device)
    car, contacts, _ = physics_tick(cfg, tb, tid, rules.index, car, steer_cmd, pedal)
    rules, _ = update_rules(tb, tid, rules, car.pos, contacts, active)
    flat = car.flat().cpu().tolist()
    return [{"state": flat[k], "steer": float(car.steer[k]), "contacts": int(contacts[k]), "index": int(rules.index[k]),
             "s": float(rules.s[k]), "finished": bool(rules.finished[k]), "out": bool(rules.out[k])} for k in range(n)]


def port_sequence(track: dict, start: dict, actions: list[int], device="cpu", dtype=torch.float64, cfg=None) -> list[dict]:
    """Run one action sequence tick by tick through the port, stopping when the run ends."""
    cfg = cfg or load_vehicle_config()
    tb = TrackBatch.from_tracks([track], device=device, dtype=dtype)
    tid = torch.zeros(1, dtype=torch.long, device=device)
    car = CarState.from_flat(torch.tensor([start["state"]], dtype=dtype, device=device),
                             torch.tensor([start["steer"]], dtype=dtype, device=device))
    rules = reset_rules(tb, tid, car.pos, torch.tensor([start["index"]], device=device))
    from .controls import decode_action
    out = []
    for a in actions:
        steer_cmd, pedal = decode_action(torch.tensor([a], device=device))
        for _ in range(cfg.ticks_per_action):
            if bool(rules.finished[0] | rules.out[0]):
                break
            car, contacts, _ = physics_tick(cfg, tb, tid, rules.index, car, steer_cmd, pedal)
            rules, _ = update_rules(tb, tid, rules, car.pos, contacts, torch.ones(1, dtype=torch.bool, device=device))
            out.append({"state": car.flat()[0].tolist(), "steer": float(car.steer[0]), "contacts": int(contacts[0]),
                        "index": int(rules.index[0]), "s": float(rules.s[0]), "finished": bool(rules.finished[0]),
                        "out": bool(rules.out[0])})
        if bool(rules.finished[0] | rules.out[0]):
            break
    return out
