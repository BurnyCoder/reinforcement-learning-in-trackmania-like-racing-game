"""One evaluator for every driver: PPO and PQN agents, the scripted driver, random and untrained baselines.

Global context: every reported score comes from here, so all drivers are scored identically
(reports/preregistration.md, section 4): one run per track from the standing start, the policy's
most probable action, a time cap of 3 × the track's reference time, and the success-weighted
speed score ``finished · T_ref / T`` per track. The evaluator also records simple detectors for
physics exploits (section 8 of the preregistration) and, optionally, replays for the game's ghosts.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..env.vec_env import OUT_REASONS, EnvConfig, RacingVecEnv
from ..sim.math3d import quat_conj, quat_rotate
from ..utils.logging import get_logger

log = get_logger(__name__)
TIME_CAP_FACTOR = 3.0


def mechanical_energy(car, cfg, g: float) -> Tensor:
    """Mechanical energy per kilogram [N]: translational + gravitational + rotational kinetic energy.

    Rotational energy is ½·Σ I_j·ω_j² / m with ω in the chassis frame (``Body.invInertia`` is the
    diagonal of the box's principal inertia, cannon-es ``Box.calculateInertia``). It must be counted:
    in a crash the chassis spins up and later turns that spin back into linear speed, which looked like
    an energy gain while coasting before rotation was included (specialist seed 0, track val-1000079).
    Spring energy in the suspension is not counted (the wheel compressions are not part of the state),
    so a flagged run is still confirmed by hand (preregistration, section 8).
    """
    inertia = 1.0 / torch.tensor(cfg.inv_inertia, device=car.vel.device, dtype=car.vel.dtype)  # principal moments (kg m²)
    w_local = quat_rotate(quat_conj(car.quat), car.angvel)  # angular velocity in the chassis frame
    rot = 0.5 * (inertia * w_local ** 2).sum(1) / cfg.mass  # ½ωᵀIω per kg
    return 0.5 * (car.vel ** 2).sum(1) + g * car.pos[:, 1] + rot


def evaluate(policy, cfg, tracks, device="cuda", dtype=torch.float32, ref_ticks=None, record_replays=False,
             max_ticks_default: int = 30000, env_cfg: EnvConfig | None = None, compile: bool = False,
             log_access: str | None = None, fresh_compile: bool = True) -> list[dict]:
    """Run ``policy(env) -> actions`` once on every track of ``tracks`` (a TrackBatch).

    ``ref_ticks``: reference finishing ticks per track (default: the TrackBatch's ``ref_ticks``);
    tracks without a reference use ``max_ticks_default`` as the cap and get no score.
    ``log_access``: name of a protected set (test, oor); every evaluation on it is appended to
    ``results/test_access_log.jsonl`` (preregistration, section 3).
    ``fresh_compile``: with ``compile``, reset the in-process compiler state first (``torch.compiler.reset``) so
    the simulator is compiled for exactly this batch size. Without it, a second evaluation in the
    same process reuses a kernel recompiled for dynamic shapes, whose float32 rounding differs; on
    chaotic tracks that changes outcomes (741 of 1,000 training-sample tracks ended on a different
    tick, reports/plan.md amendments). Training's monitoring validation passes False to keep its
    compiled training step.
    Returns one dictionary per track.
    """
    if compile and fresh_compile:
        torch.compiler.reset()  # public API, https://docs.pytorch.org/docs/stable/generated/torch.compiler.reset.html
    if log_access:
        _log_access(log_access, getattr(policy, "name", type(policy).__name__), tracks.num_tracks)
    T = tracks.num_tracks
    ecfg = env_cfg or EnvConfig()
    ecfg = EnvConfig(**{**ecfg.__dict__, "random_start_prob": 0.0, "max_steps": 10**9})
    env = RacingVecEnv(cfg, tracks, T, env_cfg=ecfg, device=device, dtype=dtype,
                       assign=torch.arange(T), auto_reset=False, compile=compile)
    ref = (ref_ticks if ref_ticks is not None else env.tracks.ref_ticks).to(env.device).float()
    cap = torch.where(ref > 0, torch.ceil(ref * TIME_CAP_FACTOR), torch.full_like(ref, max_ticks_default)).long()
    top = cfg.controls["top_speed"]
    max_speed = torch.zeros(T, device=env.device)
    coast_energy_gain = torch.zeros(T, device=env.device)
    replays = [[] for _ in range(T)] if record_replays else None
    g = -cfg.gravity[1]
    if record_replays:  # states[0] is the start state; states[k] is the state after k agent steps
        st0 = torch.cat((env.car.pos, env.car.quat, env.car.steer.unsqueeze(1)), 1).float().cpu().numpy()
        for i in range(T):
            replays[i].append([round(float(x), 4) for x in st0[i]])
    with torch.no_grad():
        while True:
            alive = ~(env.rules.finished | env.rules.out) & (env.rules.ticks < cap)
            if not bool(alive.any()):
                break
            energy_before = mechanical_energy(env.car, cfg, g)
            action = policy(env)
            env.step(action)
            speed = torch.linalg.norm(env.car.vel, dim=1)
            max_speed = torch.where(alive, torch.maximum(max_speed, speed.float()), max_speed)
            energy_after = mechanical_energy(env.car, cfg, g)
            coasting = (action % 3) == 1  # coast pedal only (brake at standstill engages reverse, which adds energy)
            gain = torch.where(alive & coasting, (energy_after - energy_before).float(), torch.zeros_like(max_speed))
            coast_energy_gain = torch.maximum(coast_energy_gain, gain)
            if record_replays:
                st = torch.cat((env.car.pos, env.car.quat, env.car.steer.unsqueeze(1)), 1).float().cpu().numpy()
                for i in alive.nonzero().squeeze(1).tolist():
                    replays[i].append([round(float(x), 4) for x in st[i]])
    finished = env.rules.finished & (env.rules.ticks <= cap)
    ticks = env.rules.ticks
    out = []
    for i in range(T):
        f = bool(finished[i])
        r = float(ref[i])
        score = (r / float(ticks[i])) if (f and r > 0) else 0.0
        reason = "finished" if f else ("timeout" if not bool(env.rules.out[i]) else OUT_REASONS[int(env.rules.out_reason[i])])
        row = {"track": tracks.names[i], "finished": int(f), "ticks": int(ticks[i]), "time": float(ticks[i]) * cfg.dt,
               "ref_ticks": int(r), "score": score, "end": reason, "progress": float(env.rules.s[i]),
               "max_speed": float(max_speed[i]), "flag_overspeed": int(float(max_speed[i]) > 1.2 * top),
               "max_coast_energy_gain": float(coast_energy_gain[i]),
               "flag_energy_gain": int(float(coast_energy_gain[i]) > 2.0)}
        if record_replays:
            row["replay"] = replays[i]
        out.append(row)
    return out


def summarise(rows: list[dict]) -> dict:
    """Success-weighted speed (mean score), finish rate and mean time ratio on finished tracks."""
    s = np.array([r["score"] for r in rows]); f = np.array([r["finished"] for r in rows])
    ratio = np.array([r["ref_ticks"] / r["ticks"] for r in rows if r["finished"] and r["ref_ticks"] > 0])
    return {"sws": float(s.mean()) if len(s) else math.nan, "finish_rate": float(f.mean()) if len(f) else math.nan,
            "speed_ratio_finished": float(ratio.mean()) if len(ratio) else math.nan, "n_tracks": len(rows),
            "flags_overspeed": int(sum(r["flag_overspeed"] for r in rows)),
            "flags_energy_gain": int(sum(r["flag_energy_gain"] for r in rows))}


def write_eval_csv(rows: list[dict], path: Path):
    """Write per-track rows (without replays) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if k != "replay"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


def _log_access(split: str, who: str, n: int):
    from datetime import datetime, timezone
    from ..utils.paths import RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "test_access_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "split": split, "policy": who, "tracks": n}) + "\n")
    log.info("protected set '%s' read by %s (%d tracks); logged", split, who, n)
