"""Certify generated tracks by driving them, and tune the scripted driver on validation candidates.

Global context: a generated track enters a set only if the frozen scripted driver finishes it in
the simulator at three speed settings (0.9, 1.0, 1.1 × its target speeds); its finishing time at
1.0 becomes the track's reference time ``T_ref`` (reports/preregistration.md, sections 3–4). This
is an existence proof that the track can be finished with this car, in this deterministic simulator.
The scripted driver's two parameters are tuned first, on validation candidates only.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import torch

from ..drivers.scripted import ScriptedDriver, ScriptedParams
from ..env.vec_env import EnvConfig, RacingVecEnv
from ..sim.track_tensors import TrackBatch
from ..utils.logging import get_logger

log = get_logger(__name__)
SPEED_SCALES = (0.9, 1.0, 1.1)


class _MultiParamDriver:
    """Scripted drivers with different parameters for different cars of one environment."""

    def __init__(self, env, params_per_car: list[ScriptedParams]):
        self.groups = {}
        for i, p in enumerate(params_per_car):
            self.groups.setdefault(p, []).append(i)
        self.drivers = {p: ScriptedDriver(env, p) for p in self.groups}
        self.idx = {p: torch.tensor(ix, device=env.device) for p, ix in self.groups.items()}

    def __call__(self, env):
        out = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for p, d in self.drivers.items():
            a = d(env)
            out[self.idx[p]] = a[self.idx[p]]
        return out


def drive_runs(cfg, tracks: TrackBatch, params_per_run: list[ScriptedParams], track_of_run: torch.Tensor,
               device="cuda", max_ticks: int = 20000, compile: bool = True):
    """Drive each run (track ``track_of_run[i]`` with ``params_per_run[i]``) once from the standing start.

    Returns ``finished [R]`` (bool numpy), ``ticks [R]`` (int numpy) and the action sequences at the
    agent-step level (list of lists) for later storage of reference runs.
    """
    R = len(params_per_run)
    ecfg = EnvConfig(random_start_prob=0.0, max_steps=10**9)
    env = RacingVecEnv(cfg, tracks, R, env_cfg=ecfg, device=device, assign=track_of_run, auto_reset=False, compile=compile)
    driver = _MultiParamDriver(env, params_per_run)
    actions = []
    with torch.no_grad():
        while True:
            alive = ~(env.rules.finished | env.rules.out) & (env.rules.ticks < max_ticks)
            if not bool(alive.any()):
                break
            a = driver(env)
            actions.append(torch.where(alive, a, torch.full_like(a, -1)).cpu())
            env.step(a)
    fin = env.rules.finished.cpu().numpy()
    ticks = env.rules.ticks.cpu().numpy()
    acts = torch.stack(actions, 1).numpy() if actions else np.zeros((R, 0), int)
    seqs = [row[row >= 0].tolist() for row in acts]
    return fin, ticks, seqs


def certify(cfg, tracks: TrackBatch, params: ScriptedParams, device="cuda", chunk: int = 2048, compile: bool = True):
    """Certification of every track: finished at all three speed scales.

    Returns ``passed [T]`` (bool), ``ref_ticks [T]`` (ticks at scale 1.0; 0 if not finished) and the
    scale-1.0 action sequences.
    """
    T = tracks.num_tracks
    passed = np.zeros(T, bool); ref = np.zeros(T, np.int64); seqs = [None] * T
    for lo in range(0, T, chunk):
        ids = list(range(lo, min(T, lo + chunk)))
        sub = tracks.subset(ids)
        k = len(ids)
        runs = [dataclasses.replace(params, speed_scale=s) for s in SPEED_SCALES for _ in range(k)]
        tor = torch.arange(k).repeat(len(SPEED_SCALES))
        fin, ticks, sq = drive_runs(cfg, sub, runs, tor, device=device, compile=compile)
        fin = fin.reshape(len(SPEED_SCALES), k); ticks = ticks.reshape(len(SPEED_SCALES), k)
        passed[lo:lo + k] = fin.all(0)
        ref[lo:lo + k] = np.where(fin[1], ticks[1], 0)
        for j in range(k):
            seqs[lo + j] = sq[k + j]
        log.info("certified tracks %d-%d: %d/%d passed", lo, lo + k - 1, int(fin.all(0).sum()), k)
    return passed, ref, seqs


def tune_scripted(cfg, val_tracks: TrackBatch, device="cuda", grid=None, compile: bool = True):
    """Grid search over (a_lat, lookahead_gain) on validation candidates.

    Objective (fixed before any result, reports/preregistration.md, section 7): highest share of
    tracks finished at all three speed scales; among settings within 1 percentage point of the best
    share, the lowest mean time at scale 1.0 on the tracks every such setting finishes.
    Returns ``(best_params, table)``.
    """
    grid = grid or {"a_lat": (6.0, 8.0, 10.0, 12.0), "lookahead_gain": (0.3, 0.45, 0.6)}
    settings = [ScriptedParams(a_lat=a, lookahead_gain=g) for a, g in itertools.product(grid["a_lat"], grid["lookahead_gain"])]
    T = val_tracks.num_tracks
    runs, tor = [], []
    for p in settings:
        for s in SPEED_SCALES:
            runs += [dataclasses.replace(p, speed_scale=s)] * T
            tor += list(range(T))
    fin, ticks, _ = drive_runs(cfg, val_tracks, runs, torch.tensor(tor), device=device, compile=compile)
    fin = fin.reshape(len(settings), len(SPEED_SCALES), T); ticks = ticks.reshape(len(settings), len(SPEED_SCALES), T)
    share = fin.all(1).mean(1)
    best_share = share.max()
    cand = [i for i in range(len(settings)) if share[i] >= best_share - 0.01]
    common = np.all([fin[i, 1] for i in cand], axis=0)
    mean_time = {i: float(ticks[i, 1][common].mean() * cfg.dt) if common.any() else float("inf") for i in cand}
    best = min(cand, key=lambda i: mean_time[i])
    table = [{"a_lat": settings[i].a_lat, "lookahead_gain": settings[i].lookahead_gain,
              "finish_all_scales": float(share[i]), "finish_scale_1": float(fin[i, 1].mean()),
              "mean_time_common": mean_time.get(i, None), "chosen": i == best} for i in range(len(settings))]
    return settings[best], table
