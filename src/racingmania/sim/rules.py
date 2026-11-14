"""Batched port of ``web/js/core/rules.js``: progress, checkpoints, finish and out conditions.

Global context: these rules define the reward (metres of centreline progress) and the episode
ends for the RL agent, and they are identical to what players see in the browser game
(docs/formats.md, "Rules"). Each function mirrors its JavaScript namesake operation by operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

MAX_AIR_TICKS = 500
STALL_TICKS = 200
STALL_DISTANCE = 0.5
SEARCH_OFFSETS = tuple(range(-2, 7))  # nearest-sample window [index - 2, index + 6]


@dataclass
class RuleState:
    """Per-car rule state ``[N]``; mirrors the fields of ``RaceRules`` in rules.js."""

    index: Tensor  # nearest centreline sample (long)
    s: Tensor  # progress (m)
    anchor_s: Tensor  # progress at the last stall-timer reset
    stall_ticks: Tensor  # long
    air_ticks: Tensor  # long
    next_cp: Tensor  # checkpoints passed (long)
    ticks: Tensor  # ticks since the start of the run (long)
    finished: Tensor  # bool
    out: Tensor  # bool
    out_reason: Tensor  # 0 none, 1 fell, 2 airborne, 3 stalled (long)

    def clone(self) -> "RuleState":
        return RuleState(**{k: v.clone() for k, v in self.__dict__.items()})


def nearest_index(tracks, track_id: Tensor, p: Tensor, index: Tensor) -> Tensor:
    """rules.js ``nearestIndex``: argmin of squared distance over the window (first minimum wins)."""
    off = torch.tensor(SEARCH_OFFSETS, device=index.device)
    j = index.unsqueeze(1) + off  # [N, 9]
    n = tracks.n[track_id].unsqueeze(1)
    valid = (j >= 0) & (j <= n - 1)
    jc = j.clamp(0, tracks.center.shape[1] - 1)
    c = tracks.center[track_id.unsqueeze(1).expand_as(jc), jc]  # [N, 9, 3]
    dx = p[:, 0:1] - c[..., 0]; dy = p[:, 1:2] - c[..., 1]; dz = p[:, 2:3] - c[..., 2]
    d = dx * dx + dy * dy + dz * dz
    d = torch.where(valid, d, torch.full_like(d, float("inf")))
    return jc.gather(1, torch.argmin(d, dim=1, keepdim=True)).squeeze(1)


def _project(a: Tensor, b: Tensor, p: Tensor):
    """rules.js ``projectOnSegment``: clamped parameter t and squared distance."""
    ux = b[:, 0] - a[:, 0]; uy = b[:, 1] - a[:, 1]; uz = b[:, 2] - a[:, 2]
    wx = p[:, 0] - a[:, 0]; wy = p[:, 1] - a[:, 1]; wz = p[:, 2] - a[:, 2]
    uu = ux * ux + uy * uy + uz * uz
    t = (wx * ux + wy * uy + wz * uz) / uu
    t = torch.clamp(t, 0, 1)  # Math.min(Math.max(t, 0), 1)
    ex = wx - t * ux; ey = wy - t * uy; ez = wz - t * uz
    return t, ex * ex + ey * ey + ez * ez


def progress_at(tracks, track_id: Tensor, p: Tensor, j: Tensor) -> Tensor:
    """rules.js ``progressAt``: nearer projection onto segment j-1 or j (ties toward j-1)."""
    n = tracks.n[track_id]
    M = tracks.center.shape[1]
    best_s = torch.zeros_like(p[:, 0]); best_d = torch.full_like(p[:, 0], float("inf"))
    for k in (j - 1, j):
        ok = (k >= 0) & (k <= n - 2)
        kc = k.clamp(0, M - 2)
        a = tracks.center[track_id, kc]; b = tracks.center[track_id, kc + 1]
        t, d = _project(a, b, p)
        c0 = tracks.cum[track_id, kc]; c1 = tracks.cum[track_id, kc + 1]
        s = c0 + t * (c1 - c0)
        better = ok & (d < best_d)
        best_s = torch.where(better, s, best_s); best_d = torch.where(better, d, best_d)
    return best_s


def reset_rules(tracks, track_id: Tensor, p: Tensor, index: Tensor) -> RuleState:
    """``RaceRules.reset``: index snapped with the window search, progress measured at p."""
    idx = nearest_index(tracks, track_id, p, index)
    s = progress_at(tracks, track_id, p, idx)
    zl = torch.zeros_like(idx)
    zb = torch.zeros_like(idx, dtype=torch.bool)
    return RuleState(index=idx, s=s, anchor_s=s.clone(), stall_ticks=zl.clone(), air_ticks=zl.clone(),
                     next_cp=zl.clone(), ticks=zl.clone(), finished=zb.clone(), out=zb.clone(), out_reason=zl.clone())


def update_rules(tracks, track_id: Tensor, st: RuleState, p: Tensor, contacts: Tensor, active: Tensor):
    """``RaceRules.update`` for one tick, only for cars where ``active`` (others are left unchanged).

    Returns ``(new_state, gained)`` where ``gained`` is the progress during this tick (0 if inactive).
    """
    ticks = st.ticks + 1
    index = nearest_index(tracks, track_id, p, st.index)
    s = progress_at(tracks, track_id, p, index)
    gained = s - st.s

    passed = (tracks.cp_s[track_id] <= s.unsqueeze(1)).sum(dim=1)  # cp_s sorted, padding +inf
    next_cp = torch.maximum(st.next_cp, passed)  # equals the while-loop in rules.js
    finished = next_cp >= tracks.n_cp[track_id]

    fell = ~finished & (p[:, 1] < tracks.floor[track_id, index])
    air_ticks = torch.where(contacts == 0, st.air_ticks + 1, torch.zeros_like(st.air_ticks))
    airborne = ~finished & ~fell & (air_ticks >= MAX_AIR_TICKS)
    progressed = s >= st.anchor_s + STALL_DISTANCE
    anchor_s = torch.where(progressed, s, st.anchor_s)
    stall_ticks = torch.where(progressed, torch.zeros_like(st.stall_ticks), st.stall_ticks + 1)
    stalled = ~finished & ~fell & ~airborne & (stall_ticks >= STALL_TICKS)
    out = fell | airborne | stalled
    reason = torch.where(fell, 1, torch.where(airborne, 2, torch.where(stalled, 3, 0)))

    # In rules.js, later counters are not updated once an earlier condition ends the tick.
    air_ticks = torch.where(finished | fell, st.air_ticks, air_ticks)
    anchor_s = torch.where(finished | fell | airborne, st.anchor_s, anchor_s)
    stall_ticks = torch.where(finished | fell | airborne, st.stall_ticks, stall_ticks)

    a = active
    new = RuleState(
        index=torch.where(a, index, st.index), s=torch.where(a, s, st.s), anchor_s=torch.where(a, anchor_s, st.anchor_s),
        stall_ticks=torch.where(a, stall_ticks, st.stall_ticks), air_ticks=torch.where(a, air_ticks, st.air_ticks),
        next_cp=torch.where(a, next_cp, st.next_cp), ticks=torch.where(a, ticks, st.ticks),
        finished=torch.where(a, finished, st.finished), out=torch.where(a, out, st.out),
        out_reason=torch.where(a, reason, st.out_reason))
    return new, torch.where(a, gained, torch.zeros_like(gained))
