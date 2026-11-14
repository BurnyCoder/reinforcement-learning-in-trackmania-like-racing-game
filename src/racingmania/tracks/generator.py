"""Seeded procedural generator: a random segment plan → a ribbon track (``ribbon.build_track``).

Global context: every generated track is identified by an integer seed. The seed ranges of the
training pool, validation, test and out-of-range sets are disjoint (reports/preregistration.md,
section 3); ``split_of_seed`` enforces that. A generated plan is only a candidate: it becomes part
of a set after it passes the geometric checks here and the driving certification in
``validate.py`` (a frozen scripted driver must finish it).

Generator design: TORCS-style straights and arcs (https://torcs.sourceforge.net/api/track_manual.html)
with smooth height changes, banking toward the inside of arcs, and parabolic-kicker jumps on long
straights. Ranges are in ``GeneratorParams``; the out-of-range test set pushes exactly one factor
beyond its training range.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .ribbon import SPACING, build_track, crossing_ok

GENERATOR_VERSION = 1

# Disjoint seed ranges (reports/preregistration.md, section 3).
SPLITS = {
    "train": (0, 50_000),
    "val": (1_000_000, 1_000_200),
    "test": (2_000_000, 2_001_000),
    "oor": (3_000_000, 3_000_200),
}


def split_of_seed(seed: int) -> str:
    """Name of the set a seed belongs to; raises if it belongs to none."""
    for name, (lo, hi) in SPLITS.items():
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} is outside every split")


@dataclass(frozen=True)
class GeneratorParams:
    """Ranges the generator samples from (training distribution)."""

    length: tuple[float, float] = (600.0, 1400.0)  # total length before the final straight (m)
    radius: tuple[float, float] = (22.0, 120.0)  # arc radius (m)
    angle: tuple[float, float] = (20.0, 150.0)  # arc angle (deg)
    straight: tuple[float, float] = (30.0, 140.0)  # straight length (m)
    grade: tuple[float, float] = (0.0, 0.06)  # |height change| / length per segment
    bank: tuple[float, float] = (0.0, 14.0)  # arc banking toward the inside (deg)
    p_bank: float = 0.6  # probability an arc is banked
    p_jump: float = 0.4  # probability a long straight gets a jump
    jump_min_straight: float = 80.0  # only straights at least this long get a jump
    gap: tuple[float, float] = (6.0, 14.0)  # jump gap (m)
    kicker_h: tuple[float, float] = (0.8, 1.6)  # kicker height (m)
    kicker_len: tuple[float, float] = (10.0, 16.0)
    drop: tuple[float, float] = (0.0, 2.0)  # landing lower than the lip by this much (m)
    landing_len: tuple[float, float] = (22.0, 30.0)


TRAIN_PARAMS = GeneratorParams()
# Out-of-range variants: one factor beyond the training range each (preregistration, section 3).
OOR_VARIANTS = {
    "tight_turns": dict(radius=(14.0, 22.0)),
    "steep_hills": dict(grade=(0.06, 0.10)),
    "long_jumps": dict(gap=(14.0, 20.0), p_jump=0.8),
    "long_track": dict(length=(1400.0, 2400.0)),
}


def params_for_seed(seed: int) -> tuple[GeneratorParams, str]:
    """Training ranges for train/val/test seeds; one out-of-range factor for oor seeds."""
    if split_of_seed(seed) == "oor":
        name = sorted(OOR_VARIANTS)[seed % len(OOR_VARIANTS)]
        return dataclasses.replace(TRAIN_PARAMS, **OOR_VARIANTS[name]), name
    return TRAIN_PARAMS, "train_ranges"


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


def random_plan(rng: np.random.Generator, p: GeneratorParams) -> list[dict]:
    """Sample a segment plan: flat start straight, random middle, flat finish straight."""
    plan = [{"kind": "straight", "length": 40.0, "dh": 0.0}]
    target = _u(rng, p.length)
    total = 40.0
    last_kind = "straight"
    while total < target:
        make_arc = last_kind == "straight" or rng.random() < 0.55
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if make_arc:
            radius, angle = _u(rng, p.radius), _u(rng, p.angle)
            length = radius * np.radians(angle)
            seg = {"kind": "arc", "radius": round(radius, 2), "angle": round(angle, 2), "dir": int(rng.choice([-1, 1])),
                   "dh": round(sign * _u(rng, p.grade) * length, 2),
                   "bank": round(_u(rng, p.bank), 2) if rng.random() < p.p_bank else 0.0}
        else:
            length = _u(rng, p.straight)
            seg = {"kind": "straight", "length": round(length, 2), "dh": round(sign * _u(rng, p.grade) * length, 2)}
            if length >= p.jump_min_straight and rng.random() < p.p_jump:
                lk, lg, ll = _u(rng, p.kicker_len), _u(rng, p.gap), _u(rng, p.landing_len)
                need = 30.0 + lk + lg + ll + 10.0  # run-up + kicker + gap + landing + run-out
                if need > length:
                    length = need
                seg["length"] = round(length, 2)
                seg["dh"] = 0.0  # jumps sit on flat straights so the take-off speed is predictable
                seg["jump"] = {"at": float(SPACING * round(30.0 / SPACING)), "kicker_len": round(lk, 2),
                               "kicker_h": round(_u(rng, p.kicker_h), 2), "gap": round(lg, 2),
                               "drop": round(_u(rng, p.drop), 2), "landing_len": round(ll, 2)}
        plan.append(seg)
        total += length
        last_kind = seg["kind"]
    plan.append({"kind": "straight", "length": 80.0, "dh": 0.0})  # finish straight with run-out
    return plan


def generate_track(seed: int, max_tries: int = 50) -> dict | None:
    """Geometrically valid track for ``seed`` (deterministic), or None after ``max_tries`` attempts.

    Attempt ``i`` uses ``np.random.default_rng([seed, i])``, so the same seed always yields the same
    track. Certification by driving happens separately (``validate.py``).
    """
    split = split_of_seed(seed)
    params, variant = params_for_seed(seed)
    for attempt in range(max_tries):
        rng = np.random.default_rng([seed, attempt])
        plan = random_plan(rng, params)
        track = build_track(plan, name=f"{split}-{seed}", seed=seed,
                            generator_info={"version": GENERATOR_VERSION, "split": split, "variant": variant,
                                            "attempt": attempt})
        heights = np.asarray(track["center"])[:, 1]
        if len(track["vertices"]) > 32000 or heights.max() - heights.min() > 120:
            continue
        if crossing_ok(track):
            return track
    return None


LIGHT_KEYS = ("name", "seed", "center", "left", "gap", "floor", "checkpoints", "start", "half_width", "length", "generator")


def generate_light(seed: int) -> dict | None:
    """``generate_track`` keeping only what track pools need, for worker processes (small to pickle).
    This module imports no PyTorch, so worker processes stay small."""
    t = generate_track(seed)
    if t is None:
        return None
    out = {k: t[k] for k in LIGHT_KEYS}
    # Compact NumPy arrays instead of nested lists: ~8x less memory when 50,000 tracks are held at once.
    for k in ("center", "left", "floor"):
        out[k] = np.asarray(t[k], dtype=np.float64)
    out["gap"] = np.asarray(t["gap"], dtype=np.uint8)
    return out
