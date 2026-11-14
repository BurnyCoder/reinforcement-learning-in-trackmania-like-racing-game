"""Where do agents fall off? Turn sharpness and jumps just before each failure, against random places.

Local purpose: for every failed run of an evaluation CSV, find the sharpest horizontal turn in the
60 m of centreline before the point where the car left the road (its last arc-length progress), and
whether a jump (gap segment) lies in that stretch; compare with the same statistics at uniformly
random arc lengths on the same tracks. A failure position concentrated after tight turns, against a
broad random distribution, is (correlational) evidence that the agent fails by entering corners too
fast.

Global context: the learnings for the first models (reports/notes/first-models.md) use this to name
the generalist's failure mode, as the fix loop's "watch replays to name the failure" step asks
(reports/plan.md). Geometry follows docs/formats.md ("Track file"): ``center`` every ~2 m, ``gap[i]=1``
for a segment without road.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

WINDOW_M = 60.0  # metres of track before the failure point that are searched for the sharpest turn


def _geometry(pool: dict, i: int):
    """Cumulative arc length [M], horizontal curvature at interior samples [M-2] and gap mask [M-1] of track i."""
    n = int(pool["n"][i])
    c = np.asarray(pool["center"][i][:n], float)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)  # segment lengths
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    heading = np.unwrap(np.arctan2(np.diff(c[:, 2]), np.diff(c[:, 0])))  # horizontal heading of each segment (x–z plane)
    curv = np.abs(np.diff(heading)) / np.maximum(seg[1:], 1e-6)  # heading change per metre = 1 / turn radius
    gap = np.asarray(pool["gap"][i][:n - 1]).astype(bool)
    return cum, curv, gap


def stats_before(pool: dict, i: int, s: float) -> tuple[float, bool]:
    """Sharpest turn radius (m) and whether a jump lies in the ``WINDOW_M`` metres before arc length ``s``."""
    cum, curv, gap = _geometry(pool, i)
    j = int(np.searchsorted(cum, s))
    lo = int(np.searchsorted(cum, s - WINDOW_M))
    window = curv[max(0, lo - 1): max(1, j - 1)]  # interior samples between s − 60 m and s
    kmax = float(window.max()) if window.size else 0.0
    return (1.0 / kmax if kmax > 0 else float("inf")), bool(gap[max(0, lo - 1): j].any())


def summarise_positions(radii: np.ndarray, jumps: np.ndarray) -> dict:
    return {"n": int(len(radii)), "median_radius_m": float(np.median(radii)) if len(radii) else float("nan"),
            "share_radius_below_40m": float(np.mean(radii < 40)) if len(radii) else float("nan"),
            "share_radius_below_60m": float(np.mean(radii < 60)) if len(radii) else float("nan"),
            "share_jump_in_window": float(np.mean(jumps)) if len(jumps) else float("nan")}


def failure_positions(eval_csv: Path, pool: dict, random_per_track: int = 5, seed: int = 0) -> dict:
    """Failure-position statistics of one evaluation CSV against random places on the same tracks."""
    rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
    index = {name: k for k, name in enumerate(pool["names"])}
    rng = np.random.default_rng(seed)
    fail = [(index[r["track"]], float(r["progress"])) for r in rows if r["end"] in ("fell", "stalled")]
    rand = []
    for r in rows:
        i = index[r["track"]]
        length = float(_geometry(pool, i)[0][-1])
        rand += [(i, s) for s in rng.uniform(WINDOW_M, length, random_per_track)]
    out = {"tracks": len(rows), "window_m": WINDOW_M}
    for key, pts in (("failures", fail), ("random", rand)):
        st = [stats_before(pool, i, s) for i, s in pts]
        out[key] = summarise_positions(np.array([x[0] for x in st]), np.array([x[1] for x in st]))
    return out
