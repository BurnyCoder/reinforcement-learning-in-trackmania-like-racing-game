"""Per-track features (jumps, banking, hills) for the "finish rate by track feature" analysis.

Local purpose: describe every evaluation track by a few numbers computed from its geometry:
the number of jumps (runs of gap segments), the steepest banking (degrees) and the height range
(metres), plus the out-of-range variant for the out-of-range set. Campaign tracks are read from
``web/assets/tracks/*.json``; generated sets from their pools under ``RACINGMANIA_DATA_DIR`` (the
first time only), cached as ``results/tracks/<set>_features.csv`` so later builds and clones of the
repository do not need the large pool files.

Global context: the preregistration lists "finish rate by track feature (jumps, banking, hills)"
among the secondary numbers of every run (section 6). Geometry conventions are those of
docs/formats.md ("Track file"): ``left`` is the unit vector to the left edge including banking,
so the bank angle is ``asin(|left_y|)``; ``gap[i] = 1`` marks a segment without road surface.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)
FEATURE_COLUMNS = ["track", "jumps", "max_bank_deg", "height_range_m", "length_m", "variant"]
GENERATED_SETS = ("val", "test", "oor", "test_unfiltered")


def geometry_features(center, left, gap) -> dict:
    """Features of one track from its centreline (M×3), left vectors (M×3) and gap mask (M−1)."""
    c = np.asarray(center, float)
    l = np.asarray(left, float)
    g = np.asarray(gap, int)
    starts = int(np.sum((g[1:] == 1) & (g[:-1] == 0)) + (g[0] == 1 if g.size else 0))  # each jump is one run of gap segments
    bank = np.degrees(np.arcsin(np.clip(np.abs(l[:, 1]), 0.0, 1.0)))
    return {"jumps": starts, "max_bank_deg": float(bank.max()) if bank.size else 0.0,
            "height_range_m": float(c[:, 1].max() - c[:, 1].min()) if c.size else 0.0,
            "length_m": float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0}


def _read_cache(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["track"]] = {"jumps": int(row["jumps"]), "max_bank_deg": float(row["max_bank_deg"]),
                                 "height_range_m": float(row["height_range_m"]), "length_m": float(row["length_m"]),
                                 "variant": row.get("variant", "")}
    return out


def _from_pool(split: str) -> dict[str, dict] | None:
    """Compute features from a pool file if it exists (needs torch; returns None otherwise)."""
    from ..utils.paths import data_dir
    p = data_dir() / "tracks" / f"{split}.pt"  # same location as tracks.store.pool_path, checked before importing torch
    if not p.exists():
        return None
    try:
        from ..tracks.store import load_pool
    except Exception as e:  # torch or the simulator package not importable here
        log.info("track features for %s skipped: %s", split, e)
        return None
    pool = load_pool(p)
    variant_of = None
    if split == "oor":
        from ..tracks.generator import params_for_seed
        variant_of = lambda seed: params_for_seed(int(seed))[1]
    out = {}
    for i, name in enumerate(pool["names"]):
        n = int(pool["n"][i])
        f = geometry_features(pool["center"][i, :n], pool["left"][i, :n], pool["gap"][i, : n - 1])
        f["variant"] = variant_of(pool["seeds"][i]) if variant_of else ""
        out[name] = f
    return out


def load_features(results_dir: Path, repo: Path) -> dict[str, dict]:
    """Track name -> features, for the campaign tracks and every generated set available."""
    feats = {}
    for p in sorted((repo / "web" / "assets" / "tracks").glob("campaign-*.json")):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
            feats[t["name"]] = {**geometry_features(t["center"], t["left"], t["gap"]), "variant": "campaign"}
        except (OSError, KeyError, json.JSONDecodeError) as e:
            log.warning("campaign track %s unreadable: %s", p, e)
    for split in GENERATED_SETS:
        cache = results_dir / "tracks" / f"{split}_features.csv"
        if cache.exists():
            feats.update(_read_cache(cache))
            continue
        f = _from_pool(split)
        if not f:
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS)
            w.writeheader()
            for name, v in f.items():
                w.writerow({"track": name, **{k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in FEATURE_COLUMNS[1:]}})
        log.info("track features of %s cached in %s (%d tracks)", split, cache, len(f))
        feats.update(f)
    return feats


def feature_bins(values: dict[str, float], feature: str) -> tuple[dict[str, str], list[str]]:
    """Assign each track to a bin: 0 / 1 / 2+ jumps, or tertiles (low / mid / high) of a continuous feature.

    Tertiles are computed on the tracks given (one evaluation set), so the bins hold similar numbers
    of tracks; their numeric ranges are part of the label so the reader sees what "high" means.
    """
    if not values:
        return {}, []
    if feature == "jumps":
        lab = lambda v: "0" if v == 0 else ("1" if v == 1 else "2+")
        return {k: lab(v) for k, v in values.items()}, [b for b in ("0", "1", "2+") if any(lab(v) == b for v in values.values())]
    arr = np.array(list(values.values()), float)
    q1, q2 = np.quantile(arr, [1 / 3, 2 / 3])
    lo, hi = float(arr.min()), float(arr.max())
    if math.isclose(q1, q2) or math.isclose(lo, hi):
        name = f"all ({lo:.1f}-{hi:.1f})"
        return {k: name for k in values}, [name]
    names = [f"low ({lo:.1f}-{q1:.1f})", f"mid ({q1:.1f}-{q2:.1f})", f"high ({q2:.1f}-{hi:.1f})"]
    pick = lambda v: names[0] if v <= q1 else (names[1] if v <= q2 else names[2])
    return {k: pick(v) for k, v in values.items()}, names
