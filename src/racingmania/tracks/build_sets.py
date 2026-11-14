"""Build the track sets: tune the scripted driver, generate and certify every set, save pools.

Global context: this is phase 1 of the pipeline (``scripts/01_generate_tracks.py``). Order, fixed in
reports/preregistration.md (sections 3 and 7): (1) tune the scripted driver's two parameters on
validation candidates; (2) generate candidates for every seed range in parallel; (3) certify each
candidate by driving it at three speed settings; (4) save the certified tracks of each set as a pool
with their reference times; (5) write reference runs into the campaign track files used by the game.
Counts of generated, geometrically rejected and certification-rejected tracks are saved for the report.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict

import numpy as np

from ..drivers.scripted import ScriptedParams
from ..sim.vehicle_config import load_vehicle_config
from ..utils.logging import get_logger
from ..utils.paths import RESULTS_DIR, WEB_ASSETS
from .campaign import CAMPAIGN_PLANS, campaign_track
from .generator import SPLITS, generate_track
from .store import batch_from_pool, pack, pool_path, save_pool
from .validate import certify, tune_scripted

log = get_logger(__name__)
LIGHT_KEYS = ("name", "seed", "center", "left", "gap", "floor", "checkpoints", "start", "half_width", "length", "generator")


def _light(seed: int):
    """Generate one track and keep only what pools need (keeps inter-process traffic small)."""
    t = generate_track(seed)
    return None if t is None else {k: t[k] for k in LIGHT_KEYS}


def generate_many(seeds, processes: int = 14) -> tuple[list[dict], int]:
    """Generate tracks for ``seeds`` in parallel; returns the valid ones and the number rejected."""
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes) as pool:
        out = pool.map(_light, list(seeds), chunksize=32)
    good = [t for t in out if t is not None]
    log.info("generated %d/%d tracks in %.0fs", len(good), len(out), time.time() - t0)
    return good, len(out) - len(good)


def tracks_summary_dir():
    d = RESULTS_DIR / "tracks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tune(device="cuda", processes: int = 14) -> ScriptedParams:
    """Tune the scripted driver on validation candidates and save the table."""
    cfg = load_vehicle_config()
    lo, hi = SPLITS["val"]
    cands, rejected = generate_many(range(lo, hi), processes)
    params, table = tune_scripted(cfg, batch_from_pool(pack(cands), device=device), device=device)
    (tracks_summary_dir() / "scripted_tuning.json").write_text(json.dumps(
        {"chosen": asdict(params), "grid": table, "n_candidates": len(cands), "geometric_rejects": rejected}, indent=2))
    log.info("scripted driver tuned: %s", params)
    return params


def load_tuned() -> ScriptedParams:
    d = json.loads((tracks_summary_dir() / "scripted_tuning.json").read_text())["chosen"]
    return ScriptedParams(**d)


def build_split(split: str, params: ScriptedParams, device="cuda", processes: int = 14, limit: int | None = None,
                keep_failed: bool = False):
    """Generate, certify and save one set. ``keep_failed`` keeps uncertified tracks (unfiltered test set)."""
    cfg = load_vehicle_config()
    lo, hi = SPLITS["test" if split == "test_unfiltered" else split]
    if split == "test_unfiltered":
        hi = lo + 200
    if limit:
        hi = min(hi, lo + limit)
    cands, geo_rejects = generate_many(range(lo, hi), processes)
    tb = batch_from_pool(pack(cands), device=device)
    passed, ref, _ = certify(cfg, tb, params, device=device)
    keep = np.ones(len(cands), bool) if keep_failed else passed
    chosen = [c for c, k in zip(cands, keep) if k]
    pool = pack(chosen, ref_ticks=[int(r) for r, k in zip(ref, keep) if k])
    meta = {"split": split, "seed_range": [lo, hi], "generated": hi - lo, "geometric_rejects": geo_rejects,
            "certification_rejects": int((~passed).sum()), "kept": len(chosen), "scripted": asdict(params)}
    save_pool(pool, pool_path(split), meta)
    (tracks_summary_dir() / f"{split}_summary.json").write_text(json.dumps(
        {**meta, "mean_length_m": float(np.mean([c["length"] for c in chosen])),
         "mean_ref_time_s": float(np.mean([r for r in pool["ref_ticks"] if r > 0]) * cfg.dt)}, indent=2))
    log.info("%s: kept %d of %d seeds", split, len(chosen), hi - lo)
    return meta


def build_campaign(params: ScriptedParams, device="cuda"):
    """Certify the campaign tracks and store their reference runs in the game's track files."""
    cfg = load_vehicle_config()
    tracks = [campaign_track(n) for n in sorted(CAMPAIGN_PLANS)]
    passed, ref, seqs = certify(cfg, batch_from_pool(pack(tracks), device=device), params, device=device)
    for t, ok, r, s in zip(tracks, passed, ref, seqs):
        t["reference"] = {"ticks": int(r), "time": round(float(r) * cfg.dt, 2), "actions": s, "certified": bool(ok)}
        (WEB_ASSETS / "tracks" / f"{t['name']}.json").write_text(json.dumps(t), encoding="utf-8")
        log.info("%s: reference %.2f s (certified at all scales: %s)", t["name"], r * cfg.dt, ok)
    return {t["name"]: t["reference"]["time"] for t in tracks}
