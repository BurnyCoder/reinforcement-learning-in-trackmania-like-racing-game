"""``web/assets/data/results.json``: everything the results explorer page (web/results.html) shows.

Local purpose: serialise the analysis for the browser: run statuses, configuration summaries with
intervals, hypotheses, seed sensitivity, the training-set-size curve, feature finish rates,
specialist times, training curves (down-sampled to at most ``MAX_POINTS`` points per curve) and a
column-oriented per-track table (seed-averaged per configuration) for the sortable track view.

Global context: written by ``build.build_all``; read by ``web/js/results/*.js`` with a relative
URL, so the site also works on GitHub Pages under a sub-path. NaN becomes ``null``.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np

from . import narrative
from . import taxonomy as tx
from .render import REPO_URL, nan_to_none

MAX_POINTS = 300


def _downsample(x: np.ndarray, *ys: np.ndarray):
    """Evenly spaced subset of at most MAX_POINTS indices (always keeping the last point)."""
    n = len(x)
    idx = np.unique(np.r_[np.linspace(0, n - 1, min(n, MAX_POINTS)).round().astype(int), n - 1]) if n else np.array([], int)
    r = lambda a: [None if (v != v) else round(float(v), 5) for v in a[idx]]
    return [r(x)] + [r(y) for y in ys]


def curves(res) -> dict:
    out = {}
    for r in res.runs:
        tr, va = res.train_curves.get(r["rid"], {}), res.val_curves.get(r["rid"], {})
        if not tr and not va:
            continue
        d = {"key": r["key"], "label": r["label"], "seed": r["seed"], "category": r["category"]}
        if "global_step" in tr:
            cols = [c for c in ("ep_return", "ep_finish_rate", "ep_progress", "entropy", "sps") if c in tr]
            vals = _downsample(tr["global_step"], *[tr[c] for c in cols])
            d["train"] = {"step": vals[0], **dict(zip(cols, vals[1:]))}
        if "global_step" in va:
            cols = [c for c in ("sws", "finish_rate") if c in va]
            vals = _downsample(va["global_step"], *[va[c] for c in cols])
            d["val"] = {"step": vals[0], **dict(zip(cols, vals[1:]))}
        out[r["rid"]] = d
    return out


def per_track(res, an) -> dict:
    """set -> {tracks, ref_time, methods: key -> {score, finished, time, n_seeds}} (seed-averaged)."""
    out = {}
    for s in tx.SETS:
        tracks, ref = {}, {}
        methods = {}
        for c in an.configs:
            rids = [r for r in c["rids"] if s in res.evals.get(r, {})]
            if not rids:
                continue
            agg: dict = {}
            for rid in rids:
                for row in res.evals[rid][s]:
                    t = row["track"]
                    tracks[t] = True
                    if row["ticks"] and row["ref_ticks"]:
                        ref[t] = row["ref_ticks"] * row["time"] / row["ticks"]
                    a = agg.setdefault(t, {"score": [], "finished": [], "time": []})
                    a["score"].append(float(row["score"]))
                    a["finished"].append(float(row["finished"]))
                    if row["finished"]:
                        a["time"].append(float(row["time"]))
            methods[c["key"]] = (agg, len(rids))
        if not methods:
            continue
        names = sorted(tracks)
        out[s] = {"tracks": names, "ref_time": [round(ref[t], 3) if t in ref else None for t in names],
                  "features": [nan_to_none(res.features.get(t)) for t in names], "methods": {}}
        for k, (agg, n) in methods.items():
            col = lambda f, fn: [round(fn(agg[t][f]), 4) if t in agg and agg[t][f] else None for t in names]
            out[s]["methods"][k] = {"score": col("score", np.mean), "finished": col("finished", np.mean),
                                    "time": col("time", np.mean), "n_seeds": n}
    return out


def results_json(res, an, figs) -> dict:
    runs = [{k: r.get(k) for k in ("rid", "key", "label", "experiment", "seed", "track", "status", "phase", "group",
                                   "category", "global_step", "wall_hours", "physics", "updated")} for r in res.runs + res.queued]
    configs = [{k: c.get(k) for k in ("key", "label", "category", "group", "experiment", "track", "pilot", "physics",
                                      "n_train_tracks", "seeds", "statuses", "algo", "architecture", "objective",
                                      "observation", "training_data", "start_states", "description")} for c in an.configs]
    data = {
        "generated": date.today().isoformat(), "repo": REPO_URL, "primary_set": an.primary_set, "paused": res.paused,
        "sets": tx.SET_LABELS, "set_short": tx.SET_SHORT, "statuses": tx.STATUSES, "status_text": tx.STATUS_TEXT,
        "categories": tx.CATEGORY_LABELS, "groups": [{"slug": g[0], "title": g[1], "question": g[2]} for g in tx.GROUPS],
        "headline": narrative.headline(an), "status_counts": an.status_counts, "runs": runs, "configs": configs,
        "summaries": an.summaries, "hypotheses": an.hypotheses, "cross_checks": an.cross_checks,
        "seed_sensitivity": an.seed_sensitivity, "single_vs_multi": an.single_vs_multi, "size_curve": an.size_curve,
        "feature_rates": an.feature_rates, "specialists": an.specialists, "flags": an.flags, "curves": curves(res),
        "per_track": per_track(res, an), "track_sets": res.track_sets, "exploits": res.exploits,
        "figures": {k: v["png"] for k, v in figs.items() if k != "curves"},
    }
    return nan_to_none(data)


def write(res, an, figs) -> str:
    res.paths.web_data.mkdir(parents=True, exist_ok=True)
    p = res.paths.web_data / "results.json"
    p.write_text(json.dumps(results_json(res, an, figs), separators=(",", ":"), allow_nan=False), encoding="utf-8", newline="\n")
    return str(p)

