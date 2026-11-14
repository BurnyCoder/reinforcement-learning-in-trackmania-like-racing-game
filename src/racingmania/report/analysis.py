"""The preregistered analysis: per-configuration scores, hypotheses H1-H3, seed sensitivity, curves.

Local purpose: turn the tidy tables of ``collect.Results`` into numbers: for every configuration
(an experiment, its training track if any, pilot or not, physics version; seeds pooled) and every
evaluation set its SWS and finish rate with two-level bootstrap intervals; the three hypothesis
tests exactly as reports/preregistration.md (section 5, section 6 and the amendments) specifies;
the seed-sensitivity numbers; the training-set-size curve; finish rate by track feature; and the
specialist-versus-generalist times on the campaign tracks.

Global context: tables, figures, reports, README block, paper, website and model cards all read
from the ``Analysis`` object built here, so the decision rules live in one place.
Decision rule per hypothesis (preregistration, section 5 and amendment 1):
- "refuted" if the upper end of the 95 % bootstrap interval lies below the threshold
  (1.0 for H1; a difference of 0.10 for H2 and H3);
- otherwise "confirmed" if the Holm-adjusted one-sided t-test p-value is below 0.05;
- otherwise "inconclusive";
- "not yet tested" until all preregistered seeds (10 generalist, 5 single-track, 5 blind) have
  test results; numbers computed from fewer seeds are shown and labelled preliminary.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..eval import stats
from ..utils.logging import get_logger
from . import taxonomy as tx
from .features import feature_bins

log = get_logger(__name__)

ALPHA = 0.05  # family-wise error rate for H1-H3 (preregistration, section 5)
SMALLEST_EFFECT = 0.10  # smallest SWS difference treated as meaningful (section 5)
H1_THRESHOLD = 1.0  # SWS of the scripted driver by construction (section 4)
CONFIRM_SET = "test"  # hypotheses are tested on the test set only (section 5)
HYPOTHESES = [
    # id, statement, treatment config, comparison config (None = constant), threshold, required seeds
    ("H1", "The generalist drives unseen tracks better than the scripted driver", "generalist", None, H1_THRESHOLD,
     {"generalist": 10}),
    ("H2", "Track diversity is what generalises (50,000 tracks vs 1 track)", "generalist", "size-1", SMALLEST_EFFECT,
     {"generalist": 10, "size-1": 5}),
    ("H3", "The agent uses the look-ahead (generalist vs blind control)", "generalist", "blind", SMALLEST_EFFECT,
     {"generalist": 10, "blind": 5}),
]
FEATURE_METHODS = ["generalist", "blind", "size-1", "pqn", "generalist-pilot"]  # at most 4 are drawn


@dataclass
class Analysis:
    configs: list = field(default_factory=list)  # one dict per configuration, display order
    summaries: dict = field(default_factory=dict)  # key -> set -> summary dict
    primary_set: str | None = None  # "test" once any test result exists, else "val"
    hypotheses: list = field(default_factory=list)
    cross_checks: list = field(default_factory=list)
    seed_sensitivity: list = field(default_factory=list)
    single_vs_multi: list = field(default_factory=list)
    size_curve: list = field(default_factory=list)
    feature_rates: dict = field(default_factory=dict)
    specialists: list = field(default_factory=list)
    status_counts: dict = field(default_factory=dict)  # phase -> status -> count
    flags: list = field(default_factory=list)

    def config(self, key: str) -> dict | None:
        return next((c for c in self.configs if c["key"] == key), None)

    def summary(self, key: str, s: str) -> dict | None:
        return self.summaries.get(key, {}).get(s)


# ----------------------------------------------------------------------------- matrices
def matrix(res, rids: list[str], set_name: str, column: str = "score", tracks: list[str] | None = None):
    """Seeds × tracks array of ``column`` for the runs ``rids`` on one set.

    Rows follow ``rids`` (runs without this set are dropped); columns are the tracks every run has
    (sorted by name) or ``tracks`` if given. Returns (array, tracks, kept rids).
    """
    per = []
    for rid in rids:
        rows = res.evals.get(rid, {}).get(set_name)
        if rows:
            per.append((rid, {r["track"]: float(r[column]) for r in rows}))
    if not per:
        return np.zeros((0, 0)), [], []
    if tracks is None:
        common = set(per[0][1])
        for _, d in per[1:]:
            common &= set(d)
        dropped = max(len(d) for _, d in per) - len(common)
        if dropped:
            log.warning("%s on %s: %d tracks not shared by all seeds were dropped", rids[0], set_name, dropped)
        tracks = sorted(common)
    M = np.array([[d.get(t, math.nan) for t in tracks] for _, d in per], float)
    return M, tracks, [rid for rid, _ in per]


def _common_tracks(res, rids_a, rids_b, s):
    _, ta, _ = matrix(res, rids_a, s)
    _, tb, _ = matrix(res, rids_b, s)
    return sorted(set(ta) & set(tb))


def summarise_config(res, cfg: dict, s: str) -> dict | None:
    """SWS, finish rate (both with 95 % intervals), IQM, time ratio and flags of one configuration on one set."""
    S, tracks, rids = matrix(res, cfg["rids"], s, "score")
    if not rids:
        return None
    F, _, _ = matrix(res, rids, s, "finished", tracks)
    ticks, _, _ = matrix(res, rids, s, "ticks", tracks)
    ref, _, _ = matrix(res, rids, s, "ref_ticks", tracks)
    sws_ci, fin_ci = stats.bootstrap_mean(S), stats.bootstrap_mean(F)
    fin_mask = (F > 0) & (ref > 0)
    seeds = [res.run(r)["seed"] for r in rids]
    flags_o = sum(int(sum(x.get("flag_overspeed", 0) or 0 for x in res.evals[r][s])) for r in rids)
    flags_e = sum(int(sum(x.get("flag_energy_gain", 0) or 0 for x in res.evals[r][s])) for r in rids)
    return {"set": s, "sws": sws_ci.estimate, "sws_lo": sws_ci.lo, "sws_hi": sws_ci.hi,
            "finish": fin_ci.estimate, "finish_lo": fin_ci.lo, "finish_hi": fin_ci.hi,
            "n_seeds": len(rids), "n_tracks": len(tracks), "iqm": stats.iqm(S),
            "time_ratio_finished": float((ticks[fin_mask] / ref[fin_mask]).mean()) if fin_mask.any() else math.nan,
            "per_seed": [{"seed": sd, "rid": r, "sws": float(S[i].mean()), "finish": float(F[i].mean())}
                         for i, (sd, r) in enumerate(zip(seeds, rids))],
            "flags_overspeed": flags_o, "flags_energy_gain": flags_e}


# ----------------------------------------------------------------------------- configurations
def build_configs(res) -> list[dict]:
    """Group runs (and queued run-matrix entries) into configurations, in display order."""
    groups: dict[str, dict] = {}
    for r in res.runs + res.queued:
        c = groups.setdefault(r["key"], {
            "key": r["key"], "label": r["label"], "experiment": r["experiment"], "track": r.get("track"),
            "pilot": r["pilot"], "physics": r["physics"], "group": r["group"], "category": r["category"],
            "algo": r.get("algo"), "architecture": r["architecture"], "objective": r["objective"],
            "observation": r["observation"], "training_data": r["training_data"], "start_states": r["start_states"],
            "description": r.get("description") or r.get("exp_cfg", {}).get("description", ""),
            "n_train_tracks": r.get("n_train_tracks"), "rids": [], "queued": [], "statuses": defaultdict(int),
            "order": (tx.CATEGORIES.index(r["category"]), r["phase_order"], r["matrix_order"] if r["matrix_order"] >= 0 else 10**6),
            "wall_hours": 0.0, "global_step": []})
        c["statuses"][r["status"]] += 1
        if r["status"] == "not started":
            c["queued"].append(r["rid"])
        else:
            c["rids"].append(r["rid"])
            if isinstance(r.get("wall_hours"), (int, float)):
                c["wall_hours"] += float(r["wall_hours"])
            if isinstance(r.get("global_step"), (int, float)):
                c["global_step"].append(int(r["global_step"]))
            for k in ("architecture", "objective", "observation", "training_data", "start_states"):
                c[k] = r[k]  # a registered run's logged config beats the experiment file
    out = sorted(groups.values(), key=lambda c: c["order"])
    for c in out:
        c["statuses"] = dict(c["statuses"])
        c["seeds"] = sorted({res.run(r)["seed"] for r in c["rids"]})
    return out


def _decide(tested: bool, ci_hi: float, threshold: float, p_holm: float) -> str:
    if not tested:
        return "not yet tested"
    if ci_hi == ci_hi and ci_hi < threshold:  # interval entirely below the threshold
        return "refuted"
    if p_holm == p_holm and p_holm < ALPHA:
        return "confirmed"
    return "inconclusive"


def hypotheses(res, an: Analysis) -> list[dict]:
    """H1-H3 on the test set, computed exactly as preregistered (see the module docstring)."""
    out = []
    for hid, text, treat, comp, thr, required in HYPOTHESES:
        ct, cc = an.config(treat), an.config(comp) if comp else None
        rt = [r for r in (ct or {}).get("rids", []) if CONFIRM_SET in res.evals.get(r, {})]
        rc = [r for r in (cc or {}).get("rids", []) if CONFIRM_SET in res.evals.get(r, {})]
        have = {treat: len(rt), **({comp: len(rc)} if comp else {})}
        tested = all(have.get(k, 0) >= n for k, n in required.items())
        row = {"id": hid, "statement": text, "threshold": thr, "set": CONFIRM_SET, "seeds": have,
               "required": required, "tested": tested, "estimate": math.nan, "lo": math.nan, "hi": math.nan,
               "statistic": math.nan, "df": math.nan, "p": math.nan, "n_tracks": 0,
               "test": "one-sided one-sample t-test vs 1.0" if comp is None else f"one-sided Welch t-test, difference > {thr}"}
        if comp is None and rt:
            M, tracks, _ = matrix(res, rt, CONFIRM_SET)
            ci = stats.bootstrap_mean(M)
            t = stats.one_sample_t(M.mean(1), thr)
            row.update(estimate=ci.estimate, lo=ci.lo, hi=ci.hi, n_tracks=len(tracks), statistic=t.statistic, df=t.df, p=t.p)
        elif comp is not None and rt and rc:
            tracks = _common_tracks(res, rt, rc, CONFIRM_SET)
            A, _, _ = matrix(res, rt, CONFIRM_SET, tracks=tracks)
            B, _, _ = matrix(res, rc, CONFIRM_SET, tracks=tracks)
            ci = stats.bootstrap_diff(A, B)
            t = stats.welch_t(A.mean(1), B.mean(1), delta=thr)
            row.update(estimate=ci.estimate, lo=ci.lo, hi=ci.hi, n_tracks=len(tracks), statistic=t.statistic, df=t.df, p=t.p)
        out.append(row)
    adj = stats.holm([h["p"] for h in out])
    for h, pa in zip(out, adj):
        h["p_holm"] = float(pa)
        h["decision"] = _decide(h["tested"], h["hi"], h["threshold"], h["p_holm"])
        h["preliminary"] = (not h["tested"]) and h["estimate"] == h["estimate"]
    return out


def cross_checks(res, an: Analysis, s: str) -> list[dict]:
    """IQM, probability of improvement and the time ratio on tracks both agents finish, generalist vs others."""
    g = an.config("generalist")
    out = []
    if not g or not g["rids"]:
        return out
    for other in ("size-1", "blind", "pqn", "baseline-scripted", "baseline-random", "baseline-untrained"):
        o = an.config(other)
        if not o or not o["rids"]:
            continue
        tracks = _common_tracks(res, g["rids"], o["rids"], s)
        if not tracks:
            continue
        A, _, ra = matrix(res, g["rids"], s, tracks=tracks)
        B, _, rb = matrix(res, o["rids"], s, tracks=tracks)
        FA, _, _ = matrix(res, ra, s, "finished", tracks)
        FB, _, _ = matrix(res, rb, s, "finished", tracks)
        TA, _, _ = matrix(res, ra, s, "ticks", tracks)
        TB, _, _ = matrix(res, rb, s, "ticks", tracks)
        both = (FA.min(0) > 0) & (FB.min(0) > 0)  # tracks every seed of both agents finished
        ratio = float((TA[:, both].mean(0) / TB[:, both].mean(0)).mean()) if both.any() else math.nan
        out.append({"a": "generalist", "b": other, "label_b": o["label"], "set": s, "n_tracks": len(tracks),
                    "seeds_a": len(ra), "seeds_b": len(rb), "iqm_a": stats.iqm(A), "iqm_b": stats.iqm(B),
                    "prob_improvement": stats.probability_of_improvement(A, B),
                    "time_ratio_both_finished": ratio, "n_both_finished": int(both.sum())})
    return out


def seed_sensitivity(res, an: Analysis) -> tuple[list[dict], list[dict]]:
    """Spread over seeds and variance shares per configuration (test set), and single- vs multi-seed signs."""
    rows, checks = [], []
    g = an.config("generalist")
    for c in an.configs:
        rids = [r for r in c["rids"] if CONFIRM_SET in res.evals.get(r, {})]
        if len(rids) < 2:
            continue
        M, tracks, _ = matrix(res, rids, CONFIRM_SET)
        rows.append({"key": c["key"], "label": c["label"], **stats.seed_spread(M.mean(1)),
                     **{k: v for k, v in stats.variance_components(M).items() if k.startswith(("share", "var"))},
                     "n_tracks": len(tracks)})
        if g and c["key"] != "generalist" and not c["key"].startswith("baseline-"):
            grids = [r for r in g["rids"] if CONFIRM_SET in res.evals.get(r, {})]
            if len(grids) < 2:
                continue
            common = _common_tracks(res, rids, grids, CONFIRM_SET)
            A, _, ra = matrix(res, rids, CONFIRM_SET, tracks=common)
            B, _, rb = matrix(res, grids, CONFIRM_SET, tracks=common)
            multi = stats.bootstrap_diff(A, B)
            s1a = [i for i, r in enumerate(ra) if res.run(r)["seed"] == 1]
            s1b = [i for i, r in enumerate(rb) if res.run(r)["seed"] == 1]
            single = float(A[s1a[0]].mean() - B[s1b[0]].mean()) if s1a and s1b else math.nan
            checks.append({"key": c["key"], "label": c["label"], "single_seed_diff": single, "multi_seed_diff": multi.estimate,
                           "lo": multi.lo, "hi": multi.hi, "n_seeds": len(ra), "n_seeds_generalist": len(rb),
                           "sign_holds": (np.sign(single) == np.sign(multi.estimate)) if single == single else None})
    return rows, checks


def size_curve(res, an: Analysis, s: str) -> list[dict]:
    """Score and train-test gap against the number of training tracks (log x-axis in the figure).

    The gap uses the run's own training tracks: the optional ``train_own`` set if evaluated,
    otherwise ``train_sample`` only for configurations trained on the whole pool (for the
    size-N ablations the first 1,000 pool tracks are mostly not their training tracks).
    """
    out = []
    keys = [c["key"] for c in an.configs if tx.size_of(c["experiment"]) and not c["pilot"] and c["physics"] == "v1"]
    keys += ["generalist"]
    for k in keys:
        c = an.config(k)
        if not c:
            continue
        rids = [r for r in c["rids"] if s in res.evals.get(r, {})]
        if not rids:
            continue
        M, tracks, rids = matrix(res, rids, s)
        ci = stats.bootstrap_mean(M)
        ns = [res.run(r).get("n_train_tracks") for r in rids]
        n = int(np.median([x for x in ns if x])) if any(ns) else (tx.size_of(c["experiment"]) or 0)
        own = "train_own" if all("train_own" in res.evals.get(r, {}) for r in rids) else (
            "train_sample" if (k == "generalist" and all("train_sample" in res.evals.get(r, {}) for r in rids)) else None)
        gap = None
        if own:
            O, _, _ = matrix(res, rids, own)
            gap = stats.bootstrap_set_gap(O, M).as_dict()
            gap["own_set"] = own
        out.append({"key": k, "label": c["label"], "n_train_tracks": n, "set": s, "sws": ci.estimate, "lo": ci.lo,
                    "hi": ci.hi, "n_seeds": len(rids), "n_tracks": len(tracks), "gap": gap})
    return sorted(out, key=lambda d: d["n_train_tracks"])


def feature_rates(res, an: Analysis, s: str) -> dict:
    """Finish rate per feature bin (and per out-of-range variant) for up to four configurations."""
    keys = [k for k in FEATURE_METHODS if an.config(k) and any(s in res.evals.get(r, {}) for r in an.config(k)["rids"])][:4]
    out = {"set": s, "methods": [], "features": {}, "oor_variants": None}
    if not keys or not res.features:
        return out
    out["methods"] = [{"key": k, "label": an.config(k)["label"]} for k in keys]
    for feat in ("jumps", "max_bank_deg", "height_range_m"):
        per_method = {}
        for k in keys:
            F, tracks, rids = matrix(res, an.config(k)["rids"], s, "finished")
            vals = {t: res.features[t][feat] for t in tracks if t in res.features}
            if not vals:
                continue
            bins, order = feature_bins(vals, feat)
            per_method[k] = {"order": order, "bins": {}}
            for b in order:
                idx = [i for i, t in enumerate(tracks) if bins.get(t) == b]
                ci = stats.bootstrap_mean(F[:, idx])
                per_method[k]["bins"][b] = {"rate": ci.estimate, "lo": ci.lo, "hi": ci.hi, "n_tracks": len(idx), "n_seeds": len(rids)}
        if per_method:
            out["features"][feat] = per_method
    oor = {}
    for k in keys:
        F, tracks, rids = matrix(res, an.config(k)["rids"], "oor", "finished")
        if not rids:
            continue
        variants = sorted({res.features.get(t, {}).get("variant", "") for t in tracks} - {""})
        if variants:
            oor[k] = {}
            for v in variants:
                idx = [i for i, t in enumerate(tracks) if res.features.get(t, {}).get("variant") == v]
                ci = stats.bootstrap_mean(F[:, idx])
                oor[k][v] = {"rate": ci.estimate, "lo": ci.lo, "hi": ci.hi, "n_tracks": len(idx), "n_seeds": len(rids)}
    out["oor_variants"] = oor or None
    return out


def specialists(res, an: Analysis) -> list[dict]:
    """Per campaign track: the scripted reference time and each agent's per-seed finishing times."""
    tracks = sorted({row["track"] for rid in res.evals for row in res.evals[rid].get("campaign", [])})
    out = []
    for t in tracks:
        ref_time, entries = math.nan, []
        cand = []
        for c in an.configs:
            if c["experiment"] in ("specialist", "specialist-finetune"):
                if c.get("track") == t:
                    cand.append((c, "own track"))
            elif c["key"] in ("generalist", "generalist-pilot", "baseline-scripted", "baseline-cem"):
                cand.append((c, "zero-shot" if c["experiment"] == "generalist" else "reference"))
        for c, kind in cand:
            seeds = []
            for rid in c["rids"]:
                row = next((x for x in res.evals.get(rid, {}).get("campaign", []) if x["track"] == t), None)
                if row is None:
                    continue
                dt = row["time"] / row["ticks"] if row["ticks"] else math.nan
                if row["ref_ticks"] and dt == dt:
                    ref_time = row["ref_ticks"] * dt
                seeds.append({"seed": res.run(rid)["seed"], "finished": int(row["finished"]), "time": float(row["time"])})
            if seeds:
                fin = [x["time"] for x in seeds if x["finished"]]
                entries.append({"key": c["key"], "label": c["label"], "kind": kind, "seeds": seeds,
                                "n": len(seeds), "n_finished": len(fin), "mean_time": float(np.mean(fin)) if fin else math.nan})
        if entries:
            spec = next((e for e in entries if e["key"] == f"specialist-{t}"), None)
            gen = next((e for e in entries if e["key"] == "generalist"), None)
            gap = (gen["mean_time"] / spec["mean_time"] - 1.0) if spec and gen and spec["n_finished"] and gen["n_finished"] else math.nan
            out.append({"track": t, "ref_time": ref_time, "entries": entries, "generalist_slower_pct": 100 * gap})
    return out


def status_counts(res) -> dict:
    out: dict = {}
    for r in res.runs + res.queued:
        ph = r.get("phase") or "unassigned"
        out.setdefault(ph, {s: 0 for s in tx.STATUSES})
        out[ph][r["status"] if r["status"] in tx.STATUSES else "planned"] += 1
    order = sorted(out, key=lambda p: (p != "baselines", p))
    return {p: out[p] for p in order}


def analyze(res) -> Analysis:
    """Run the whole analysis (see the module docstring)."""
    an = Analysis()
    an.configs = build_configs(res)
    for c in an.configs:
        for s in sorted({s for r in c["rids"] for s in res.evals.get(r, {})}):
            summ = summarise_config(res, c, s)
            if summ:
                an.summaries.setdefault(c["key"], {})[s] = summ
    has = lambda s: any(s in d for d in an.summaries.values())
    an.primary_set = "test" if has("test") else ("val" if has("val") else None)
    an.hypotheses = hypotheses(res, an)
    if an.primary_set:
        an.cross_checks = cross_checks(res, an, an.primary_set)
        an.size_curve = size_curve(res, an, an.primary_set)
        an.feature_rates = feature_rates(res, an, an.primary_set)
    an.seed_sensitivity, an.single_vs_multi = seed_sensitivity(res, an)
    an.specialists = specialists(res, an)
    an.status_counts = status_counts(res)
    an.flags = [{"key": k, "label": an.config(k)["label"], "set": s, "overspeed": v["flags_overspeed"],
                 "energy_gain": v["flags_energy_gain"], "n_seeds": v["n_seeds"], "n_tracks": v["n_tracks"]}
                for k, d in an.summaries.items() for s, v in d.items() if v["flags_overspeed"] or v["flags_energy_gain"]]
    log.info("analysis: %d configurations, %d with scores, primary set %s", len(an.configs), len(an.summaries), an.primary_set)
    return an
