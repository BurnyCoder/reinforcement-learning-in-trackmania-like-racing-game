"""Markdown tables built from the analysis: runs, statuses, scores, hypotheses, seeds, exploits.

Local purpose: formatting only. Every function takes ``collect.Results`` and/or
``analysis.Analysis`` and returns a Markdown string; missing values print as an em dash and empty
tables become a sentence saying the results are not yet available.

Global context: the same tables appear in reports/summary.md, the per-experiment reports, the
README results block and the paper (``render.py``), so a number always prints the same way.
"""

from __future__ import annotations

import math

from . import taxonomy as tx

NA = "—"
NOT_YET = "_Not yet available: the runs behind this table have no results yet._"


def num(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return NA
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:.{digits}f}"


def pct(x, digits: int = 1) -> str:
    return NA if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f} %"


def ci(est, lo, hi, digits: int = 3) -> str:
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return NA
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return num(est, digits)
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def ci_pct(est, lo, hi, digits: int = 1) -> str:
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return NA
    return f"{100 * est:.{digits}f} % [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"


def pval(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return NA
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def md_table(headers: list[str], rows: list[list], empty: str = NOT_YET) -> str:
    if not rows:
        return empty
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- run inventory
def networks_table(an, res, keys: list[str] | None = None) -> str:
    """All configurations: architecture, algorithm, objective, data, start states, status, scores."""
    rows = []
    for c in an.configs:
        if keys is not None and c["key"] not in keys:
            continue
        st = ", ".join(f"{n} {s}" for s, n in c["statuses"].items())
        sc = an.summaries.get(c["key"], {})
        score = "; ".join(f"{tx.SET_SHORT.get(s, s)} {num(v['sws'])} / {pct(v['finish'], 0)}" for s, v in
                          sorted(sc.items(), key=lambda kv: tx.SETS.index(kv[0]) if kv[0] in tx.SETS else 99)) or NA
        rows.append([c["label"], tx.CATEGORY_LABELS[c["category"]], (c.get("algo") or NA).upper() if c.get("algo") else NA,
                     c["architecture"], c["objective"], c["observation"], c["training_data"], c["start_states"],
                     ", ".join(map(str, c["seeds"])) or NA, st, score])
    return md_table(["Configuration", "Role", "Algorithm", "Architecture", "Objective", "Observation", "Training data",
                     "Start states", "Seeds with results", "Status (runs)", "SWS / finish rate per set"], rows)


def status_table(an) -> str:
    rows = [[ph, *[d.get(s, 0) for s in tx.STATUSES], sum(d.values())] for ph, d in an.status_counts.items()]
    if rows:
        tot = [sum(r[i] for r in rows) for i in range(1, len(tx.STATUSES) + 2)]
        rows.append(["**all**", *tot])
    return md_table(["Phase", *tx.STATUSES, "total"], rows, "_No runs are registered or planned._")


def run_list(res, group: str) -> str:
    """Every run of one group with its status, steps, hours and commit (from run.json)."""
    rows = []
    for r in res.runs + res.queued:
        if r["group"] != group:
            continue
        rows.append([f"`{r['rid']}`", r["status"], r.get("phase") or NA, num(r.get("global_step")) if r.get("global_step") else NA,
                     num(r.get("wall_hours"), 2) if r.get("wall_hours") else NA, r.get("physics") or NA,
                     (r.get("git_commit") or NA)[:10], ", ".join(r.get("eval_sets", [])) or NA])
    return md_table(["Run", "Status", "Phase", "Agent steps", "Wall hours", "Physics", "Commit", "Evaluated on"], rows,
                    "_No runs in this group yet._")


def config_table(res, group: str) -> str:
    """Logged configuration of the group's runs: the settings that differ from the standard recipe, and the key ones."""
    keys = ["total_steps", "num_envs", "num_steps", "lr", "gamma", "gae_lambda", "ent_coef", "clip_coef", "hidden",
            "action_mode", "weight_decay", "loaded_tracks"]
    env_keys = ["window_k", "blind", "time_cost", "fall_penalty", "random_start_prob", "max_steps", "hint_weight", "scripted_starts"]
    rows = []
    for r in res.runs:
        if r["group"] != group or not (r.get("ppo") or r.get("env")):
            continue
        p, e = r.get("ppo") or {}, r.get("env") or {}
        rows.append([f"`{r['rid']}`", *[str(p.get(k, NA)) for k in keys], *[str(e.get(k, NA)) for k in env_keys],
                     str(r.get("n_train_tracks") or NA)])
    return md_table(["Run", *keys, *env_keys, "training tracks"], rows,
                    "_No run of this group has logged its configuration yet (the experiment files are in `configs/experiments/`)._")


# ----------------------------------------------------------------------------- scores
def eval_table(an, keys: list[str] | None = None, sets: list[str] | None = None) -> str:
    """Configurations × sets: SWS with 95 % interval, finish rate, seeds."""
    sets = sets or [s for s in tx.SETS if any(s in d for d in an.summaries.values())]
    rows = []
    for c in an.configs:
        if c["key"] not in an.summaries or (keys is not None and c["key"] not in keys):
            continue
        d = an.summaries[c["key"]]
        cells = []
        for s in sets:
            v = d.get(s)
            cells += [ci(v["sws"], v["sws_lo"], v["sws_hi"]) if v else NA, pct(v["finish"]) if v else NA]
        n = max(v["n_seeds"] for v in d.values())
        rows.append([c["label"], tx.CATEGORY_LABELS[c["category"]], n, *cells])
    heads = ["Configuration", "Role", "Seeds"]
    for s in sets:
        heads += [f"{tx.SET_SHORT.get(s, s)} SWS [95 % CI]", f"{tx.SET_SHORT.get(s, s)} finished"]
    return md_table(heads, rows)


def track_counts_note(an) -> str:
    parts = []
    for s in tx.SETS:
        ns = {v["n_tracks"] for d in an.summaries.values() for k, v in d.items() if k == s}
        if ns:
            parts.append(f"{tx.SET_LABELS[s]}: {', '.join(f'{n:,}' for n in sorted(ns))} tracks")
    return "; ".join(parts)


def hypothesis_table(an, compact: bool = False) -> str:
    """H1-H3 with estimate, interval, test statistic, raw and Holm p-values and decision.

    ``compact`` drops the statement column (the paper states the hypotheses in its text).
    """
    rows = []
    for h in an.hypotheses:
        seeds = ", ".join(f"{k} {v}/{h['required'][k]}" for k, v in h["seeds"].items())
        est = ci(h["estimate"], h["lo"], h["hi"])
        if h["preliminary"]:
            est += " (preliminary)"
        stat = NA if h["statistic"] != h["statistic"] else f"t = {h['statistic']:.3f}, df = {h['df']:.1f}"
        quantity = f"{'SWS' if h['id'] == 'H1' else 'SWS difference'} > {h['threshold']}"
        row = [h["id"]] + ([] if compact else [h["statement"]]) + [quantity, seeds, est, stat, pval(h["p"]), pval(h["p_holm"]),
                                                                  f"**{h['decision']}**", h.get("reading", "")]
        rows.append(row)
    heads = [""] + ([] if compact else ["Hypothesis"]) + ["Tested quantity", "Seeds (have/required)", "Estimate [95 % CI]",
                                                         "Test statistic", "p (one-sided)", "p (Holm)", "Decision",
                                                         "Preliminary reading (not a test)"]
    return md_table(heads, rows)


def cross_check_table(an) -> str:
    rows = [[c["label_b"], c["set"], f"{c['seeds_a']} / {c['seeds_b']}", num(c["iqm_a"]), num(c["iqm_b"]),
             num(c["prob_improvement"]), num(c["time_ratio_both_finished"]), num(c["n_both_finished"])]
            for c in an.cross_checks]
    return md_table(["Generalist vs", "Set", "Seeds (generalist / other)", "IQM generalist", "IQM other",
                     "P(generalist > other)", "Time ratio where both finish", "Tracks both finish"], rows)


def seed_table(an) -> str:
    rows = [[s["label"], s["n"], num(s["mean"]), num(s["min"]), num(s["max"]), num(s["std"]),
             pct(s.get("share_seed")), pct(s.get("share_track")), pct(s.get("share_interaction")), num(s["n_tracks"])]
            for s in an.seed_sensitivity]
    return md_table(["Configuration", "Seeds", "Mean SWS", "Min", "Max", "SD", "Variance: seeds", "Variance: tracks",
                     "Variance: seed x track", "Tracks"], rows,
                    "_Not yet available: no configuration has test results from two or more seeds._")


def single_multi_table(an) -> str:
    rows = [[c["label"], num(c["single_seed_diff"]), ci(c["multi_seed_diff"], c["lo"], c["hi"]),
             f"{c['n_seeds']} / {c['n_seeds_generalist']}", NA if c["sign_holds"] is None else ("yes" if c["sign_holds"] else "no")]
            for c in an.single_vs_multi]
    return md_table(["Configuration", "Seed-1 difference vs generalist", "All-seed difference [95 % CI]",
                     "Seeds (config / generalist)", "Sign holds"], rows,
                    "_Not yet available: needs test results from two or more seeds of a configuration and of the generalist._")


def size_table(an) -> str:
    rows = []
    for d in an.size_curve:
        g = d.get("gap")
        rows.append([f"{d['n_train_tracks']:,}", d["label"], d["n_seeds"], ci(d["sws"], d["lo"], d["hi"]),
                     ci(g["estimate"], g["lo"], g["hi"]) if g else NA, (g or {}).get("own_set", NA)])
    return md_table(["Training tracks", "Configuration", "Seeds", f"SWS [95 % CI]", "Train - test gap [95 % CI]",
                     "Training score from"], rows)


def specialist_table(an) -> str:
    rows = []
    for t in an.specialists:
        for e in t["entries"]:
            times = ", ".join(f"s{x['seed']}: {x['time']:.2f}" if x["finished"] else f"s{x['seed']}: DNF" for x in e["seeds"])
            rows.append([t["track"], num(t["ref_time"], 2), e["label"], e["kind"], f"{e['n_finished']}/{e['n']}",
                         num(e["mean_time"], 2), times])
    return md_table(["Campaign track", "Scripted time (s)", "Agent", "Kind", "Finished", "Mean time (s)", "Per seed (s)"], rows)


def exploits_table(res) -> str:
    if res.exploits is None:
        return "_No exploit list yet (`results/exploits.json` does not exist)._"
    cols = ["title", "description", "intended goal", "behaviour", "runs", "first_step", "seeds", "time_gained", "replay"]
    rows = [[(", ".join(map(str, e.get(k))) if isinstance(e.get(k), list) else e.get(k, NA)) for k in cols] for e in res.exploits]
    return md_table(["Title", "Description", "Intended goal", "Behaviour", "Runs", "First step", "Seeds", "Time gained", "Replay"],
                    rows, "_The exploit list is empty: no exploit has been confirmed._")


def flags_table(an) -> str:
    rows = [[f["label"], tx.SET_SHORT.get(f["set"], f["set"]), f["overspeed"], f["energy_gain"], f["n_seeds"], f["n_tracks"]]
            for f in an.flags]
    return md_table(["Configuration", "Set", "Overspeed flags", "Coasting energy-gain flags", "Seeds", "Tracks"], rows,
                    "_No evaluation has raised an exploit flag._")


def track_sets_table(res) -> str:
    rows = []
    for split, d in res.track_sets.items():
        lo_hi = d.get("seed_range") or [None, None]
        rows.append([split, f"{lo_hi[0]:,} - {lo_hi[1] - 1:,}" if lo_hi[0] is not None else NA, num(d.get("generated")),
                     num(d.get("geometric_rejects")), num(d.get("certification_rejects")), num(d.get("kept")),
                     num(d.get("mean_length_m"), 0), num(d.get("mean_ref_time_s"), 1)])
    return md_table(["Set", "Seeds", "Generated", "Geometric rejects", "Certification rejects", "Kept", "Mean length (m)",
                     "Mean reference time (s)"], rows, "_Track sets have not been generated yet._")


def bench_table(res) -> str:
    """Simulator throughput per platform and number of parallel cars (results/bench/<platform>.json)."""
    rows = []
    for b in res.bench:
        for r in b.get("runs") or []:
            rows.append([b.get("platform", b["file"]), b.get("torch", NA), b.get("gpu", NA), str(b.get("compile", NA)),
                         num(r.get("envs")), num(r.get("agent_steps_per_s"), 0), num(r.get("ms_per_step"), 2), num(r.get("warmup_s"), 1)])
    return md_table(["Platform", "PyTorch", "GPU", "torch.compile", "Parallel cars", "Agent steps / s", "ms per step", "Warm-up (s)"],
                    rows, "_No simulator benchmark recorded yet (`results/bench/`)._")


def access_log_table(res) -> str:
    counts: dict = {}
    for e in res.access_log:
        k = (e.get("split"), e.get("policy"))
        counts[k] = counts.get(k, 0) + 1
    rows = [[s, p, n] for (s, p), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))]
    return md_table(["Protected set", "Policy", "Reads"], rows, "_The protected test sets have not been read yet._")


def glossary_table(vehicle: dict | None = None) -> str:
    v = vehicle or {}
    tpa, dt = v.get("ticks_per_action"), v.get("dt")
    step = (f"One decision of the agent: {tpa} physics ticks of {dt} s." if tpa and dt else "One decision of the agent (several physics ticks).")
    held = (f"One keyboard input held for {tpa * dt:g} s." if tpa and dt else "One keyboard input held for a fraction of a second.")
    rows = [
        ["Success-weighted speed (SWS)", "Mean over tracks of finished x (reference time / agent time); 0 for an unfinished track. Adapted from SPL (Anderson et al. 2018).",
         "1.0 = finishes every track as fast as the scripted driver; above 1 = faster; 0 = finishes nothing."],
        ["Finish rate", "Share of tracks finished within 3x the reference time without falling off or stalling.", "How often the car gets to the end."],
        ["Scripted driver", "Pure-pursuit steering with a curvature speed profile, 2 parameters tuned on validation tracks, then frozen.", "A hand-written autopilot used as the yardstick."],
        ["Reference time", "The scripted driver's finishing time on a track (it certifies that every kept track can be finished).", "The time to beat."],
        ["Generalist", "One PPO policy trained on the whole generated training pool.", "One driver that has practised on many tracks."],
        ["Specialist", "A policy trained on one campaign track only.", "A driver that has practised one track."],
        ["Blind control", "The generalist recipe without the upcoming-track points in the input.", "The same driver, but it cannot see the road ahead."],
        ["Training-set size ablation", "The generalist recipe trained on only N tracks (1 to 10,000).", "Does practising on more different tracks help on new ones?"],
        ["PPO / PQN", "Proximal Policy Optimization (Schulman et al. 2017) / Parallelised Q-Network (Gallici et al. 2025).", "Two ways to learn from trial and error."],
        ["Seed", "One independent training run with its own random initialisation and data order.", "Repeating the experiment to see how much luck matters."],
        ["Test / validation / out-of-range / campaign", "Disjoint track sets: test (never used for decisions), validation (development decisions), out-of-range (one generator factor beyond training), campaign (5 hand-made).", "Exams with tracks the driver never practised on."],
        ["Train-test gap", "Score on the run's own training tracks minus score on test tracks (Kirk et al. 2023).", "How much better it drives tracks it practised on."],
        ["95 % CI (two-level bootstrap)", "Percentile interval from 10,000 resamples of seeds and tracks (Owen 2007).", "The range the true score plausibly lies in."],
        ["Holm correction", "Step-down adjustment of p-values for testing three hypotheses (Holm 1979).", "Guards against lucky findings when testing several things."],
        ["Welch's t-test", "t-test without assuming equal variances, on per-seed scores (Colas et al. 2019).", "Is the difference between two groups of runs bigger than chance?"],
        ["IQM, probability of improvement", "Interquartile mean; share of run pairs where one agent beats the other, averaged over tracks (Agarwal et al. 2021).", "Robust averages and head-to-head win rates."],
        ["Exploit flag", "Evaluator detector: speed above 1.2x top speed, or energy gained while coasting.", "A hint that the agent found a bug in the physics."],
        ["Physics version", "Tag of the simulator; v1 is the unmodified cannon-es port; patched versions fix exploits.", "Which rules of the game world were used."],
        ["Agent step", step, held],
    ]
    return md_table(["Term", "Technical meaning", "In plain words"], rows)
