"""Write the Markdown outputs: summary, per-experiment reports, README results block and the paper.

Local purpose: assemble tables (``tables.py``), figures (``figures.py``) and generated sentences
(``narrative.py``) into files. Hand-written prose is limited to methodology descriptions here and
to optional notes files (``reports/notes/<group>.md``, ``reports/notes/<experiment>.md``,
``reports/notes/paper-<section>.md``) that are included verbatim; numbers always come from the
analysis.

Global context: the outputs are reports/summary.md, reports/experiments/<group>.md,
reports/readme_results.md (also spliced into README.md between the RESULTS markers, if present)
and paper/paper.md from the Jinja2 template paper/paper.md.j2.
"""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path

import jinja2

from ..eval import stats
from ..utils.logging import get_logger
from . import analysis as A
from . import narrative, tables as T
from . import taxonomy as tx

log = get_logger(__name__)
REPO_URL = "https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game"
MARK_START, MARK_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# Methodology of each experiment group: design only (what is trained, on what, compared with what).
GROUP_METHODS = {
    "first-models": "The first specialist (campaign track 1) and the first generalist (training pool) are trained with seed 0 "
                    "before the recipe is frozen, and judged only on campaign track 1 and the validation tracks by the "
                    "preregistered learning gate (preregistration, section 7). They are reported separately from all later runs.",
    "baselines": "Training-free drivers scored by the same evaluator: uniformly random actions and an untrained network "
                 "(the floor; several random seeds), and the frozen scripted driver (pure pursuit + curvature speed profile, "
                 "two parameters tuned on validation candidates), whose times are the reference of the metric.",
    "generalist": "The standard recipe (PPO, separate 2-layer actor and critic, two 3-way action heads, progress reward, "
                  "standing and random starts) trained on the whole generated training pool; evaluated with the checkpoint "
                  "at the end of the step budget, most probable action, one run per track from a standing start.",
    "blind-control": "The generalist recipe with the upcoming-track points removed from the observation (the nearest centreline "
                     "point stays). Generalist minus blind on the test tracks is H3.",
    "training-set-size": "The generalist recipe trained on only N tracks drawn from the training pool with the run's seed. "
                         "SWS(whole pool) minus SWS(1 track) on the test tracks is H2; the curve over N and the train-test gap "
                         "follow Kirk et al. 2023 and Cobbe et al. 2019/2020.",
    "specialists": "The standard recipe trained on one campaign track each (from scratch, and fine-tuned from the seed-1 "
                   "generalist), compared on that track with the generalist zero-shot and the scripted driver, and also "
                   "scored on the test tracks.",
    "pqn": "PQN (Gallici et al. 2025, adapted from CleanRL's pqn.py) with the same inputs, reward, rollout size, normaliser and "
           "step budget, and a single 9-way action head; the PPO ablation with a 9-way head isolates the action parameterisation.",
    "ablations": "Exploratory one-change variants of the standard recipe: continuous actions, 5 or 40 look-ahead points, a "
                 "per-step time cost, 10 Hz actions, a penalty when the run ends by falling, and a single 9-way head. Each "
                 "is compared with the generalist.",
    "imitation-hybrids": "Phase 4 only: behaviour cloning and DAgger on the scripted driver's actions, PPO warm-started from "
                         "behaviour cloning, PPO with a reward for matching the scripted driver, and PPO with mid-track starts "
                         "taken from the scripted driver's runs. Each is compared with the generalist that learned on its own.",
    "hyperparameters": "Descriptive sensitivity: the standard recipe with one setting changed (learning rate / 3, discount 0.995, "
                       "3x512 networks, weight decay 1e-5). None of these replaces the frozen recipe.",
    "patched-physics": "Only if an agent used an exploit: earlier models re-scored on the patched physics, and a generalist and "
                       "specialists retrained on it; the confirmatory analysis stays on physics v1 (preregistration, section 8).",
    "seed-sensitivity": "Earlier runs repeated with new seeds and nothing else changed. For every configuration: minimum, "
                        "maximum and standard deviation over seeds and the share of variance from seeds versus tracks; whether "
                        "each single-seed comparison keeps its sign; and the formal tests of H1-H3.",
}
GROUP_COMPARE = {  # configurations shown next to a group's own rows in its results table
    "generalist": ["baseline-scripted", "baseline-random", "baseline-untrained", "blind", "size-1"],
    "blind-control": ["generalist", "baseline-scripted"],
    "training-set-size": ["generalist", "baseline-scripted"],
    "pqn": ["generalist", "ablation-joint9"],
    "ablations": ["generalist"],
    "imitation-hybrids": ["generalist", "baseline-scripted"],
    "hyperparameters": ["generalist"],
    "patched-physics": ["generalist"],
    "specialists": ["generalist", "baseline-scripted"],
    "first-models": ["baseline-scripted", "baseline-random", "baseline-untrained"],
}


def _note(paths, name: str) -> str | None:
    p = paths.notes / f"{name}.md"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def _fig_md(fig: dict | None, rel: str, ext: str = "png") -> str:
    if not fig:
        return "_Figure not yet available: its runs have no results yet._"
    return f"![{fig['caption']}]({rel}{fig[ext]})\n\n*{fig['caption']}*"


def _paper_fig(fig: dict | None, width: str = "100%") -> str:
    """Figure for the paper: pandoc makes a lone image with alt text a numbered figure with that caption."""
    if not fig:
        return "_Figure not yet available: its runs have no results yet._"
    return f"![{fig['caption']}](../reports/figures/{fig['svg']}){{width={width}}}"


def _generated_line(paths) -> str:
    return (f"_Generated by `uv run python scripts/07_make_report.py` from the files in `results/` on {date.today().isoformat()}. "
            "Do not edit by hand; hand-written notes go in `reports/notes/`._")


# ----------------------------------------------------------------------------- learning gate
def learning_gate(an) -> str:
    """The preregistered learning gate for the two first models (section 7), from their scores."""
    rows = []
    sp = next((c for c in an.configs if c["key"].startswith("specialist-") and c["key"].endswith("-pilot")), None)
    if sp:
        v = an.summary(sp["key"], "campaign")
        own = None
        if v:
            own = next((e for t in an.specialists for e in t["entries"] if e["key"] == sp["key"] and t["track"] == sp["track"]), None)
        if own and own["seeds"]:
            x = own["seeds"][0]
            ref = next(t["ref_time"] for t in an.specialists if t["track"] == sp["track"])
            ok = bool(x["finished"]) and ref == ref and x["time"] <= ref
            rows.append([sp["label"], "finishes its track and is at least as fast as the scripted driver",
                         f"finished: {'yes' if x['finished'] else 'no'}; time {T.num(x['time'], 2)} s vs reference {T.num(ref, 2)} s",
                         "**passed**" if ok else "**not passed**"])
        else:
            rows.append([sp["label"], "finishes its track and is at least as fast as the scripted driver", T.NA, "not yet evaluated"])
    gp = an.config("generalist-pilot")
    if gp:
        v = an.summary("generalist-pilot", "val")
        if v:
            ok = v["finish"] >= 0.8 and v["sws"] >= 0.8
            rows.append([gp["label"], "finishes >= 80 % of validation tracks with SWS >= 0.8",
                         f"finished {T.pct(v['finish'])}, SWS {T.num(v['sws'])}", "**passed**" if ok else "**not passed**"])
        else:
            rows.append([gp["label"], "finishes >= 80 % of validation tracks with SWS >= 0.8", T.NA, "not yet evaluated"])
    return T.md_table(["Model", "Bar (preregistration, section 7)", "Result", "Gate"], rows, "_The first models are not registered yet._")


# ----------------------------------------------------------------------------- experiment reports
def experiment_report(res, an, figs, group: str) -> str:
    slug, title, question = next(g for g in tx.GROUPS if g[0] == group)
    paths = res.paths
    own = [c["key"] for c in an.configs if c["group"] == group]
    compare = [k for k in GROUP_COMPARE.get(group, []) if an.config(k)]
    s = an.primary_set
    parts = [f"# {title}", _generated_line(paths), f"**Question.** {question}", "## Methodology", GROUP_METHODS.get(group, ""),
             "Metric: success-weighted speed (SWS) and finish rate; intervals are 95 % two-level bootstrap intervals "
             f"({stats.N_BOOT:,} resamples of seeds and tracks). See `reports/preregistration.md` for the binding protocol."]
    if group != "seed-sensitivity":
        parts += ["## Runs", T.run_list(res, group), "## Configuration (logged in each run.json)", T.config_table(res, group)]
    parts.append("## Results")
    if group == "first-models":
        parts += ["### Learning gate", learning_gate(an)]
    if group == "baselines":
        parts += ["### Track sets", T.track_sets_table(res)]
        if res.scripted_tuning:
            ch = res.scripted_tuning.get("chosen", {})
            parts.append("Scripted driver parameters chosen on validation candidates: " +
                         ", ".join(f"`{k}` = {v}" for k, v in ch.items()) + ".")
    if group == "seed-sensitivity":
        parts += ["### Spread over seeds (test set)", T.seed_table(an), _fig_md(figs.get("seed_spread"), "../figures/"),
                  "### Do single-seed comparisons hold?", T.single_multi_table(an),
                  "### Hypotheses", T.hypothesis_table(an), "### Cross-checks (Agarwal et al. 2021)", T.cross_check_table(an)]
    else:
        parts += ["### Scores", T.eval_table(an, keys=own + compare)]
        if s:
            parts.append(f"Rows from other groups are shown for comparison. Track counts: {T.track_counts_note(an)}.")
    if group == "generalist":
        parts += ["### Comparison of all configurations", _fig_md(figs.get("sws_by_method"), "../figures/"),
                  "### Hypotheses", T.hypothesis_table(an), "### Cross-checks (Agarwal et al. 2021)", T.cross_check_table(an),
                  "### Finish rate by track feature", _fig_md(figs.get("finish_by_feature"), "../figures/"),
                  "### Exploit flags", T.flags_table(an)]
    if group == "training-set-size":
        parts += ["### Score and train-test gap against the number of training tracks", T.size_table(an),
                  _fig_md(figs.get("training_set_size"), "../figures/")]
    if group in ("specialists", "first-models"):
        parts += ["### Campaign tracks", T.specialist_table(an), _fig_md(figs.get("specialists_campaign"), "../figures/")]
    if group in ("blind-control", "training-set-size"):
        parts += ["### Hypotheses", T.hypothesis_table(an)]
    if group == "patched-physics":
        parts += ["### Exploits", T.exploits_table(res)]
    curves = [figs["curves"][k] for k in own if k in figs.get("curves", {})]
    if curves:
        parts.append("### Training curves")
        parts += [_fig_md(f, "../figures/") for f in curves]
    parts.append("## Learnings")
    notes = [n for n in [_note(paths, group)] + [_note(paths, e) for e in sorted({c["experiment"] for c in an.configs if c["group"] == group})] if n]
    parts += notes or [f"_No hand-written notes yet (add `reports/notes/{group}.md`)._"]
    return "\n\n".join(p for p in parts if p) + "\n"


def write_experiment_reports(res, an, figs) -> list[str]:
    res.paths.experiments.mkdir(parents=True, exist_ok=True)
    written = []
    for slug, _, _ in tx.GROUPS:
        (res.paths.experiments / f"{slug}.md").write_text(experiment_report(res, an, figs, slug), encoding="utf-8", newline="\n")
        written.append(slug)
    return written


# ----------------------------------------------------------------------------- summary
def summary(res, an, figs) -> str:
    total = len(res.runs) + len(res.queued)
    evaluated = sum(1 for r in res.runs if r["status"] == "evaluated")
    links = []
    for slug, title, q in tx.GROUPS:
        st = {}
        for c in an.configs:
            if c["group"] == slug:
                for k, v in c["statuses"].items():
                    st[k] = st.get(k, 0) + v
        links.append([f"[{title}](experiments/{slug}.md)", q, ", ".join(f"{v} {k}" for k, v in st.items()) or T.NA])
    n_eval_files = sum(len(v) for v in res.evals.values())
    parts = [
        "# Summary of all experiments", _generated_line(res.paths),
        "## Where things stand",
        f"{evaluated} of {total} runs (registered runs plus run-matrix entries) are evaluated. Status per phase:",
        T.status_table(an),
        "Statuses: " + "; ".join(f"**{k}** = {v}" for k, v in tx.STATUS_TEXT.items()) + ".",
        "## Headline results in plain words", "\n".join(f"- {x}" for x in narrative.headline(an)),
        "## Hypotheses (preregistered)", T.hypothesis_table(an),
        "Decision rule: refuted if the 95 % interval lies below the threshold; confirmed if the Holm-adjusted one-sided "
        "t-test p-value is below 0.05; otherwise inconclusive; not yet tested until all preregistered seeds have test results.",
        "## Main evaluation table", T.eval_table(an),
        f"Track counts: {T.track_counts_note(an) or T.NA}.",
        _fig_md(figs.get("sws_by_method"), "figures/"),
        "## All networks and runs", T.networks_table(an, res),
        "## Experiment reports", T.md_table(["Report", "Question", "Runs"], links),
        "## Track sets", T.track_sets_table(res), "### Reads of the protected test sets", T.access_log_table(res),
        "## Simulator throughput", T.bench_table(res),
        "## Physics exploits", T.exploits_table(res), "### Evaluator flags", T.flags_table(an),
        "## Key terms", T.glossary_table(res.vehicle),
        "## Data integrity",
        f"{n_eval_files} evaluation files read; {res.score_mismatches} per-track scores disagree with "
        "finished x reference ticks / ticks (they are reported as written by the evaluator).",
    ]
    note = _note(res.paths, "summary")
    if note:
        parts += ["## Notes", note]
    return "\n\n".join(parts) + "\n"


# ----------------------------------------------------------------------------- README block
def readme_block(res, an, figs) -> str:
    s = an.primary_set
    sets = [x for x in ["test", "oor", "campaign", "val", "train_sample"] if any(x in d for d in an.summaries.values())]
    total = len(res.runs) + len(res.queued)
    evaluated = sum(1 for r in res.runs if r["status"] == "evaluated")
    parts = [f"_Results generated from `results/` by `scripts/07_make_report.py` on {date.today().isoformat()}; "
             f"{evaluated} of {total} planned runs evaluated."
             + (f" Training was paused on {res.paused['date']}; the remaining runs are postponed." if res.paused else "")
             + " Full tables: [`reports/summary.md`](reports/summary.md)._",
             "\n".join(f"- {x}" for x in narrative.headline(an))]
    if an.summaries:
        parts += ["**Main evaluation table** (SWS = success-weighted speed, 1.0 = as fast as the scripted driver on every "
                  "track; 95 % two-level bootstrap intervals over seeds and tracks):", T.eval_table(an, sets=sets)]
    parts += ["**Preregistered hypotheses:**", T.hypothesis_table(an)]
    if figs.get("sws_by_method"):
        parts.append(f"![{figs['sws_by_method']['caption']}](reports/figures/{figs['sws_by_method']['png']})")
    return "\n\n".join(parts) + "\n"


def splice_readme(readme: Path, block: str) -> bool:
    """Replace the text between the RESULTS markers of README.md; returns False if the markers are absent."""
    if not readme.exists():
        return False
    raw = readme.read_bytes().decode("utf-8")  # raw, so CRLF in the working copy is seen and normalised
    text = raw.replace("\r\n", "\n")
    if MARK_START not in text or MARK_END not in text:
        return False
    new = re.sub(re.escape(MARK_START) + r".*?" + re.escape(MARK_END), lambda m: f"{MARK_START}\n{block}\n{MARK_END}", text,
                 count=1, flags=re.S)
    if new != raw:  # only the text between the markers changes; LF endings as in .gitattributes
        readme.write_text(new, encoding="utf-8", newline="\n")
    return True


# ----------------------------------------------------------------------------- paper
def _defaults(res) -> dict:
    """Hyperparameters, environment settings and generator ranges for the appendices.

    Taken from a logged run (generalist first) when one exists, otherwise from the code's own
    defaults (imports guarded: they need torch, which a report-only environment may lack).
    """
    out = {"ppo": None, "env": None, "ppo_source": None, "generator": None, "oor": None, "splits": None}
    run = next((r for k in ("generalist", "generalist-pilot") for r in res.runs if r["key"] == k and r.get("ppo")), None)
    if run:
        out["ppo"], out["env"], out["ppo_source"] = run["ppo"], run.get("env"), f"logged in `results/{run['rid']}/run.json`"
    else:
        try:
            from dataclasses import asdict
            from ..env.vec_env import EnvConfig
            from ..rl.ppo import PPOConfig
            out["ppo"], out["env"], out["ppo_source"] = asdict(PPOConfig()), asdict(EnvConfig()), "code defaults (`PPOConfig`, `EnvConfig`); no run logged yet"
        except Exception as e:  # noqa: BLE001
            log.info("PPO defaults unavailable for the paper: %s", e)
    try:
        from dataclasses import asdict
        from ..tracks.generator import OOR_VARIANTS, SPLITS, TRAIN_PARAMS
        out["generator"], out["oor"], out["splits"] = asdict(TRAIN_PARAMS), OOR_VARIANTS, SPLITS
    except Exception as e:  # noqa: BLE001
        log.info("generator ranges unavailable for the paper: %s", e)
    return out


def _kv_table(d: dict | None, name: str) -> str:
    if not d:
        return f"_{name} not available._"
    return T.md_table(["Setting", "Value"], [[f"`{k}`", f"`{v}`"] for k, v in d.items()])


def paper_context(res, an, figs) -> dict:
    s = an.primary_set
    d = _defaults(res)
    rel = _paper_fig
    paper_sets = [x for x in [s, "oor", "campaign"] if x and any(x in v for v in an.summaries.values())]
    gen = d["generator"]
    gen_rows = [[f"`{k}`", f"{v}", ", ".join(f"{n}: {o[k]}" for n, o in (d["oor"] or {}).items() if k in o) or "-"]
                for k, v in gen.items()] if gen else []
    notes = {p.stem.removeprefix("paper-"): p.read_text(encoding="utf-8").strip() for p in sorted(res.paths.notes.glob("paper-*.md"))} \
        if res.paths.notes.exists() else {}
    return {
        "date": date.today().isoformat(), "repo_url": REPO_URL,
        "n_runs": len(res.runs) + len(res.queued), "n_evaluated": sum(1 for r in res.runs if r["status"] == "evaluated"),
        "paused": res.paused, "has_results": bool(an.summaries), "primary_set": s, "primary_is_test": s == "test",
        "headline": narrative.headline(an, detail=False),
        "hypotheses_tested": all(h["tested"] for h in an.hypotheses), "any_tested": any(h["tested"] for h in an.hypotheses),
        "preliminary_hypotheses": narrative.preliminary_hypotheses(an),
        "status_table": T.status_table(an), "eval_table": T.eval_table(an, sets=paper_sets) if an.summaries else T.NOT_YET,
        "track_counts": T.track_counts_note(an), "hypothesis_table": T.hypothesis_table(an, compact=True),
        "cross_check_table": T.cross_check_table(an), "seed_table": T.seed_table(an), "single_multi_table": T.single_multi_table(an),
        "size_table": T.size_table(an), "specialist_table": T.specialist_table(an), "exploits_table": T.exploits_table(res),
        "flags_table": T.flags_table(an), "track_sets_table": T.track_sets_table(res), "learning_gate": learning_gate(an),
        "fig_methods": rel(figs.get("sws_by_method")), "fig_size": rel(figs.get("training_set_size"), "70%"),
        "fig_features": rel(figs.get("finish_by_feature")), "fig_specialists": rel(figs.get("specialists_campaign")),
        "fig_seeds": rel(figs.get("seed_spread"), "80%"),
        "fig_curves_generalist": rel(figs.get("curves", {}).get("generalist") or figs.get("curves", {}).get("generalist-pilot")),
        "ppo_table": _kv_table(d["ppo"], "PPO settings"), "ppo_source": d["ppo_source"] or "not available",
        "env_table": _kv_table(d["env"], "Environment settings"),
        "generator_table": T.md_table(["Range", "Training distribution", "Out-of-range variants"], gen_rows, "_Generator ranges not available._"),
        "splits": d["splits"], "vehicle": res.vehicle, "dt": res.vehicle.get("dt"), "ticks_per_action": res.vehicle.get("ticks_per_action"),
        "parity": res.parity, "bench_table": T.bench_table(res), "has_bench": bool(res.bench), "n_boot": stats.N_BOOT, "alpha": A.ALPHA, "smallest_effect": A.SMALLEST_EFFECT,
        "notes": notes, "scripted_tuning": (res.scripted_tuning or {}).get("chosen"),
    }


def render_paper(res, an, figs) -> str | None:
    tpl_path = res.paths.template
    if not tpl_path.exists():
        log.warning("paper template %s missing; paper skipped", tpl_path)
        return None
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(tpl_path.parent)), undefined=jinja2.StrictUndefined,
                             keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
    text = env.get_template(tpl_path.name).render(**paper_context(res, an, figs))
    res.paths.paper.mkdir(parents=True, exist_ok=True)
    out = res.paths.paper / "paper.md"
    out.write_text(text, encoding="utf-8", newline="\n")
    return str(out)


def nan_to_none(x):
    """JSON-safe copy of nested data: NaN and infinities become None."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {str(k): nan_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [nan_to_none(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        return nan_to_none(x.item())
    return x
