"""Read every result file into tidy in-memory tables for the report.

Local purpose: the only module of the report pipeline that reads ``results/``. It finds every
``run.json`` (trained runs under ``results/<experiment>[-<track>]/seed<k>/``, baselines under
``results/baselines/<name>/seed<k>/``), the per-track evaluation CSVs, the training and validation
curves, the track-set summaries, the test-access log and the optional exploit list, and attaches
the report vocabulary from ``taxonomy.py`` to each run. Missing files are normal (runs that have not
happened yet) and simply produce empty tables.

Global context: ``build.build_all`` calls ``collect()`` once; analysis, tables, figures, reports,
the README block, the paper, the website data and the Hugging Face cards are all computed from the
returned ``Results`` object, so every number traces back to a file listed in docs/formats.md
("Results"). The results directory can be redirected with ``RACINGMANIA_RESULTS_DIR`` (tests use a
temporary copy) and the output root with ``RACINGMANIA_REPORT_ROOT``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..eval.stats import track_scores
from ..utils.logging import get_logger
from ..utils.paths import REPO_ROOT, RESULTS_DIR
from . import taxonomy as tx
from .features import load_features

log = get_logger(__name__)


@dataclass
class ReportPaths:
    """Where results are read from (``results``) and where report files are written (``out``).

    Sources that are part of the code base (run matrix, experiment configs, paper template,
    literature review, vehicle and campaign files) are always read from the repository itself.
    """

    results: Path
    out: Path
    repo: Path = REPO_ROOT

    @classmethod
    def resolve(cls, results_dir=None, out_root=None) -> "ReportPaths":
        results = Path(results_dir or os.environ.get("RACINGMANIA_RESULTS_DIR") or RESULTS_DIR)
        out = Path(out_root or os.environ.get("RACINGMANIA_REPORT_ROOT") or REPO_ROOT)
        return cls(results=results, out=out)

    @property
    def reports(self) -> Path:
        return self.out / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def experiments(self) -> Path:
        return self.reports / "experiments"

    @property
    def notes(self) -> Path:
        return self.reports / "notes"

    @property
    def paper(self) -> Path:
        return self.out / "paper"

    @property
    def web_data(self) -> Path:
        return self.out / "web" / "assets" / "data"

    @property
    def readme(self) -> Path:
        return self.out / "README.md"

    @property
    def template(self) -> Path:
        return self.repo / "paper" / "paper.md.j2"


@dataclass
class Results:
    """Everything read from the result files, plus the static project facts the report needs."""

    paths: ReportPaths
    runs: list[dict] = field(default_factory=list)  # one record per run.json
    queued: list[dict] = field(default_factory=list)  # run-matrix entries without a run.json yet
    evals: dict = field(default_factory=dict)  # rid -> set -> list of per-track rows
    train_curves: dict = field(default_factory=dict)  # rid -> column -> array (train_metrics.csv)
    val_curves: dict = field(default_factory=dict)  # rid -> column -> array (val_metrics.csv)
    track_sets: dict = field(default_factory=dict)  # split -> results/tracks/<split>_summary.json
    scripted_tuning: dict | None = None
    access_log: list = field(default_factory=list)
    exploits: list | None = None
    parity: dict | None = None
    bench: list = field(default_factory=list)  # results/bench/<platform>.json: simulator throughput
    vehicle: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)  # track name -> geometry features (features.py)
    experiments: dict = field(default_factory=dict)  # experiment name -> configs/experiments/<name>.yaml
    matrix: list = field(default_factory=list)  # flattened configs/run_matrix.yaml
    paused: dict | None = None  # configs/run_matrix.yaml "paused": {date, note} while training is paused
    score_mismatches: int = 0  # rows whose score column disagrees with finished * ref / ticks

    def run(self, rid: str) -> dict | None:
        return next((r for r in self.runs if r["rid"] == rid), None)


# ----------------------------------------------------------------------------- small readers
def _num(v: str):
    """CSV cell -> int, float or the original string (empty cell -> NaN)."""
    if v is None or v == "":
        return math.nan
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def read_rows(path: Path) -> list[dict]:
    """A CSV file as a list of dicts with typed values."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def read_columns(path: Path) -> dict[str, np.ndarray]:
    """A numeric CSV file as column arrays (non-numbers become NaN)."""
    rows = read_rows(path)
    if not rows:
        return {}
    cols = {}
    for k in rows[0]:
        vals = [r.get(k) for r in rows]
        cols[k] = np.array([v if isinstance(v, (int, float)) else math.nan for v in vals], float)
    return cols


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("could not read %s: %s", path, e)
        return None


def load_pause(repo: Path) -> dict | None:
    """The run matrix's ``paused`` block ({date, note}) while training is paused, else None."""
    p = repo / "configs" / "run_matrix.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("paused") if p.exists() else None


def load_matrix(repo: Path) -> list[dict]:
    """``configs/run_matrix.yaml`` flattened to one dict per run, in the preregistered order."""
    p = repo / "configs" / "run_matrix.yaml"
    if not p.exists():
        return []
    m = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for phase in m.get("phases", []):
        for r in phase.get("runs") or []:
            out.append({**r, "phase": phase["name"], "order": len(out)})
    return out


def load_experiments(repo: Path) -> dict[str, dict]:
    d = repo / "configs" / "experiments"
    return {p.stem: (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) for p in sorted(d.glob("*.yaml"))} if d.exists() else {}


# ----------------------------------------------------------------------------- runs
def _decorate(rec: dict, exp_cfg: dict, phase_order: dict) -> dict:
    """Attach group, category, configuration key, label and descriptions to a run record."""
    e, track, physics = rec["experiment"], rec.get("track"), rec.get("physics") or "v1"
    baseline = e.startswith("baseline-")
    rec["is_baseline"] = baseline
    rec["pilot"] = (not baseline) and (rec.get("phase") == "1-first-models" or (
        not rec.get("phase") and rec.get("seed") == 0 and e in ("generalist", "specialist")))
    rec["physics"] = physics
    rec["exp_cfg"] = exp_cfg
    rec["group"] = tx.group_of(e, rec["pilot"], physics)
    rec["category"] = tx.category_of(e)
    rec["key"] = tx.config_key(e, track, rec["pilot"], physics)
    rec["label"] = tx.label_of(e, track, rec["pilot"], physics, rec.get("description") or exp_cfg.get("description", ""))
    if not rec.get("n_train_tracks"):
        n = (exp_cfg.get("tracks") or {}).get("n_tracks") or tx.size_of(e)
        rec["n_train_tracks"] = n
    rec["architecture"] = tx.architecture(rec)
    rec["objective"] = tx.objective(rec)
    rec["observation"] = tx.observation(rec)
    rec["training_data"] = tx.training_data(rec)
    rec["start_states"] = tx.start_states(rec)
    rec["phase_order"] = phase_order.get(rec.get("phase"), 99)
    return rec


def read_runs(paths: ReportPaths, matrix: list[dict], experiments: dict, paused: bool = False) -> tuple[list[dict], list[dict]]:
    """All registered runs, and the run-matrix entries that have no run.json yet ("postponed" while paused)."""
    by_key = {(m["experiment"], int(m["seed"]), m.get("track")): m for m in matrix}
    phase_order = {}
    for m in matrix:
        phase_order.setdefault(m["phase"], len(phase_order))
    phase_order["baselines"] = -1
    runs, seen = [], set()
    if paths.results.exists():
        for p in sorted(paths.results.rglob("run.json")):
            d = _json(p)
            if not isinstance(d, dict):
                continue
            rid = p.parent.relative_to(paths.results).as_posix()
            experiment = d.get("experiment") or rid.split("/")[0]
            seed = d.get("seed")
            if seed is None:
                seed = int(p.parent.name.replace("seed", "")) if p.parent.name.startswith("seed") else 0
            track = d.get("track") or None
            m = by_key.get((experiment, int(seed), track))
            rec = {**d, "rid": rid, "experiment": experiment, "seed": int(seed), "track": track,
                   "status": d.get("status", "planned"),
                   "phase": d.get("phase") or (m["phase"] if m else ("baselines" if experiment.startswith("baseline-") else None)),
                   "matrix_order": m["order"] if m else -1}
            name = experiment.removeprefix("baseline-") if experiment.startswith("baseline-") else experiment
            runs.append(_decorate(rec, experiments.get(name, {}), phase_order))
            seen.add((experiment, int(seed), track))
    queued = []
    for m in matrix:
        if (m["experiment"], int(m["seed"]), m.get("track")) in seen:
            continue
        rid = f"{m['experiment']}-{m['track']}/seed{m['seed']}" if m.get("track") else f"{m['experiment']}/seed{m['seed']}"
        rec = {"rid": rid, "experiment": m["experiment"], "seed": int(m["seed"]), "track": m.get("track"),
               "status": "postponed" if paused else "not started", "phase": m["phase"], "matrix_order": m["order"],
               "algo": experiments.get(m["experiment"], {}).get("algo"),
               "description": experiments.get(m["experiment"], {}).get("description", "")}
        queued.append(_decorate(rec, experiments.get(m["experiment"], {}), phase_order))
    order = lambda r: (r["phase_order"], r["matrix_order"] if r["matrix_order"] >= 0 else 10**6, r["experiment"],
                       r.get("track") or "", r["seed"])
    return sorted(runs, key=order), sorted(queued, key=order)


# ----------------------------------------------------------------------------- everything
def collect(paths: ReportPaths) -> Results:
    """Read all result files (see the module docstring)."""
    res = Results(paths=paths)
    res.matrix = load_matrix(paths.repo)
    res.experiments = load_experiments(paths.repo)
    res.paused = load_pause(paths.repo)
    res.runs, res.queued = read_runs(paths, res.matrix, res.experiments, paused=bool(res.paused))
    for r in res.runs:
        d = paths.results / r["rid"]
        sets = {}
        for f in sorted(d.glob("eval_*.csv")):
            rows = read_rows(f)
            if not rows:
                continue
            recomputed = track_scores([x["finished"] for x in rows], [x["ref_ticks"] for x in rows], [x["ticks"] for x in rows])
            bad = int(np.sum(np.abs(recomputed - np.array([float(x["score"]) for x in rows])) > 1e-6))
            if bad:
                log.warning("%s: %d score values disagree with finished * ref_ticks / ticks", f, bad)
                res.score_mismatches += bad
            sets[f.stem.removeprefix("eval_")] = rows
        if sets:
            res.evals[r["rid"]] = sets
        r["eval_sets"] = [s for s in tx.SETS if s in sets] + sorted(s for s in sets if s not in tx.SETS)
        if (d / "train_metrics.csv").exists():
            res.train_curves[r["rid"]] = read_columns(d / "train_metrics.csv")
        if (d / "val_metrics.csv").exists():
            res.val_curves[r["rid"]] = read_columns(d / "val_metrics.csv")
    tdir = paths.results / "tracks"
    for p in sorted(tdir.glob("*_summary.json")) if tdir.exists() else []:
        d = _json(p)
        if isinstance(d, dict):
            res.track_sets[p.stem.removesuffix("_summary")] = d
    if (tdir / "scripted_tuning.json").exists():
        res.scripted_tuning = _json(tdir / "scripted_tuning.json")
    log_path = paths.results / "test_access_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                res.access_log.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("bad line in %s: %r", log_path, line[:80])
    if (paths.results / "exploits.json").exists():
        ex = _json(paths.results / "exploits.json")
        res.exploits = ex if isinstance(ex, list) else None
    bdir = paths.results / "bench"
    for p in sorted(bdir.glob("*.json")) if bdir.exists() else []:
        d = _json(p)
        if isinstance(d, dict):
            res.bench.append({**d, "file": p.name})
    if (paths.results / "parity.json").exists():
        res.parity = _json(paths.results / "parity.json")
    vpath = paths.repo / "web" / "assets" / "vehicle.json"
    res.vehicle = (_json(vpath) or {}) if vpath.exists() else {}
    res.features = load_features(paths.results, paths.repo)
    log.info("collected %d runs (%d with evaluations), %d queued matrix entries, %d track-set summaries",
             len(res.runs), len(res.evals), len(res.queued), len(res.track_sets))
    return res
