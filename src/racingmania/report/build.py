"""One call regenerates every report artifact from the result files: ``build_all()``.

Local purpose: the orchestrator of the report pipeline. Phases, in order:
1. ``collect``   read results/ into tidy tables (collect.py)
2. ``analyze``   per-configuration scores with intervals, H1-H3, seed sensitivity, curves (analysis.py)
3. ``figures``   PNG + SVG into reports/figures/ (figures.py)
4. ``reports``   reports/summary.md and reports/experiments/<group>.md (render.py)
5. ``readme``    reports/readme_results.md, spliced into README.md between the RESULTS markers
6. ``paper``     paper/paper.md from paper/paper.md.j2, then paper/paper.pdf (render.py, pdf.py)
7. ``website``   web/assets/data/results.json (webdata.py)
8. ``cards``     results/HF_DATASET_CARD.md (cards.py); ``model_card(rid)`` for each model repository
The build is idempotent and works with zero, partial or complete results.

Global context: ``scripts/07_make_report.py`` calls ``build_all()``; ``publish.publish_all`` calls
it before every push and uploads ``model_card(rid)`` with each model (reports/plan.md, "Publishing
rule"). Inputs can be redirected with ``RACINGMANIA_RESULTS_DIR`` and outputs with
``RACINGMANIA_REPORT_ROOT`` (or the arguments of ``build_all``).
"""

from __future__ import annotations

import re
import time

from ..utils.logging import get_logger
from . import cards, figures, render, webdata
from .analysis import analyze
from .collect import ReportPaths, collect
from .pdf import build_pdf

log = get_logger(__name__)
FIGURE_FILES = re.compile(r"^(curves_.+|sws_by_method|training_set_size|finish_by_feature|specialists_campaign|seed_spread)\.(png|svg)$")


def _clean_stale_figures(paths: ReportPaths, figs: dict):
    """Remove figure files this pipeline wrote earlier but no longer produces (their data went away)."""
    keep = {f[e] for f in list(figs.get("curves", {}).values()) + [v for k, v in figs.items() if k != "curves"] for e in ("png", "svg")}
    if paths.figures.exists():
        for p in paths.figures.iterdir():
            if FIGURE_FILES.match(p.name) and p.name not in keep:
                p.unlink()


def build_all(results_dir=None, out_root=None, pdf: bool = True) -> dict:
    """Regenerate every table, figure, report, the README block, the paper, the site data and the dataset card."""
    t0 = time.time()
    paths = ReportPaths.resolve(results_dir, out_root)
    log.info("report build: results from %s, outputs under %s", paths.results, paths.out)
    res = collect(paths)
    an = analyze(res)
    figs = figures.make_all(res, an, paths.figures)
    _clean_stale_figures(paths, figs)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "summary.md").write_text(render.summary(res, an, figs), encoding="utf-8", newline="\n")
    groups = render.write_experiment_reports(res, an, figs)
    block = render.readme_block(res, an, figs)
    (paths.reports / "readme_results.md").write_text(block, encoding="utf-8", newline="\n")
    spliced = render.splice_readme(paths.readme, block)
    if not spliced:
        log.info("README.md has no %s ... %s markers; results block written to reports/readme_results.md only",
                 render.MARK_START, render.MARK_END)
    paper = render.render_paper(res, an, figs)
    pdf_path = build_pdf(paths.paper / "paper.md", paths.out) if (paper and pdf) else None
    site = webdata.write(res, an, figs)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "HF_DATASET_CARD.md").write_text(cards.dataset_card(res, an), encoding="utf-8", newline="\n")
    out = {"summary": str(paths.reports / "summary.md"), "experiments": groups, "readme_spliced": spliced, "paper": paper,
           "pdf": str(pdf_path) if pdf_path else None, "results_json": site, "figures": sorted(
               [f["png"] for f in figs.get("curves", {}).values()] + [v["png"] for k, v in figs.items() if k != "curves"]),
           "hypotheses": {h["id"]: h["decision"] for h in an.hypotheses}, "n_runs": len(res.runs), "n_evaluated_runs": len(res.evals)}
    log.info("report build done in %.1f s: %d runs, %d evaluated, %d figures, PDF %s", time.time() - t0, len(res.runs),
             len(res.evals), len(out["figures"]), "written" if pdf_path else "skipped")
    return out


def model_card(rid: str, results_dir=None) -> str:
    """Hugging Face model card of run ``rid`` (e.g. ``generalist/seed1``), from the current result files."""
    res = collect(ReportPaths.resolve(results_dir, None))
    return cards.model_card(res, rid)
