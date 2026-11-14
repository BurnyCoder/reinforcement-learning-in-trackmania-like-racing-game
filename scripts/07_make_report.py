"""Phase 7: regenerate every table, figure, report, the README results block, the paper and the site data.

Usage: uv run python scripts/07_make_report.py [--results DIR] [--out DIR] [--no-pdf]

Reads only the files in results/ (see docs/formats.md, "Results") and writes reports/summary.md,
reports/experiments/*.md, reports/figures/, reports/readme_results.md (and README.md between the
RESULTS markers), paper/paper.md and paper/paper.pdf, web/assets/data/results.json and
results/HF_DATASET_CARD.md. Idempotent; works with no results at all ("not yet available").
"""
import argparse
import json

import _common  # noqa: F401  (timestamped logging)

from racingmania.report.build import build_all

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default=None, help="results directory (default: $RACINGMANIA_RESULTS_DIR or results/)")
ap.add_argument("--out", default=None, help="output root (default: $RACINGMANIA_REPORT_ROOT or the repository)")
ap.add_argument("--no-pdf", action="store_true", help="skip the pandoc + typst PDF step")
a = ap.parse_args()
print(json.dumps(build_all(results_dir=a.results, out_root=a.out, pdf=not a.no_pdf), indent=2))
