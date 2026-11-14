"""Build paper/paper.pdf from paper/paper.md with pandoc and typst; never fail the report build.

Local purpose: pandoc converts the Markdown paper to a standalone typst file
(``pandoc paper.md -s -t typst``, https://pandoc.org/MANUAL.html#option--to), then typst compiles it
with ``--root`` set to the output root so the figures under ``reports/figures/`` (referenced as
``../reports/figures/...``) are inside typst's project tree (``typst compile --help``, typst 0.15:
"--root <DIR>  Configures the project root (for absolute paths)"; files outside the root cannot be
read). The one-step ``pandoc --pdf-engine=typst`` was tried
first; on Windows it rewrites image paths to absolute ``C:/...`` paths that typst rejects, hence two
steps. If the typst executable is missing, the ``typst`` Python package (``typst.compile``) is used
if installed. If neither works, a warning is logged and the PDF is skipped.

Global context: called at the end of ``build.build_all``. Install hints: ``winget install --id
Typst.Typst`` and ``winget install --id JohnMacFarlane.Pandoc`` (Windows), or ``uv add typst``.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)


def _find(exe: str, extra_globs: list[str]) -> str | None:
    p = shutil.which(exe)
    if p:
        return p
    for g in extra_globs:
        hits = sorted(glob.glob(os.path.expandvars(g), recursive=True))
        if hits:
            return hits[-1]
    return None


def find_pandoc() -> str | None:
    return _find("pandoc", [r"%ProgramFiles%\Pandoc\pandoc.exe", r"%LOCALAPPDATA%\Pandoc\pandoc.exe"])


def find_typst() -> str | None:
    # winget installs Typst under its package folder and may not have refreshed PATH in this process
    return _find("typst", [r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Typst.Typst_*\**\typst.exe",
                           r"%LOCALAPPDATA%\Microsoft\WinGet\Links\typst.exe"])


def build_pdf(paper_md: Path, root: Path, timeout: int = 600) -> Path | None:
    """paper.md -> paper.pdf next to it; returns the PDF path or None (with a logged warning)."""
    paper_md = Path(paper_md)
    pandoc = find_pandoc()
    if not pandoc:
        log.warning("PDF skipped: pandoc not found (install: winget install --id JohnMacFarlane.Pandoc)")
        return None
    typ = paper_md.with_suffix(".tmp.typ")  # next to paper.md so relative image paths keep working
    pdf = paper_md.with_suffix(".pdf")
    try:
        subprocess.run([pandoc, paper_md.name, "-s", "-t", "typst", "-o", typ.name], cwd=paper_md.parent, check=True,
                       capture_output=True, text=True, timeout=timeout)
        typst = find_typst()
        if typst:
            r = subprocess.run([typst, "compile", "--root", str(root), typ.name, pdf.name], cwd=paper_md.parent,
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0:
                log.warning("PDF skipped: typst failed:\n%s", r.stderr[-3000:])
                return None
        else:
            try:
                import typst as typst_py  # optional fallback: the typst Python package
            except ImportError:
                log.warning("PDF skipped: typst not found (install: winget install --id Typst.Typst, or uv add typst)")
                return None
            typst_py.compile(str(typ), output=str(pdf), root=str(root))
        log.info("paper PDF written: %s", pdf)
        return pdf
    except subprocess.CalledProcessError as e:
        log.warning("PDF skipped: pandoc failed:\n%s", (e.stderr or "")[-3000:])
    except Exception as e:  # noqa: BLE001  (the report build must never fail because of the PDF)
        log.warning("PDF skipped: %s", e)
    finally:
        typ.unlink(missing_ok=True)
    return None
