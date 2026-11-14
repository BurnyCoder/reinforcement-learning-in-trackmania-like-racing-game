"""Publish the project state to GitHub (branch → pull request → merge) and to Hugging Face.

Global context: the publishing rule (reports/plan.md, "Publishing rule") pushes everything three
times per model: before its training, after its training and after its evaluation. Each push first
regenerates every table, figure, report, the README results block, the paper and the website data
from the result files (``report.build_all``), then commits and merges through a short-lived branch,
and mirrors models and results to one Hugging Face collection. Credentials come from ``gh auth`` and
``hf auth`` logins; nothing secret is written to the repository (``.env`` is git-ignored).

Hugging Face API: ``HfApi.create_repo``, ``upload_folder``, ``create_collection``,
``add_collection_item`` (https://huggingface.co/docs/huggingface_hub/guides/collections).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .utils.logging import get_logger
from .utils.paths import REPO_ROOT, RESULTS_DIR, data_dir

log = get_logger(__name__)
PUBLISH_PATHS = ["results", "reports", "paper", "README.md", "docs", "web/assets"]
RESULTS_REPO_NAME = "racingmania-results"


def _git(*args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check, capture_output=True, text=True).stdout.strip()


def _gh(*args) -> str:
    return subprocess.run(["gh", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def publish_github(message: str) -> bool:
    """Commit the publishable paths on a new branch, open a pull request and merge it. Returns True if
    anything was committed."""
    _git("checkout", "main"); _git("pull", "--ff-only", check=False)
    _git("add", *[p for p in PUBLISH_PATHS if (REPO_ROOT / p).exists()])
    if not _git("diff", "--cached", "--name-only"):
        log.info("github: nothing to publish")
        return False
    branch = "results/" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    _git("checkout", "-b", branch)
    _git("commit", "-m", message)
    _git("push", "-u", "origin", branch)
    _gh("pr", "create", "--title", message.splitlines()[0][:70], "--body", message, "--base", "main", "--head", branch)
    _gh("pr", "merge", branch, "--merge", "--delete-branch")
    _git("checkout", "main"); _git("pull", "--ff-only", check=False)
    log.info("github: published (%s)", message.splitlines()[0])
    return True


def _hf():
    from huggingface_hub import HfApi
    return HfApi()


def _namespace() -> str:
    return os.environ.get("HF_NAMESPACE") or _hf().whoami()["name"]


_collection = None  # slug of the project's collection, looked up once per process


def _collection_slug(api) -> str:
    """Slug of the project's Hugging Face collection, found with read-only calls and created only if missing.

    ``create_collection(exists_ok=True)`` counts against Hugging Face's limit of 100 collection creations per
    day even when the collection exists; calling it on every publish hit that limit on 2026-11-14 and stopped
    the run queue (reports/notes/phase-2.md). ``HfApi.list_collections(owner=...)`` is a read."""
    global _collection
    if _collection is None:
        title = os.environ.get("HF_COLLECTION_TITLE", "RacingMania RL")
        _collection = os.environ.get("HF_COLLECTION_SLUG") or next(
            (c.slug for c in api.list_collections(owner=_namespace(), limit=100) if c.title == title), None)
        if _collection is None:
            _collection = api.create_collection(title=title, namespace=_namespace(),
                description="Reinforcement-learning drivers for RacingMania, an original 3D time-attack racing game").slug
    return _collection


def model_repo_id(rid: str) -> str:
    return f"{_namespace()}/racingmania-{rid.replace('/', '-')}"


def publish_model(rid: str, status: str, card: str, files: dict[str, Path]):
    """Create/update the model repository of run ``rid`` with ``files`` ({path in repo: local path})."""
    api = _hf()
    repo = model_repo_id(rid)
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "README.md").write_text(card, encoding="utf-8")
        for dst, src in files.items():
            if Path(src).exists():
                (d / dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d / dst)
        api.upload_folder(folder_path=str(d), repo_id=repo, repo_type="model", commit_message=f"{rid}: {status}")
    api.add_collection_item(_collection_slug(api), item_id=repo, item_type="model", exists_ok=True)
    log.info("huggingface: %s updated (%s)", repo, status)
    return repo


def publish_results_dataset(message: str):
    """Mirror results, reports, the paper and the plan to one dataset repository in the collection."""
    api = _hf()
    repo = f"{_namespace()}/{RESULTS_REPO_NAME}"
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    for sub in ("results", "reports", "paper"):
        p = REPO_ROOT / sub
        if p.exists():
            api.upload_folder(folder_path=str(p), path_in_repo=sub, repo_id=repo, repo_type="dataset", commit_message=message,
                              ignore_patterns=["*.log"])
    readme = REPO_ROOT / "results" / "HF_DATASET_CARD.md"
    if readme.exists():
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                        commit_message=message)
    api.add_collection_item(_collection_slug(api), item_id=repo, item_type="dataset", exists_ok=True)
    log.info("huggingface: dataset %s updated", repo)


def publish_all(message: str, rid: str | None = None, status: str | None = None):
    """Regenerate every report artifact, then push to GitHub and Hugging Face."""
    from .report.build import build_all, model_card
    build_all()
    publish_github(message)
    publish_results_dataset(message)
    if rid:
        res = RESULTS_DIR / rid
        ckpt = data_dir() / "checkpoints" / rid
        files = {"final.pt": ckpt / "final.pt", "policy.onnx": res / "policy.onnx", "run.json": res / "run.json",
                 "train_metrics.csv": res / "train_metrics.csv", "val_metrics.csv": res / "val_metrics.csv"}
        files.update({f"eval/{p.name}": p for p in res.glob("eval_*.csv")})
        publish_model(rid, status or "update", model_card(rid), files)
