"""Resumable run queue: train, evaluate and publish every run of the run matrix in order.

Global context: ``configs/run_matrix.yaml`` lists the runs of each phase in the preregistered order
(reports/plan.md, "Runs, by phase"). For every run not yet evaluated the queue (1) publishes the
project with the run registered as planned, (2) trains it (resuming from its checkpoint after any
interruption), (3) publishes again, (4) evaluates it, (5) publishes again — the publishing rule.
The queue runs on Windows; GPU work runs in WSL2, where ``torch.compile`` makes the simulator ~10×
faster, through ``wsl -e bash -lc``. Because every step is idempotent and statuses live in
``run.json``, rerunning the queue after a crash or reboot continues where it stopped.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .runs import run_dirs, run_id, update_run_json
from .utils.logging import get_logger
from .utils.paths import REPO_ROOT

log = get_logger(__name__)
MATRIX_FILE = REPO_ROOT / "configs" / "run_matrix.yaml"
WSL_REPO = "/mnt/c/Users/Happy/Documents/projects/simplified-trackmania-reinforcement-learning"


def wsl_repo() -> str:
    return os.environ.get("RACINGMANIA_WSL_REPO", WSL_REPO)


def run_gpu(script_args: list[str]):
    """Run ``scripts/<...>.py args`` on the GPU: inside WSL from Windows, directly on Linux."""
    if os.name == "nt":
        inner = (f"cd {shlex.quote(wsl_repo())} && export UV_PROJECT_ENVIRONMENT=$HOME/venvs/racingmania "
                 f"TRITON_DEFAULT_FP_FUSION=1 && uv run python {' '.join(shlex.quote(a) for a in script_args)}")
        cmd = ["wsl", "-e", "bash", "-lc", inner]
    else:
        cmd = [sys.executable, *script_args]
    log.info("running: %s", " ".join(script_args))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load_matrix() -> list[dict]:
    m = yaml.safe_load(MATRIX_FILE.read_text(encoding="utf-8"))
    runs = []
    for phase in m["phases"]:
        for r in phase["runs"]:
            runs.append({**r, "phase": phase["name"]})
    return runs


def status(r: dict) -> str:
    res, _, _ = run_dirs(run_id(r["experiment"], r["seed"], r.get("track")))
    p = res / "run.json"
    return json.loads(p.read_text()).get("status", "none") if p.exists() else "none"


def run_queue(phases: list[str] | None = None, publish: bool = True, limit: int | None = None):
    from .publish import publish_all
    done = 0
    for r in load_matrix():
        if phases and r["phase"] not in phases:
            continue
        rid = run_id(r["experiment"], r["seed"], r.get("track"))
        st = status(r)
        if st == "evaluated":
            continue
        res, _, _ = run_dirs(rid)
        track_args = ["--track", r["track"]] if r.get("track") else []
        if st in ("none", "planned"):
            update_run_json(res, experiment=r["experiment"], seed=r["seed"], track=r.get("track"), status="planned",
                            phase=r["phase"])
            if publish:
                publish_all(f"Register run {rid} ({r['phase']}) before training", rid, "planned")
        if st in ("none", "planned", "training"):
            run_gpu(["scripts/02_train.py", "--experiment", r["experiment"], "--seed", str(r["seed"]), *track_args])
            if publish:
                publish_all(f"Run {rid} trained", rid, "trained")
        run_gpu(["scripts/03_evaluate.py", "--experiment", r["experiment"], "--seed", str(r["seed"]), *track_args,
                 "--sets", *r.get("eval_sets", ["val", "train_sample", "campaign", "test", "oor"])])
        run_gpu(["scripts/06_export_web.py", "--run", rid])
        if publish:
            publish_all(f"Run {rid} evaluated", rid, "evaluated")
        done += 1
        if limit and done >= limit:
            break
