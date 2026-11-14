"""Run-level phases: train one (experiment, seed) run, evaluate it, record its status.

Global context: every trained model in the run matrix (reports/plan.md, "Runs, by phase") is an
experiment config from ``configs/experiments/<name>.yaml`` plus a seed. This module turns that pair
into a training run (``train_run``) and an evaluation (``evaluate_run``), and keeps
``results/<experiment>/seed<seed>/run.json`` up to date (planned → training → trained → evaluated)
so the README, paper and website can show unfinished runs as unfinished.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .env.vec_env import EnvConfig
from .sim.vehicle_config import load_vehicle_config
from .tracks.store import batch_from_pool, load_pool, pack, pool_path
from .utils.logging import get_logger, setup_logging
from .utils.paths import REPO_ROOT, RESULTS_DIR, WEB_ASSETS, data_dir

log = get_logger(__name__)
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"
PHYSICS_VERSION = "v1"  # bumped when a physics patch is applied (preregistration, section 8)


def load_experiment(name: str) -> dict:
    exp = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    exp["name"] = name
    return exp


def run_id(experiment: str, seed: int, track: str | None = None) -> str:
    return f"{experiment}-{track}/seed{seed}" if track else f"{experiment}/seed{seed}"


def run_dirs(rid: str) -> tuple[Path, Path, Path]:
    """(results dir in the repository, checkpoint dir, TensorBoard dir) of a run."""
    return RESULTS_DIR / rid, data_dir() / "checkpoints" / rid, data_dir() / "tensorboard" / rid


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def update_run_json(results_dir: Path, **fields):
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / "run.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(fields)
    d["updated"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    return d


def hardware() -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def campaign_pool(names: list[str]) -> dict:
    """Pack campaign track files (with their reference runs) into a pool."""
    tracks = [json.loads((WEB_ASSETS / "tracks" / f"{n}.json").read_text()) for n in names]
    return pack(tracks)


def tracks_for(spec: dict, seed: int):
    """Pool and allowed track indices for a ``tracks`` / ``validation`` spec of an experiment."""
    if "campaign" in spec:
        names = spec["campaign"] if isinstance(spec["campaign"], list) else [spec["campaign"]]
        return campaign_pool(names), None
    pool = load_pool(pool_path(spec["pool"]))
    ids = None
    if spec.get("n_tracks"):
        ids = np.sort(np.random.default_rng([seed, 7919]).choice(len(pool["n"]), int(spec["n_tracks"]), replace=False))
    return pool, ids


def vehicle_for(exp: dict):
    cfg = load_vehicle_config()
    over = exp.get("vehicle", {})
    return dataclasses.replace(cfg, **over) if over else cfg


def train_run(experiment: str, seed: int, track: str | None = None, resume: bool = True,
              stop_after_steps: int | None = None, overrides: dict | None = None, device="cuda") -> Path:
    """Train one run to its step budget (or until ``stop_after_steps``), resuming if a checkpoint exists."""
    from torch.utils.tensorboard import SummaryWriter
    from .rl.ppo import PPOConfig, PPOTrainer
    from .rl.pqn import PQNConfig, PQNTrainer
    exp = load_experiment(experiment)
    if track:  # specialists name their campaign track per run
        exp["tracks"] = {"campaign": track}; exp["validation"] = {"campaign": track}
    rid = run_id(experiment, seed, track)
    res_dir, ckpt_dir, tb_dir = run_dirs(rid)
    setup_logging(res_dir / "train.log")
    cfg = vehicle_for(exp)
    env_cfg = EnvConfig(**exp.get("env", {}))
    algo = exp["algo"]
    pp = {**exp.get("pqn" if algo == "pqn" else "ppo", {}), **(overrides or {}), "seed": seed}
    if "hidden" in pp:
        pp["hidden"] = tuple(pp["hidden"])
    ppo = PQNConfig(**pp) if algo == "pqn" else PPOConfig(**pp)
    pool, ids = tracks_for(exp["tracks"], seed)
    val_pool, _ = tracks_for(exp["validation"], seed)
    val_tracks = batch_from_pool(val_pool, device=device)
    update_run_json(res_dir, experiment=experiment, seed=seed, track=track, status="training", algo=exp["algo"],
                    description=exp.get("description", ""), git_commit=git_commit(), physics=PHYSICS_VERSION,
                    hardware=hardware(), ppo=dataclasses.asdict(ppo), env=dataclasses.asdict(env_cfg),
                    n_train_tracks=int(len(ids) if ids is not None else len(pool["n"])),
                    vehicle_overrides=exp.get("vehicle", {}), checkpoint_dir=str(ckpt_dir), tensorboard_dir=str(tb_dir))
    if algo.startswith("imitation"):
        return _train_imitation_run(exp, seed, cfg, env_cfg, pool, ids, res_dir, ckpt_dir, device)
    init = None
    if exp.get("init_from"):  # fine-tuning from another run's final weights
        init = str(data_dir() / "checkpoints" / exp["init_from"] / "final.pt")
    elif exp.get("bc_init"):  # phase-4 warm start: behaviour cloning first, then PPO
        init = str(_bc_warm_start(exp, seed, cfg, env_cfg, pool, ids, ckpt_dir, device))
    Trainer = PQNTrainer if algo == "pqn" else PPOTrainer
    trainer = Trainer(cfg, ppo, env_cfg, pool, res_dir, ckpt_dir, val_tracks=val_tracks, train_ids=ids,
                      device=device, writer=SummaryWriter(str(tb_dir)), init_agent=init)
    if resume and (ckpt_dir / "latest.pt").exists():
        trainer.load(ckpt_dir / "latest.pt")
    t0 = time.time()
    trainer.train(stop_after_steps=stop_after_steps)
    done = trainer.global_step >= ppo.total_steps
    update_run_json(res_dir, status="trained" if done else "training", global_step=trainer.global_step,
                    iterations=trainer.iteration, wall_hours=trainer.wall / 3600, last_session_hours=(time.time() - t0) / 3600)
    if done:
        torch.save({"agent": trainer.agent.state_dict(), "norm": trainer.norm.state_dict(), "ppo": dataclasses.asdict(ppo),
                    "env_cfg": dataclasses.asdict(env_cfg), "obs_dim": trainer.obs_dim, "algo": algo}, ckpt_dir / "final.pt")
    return res_dir


def _imitation_data(exp, cfg, env_cfg, pool, ids, device):
    from .tracks.build_sets import load_tuned
    sub_ids = ids if ids is not None else np.random.default_rng(0).choice(len(pool["n"]), min(4096, len(pool["n"])), replace=False)
    return batch_from_pool(pool, np.sort(sub_ids), device=device), load_tuned()


def _bc_warm_start(exp, seed, cfg, env_cfg, pool, ids, ckpt_dir, device) -> Path:
    from .rl.imitation import train_imitation
    path = ckpt_dir / "bc_init.pt"
    if not path.exists():
        tb, params = _imitation_data(exp, cfg, env_cfg, pool, ids, device)
        agent, norm, _ = train_imitation(cfg, tb, env_cfg, params, "bc", device=device, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent.state_dict(), "norm": norm.state_dict()}, path)
    return path


def _train_imitation_run(exp, seed, cfg, env_cfg, pool, ids, res_dir, ckpt_dir, device) -> Path:
    """Pure imitation runs (phase 4): BC or DAgger, saved in the same format as PPO runs."""
    import csv
    from .rl.imitation import train_imitation
    method = "dagger" if "dagger" in exp["algo"] else "bc"
    tb, params = _imitation_data(exp, cfg, env_cfg, pool, ids, device)
    t0 = time.time()
    agent, norm, rows = train_imitation(cfg, tb, env_cfg, params, method, device=device, seed=seed)
    with open(res_dir / "imitation_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "norm": norm.state_dict(), "ppo": {"hidden": [256, 256], "action_mode": "factored"},
                "env_cfg": dataclasses.asdict(env_cfg), "algo": exp["algo"]}, ckpt_dir / "final.pt")
    update_run_json(res_dir, status="trained", wall_hours=(time.time() - t0) / 3600, imitation=rows[-1])
    return res_dir


def load_policy(experiment: str, seed: int, track: str | None = None, device="cuda", which: str = "final"):
    """Rebuild a trained agent for evaluation (most probable action) from its checkpoint."""
    from .rl.networks import ActorCritic
    from .rl.normalizer import RunningNorm
    rid = run_id(experiment, seed, track)
    _, ckpt_dir, _ = run_dirs(rid)
    sd = torch.load(ckpt_dir / f"{which}.pt", map_location=device, weights_only=False)
    ppo, env_cfg = sd["ppo"], EnvConfig(**sd["env_cfg"])
    from .env.obs import obs_size
    if sd.get("algo") == "pqn":
        from .rl.pqn import QAgent
        agent = QAgent(obs_size(env_cfg.window_k), tuple(ppo["hidden"])).to(device)
    else:
        agent = ActorCritic(obs_size(env_cfg.window_k), tuple(ppo["hidden"]), ppo["action_mode"]).to(device)
    agent.load_state_dict(sd["agent"]); agent.eval()
    norm = RunningNorm(obs_size(env_cfg.window_k)).to(device); norm.load_state_dict(sd["norm"]); norm.frozen = True

    def act(env):
        with torch.no_grad():
            return agent.act(norm(env.observe().float()), deterministic=True)[1]
    act.name = rid
    return act, agent, norm, env_cfg


def evaluate_run(experiment: str, seed: int, sets: list[str], track: str | None = None, device="cuda",
                 record_replays_on: tuple[str, ...] = ("campaign",)) -> dict:
    """Evaluate a trained run on track sets ("val", "train_sample", "test", "oor", "campaign")."""
    from .eval.evaluator import evaluate, summarise, write_eval_csv
    exp = load_experiment(experiment)
    rid = run_id(experiment, seed, track)
    res_dir, ckpt_dir, _ = run_dirs(rid)
    setup_logging(res_dir / "eval.log")
    policy, _, _, env_cfg = load_policy(experiment, seed, track, device)
    cfg = vehicle_for(exp)
    summary = {}
    for s in sets:
        tb, protected = eval_tracks(s, device)
        rows = evaluate(policy, cfg, tb, device=device, env_cfg=env_cfg, compile=True,
                        record_replays=s in record_replays_on, log_access=s if protected else None)
        write_eval_csv(rows, res_dir / f"eval_{s}.csv")
        if s in record_replays_on:
            save_replays(rows, res_dir / "replays", cfg.dt)
        summary[s] = summarise(rows)
        log.info("%s on %s: %s", rid, s, summary[s])
    update_run_json(res_dir, status="evaluated", eval_summary=summary)
    return summary


def eval_tracks(name: str, device="cuda"):
    """TrackBatch for an evaluation set and whether reading it must be logged."""
    if name == "campaign":
        from .tracks.campaign import CAMPAIGN_PLANS
        return batch_from_pool(campaign_pool(sorted(CAMPAIGN_PLANS)), device=device), False
    if name == "train_sample":
        pool = load_pool(pool_path("train"))
        return batch_from_pool(pool, np.arange(min(1000, len(pool["n"]))), device=device), False
    pool = load_pool(pool_path(name))
    return batch_from_pool(pool, device=device), name in ("test", "oor", "test_unfiltered")


def save_replays(rows: list[dict], out_dir: Path, dt: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (out_dir / f"{r['track']}.json").write_text(json.dumps(
            {"track": r["track"], "finished": r["finished"], "time": r["time"], "states": r["replay"]}), encoding="utf-8")
