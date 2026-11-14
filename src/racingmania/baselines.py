"""Training-free baselines and controls, scored by the same evaluator as the trained agents.

Global context: the run table's "Training-free baselines" row (reports/plan.md): uniformly random
actions (the floor), an untrained network (10 random initialisations; shows what training adds),
and the frozen scripted driver (the reference, score 1.0 by construction on certified sets). The
CEM planner (``drivers/cem.py``) gives near-optimal times on a few tracks and is run separately.
Results go to ``results/baselines/<name>/seed<k>/`` like any other run.
"""

from __future__ import annotations

import torch

from .drivers.scripted import ScriptedDriver
from .drivers.simple import RandomDriver
from .env.obs import obs_size
from .env.vec_env import EnvConfig
from .eval.evaluator import evaluate, summarise, write_eval_csv
from .rl.networks import ActorCritic
from .rl.normalizer import RunningNorm
from .runs import eval_tracks, hardware, save_replays, update_run_json
from .sim.vehicle_config import load_vehicle_config
from .tracks.build_sets import load_tuned
from .utils.logging import get_logger
from .utils.paths import RESULTS_DIR

log = get_logger(__name__)


def untrained_agent(seed: int, device="cuda"):
    """Freshly initialised PPO actor-critic (same architecture as the standard recipe) and its default
    normaliser (zero mean, unit variance; it is never updated, so evaluation and export see the same one)."""
    torch.manual_seed(seed)
    agent = ActorCritic(obs_size()).to(device)
    norm = RunningNorm(obs_size()).to(device)
    norm.frozen = True
    return agent, norm


def untrained_policy(seed: int, device="cuda"):
    """The untrained control as a policy for the evaluator (most probable action)."""
    agent, norm = untrained_agent(seed, device)

    def act(env):
        with torch.no_grad():
            return agent.act(norm(env.observe().float()), deterministic=True)[1]
    act.name = f"untrained-seed{seed}"
    return act


class _Scripted:
    """Evaluator wrapper that builds the scripted driver lazily for the evaluator's environment."""

    def __init__(self, params):
        self.params, self.driver, self.name = params, None, "scripted"

    def __call__(self, env):
        if self.driver is None or self.driver.env is not env:
            self.driver = ScriptedDriver(env, self.params)
        return self.driver(env)


def run_baselines(sets: list[str], untrained_seeds=range(10), random_seeds=range(3), device="cuda"):
    cfg = load_vehicle_config()
    jobs = [("scripted", 0, _Scripted(load_tuned()))]
    jobs += [("random", s, RandomDriver(seed=s, device=device)) for s in random_seeds]
    jobs += [("untrained", s, untrained_policy(s, device)) for s in untrained_seeds]
    for name, seed, policy in jobs:
        res = RESULTS_DIR / "baselines" / name / f"seed{seed}"
        summary = {}
        for s in sets:
            tb, protected = eval_tracks(s, device)
            rows = evaluate(policy, cfg, tb, device=device, env_cfg=EnvConfig(), compile=True,
                            record_replays=(s == "campaign" and seed == 0), log_access=s if protected else None)
            write_eval_csv(rows, res / f"eval_{s}.csv")
            if s == "campaign" and seed == 0:
                save_replays(rows, res / "replays", cfg.dt)
            summary[s] = summarise(rows)
            log.info("baseline %s seed %d on %s: %s", name, seed, s, summary[s])
        update_run_json(res, experiment=f"baseline-{name}", seed=seed, status="evaluated", algo=name,
                        eval_summary=summary, hardware=hardware())


def run_cem(n_test: int = 25, device="cuda"):
    """CEM reference times on the 5 campaign tracks and the first ``n_test`` test tracks."""
    import numpy as np
    from .drivers.cem import CEMParams, plan_tracks
    cfg = load_vehicle_config()
    res = RESULTS_DIR / "baselines" / "cem" / "seed0"
    summary = {}
    for name in ("campaign", "test"):
        tb, protected = eval_tracks(name, device)
        if name == "test":
            tb = tb.subset(list(range(min(n_test, tb.num_tracks))))
            from .eval.evaluator import _log_access
            _log_access("test", "cem", tb.num_tracks)
        fin, ticks = plan_tracks(cfg, tb, CEMParams(), device=device)
        ref = tb.ref_ticks.cpu().numpy()
        rows = [{"track": tb.names[i], "finished": int(fin[i]), "ticks": int(ticks[i]), "time": ticks[i] * cfg.dt,
                 "ref_ticks": int(ref[i]), "score": (float(ref[i]) / ticks[i]) if fin[i] and ref[i] > 0 else 0.0,
                 "end": "finished" if fin[i] else "not_finished", "progress": float("nan"), "max_speed": float("nan"),
                 "flag_overspeed": 0, "max_coast_energy_gain": float("nan"), "flag_energy_gain": 0} for i in range(tb.num_tracks)]
        key = "campaign" if name == "campaign" else f"test{n_test}"
        write_eval_csv(rows, res / f"eval_{key}.csv")
        summary[key] = summarise(rows)
        log.info("CEM on %s: %s", key, summary[key])
    update_run_json(res, experiment="baseline-cem", seed=0, status="evaluated", algo="cem", eval_summary=summary,
                    cem=CEMParams().__dict__, hardware=hardware())
