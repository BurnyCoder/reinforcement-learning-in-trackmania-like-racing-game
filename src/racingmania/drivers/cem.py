"""CEM planner: a near-optimal reference time per track from search in the true simulator.

Global context: the run table's "CEM reference" (reports/plan.md) gives an approximate upper bound on
how fast a track can be driven with this car and these actions, to put learned and scripted times
in context. It is model-predictive control with the cross-entropy method (Rubinstein 1999; as a
planner e.g. iCEM, Pinneri et al. 2020, arXiv:2008.06389): at every agent step, sample ``P``
action sequences of length ``H`` from per-step categorical distributions, simulate them from the
current state in the deterministic simulator, refit the distributions to the best 10 %, repeat a
few times, execute the first action of the best sequence, and shift the plan by one step.
All tracks are planned in parallel: tracks × population cars in one batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..env.vec_env import EnvConfig, RacingVecEnv
from ..sim.rules import RuleState
from ..sim.step import CarState, agent_step
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CEMParams:
    population: int = 256
    horizon: int = 90  # agent steps (4.5 s); chosen on validation tracks (scripts/tune_cem.py, results/baselines/cem_horizon_tuning.txt)
    iterations: int = 3
    elite_frac: float = 0.1
    smoothing: float = 0.7  # weight of the new elite frequencies in the refit
    min_prob: float = 0.01
    finish_bonus_per_step: float = 3.0  # metres credited per step saved by finishing inside the horizon
    out_penalty: float = 1000.0


def plan_tracks(cfg, tracks, params: CEMParams = CEMParams(), device="cuda", max_ticks: int = 20000, seed: int = 0,
                compile: bool = True):
    """Drive every track of ``tracks`` once with the planner. Returns finished [T] and ticks [T] (lists)."""
    T, P, H = tracks.num_tracks, params.population, params.horizon
    gen = torch.Generator(device=device).manual_seed(seed)
    ecfg = EnvConfig(random_start_prob=0.0, max_steps=10**9)
    main = RacingVecEnv(cfg, tracks, T, ecfg, device=device, assign=torch.arange(T), auto_reset=False, compile=compile)
    plan_env = RacingVecEnv(cfg, tracks, T * P, ecfg, device=device, assign=torch.arange(T).repeat_interleave(P),
                            auto_reset=False, compile=False)
    step_fn = torch.compile(agent_step) if compile else agent_step
    probs = torch.full((T, H, 9), 1 / 9, device=device)
    rep = lambda x: x.repeat_interleave(P, dim=0)
    n_elite = max(1, int(P * params.elite_frac))
    decisions = 0
    with torch.no_grad():
        while True:
            alive = ~(main.rules.finished | main.rules.out) & (main.rules.ticks < max_ticks)
            if not bool(alive.any()):
                break
            best_seq = torch.zeros(T, H, dtype=torch.long, device=device)
            for _ in range(params.iterations):
                seqs = torch.distributions.Categorical(probs=probs.unsqueeze(1).expand(T, P, H, 9)).sample()  # [T, P, H]
                car = CarState(**{f: rep(getattr(main.car, f)) for f in ("pos", "quat", "vel", "angvel", "steer")})
                rules = RuleState(**{f: rep(getattr(main.rules, f)) for f in RuleState.__dataclass_fields__})
                s0 = rules.s.clone()
                fin_step = torch.full((T * P,), H, device=device)
                flat = seqs.reshape(T * P, H)
                for h in range(H):
                    was_done = rules.finished.clone()
                    car, rules, _, _ = step_fn(cfg, plan_env.tracks, plan_env.track_id, car, rules, flat[:, h])
                    fin_step = torch.where(rules.finished & ~was_done, torch.full_like(fin_step, h + 1), fin_step)
                score = (rules.s - s0) + rules.finished.float() * (H - fin_step) * params.finish_bonus_per_step
                score = score - rules.out.float() * params.out_penalty
                score = score.view(T, P)
                top = score.topk(n_elite, dim=1).indices  # [T, E]
                elite = seqs.gather(1, top.unsqueeze(-1).expand(-1, -1, H))  # [T, E, H]
                freq = torch.nn.functional.one_hot(elite, 9).float().mean(1)  # [T, H, 9]
                probs = (1 - params.smoothing) * probs + params.smoothing * freq
                probs = probs.clamp(min=params.min_prob); probs = probs / probs.sum(-1, keepdim=True)
                best_seq = seqs[torch.arange(T, device=device), score.argmax(1)]
            main.step(best_seq[:, 0])
            probs = torch.cat((probs[:, 1:], torch.full((T, 1, 9), 1 / 9, device=device)), dim=1)  # shift the plan
            decisions += 1
            if decisions % 100 == 0:
                log.info("CEM: %d decisions, %d/%d tracks still driving", decisions, int(alive.sum()), T)
    return main.rules.finished.cpu().tolist(), main.rules.ticks.cpu().tolist()
