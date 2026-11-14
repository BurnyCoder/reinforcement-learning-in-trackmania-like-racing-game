"""Phase 4 only: imitation of the scripted driver (behaviour cloning, DAgger) and the BC warm start.

Global context: in every other phase the agent learns on its own (reports/preregistration.md,
section 2). Phase 4 asks what help from the classical driver changes. Methods:
- Behaviour cloning: supervised learning of the scripted driver's actions from its own states.
- DAgger (Ross, Gordon & Bagnell 2011, "A Reduction of Imitation Learning and Structured Prediction
  to No-Regret Online Learning", AISTATS): roll out a mixture of learner and expert (expert share
  β_i = 0.5^i), label every visited state with the expert's action, aggregate, retrain.
- Warm start: PPO initialised from the BC weights (Fuchs et al. 2021 report that this did not help
  in Gran Turismo; we test it here).
The networks and normaliser are the standard recipe's, so the results are comparable.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ..drivers.scripted import ScriptedDriver
from ..env.obs import obs_size
from ..env.vec_env import EnvConfig, RacingVecEnv
from ..utils.logging import get_logger
from .networks import ActorCritic
from .normalizer import RunningNorm

log = get_logger(__name__)


def _rollout(env, expert, learner, norm, steps: int, beta: float, gen):
    """Collect (raw observation, expert action) pairs while acting with an expert/learner mixture."""
    obs_list, act_list = [], []
    for _ in range(steps):
        obs = env.observe().float()
        a_exp = expert(env)
        if learner is None or beta >= 1.0:
            a = a_exp
        else:
            with torch.no_grad():
                a_lrn = learner.act(norm(obs), deterministic=True)[1]
            use_exp = torch.rand(a_exp.shape, generator=gen, device=a_exp.device) < beta
            a = torch.where(use_exp, a_exp, a_lrn)
        obs_list.append(obs); act_list.append(a_exp)
        env.step(a)
    return torch.cat(obs_list), torch.cat(act_list)


def fit_bc(agent: ActorCritic, norm: RunningNorm, obs: torch.Tensor, act: torch.Tensor, epochs: int = 5,
           batch: int = 65536, lr: float = 1e-3) -> list[float]:
    """Cross-entropy on the two action heads (steer, pedal) of the actor."""
    norm.update(obs)
    opt = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    target = torch.stack((act // 3, act % 3), 1)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(obs.shape[0], device=obs.device)
        for i in range(0, obs.shape[0], batch):
            ix = perm[i:i + batch]
            logp, _ = agent.evaluate_actions(norm(obs[ix]), target[ix])
            loss = -logp.mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    return losses


def train_imitation(cfg, tb, env_cfg: EnvConfig, scripted_params, method: str = "bc", num_envs: int = 2048,
                    steps_per_round: int = 200, rounds: int = 5, device="cuda", seed: int = 0, compile: bool = True):
    """Train an actor by BC (one round of expert rollouts) or DAgger (``rounds`` rounds).

    Returns ``(agent, norm, log_rows)``; the critic is left untrained (it is only used by the warm start).
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    env = RacingVecEnv(cfg, tb, num_envs, env_cfg, device=device, seed=seed, compile=compile)
    expert = ScriptedDriver(env, scripted_params)
    agent = ActorCritic(obs_size(env_cfg.window_k)).to(device)
    norm = RunningNorm(obs_size(env_cfg.window_k)).to(device)
    all_obs, all_act, rows = [], [], []
    n_rounds = 1 if method == "bc" else rounds
    for i in range(n_rounds):
        t0 = time.time()
        beta = 1.0 if (method == "bc" or i == 0) else 0.5 ** i
        o, a = _rollout(env, expert, None if i == 0 else agent, norm, steps_per_round, beta, gen)
        all_obs.append(o); all_act.append(a)
        obs, act = torch.cat(all_obs), torch.cat(all_act)
        losses = fit_bc(agent, norm, obs, act)
        with torch.no_grad():
            pred = agent.act(norm(obs[-200000:]), deterministic=True)[1]
            acc = float((pred == act[-200000:]).float().mean())
        rows.append({"round": i, "beta": beta, "samples": int(obs.shape[0]), "bc_loss": float(np.mean(losses[-20:])),
                     "train_accuracy": acc, "seconds": time.time() - t0})
        log.info("%s round %d: %d samples, loss %.3f, accuracy %.3f", method, i, obs.shape[0], rows[-1]["bc_loss"], acc)
    return agent, norm, rows
