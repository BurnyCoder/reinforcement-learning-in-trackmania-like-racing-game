"""PPO for the GPU-vectorised racing environment, adapted from CleanRL's GPU script.

Source: CleanRL ``ppo_continuous_action_isaacgym.py`` (https://github.com/vwxyzjn/cleanrl, MIT,
with an NVIDIA BSD-3 notice; see THIRD_PARTY_NOTICES.md). Changes, each with its reason:
- two 3-way categorical action heads (keyboard actions) instead of a Gaussian, log-probabilities and
  entropies summed (CleanRL ``ppo_multidiscrete.py``);
- observation normalisation with clipping (Huang et al. 2022, "37 implementation details");
- bootstrapping on time limits: ``r += γ·V(final_obs)`` for truncated episodes (Rudin et al. 2021
  bootstrap with the critic's prediction; Gymnasium's termination/truncation rule);
- learning rate adapted to a KL target (Rudin et al. 2021; rsl_rl's rule: ÷1.5 above 2× target,
  ×1.5 below ½ target, bounded to [1e-5, 1e-2]) and an entropy bonus of 0.01;
- full checkpoint/resume (model, optimiser, normaliser, environment and RNG state) every few
  minutes, because runs last hours (project instructions);
- periodic deterministic evaluation on validation tracks for the live dashboard and the learning gate.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..env.obs import obs_size
from ..env.vec_env import EnvConfig, RacingVecEnv
from ..utils.logging import get_logger
from .networks import ActorCritic
from .normalizer import RunningNorm

log = get_logger(__name__)


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000_000  # agent steps (summed over all cars)
    num_envs: int = 16384
    num_steps: int = 32  # rollout length per car
    lr: float = 1e-3  # initial learning rate (then KL-adaptive)
    adaptive_lr: bool = True
    desired_kl: float = 0.01
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 5
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    hidden: tuple = (256, 256)
    action_mode: str = "factored"
    eval_every_minutes: float = 10.0
    checkpoint_every_minutes: float = 10.0
    rotate_tracks_every: int = 20  # iterations between reloading a new random subset of the pool
    loaded_tracks: int = 4096  # tracks on the GPU at a time when the pool is larger
    seed: int = 0
    compile: bool = True


def gae(rewards, values, dones, next_value, gamma: float, lam: float):
    """Generalised advantage estimation (Schulman et al. 2016, arXiv:1506.02438), CleanRL form.

    ``dones[t]`` = 1 if the episode ended at step t (termination or truncation; truncations are
    already bootstrapped in ``rewards``). Shapes ``[S, N]``; ``next_value [N]``.
    """
    S = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(S)):
        nv = next_value if t == S - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv


class PPOTrainer:
    """Owns the agent, the optimiser, the environment and the training loop of one run."""

    def __init__(self, cfg, ppo: PPOConfig, env_cfg: EnvConfig, pool, run_dir: Path, ckpt_dir: Path,
                 val_tracks=None, train_ids=None, device="cuda", writer=None, init_agent: str | None = None):
        self.cfg, self.ppo, self.env_cfg, self.pool = cfg, ppo, env_cfg, pool
        self.run_dir, self.ckpt_dir = Path(run_dir), Path(ckpt_dir)
        self.device = torch.device(device)
        self.writer = writer
        self.val_tracks = val_tracks
        self.train_ids = np.arange(len(pool["n"])) if train_ids is None else np.asarray(train_ids)
        self._seed_everything(ppo.seed)
        self.rng = np.random.default_rng(ppo.seed)
        self.obs_dim = obs_size(env_cfg.window_k)
        self.agent = ActorCritic(self.obs_dim, tuple(ppo.hidden), ppo.action_mode).to(self.device)
        if init_agent:  # fine-tuning (exploratory): start from another run's weights and normaliser
            sd = torch.load(init_agent, map_location=self.device, weights_only=False)
            self.agent.load_state_dict(sd["agent"])
        self.norm = RunningNorm(self.obs_dim).to(self.device)
        if init_agent:
            self.norm.load_state_dict(sd["norm"])
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=ppo.lr, eps=1e-5, weight_decay=ppo.weight_decay)
        self.lr = ppo.lr
        self.iteration, self.global_step, self.wall = 0, 0, 0.0
        self.env = RacingVecEnv(cfg, self._load_tracks(), ppo.num_envs, env_cfg, device=self.device, seed=ppo.seed,
                                compile=ppo.compile)
        self.metrics_path = self.run_dir / "train_metrics.csv"
        self.val_path = self.run_dir / "val_metrics.csv"

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def _load_tracks(self):
        """A random subset of the allowed training tracks (all of them if they fit)."""
        from ..tracks.store import batch_from_pool
        ids = self.train_ids
        if len(ids) > self.ppo.loaded_tracks:
            ids = self.rng.choice(ids, self.ppo.loaded_tracks, replace=False)
        self.loaded_ids = np.sort(ids)
        return batch_from_pool(self.pool, self.loaded_ids, device=self.device)

    # --------------------------------------------------------------- checkpoints
    def save(self, path: Path | None = None):
        path = path or self.ckpt_dir / "latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"agent": self.agent.state_dict(), "opt": self.opt.state_dict(), "norm": self.norm.state_dict(),
                    "lr": self.lr, "iteration": self.iteration, "global_step": self.global_step, "wall": self.wall,
                    "env": self.env.state_dict(), "loaded_ids": self.loaded_ids, "np_rng": self.rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "py_rng": random.getstate(), "ppo": asdict(self.ppo), "env_cfg": asdict(self.env_cfg)}, tmp)
        tmp.replace(path)  # atomic: a crash during saving never leaves a broken checkpoint

    def load(self, path: Path):
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(sd["agent"]); self.opt.load_state_dict(sd["opt"]); self.norm.load_state_dict(sd["norm"])
        self.lr, self.iteration, self.global_step, self.wall = sd["lr"], sd["iteration"], sd["global_step"], sd["wall"]
        self.rng.bit_generator.state = sd["np_rng"]
        from ..tracks.store import batch_from_pool
        self.loaded_ids = sd["loaded_ids"]
        self.env.set_tracks(batch_from_pool(self.pool, self.loaded_ids, device=self.device))
        self.env.load_state_dict(sd["env"])
        torch.set_rng_state(sd["torch_rng"]); torch.cuda.set_rng_state_all(sd["cuda_rng"]); random.setstate(sd["py_rng"])
        log.info("resumed from %s at iteration %d, %d steps", path, self.iteration, self.global_step)

    # --------------------------------------------------------------- evaluation
    def policy(self, deterministic: bool = True):
        """Evaluator policy: normalised observation → most probable action."""
        agent, norm = self.agent, self.norm

        def act(env):
            with torch.no_grad():
                return agent.act(norm(env.observe().float()), deterministic=deterministic)[1]
        act.name = "ppo"
        return act

    def validate(self) -> dict:
        from ..eval.evaluator import evaluate, summarise
        if self.val_tracks is None:
            return {}
        rows = evaluate(self.policy(), self.cfg, self.val_tracks, device=self.device, env_cfg=self.env_cfg,
                        compile=self.ppo.compile)
        s = summarise(rows)
        s.update({"iteration": self.iteration, "global_step": self.global_step, "wall_hours": self.wall / 3600})
        new = not self.val_path.exists()
        with open(self.val_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(s)); new and w.writeheader(); w.writerow(s)
        if self.writer:
            for k in ("sws", "finish_rate", "speed_ratio_finished"):
                if s[k] == s[k]:
                    self.writer.add_scalar(f"val/{k}", s[k], self.global_step)
        log.info("validation @ %d steps: SWS %.3f, finished %.1f%%", self.global_step, s["sws"], 100 * s["finish_rate"])
        return s

    # --------------------------------------------------------------- training
    def train(self, stop_after_steps: int | None = None) -> dict:
        p, env, agent, dev = self.ppo, self.env, self.agent, self.device
        N, S = p.num_envs, p.num_steps
        batch = N * S
        mb = batch // p.num_minibatches
        target = min(p.total_steps, stop_after_steps or p.total_steps)
        obs_buf = torch.zeros(S, N, self.obs_dim, device=dev)
        raw_shape = {"factored": (2,), "joint": (), "continuous": (2,)}[p.action_mode]
        raw_dtype = torch.float32 if p.action_mode == "continuous" else torch.long
        act_buf = torch.zeros(S, N, *raw_shape, device=dev, dtype=raw_dtype)
        logp_buf = torch.zeros(S, N, device=dev); rew_buf = torch.zeros(S, N, device=dev)
        done_buf = torch.zeros(S, N, device=dev); val_buf = torch.zeros(S, N, device=dev)
        next_obs_raw = env.observe().float()
        last_ckpt = last_eval = time.time()
        t_start = time.time() - self.wall
        if self.iteration == 0 and self.val_tracks is not None:
            self.validate()  # the untrained network's score, logged at step 0
        while self.global_step < target:
            it_start = time.time()
            if p.rotate_tracks_every and len(self.train_ids) > p.loaded_tracks and self.iteration > 0 \
                    and self.iteration % p.rotate_tracks_every == 0:
                env.set_tracks(self._load_tracks())
                next_obs_raw = env.reset_all().float()  # partial episodes end at a rollout boundary
            ep_stats = []
            self.norm.update(next_obs_raw)
            for t in range(S):
                obs = self.norm(next_obs_raw)
                with torch.no_grad():
                    raw, a_env, logp, _ = agent.act(obs)
                    v = agent.value(obs)
                next_obs_raw, r, term, trunc, info = env.step(a_env)
                next_obs_raw = next_obs_raw.float()
                r = r.float()
                if bool(trunc.any()):
                    with torch.no_grad():
                        v_final = agent.value(self.norm(info["final_obs"].float()))
                    r = r + p.gamma * v_final * trunc.float()  # bootstrap time-limit truncations
                obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, raw, logp, v
                rew_buf[t], done_buf[t] = r, (term | trunc).float()
                if "episode" in info:
                    ep_stats.append({k: v.cpu() for k, v in info["episode"].items()})
            self.norm.update(next_obs_raw)
            with torch.no_grad():
                next_v = agent.value(self.norm(next_obs_raw))
            adv = gae(rew_buf, val_buf, done_buf, next_v, p.gamma, p.gae_lambda)
            ret = adv + val_buf
            b_obs, b_act = obs_buf.reshape(batch, -1), act_buf.reshape(batch, *raw_shape)
            b_logp, b_adv, b_ret, b_val = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1), val_buf.reshape(-1)
            stats = {"kl": [], "clipfrac": [], "pg_loss": [], "v_loss": [], "entropy": []}
            for _ in range(p.update_epochs):
                perm = torch.randperm(batch, device=dev)
                for start in range(0, batch, mb):
                    ix = perm[start:start + mb]
                    new_logp, ent = agent.evaluate_actions(b_obs[ix], b_act[ix])
                    new_v = agent.value(b_obs[ix])
                    logratio = new_logp - b_logp[ix]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        kl = ((ratio - 1) - logratio).mean()  # Schulman's low-variance KL estimator
                        stats["kl"].append(kl.item()); stats["clipfrac"].append(((ratio - 1).abs() > p.clip_coef).float().mean().item())
                    mb_adv = (b_adv[ix] - b_adv[ix].mean()) / (b_adv[ix].std() + 1e-8)
                    pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef)).mean()
                    vl = 0.5 * ((new_v - b_ret[ix]) ** 2).mean()
                    loss = pg - p.ent_coef * ent.mean() + p.vf_coef * vl
                    if p.adaptive_lr:
                        if kl.item() > 2 * p.desired_kl:
                            self.lr = max(p.lr_min, self.lr / 1.5)
                        elif 0 < kl.item() < p.desired_kl / 2:
                            self.lr = min(p.lr_max, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                    self.opt.step()
                    stats["pg_loss"].append(pg.item()); stats["v_loss"].append(vl.item()); stats["entropy"].append(ent.mean().item())
            self.iteration += 1
            self.global_step += batch
            self.wall = time.time() - t_start
            ev = 1 - torch.var(b_ret - b_val) / (torch.var(b_ret) + 1e-8)
            row = self._log_iteration(ep_stats, stats, float(ev), batch / (time.time() - it_start))
            if time.time() - last_eval > p.eval_every_minutes * 60:
                self.validate(); last_eval = time.time()
            if time.time() - last_ckpt > p.checkpoint_every_minutes * 60:
                self.save(); last_ckpt = time.time()
        self.save()
        return row

    def _log_iteration(self, ep_stats, stats, ev: float, sps: float) -> dict:
        row = {"iteration": self.iteration, "global_step": self.global_step, "wall_hours": self.wall / 3600,
               "sps": sps, "lr": self.lr, "explained_variance": ev,
               **{k: float(np.mean(v)) for k, v in stats.items()}}
        if ep_stats:
            cat = {k: torch.cat([e[k] for e in ep_stats]) for k in ep_stats[0]}
            st = cat["standing_start"].bool()
            row.update({"episodes": len(cat["return"]), "ep_return": cat["return"].float().mean().item(),
                        "ep_finish_rate": cat["finished"].float().mean().item(),
                        "ep_finish_rate_standing": cat["finished"][st].float().mean().item() if st.any() else math.nan,
                        "ep_progress": cat["progress"].float().mean().item(),
                        "ep_fell": (cat["out_reason"] == 1).float().mean().item(),
                        "ep_stalled": (cat["out_reason"] == 3).float().mean().item(),
                        "ep_steps": cat["steps"].float().mean().item()})
        new = not self.metrics_path.exists()
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["iteration", "global_step", "wall_hours", "sps", "lr", "explained_variance", "kl", "clipfrac", "pg_loss",
                "v_loss", "entropy", "episodes", "ep_return", "ep_finish_rate", "ep_finish_rate_standing", "ep_progress",
                "ep_fell", "ep_stalled", "ep_steps"]
        with open(self.metrics_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); new and w.writeheader(); w.writerow(row)
        if self.writer:
            for k, v in row.items():
                if isinstance(v, float) and v == v:
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)
        if self.iteration % 10 == 1 or self.iteration < 5:
            log.info("it %d | %.1fM steps | %.0f sps | return %.2f | finished %.1f%% | progress %.0f m | kl %.4f | lr %.1e",
                     self.iteration, self.global_step / 1e6, sps, row.get("ep_return", math.nan),
                     100 * row.get("ep_finish_rate", math.nan), row.get("ep_progress", math.nan), row["kl"], self.lr)
        return row
