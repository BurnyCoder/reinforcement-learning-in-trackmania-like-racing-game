"""PQN baseline (parallelised Q-network with λ-returns), adapted from CleanRL's ``pqn.py``.

Source: Gallici et al., "Simplifying Deep Temporal Difference Learning" (ICLR 2025,
arXiv:2407.04811) and CleanRL ``cleanrl/pqn.py`` (MIT). PQN collects on-policy rollouts from many
parallel environments with ε-greedy exploration, computes Q(λ) targets backwards through the
rollout, and regresses a LayerNorm Q-network on them; there is no target network and no replay
buffer. Defaults follow CleanRL (λ = 0.65, ε from 1.0 to 0.05 over the first half of training,
gradient-norm clip 10, linearly annealed learning rate 2.5e-4). Changes, with reasons: the same
rollout size, observation normaliser, time-limit bootstrapping, checkpoints and validation as our
PPO (so the comparison isolates the algorithm), and a single 9-way action head because Q-learning
needs one value per joint action. The paper tested at most 1,024 environments; using 16,384 here is
untested, which is why the PQN baseline gets the same tuning effort as PPO (literature review §6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from .networks import QNetwork
from .ppo import PPOConfig, PPOTrainer


class QAgent(nn.Module):
    """Adapter giving a Q-network the ``act``/``value`` interface the trainer and evaluator use."""

    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q = QNetwork(obs_dim, tuple(hidden))
        self.action_mode = "joint"
        self.epsilon = 0.0

    def act(self, obs, deterministic: bool = False):
        qv = self.q(obs)
        greedy = qv.argmax(-1)
        if deterministic or self.epsilon <= 0:
            a = greedy
        else:
            rnd = torch.randint(0, 9, greedy.shape, device=obs.device)
            a = torch.where(torch.rand(greedy.shape, device=obs.device) < self.epsilon, rnd, greedy)
        return a, a, None, None

    def value(self, obs):
        return self.q(obs).max(-1).values

    def logits_for_export(self, obs):
        a = self.q(obs).argmax(-1)
        return nn.functional.one_hot(a // 3, 3).float(), nn.functional.one_hot(a % 3, 3).float()


@dataclass
class PQNConfig(PPOConfig):
    lr: float = 2.5e-4
    q_lambda: float = 0.65
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    max_grad_norm: float = 10.0
    adaptive_lr: bool = False


class PQNTrainer(PPOTrainer):
    def __init__(self, cfg, pqn: PQNConfig, *args, **kwargs):
        super().__init__(cfg, pqn, *args, **kwargs)
        self.agent = QAgent(self.obs_dim, pqn.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=pqn.lr, eps=1e-5)

    def train(self, stop_after_steps: int | None = None) -> dict:
        p, env, agent, dev = self.ppo, self.env, self.agent, self.device
        N, S = p.num_envs, p.num_steps
        batch, mb = N * S, (N * S) // p.num_minibatches
        target = min(p.total_steps, stop_after_steps or p.total_steps)
        obs_buf = torch.zeros(S, N, self.obs_dim, device=dev)
        act_buf = torch.zeros(S, N, dtype=torch.long, device=dev)
        rew_buf = torch.zeros(S, N, device=dev); done_buf = torch.zeros(S, N, device=dev)
        next_obs_raw = env.observe().float()
        last_ckpt = last_eval = time.time(); t_start = time.time() - self.wall
        if self.iteration == 0 and self.val_tracks is not None:
            self.validate()
        row = {}
        while self.global_step < target:
            it_start = time.time()
            frac = min(1.0, self.global_step / (p.exploration_fraction * p.total_steps))
            agent.epsilon = p.start_e + frac * (p.end_e - p.start_e)  # linear ε schedule (CleanRL pqn.py)
            self.lr = p.lr * (1.0 - self.global_step / p.total_steps)  # linear annealing
            for g in self.opt.param_groups:
                g["lr"] = self.lr
            ep_stats = []
            self.norm.update(next_obs_raw)
            for t in range(S):
                obs = self.norm(next_obs_raw)
                with torch.no_grad():
                    a, _, _, _ = agent.act(obs)
                next_obs_raw, r, term, trunc, info = env.step(a)
                next_obs_raw, r = next_obs_raw.float(), r.float()
                if bool(trunc.any()):
                    with torch.no_grad():
                        r = r + p.gamma * agent.value(self.norm(info["final_obs"].float())) * trunc.float()
                obs_buf[t], act_buf[t], rew_buf[t], done_buf[t] = obs, a, r, (term | trunc).float()
                if "episode" in info:
                    ep_stats.append({k: v.cpu() for k, v in info["episode"].items()})
            self.norm.update(next_obs_raw)
            with torch.no_grad():
                # Q(λ) returns (CleanRL pqn.py): G_t = r_t + γ (1-d_t) [λ G_{t+1} + (1-λ) max_a Q(s_{t+1}, a)].
                q_next_max = torch.zeros(S, N, device=dev)
                q_next_max[:-1] = agent.value(obs_buf[1:].reshape(-1, self.obs_dim)).reshape(S - 1, N)
                q_next_max[-1] = agent.value(self.norm(next_obs_raw))
                ret = torch.zeros(S, N, device=dev)
                g_next = q_next_max[-1]
                for t in reversed(range(S)):
                    nonterm = 1.0 - done_buf[t]
                    g_next = rew_buf[t] + p.gamma * nonterm * (p.q_lambda * g_next + (1 - p.q_lambda) * q_next_max[t])
                    ret[t] = g_next
            b_obs, b_act, b_ret = obs_buf.reshape(batch, -1), act_buf.reshape(-1), ret.reshape(-1)
            losses = []
            for _ in range(p.update_epochs):
                perm = torch.randperm(batch, device=dev)
                for start in range(0, batch, mb):
                    ix = perm[start:start + mb]
                    q = agent.q(b_obs[ix]).gather(1, b_act[ix].unsqueeze(1)).squeeze(1)
                    loss = 0.5 * ((q - b_ret[ix]) ** 2).mean()
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                    self.opt.step(); losses.append(loss.item())
            self.iteration += 1; self.global_step += batch; self.wall = time.time() - t_start
            stats = {"kl": [0.0], "clipfrac": [0.0], "pg_loss": [0.0], "v_loss": losses, "entropy": [agent.epsilon]}
            row = self._log_iteration(ep_stats, stats, float("nan"), batch / (time.time() - it_start))
            if time.time() - last_eval > p.eval_every_minutes * 60:
                self.validate(); last_eval = time.time()
            if time.time() - last_ckpt > p.checkpoint_every_minutes * 60:
                self.save(); last_ckpt = time.time()
        self.save()
        return row
