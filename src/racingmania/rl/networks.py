"""Actor-critic networks for PPO and the Q-network for PQN.

Global context: the standard recipe (reports/preregistration.md, section 7) uses separate actor and
critic MLPs with two hidden layers of 256 tanh units, as in CleanRL's GPU PPO script
(https://github.com/vwxyzjn/cleanrl, ``ppo_continuous_action_isaacgym.py``, MIT), with orthogonal
initialisation (``layer_init``: gain √2 for hidden layers, 0.01 for the policy output, 1.0 for the
value output; Huang et al. 2022, "The 37 Implementation Details of PPO"). The actor has two
3-way categorical heads (steer, pedal) whose log-probabilities and entropies are summed, the
pattern of ``ppo_multidiscrete.py`` in Huang et al.'s companion repository
(https://github.com/vwxyzjn/ppo-implementation-details, not CleanRL itself); ablations swap in one
9-way head or a Gaussian head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal weights with gain ``std`` and constant bias (CleanRL ``layer_init``)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(sizes: list[int], out_std: float, act=nn.Tanh) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [layer_init(nn.Linear(a, b)), act()]
    layers.append(layer_init(nn.Linear(sizes[-2], sizes[-1]), std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """PPO agent. ``action_mode``: "factored" (3+3 heads, default), "joint" (one 9-way head) or
    "continuous" (2 Gaussian outputs mapped to the nearest discrete action, ablation)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 256), action_mode: str = "factored"):
        super().__init__()
        self.action_mode = action_mode
        n_out = {"factored": 6, "joint": 9, "continuous": 2}[action_mode]
        self.critic = mlp([obs_dim, *hidden, 1], out_std=1.0)
        self.actor = mlp([obs_dim, *hidden, n_out], out_std=0.01)
        if action_mode == "continuous":
            self.log_std = nn.Parameter(torch.zeros(2))  # state-independent log std, initialised to 0

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(obs).squeeze(-1)

    def _dists(self, obs: Tensor):
        out = self.actor(obs)
        if self.action_mode == "factored":
            return Categorical(logits=out[:, :3]), Categorical(logits=out[:, 3:])
        if self.action_mode == "joint":
            return (Categorical(logits=out),)
        return (Normal(out, self.log_std.exp().expand_as(out)),)

    @staticmethod
    def continuous_to_action(u: Tensor) -> Tensor:
        """Map a continuous (steer, pedal) in roughly [-1, 1]² to the nearest discrete action.
        steer > 1/3 → left, < −1/3 → right; pedal > 1/3 → gas, < −1/3 → brake."""
        s = torch.where(u[:, 0] > 1 / 3, 0, torch.where(u[:, 0] < -1 / 3, 2, 1))
        p = torch.where(u[:, 1] > 1 / 3, 0, torch.where(u[:, 1] < -1 / 3, 2, 1))
        return s * 3 + p

    def act(self, obs: Tensor, deterministic: bool = False):
        """Sample (or take the most probable) action. Returns ``(raw, env_action, logprob, entropy)``;
        ``raw`` is what is stored for the update (``[B,2]`` long, ``[B]`` long or ``[B,2]`` float)."""
        dists = self._dists(obs)
        if self.action_mode == "factored":
            ds, dp = dists
            if deterministic:
                s, p = ds.probs.argmax(-1), dp.probs.argmax(-1)
            else:
                s, p = ds.sample(), dp.sample()
            raw = torch.stack((s, p), 1)
            return raw, s * 3 + p, ds.log_prob(s) + dp.log_prob(p), ds.entropy() + dp.entropy()
        if self.action_mode == "joint":
            (d,) = dists
            a = d.probs.argmax(-1) if deterministic else d.sample()
            return a, a, d.log_prob(a), d.entropy()
        (d,) = dists
        u = d.mean if deterministic else d.sample()
        return u, self.continuous_to_action(u), d.log_prob(u).sum(-1), d.entropy().sum(-1)

    def evaluate_actions(self, obs: Tensor, raw: Tensor):
        """Log-probability and entropy of stored actions under the current policy."""
        dists = self._dists(obs)
        if self.action_mode == "factored":
            ds, dp = dists
            return ds.log_prob(raw[:, 0]) + dp.log_prob(raw[:, 1]), ds.entropy() + dp.entropy()
        (d,) = dists
        if self.action_mode == "joint":
            return d.log_prob(raw), d.entropy()
        return d.log_prob(raw).sum(-1), d.entropy().sum(-1)

    def logits_for_export(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """(steer_logits, pedal_logits) for the ONNX interface (docs/formats.md, "Policy model")."""
        out = self.actor(obs)
        if self.action_mode == "factored":
            return out[:, :3], out[:, 3:]
        if self.action_mode == "joint":
            # Marginal scores: the argmax of the joint table decides; expose it as two one-hot heads.
            a = out.argmax(-1)
            s, p = a // 3, a % 3
            return nn.functional.one_hot(s, 3).float(), nn.functional.one_hot(p, 3).float()
        u = out
        a = self.continuous_to_action(u)
        return nn.functional.one_hot(a // 3, 3).float(), nn.functional.one_hot(a % 3, 3).float()


class QNetwork(nn.Module):
    """PQN Q-network: MLP with LayerNorm after each hidden layer (Gallici et al. 2025; CleanRL pqn.py)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 256), n_actions: int = 9):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)
