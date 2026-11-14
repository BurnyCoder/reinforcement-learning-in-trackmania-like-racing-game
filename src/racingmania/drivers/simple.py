"""Training-free baseline drivers: uniformly random actions, and a fixed action.

Global context: the random driver is the floor of every comparison (reports/plan.md, run table,
"Training-free"); "untrained network" baselines are PPO agents evaluated at initialisation and live
in ``rl/``. Both drivers follow the evaluator's policy interface: ``policy(env) -> actions [N]``.
"""

from __future__ import annotations

import torch


class RandomDriver:
    """Uniformly random action (0..8) every agent step, from its own seeded generator."""

    def __init__(self, seed: int = 0, device="cuda"):
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, env):
        return torch.randint(0, 9, (env.num_envs,), generator=self.gen, device=env.device)


class ConstantDriver:
    """The same action every step (e.g. 3 = no steering + gas), used in tests and sanity checks."""

    def __init__(self, action: int = 3):
        self.action = action

    def __call__(self, env):
        return torch.full((env.num_envs,), self.action, device=env.device, dtype=torch.long)
