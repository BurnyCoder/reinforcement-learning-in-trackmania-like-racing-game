"""Running observation normaliser (mean/variance with clipping), frozen at evaluation and export.

Global context: observation normalisation with clipping to [-10, 10] is one of the PPO details that
matter for continuous observations (Huang et al. 2022, "The 37 Implementation Details of PPO").
Statistics are updated from training batches with the parallel variance formula (Chan et al.,
as in OpenAI Baselines' ``RunningMeanStd``) and stored in checkpoints and inside the ONNX graph.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip, self.eps = clip, eps
        self.frozen = False

    @torch.no_grad()
    def update(self, x: Tensor):
        """Merge the statistics of batch ``x [B, dim]`` (Chan et al. parallel update)."""
        if self.frozen:
            return
        x = x.reshape(-1, x.shape[-1]).to(self.mean.dtype)
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps), -self.clip, self.clip)
