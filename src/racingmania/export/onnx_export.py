"""Export a trained policy as one ONNX graph: observation function + frozen normaliser + actor.

Global context: the browser game runs trained agents with onnxruntime-web (docs/formats.md, "Policy
model"). Putting the observation function inside the graph means JavaScript only gathers raw
inputs, so the browser cannot drift from training (the approach of mjswan, which exports
observation terms to ONNX: https://mjswan.readthedocs.io/en/latest/guides/how-it-works/).
Export uses ``torch.onnx.export`` (dynamo exporter, the default in PyTorch 2.15, opset 20); the
check compares onnxruntime's actions with PyTorch's on recorded inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..env.obs import observe

INPUT_NAMES = ["chassis", "steer", "contacts", "prev_action", "win_center", "win_left", "win_gap"]
OUTPUT_NAMES = ["steer_logits", "pedal_logits"]


class ExportedPolicy(nn.Module):
    """Raw game inputs → (steer_logits, pedal_logits)."""

    def __init__(self, agent, norm, max_steer: float):
        super().__init__()
        self.agent, self.norm, self.max_steer = agent, norm, max_steer

    def forward(self, chassis, steer, contacts, prev_action, win_center, win_left, win_gap):
        obs = observe(chassis, steer, contacts, prev_action, win_center, win_left, win_gap, self.max_steer)
        return self.agent.logits_for_export(self.norm(obs))


def export_policy(agent, norm, env, path: Path, n_check: int = 2048) -> dict:
    """Export to ``path`` and check onnxruntime against PyTorch on inputs from ``env``'s current cars.

    Returns ``{"agreement": share of identical actions, "max_abs_logit_diff": ...}``.
    """
    import onnxruntime as ort
    module = ExportedPolicy(agent.cpu().float().eval(), norm.cpu().float().eval(), env.cfg.controls["max_steer"])
    raw = {k: v[:n_check].detach().float().cpu() for k, v in env.raw_inputs().items()}
    example = tuple(raw[k][:1] for k in INPUT_NAMES)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, example, str(path), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamo=True,
                      external_data=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    agree, max_diff = 0, 0.0
    n = raw["chassis"].shape[0]
    with torch.no_grad():
        for i in range(n):
            feed = {k: raw[k][i:i + 1].numpy() for k in INPUT_NAMES}
            s_onnx, p_onnx = sess.run(OUTPUT_NAMES, feed)
            s_t, p_t = module(*[raw[k][i:i + 1] for k in INPUT_NAMES])
            agree += int(np.argmax(s_onnx) == int(s_t.argmax()) and np.argmax(p_onnx) == int(p_t.argmax()))
            max_diff = max(max_diff, float(np.abs(s_onnx - s_t.numpy()).max()), float(np.abs(p_onnx - p_t.numpy()).max()))
    return {"agreement": agree / n, "max_abs_logit_diff": max_diff, "n": n}
