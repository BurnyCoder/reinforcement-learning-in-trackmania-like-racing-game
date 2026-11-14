"""Observation function: raw game quantities → the policy's input vector.

Global context: the same function runs in training (``vec_env.py``) and inside the exported ONNX
graph used by the browser (``export/onnx_export.py``, interface in docs/formats.md "Policy model"),
so the browser never rebuilds observations. Its inputs are things the game has at every agent
step: the chassis state, the steering angle, which wheels touch the road, the previous action, and
a window of upcoming centreline samples. Design follows prior racing agents that observe the car's
own motion plus upcoming track geometry in the car's frame (Linesight, GT Sophy, Fuchs et al.
2021; reports/literature-review.md, section 4).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..sim.math3d import quat_conj, quat_rotate

WINDOW_K = 20  # number of upcoming centreline samples observed
WINDOW_STRIDE = 3  # samples between observed points (3 × 2 m = 6 m; 20 points cover 114 m ahead)

# Fixed scales that bring raw inputs near unit range before the running normaliser.
VEL_SCALE = 1 / 20.0
ANGVEL_SCALE = 1 / 2.0
POS_SCALE = 1 / 50.0


def obs_size(k: int = WINDOW_K) -> int:
    """Length of the observation vector for a window of ``k`` points."""
    return 3 + 3 + 3 + 1 + 4 + 6 + 3 * k + 3 * k + k


def observe(chassis: Tensor, steer: Tensor, contacts: Tensor, prev_action: Tensor, win_center: Tensor,
            win_left: Tensor, win_gap: Tensor, max_steer: float) -> Tensor:
    """Build observations ``[B, obs_size(K)]``.

    ``chassis [B,13]``, ``steer [B,1]``, ``contacts [B,4]`` (0/1), ``prev_action [B,2]`` (steer, pedal
    indices as floats), ``win_center [B,K,3]``, ``win_left [B,K,3]``, ``win_gap [B,K]``.
    Every geometric quantity is expressed in the chassis frame, so the observation does not depend
    on where the track is in the world or which way it faces.
    """
    pos, quat, vel, angvel = chassis[:, 0:3], chassis[:, 3:7], chassis[:, 7:10], chassis[:, 10:13]
    qc = quat_conj(quat)  # world → chassis frame (cannon-es Body.vectorToLocalFrame)
    k = win_center.shape[1]
    v_local = quat_rotate(qc, vel) * VEL_SCALE  # velocity: forward, up, right components
    w_local = quat_rotate(qc, angvel) * ANGVEL_SCALE  # spin rates about the chassis axes
    down = torch.zeros_like(vel); down[:, 1] = -1.0
    g_local = quat_rotate(qc, down)  # gravity direction: tells pitch and roll
    qk = qc.unsqueeze(1).expand(-1, k, 4)
    pts = quat_rotate(qk, win_center - pos.unsqueeze(1)) * POS_SCALE  # upcoming centreline in the car frame
    lft = quat_rotate(qk, win_left)  # road's left direction there (encodes heading and banking)
    steer_n = steer / max_steer
    choices = torch.arange(3, device=chassis.device, dtype=chassis.dtype)
    one_hot = torch.cat(((prev_action[:, 0:1] == choices).to(chassis.dtype),  # comparison, not one_hot(): exports to ONNX
                         (prev_action[:, 1:2] == choices).to(chassis.dtype)), dim=1)
    return torch.cat((v_local, w_local, g_local, steer_n, contacts, one_hot,
                      pts.flatten(1), lft.flatten(1), win_gap), dim=1)


def window_indices(index: Tensor, n: Tensor, k: int = WINDOW_K, stride: int = WINDOW_STRIDE) -> Tensor:
    """Sample indices ``min(index + stride*j, n-1)`` for ``j = 0..k-1``; ``[B, k]``."""
    j = torch.arange(k, device=index.device) * stride
    return torch.minimum(index.unsqueeze(1) + j, (n - 1).unsqueeze(1))
