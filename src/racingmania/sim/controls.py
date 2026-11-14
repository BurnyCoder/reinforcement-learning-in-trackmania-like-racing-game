"""Action → wheel inputs, a line-by-line port of ``web/js/core/controls.js``.

Global context: cannon-es has no engine, rolling resistance or top speed, so this mapping is part
of the physics contract (docs/formats.md). Human players, the in-browser AI and the training
simulator all go through it; the parity tests check it together with the physics.
"""

from __future__ import annotations

import torch
from torch import Tensor

STEER_LEFT, STEER_NONE, STEER_RIGHT = 0, 1, 2
PEDAL_GAS, PEDAL_COAST, PEDAL_BRAKE = 0, 1, 2


def decode_action(a: Tensor) -> tuple[Tensor, Tensor]:
    """Integer action 0..8 → (steer, pedal) with ``a = steer*3 + pedal``."""
    return torch.div(a, 3, rounding_mode="floor"), a % 3


def wheel_inputs(ctl: dict, dt: float, steer_angle: Tensor, steer: Tensor, pedal: Tensor, speed: Tensor):
    """Return ``(new_steer_angle, engine, brake)`` per car, mirroring ``wheelInputs`` in controls.js.

    ``speed`` is the chassis velocity projected on the chassis forward axis before the tick.
    ``engine`` is the force per drive wheel (N), ``brake`` the impulse per wheel per tick (N s).
    """
    abs_speed = speed.abs()
    one = torch.ones_like(speed)
    zero = torch.zeros_like(speed)
    steer_scale = torch.maximum(ctl["min_steer_fraction"] * one, 1 - ctl["steer_speed_falloff"] * abs_speed / ctl["top_speed"])
    direction = torch.where(steer == STEER_LEFT, one, torch.where(steer == STEER_RIGHT, -one, zero))
    target = direction * ctl["max_steer"] * steer_scale
    max_delta = ctl["steer_rate"] * dt
    delta = torch.clamp(target - steer_angle, -max_delta, max_delta)  # Math.min(Math.max(x, -m), m)
    new_steer = steer_angle + delta

    gas_engine = ctl["engine_force"] * torch.maximum(zero, 1 - speed / ctl["top_speed"])
    reverse_engine = -ctl["reverse_force"] * torch.maximum(zero, 1 + speed / ctl["reverse_top_speed"])
    moving_forward = speed > ctl["reverse_speed_threshold"]
    is_gas, is_brake = pedal == PEDAL_GAS, pedal == PEDAL_BRAKE
    engine = torch.where(is_gas, gas_engine, torch.where(is_brake & ~moving_forward, reverse_engine, zero))
    brake = torch.where(is_gas, zero, torch.where(is_brake, torch.where(moving_forward, ctl["brake_impulse"] * one, zero),
                                                  ctl["coast_brake_impulse"] * one))
    return new_steer, engine, brake
