"""One physics tick for N cars: the batched equivalent of ``RaceSim.tick`` in ``web/js/core/sim.js``.

Global context: per tick the browser game (1) maps the action to wheel inputs, (2) calls
``world.step(dt)`` — gravity, velocity damping, ``RaycastVehicle.updateVehicle``, ``Body.integrate``,
in that order (docs/physics.md) — and (3) updates the race rules. ``physics_tick`` does the same for
a batch; ``agent_step`` repeats it ``ticks_per_action`` times with one action, freezing cars whose
run ended mid-step, as ``RaceSim.step`` stops early.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .controls import decode_action, wheel_inputs
from .math3d import dot, inv_inertia_world, quat_integrate, quat_normalize, quat_rotate
from .raycast_vehicle import update_vehicle
from .rules import RuleState, update_rules


@dataclass
class CarState:
    """Physical state of N cars: the 13-number chassis state plus the rate-limited steering angle."""

    pos: Tensor  # [N, 3]
    quat: Tensor  # [N, 4] (x, y, z, w)
    vel: Tensor  # [N, 3]
    angvel: Tensor  # [N, 3]
    steer: Tensor  # [N] front-wheel angle (rad)

    @classmethod
    def from_flat(cls, s: Tensor, steer: Tensor) -> "CarState":
        """From ``[N, 13]`` = ``[px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]`` (physics.js layout)."""
        return cls(pos=s[:, 0:3], quat=s[:, 3:7], vel=s[:, 7:10], angvel=s[:, 10:13], steer=steer)

    def flat(self) -> Tensor:
        return torch.cat((self.pos, self.quat, self.vel, self.angvel), dim=1)

    def where(self, mask: Tensor, other: "CarState") -> "CarState":
        """Take ``self`` where ``mask`` [N] is True, else ``other``."""
        m3 = mask.unsqueeze(1)
        return CarState(pos=torch.where(m3, self.pos, other.pos), quat=torch.where(m3, self.quat, other.quat),
                        vel=torch.where(m3, self.vel, other.vel), angvel=torch.where(m3, self.angvel, other.angvel),
                        steer=torch.where(mask, self.steer, other.steer))


def forward_speed(car: CarState) -> Tensor:
    """physics.js ``forwardSpeed``: chassis +X rotated to world, dotted with the velocity."""
    ex = torch.zeros_like(car.pos); ex[:, 0] = 1.0
    return dot(quat_rotate(car.quat, ex), car.vel)


def physics_tick(cfg, tracks, track_id: Tensor, index: Tensor, car: CarState, steer_cmd: Tensor, pedal: Tensor):
    """Controls + one ``World.internalStep(dt)``. Returns ``(new_car, contacts [N] long, wheel_contact)``."""
    dt = cfg.dt
    speed = forward_speed(car)
    steer_angle, engine, brake = wheel_inputs(cfg.controls, dt, car.steer, steer_cmd, pedal, speed)

    g = torch.tensor(cfg.gravity, device=car.pos.device, dtype=car.pos.dtype)
    force = cfg.mass * g  # World.internalStep: f += m * g (force starts at zero)
    vel = car.vel * cfg.linear_damping_factor  # v.scale(pow(1 - linearDamping, dt))
    angvel = car.angvel * cfg.angular_damping_factor
    inv_inertia = torch.tensor(cfg.inv_inertia, device=car.pos.device, dtype=car.pos.dtype)
    inv_iw = inv_inertia_world(car.quat, inv_inertia)  # as left by the previous integrate (or reset)
    vel, angvel, wc = update_vehicle(cfg, tracks, track_id, index, car.pos, car.quat, vel, angvel, inv_iw,
                                     steer_angle, engine, brake)

    # Body.integrate: v += f * (invMass*dt); ω += dt * (I⁻¹·τ) with τ = 0 (no-op); x += v*dt;
    # q += dt/2 (ω ⊗ q); normalise every tick (quatNormalizeSkip = 0).
    i_mdt = cfg.inv_mass * dt
    vel = vel + force * i_mdt * 1.0
    pos = car.pos + vel * dt
    quat = quat_normalize(quat_integrate(car.quat, angvel, dt))
    new = CarState(pos=pos, quat=quat, vel=vel, angvel=angvel, steer=steer_angle)
    return new, wc.in_contact.sum(dim=1), wc


def agent_step(cfg, tracks, track_id: Tensor, car: CarState, rules: RuleState, action: Tensor):
    """``ticks_per_action`` ticks with the same action. Cars already done stay frozen.

    Returns ``(car, rules, gained, wheel_contact)``: ``gained`` is the progress summed over the step
    (metres); ``wheel_contact [N, 4]`` (bool) is each wheel's contact on the car's last active tick.
    """
    steer_cmd, pedal = decode_action(action)
    gained = torch.zeros_like(car.pos[:, 0])
    last_contact = None
    for _ in range(cfg.ticks_per_action):
        active = ~(rules.finished | rules.out)
        new_car, contacts, wc = physics_tick(cfg, tracks, track_id, rules.index, car, steer_cmd, pedal)
        car = new_car.where(active, car)
        rules, g = update_rules(tracks, track_id, rules, car.pos, contacts, active)
        gained = gained + g
        last_contact = wc.in_contact if last_contact is None else torch.where(active.unsqueeze(1), wc.in_contact, last_contact)
    return car, rules, gained, last_contact
