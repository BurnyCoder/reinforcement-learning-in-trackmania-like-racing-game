"""Batched port of cannon-es ``RaycastVehicle.updateVehicle`` (plus ``WheelInfo`` and the friction helpers).

Global context: this runs inside each physics tick, after velocity damping and before integration,
exactly where cannon-es's ``preStep`` event calls it (docs/physics.md). It changes the chassis
linear and angular velocity by suspension and friction impulses. Impulses are applied one wheel at
a time, in wheel order, because ``Body.applyImpulse`` updates the velocities immediately and the
floating-point sums must happen in the same order as in cannon-es.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .math3d import (cross, dot, length_squared, mat_vec, normalize, quat_conj, quat_from_axis_angle, quat_mult,
                     quat_normalize, quat_rotate)
from .ray_trimesh import raycast_closest


@dataclass
class WheelContact:
    """Per-wheel results of one ``updateVehicle`` call, ``[N, 4]`` (and ``[N, 4, 3]`` for vectors)."""

    in_contact: Tensor  # wheel.isInContact
    hit_point: Tensor  # raycastResult.hitPointWorld
    hit_normal: Tensor  # raycastResult.hitNormalWorld
    suspension_length: Tensor
    suspension_force: Tensor  # uncapped, as stored by updateSuspension
    sliding: Tensor  # wheel.sliding


def _apply_impulse(vel: Tensor, angvel: Tensor, inv_mass: float, inv_iw: Tensor, impulse: Tensor, rel: Tensor):
    """Body.applyImpulse: ``v += impulse*invMass``; ``ω += invInertiaWorld · (rel × impulse)``."""
    vel = vel + impulse * inv_mass
    angvel = angvel + mat_vec(inv_iw, cross(rel, impulse))
    return vel, angvel


def _velocity_at(vel: Tensor, angvel: Tensor, pos: Tensor, point: Tensor) -> Tensor:
    """Body.getVelocityAtWorldPoint: ``v + ω × (point - position)``."""
    return vel + cross(angvel, point - pos)


def update_vehicle(cfg, tracks, track_id: Tensor, index: Tensor, pos: Tensor, quat: Tensor, vel: Tensor,
                   angvel: Tensor, inv_iw: Tensor, steer: Tensor, engine: Tensor, brake: Tensor):
    """One ``RaycastVehicle.updateVehicle(dt)`` for N cars.

    ``cfg``: VehicleConfig; ``tracks``: TrackBatch; ``track_id``/``index`` ``[N]``: each car's track and
    nearest centreline sample (for the triangle window); ``pos [N,3]``, ``quat [N,4]``, ``vel``,
    ``angvel [N,3]``; ``inv_iw [N,3,3]`` world inverse inertia; ``steer [N]`` front-wheel angle;
    ``engine [N]`` force on each drive wheel; ``brake [N]`` impulse on each wheel.
    Returns the new ``(vel, angvel, WheelContact)``.
    """
    dt = cfg.dt
    w = cfg.wheel
    dev, dt_ = pos.device, pos.dtype
    N, W = pos.shape[0], len(cfg.connections)
    conn_local = cfg.connections_tensor(dev, dt_)  # [W, 3]
    dir_local = torch.tensor(w["direction"], device=dev, dtype=dt_)
    axle_local = torch.tensor(w["axle"], device=dev, dtype=dt_)
    zeros = torch.zeros(N, device=dev, dtype=dt_)
    front = torch.zeros(W, dtype=torch.bool, device=dev); front[list(cfg.front_wheels)] = True
    drive = torch.zeros(W, dtype=torch.bool, device=dev); drive[list(cfg.drive_wheels)] = True
    steer_w = torch.where(front, steer.unsqueeze(1), zeros.unsqueeze(1))  # [N, W] RaycastVehicle.setSteeringValue
    engine_w = torch.where(drive, engine.unsqueeze(1), zeros.unsqueeze(1))  # applyEngineForce
    brake_w = brake.unsqueeze(1).expand(N, W)  # setBrake on every wheel

    # --- updateWheelTransformWorld + updateWheelTransform: wheel frames in world space --------------
    q4 = quat.unsqueeze(1).expand(N, W, 4)
    conn_world = quat_rotate(q4, conn_local.expand(N, W, 3)) + pos.unsqueeze(1)  # pointToWorldFrame
    dir_world = quat_rotate(q4, dir_local.expand(N, W, 3))  # vectorToWorldFrame(directionLocal)
    up = -dir_local  # updateWheelTransform: up = directionLocal * -1
    steer_q = quat_from_axis_angle(up.expand(N, W, 3), steer_w)  # rotation about up by the steering angle
    # chassis.quaternion.mult(steeringOrn), then the spin about the axle (identity: the spin angle is
    # visual only because axleLocal is the rotation axis, docs/physics.md), then normalize.
    wheel_q = quat_normalize(quat_mult(q4, steer_q))

    # --- castRay: one ray per wheel against the triangles near the car -------------------------------
    raylen = w["suspension_rest_length"] + w["radius"]
    origin = conn_world
    target = conn_world + dir_world * raylen  # source + directionWorld*raylen
    a, b, c, n, valid = tracks.window_triangles(track_id, index)  # [N, K, 3]
    K = a.shape[1]
    rep = lambda t: t.unsqueeze(1).expand(N, W, *t.shape[1:]).reshape(N * W, *t.shape[1:])
    hit, dist, hit_point, hit_normal = raycast_closest(origin.reshape(-1, 3), target.reshape(-1, 3), rep(a), rep(b),
                                                      rep(c), rep(n), rep(valid))
    hit = hit.view(N, W); dist = dist.view(N, W); hit_point = hit_point.view(N, W, 3); hit_normal = hit_normal.view(N, W, 3)

    rest = w["suspension_rest_length"]
    min_len = rest - w["max_suspension_travel"]
    susp_len = torch.where(hit, torch.maximum(dist - w["radius"], torch.full_like(dist, min_len)), torch.full_like(dist, rest))
    denominator = dot(hit_normal, dir_world)  # hitNormalWorld · directionWorld
    v_at = _velocity_at(vel.unsqueeze(1), angvel.unsqueeze(1), pos.unsqueeze(1), hit_point)
    proj_vel = dot(hit_normal, v_at)
    steep = denominator >= -0.1
    inv = -1 / torch.where(steep, -torch.ones_like(denominator), denominator)
    srv = torch.where(hit, torch.where(steep, torch.zeros_like(proj_vel), proj_vel * inv), torch.zeros_like(proj_vel))
    clipped = torch.where(hit, torch.where(steep, torch.full_like(inv, 1 / 0.1), inv), torch.ones_like(inv))
    hit_normal = torch.where(hit.unsqueeze(-1), hit_normal, dir_world * -1)  # no hit: directionWorld.scale(-1)

    # --- updateSuspension ------------------------------------------------------------------------
    force = w["suspension_stiffness"] * (rest - susp_len) * clipped
    damping = torch.where(srv < 0, torch.full_like(srv, w["damping_compression"]), torch.full_like(srv, w["damping_relaxation"]))
    force = force - damping * srv
    susp_force = force * cfg.mass
    susp_force = torch.where(hit & ~(susp_force < 0), susp_force, torch.zeros_like(susp_force))  # floor at 0; 0 off the ground

    # --- suspension impulses, wheel by wheel -------------------------------------------------------
    for i in range(W):
        sf = torch.minimum(susp_force[:, i], torch.full_like(susp_force[:, i], w["max_suspension_force"]))
        imp = hit_normal[:, i] * (sf * dt).unsqueeze(-1)
        vel, angvel = _apply_impulse(vel, angvel, cfg.inv_mass, inv_iw, imp, hit_point[:, i] - pos)

    # --- updateFriction ---------------------------------------------------------------------------
    axle = quat_rotate(wheel_q, torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt_).expand(N, W, 3))  # directions[indexRightAxis]
    proj = dot(axle, hit_normal)
    axle = normalize(axle - hit_normal * proj.unsqueeze(-1))
    forward_ws = normalize(cross(hit_normal, axle))
    # resolveSingleBilateral against the static ground (invMass 0, velocity 0).
    v_at = _velocity_at(vel.unsqueeze(1), angvel.unsqueeze(1), pos.unsqueeze(1), hit_point)
    rel_vel = dot(axle, v_at - 0.0)
    mass_term = 1 / (cfg.inv_mass + 0.0)
    side = torch.where(length_squared(axle) > 1.1, torch.zeros_like(rel_vel), -0.2 * rel_vel * mass_term)
    side = torch.where(hit, side * 1.0, torch.zeros_like(side))  # sideFrictionStiffness2 = 1

    # calcRollingFriction with maxImpulse = brake (0 when not braking).
    vrel = dot(forward_ws, v_at - 0.0)
    r0 = hit_point - pos.unsqueeze(1)
    denom0 = cfg.inv_mass + dot(forward_ws, cross(mat_vec(inv_iw.unsqueeze(1), cross(r0, forward_ws)), r0))
    jac = 1.0 / (denom0 + 0.0)
    j1 = -vrel * jac
    max_imp = brake_w
    j1 = torch.where(max_imp < j1, max_imp, j1)
    j1 = torch.where(j1 < -max_imp, -max_imp, j1)
    rolling = j1 + engine_w * dt
    forward_imp = torch.where(hit, rolling, torch.zeros_like(rolling))

    maximp = susp_force * dt * w["friction_slip"]
    x = forward_imp * 0.5 / w["forward_acceleration"]
    y = side * 1 / w["side_acceleration"]
    imp_sq = x * x + y * y
    sliding = hit & (imp_sq > maximp * maximp)
    skid = torch.where(sliding, maximp / torch.sqrt(torch.where(sliding, imp_sq, torch.ones_like(imp_sq))), torch.ones_like(imp_sq))
    any_sliding = sliding.any(dim=1, keepdim=True)
    scale = any_sliding & (side != 0) & (skid < 1)
    forward_imp = torch.where(scale, forward_imp * skid, forward_imp)
    side = torch.where(scale, side * skid, side)

    # --- apply friction impulses, wheel by wheel: forward then side --------------------------------
    for i in range(W):
        rel = hit_point[:, i] - pos
        vel, angvel = _apply_impulse(vel, angvel, cfg.inv_mass, inv_iw, forward_ws[:, i] * forward_imp[:, i].unsqueeze(-1), rel)
        side_imp = axle[:, i] * side[:, i].unsqueeze(-1)
        rel_local = quat_rotate(quat_conj(quat), rel)  # vectorToLocalFrame
        rel_local = torch.stack((rel_local[:, 0], rel_local[:, 1] * w["roll_influence"], rel_local[:, 2]), dim=-1)
        rel_side = quat_rotate(quat, rel_local)  # back to world
        vel, angvel = _apply_impulse(vel, angvel, cfg.inv_mass, inv_iw, side_imp, rel_side)

    contact = WheelContact(in_contact=hit, hit_point=hit_point, hit_normal=hit_normal, suspension_length=susp_len,
                           suspension_force=susp_force, sliding=sliding)
    return vel, angvel, contact
