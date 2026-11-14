"""Scripted driver: pure-pursuit steering with a curvature-based speed profile, on discrete actions.

Global context: this classical driver certifies generated tracks (a track joins a set only if it
finishes at three speed settings) and provides the reference time ``T_ref`` of the primary metric
(reports/preregistration.md, section 4). It never trains anything and the RL agent never sees it,
except in the imitation/hybrid phase.

Method:
- Steering: pure pursuit (Coulter 1992, CMU-RI-TR-92-01): aim at a point on the centreline a
  look-ahead distance ahead; curvature ``κ = 2·y / d²`` toward it (y = lateral offset of the point in
  the car frame, d = distance), steering angle ``atan(wheelbase · κ)``. Because the game only has
  left/none/right, the driver picks the command whose rate-limited steering angle after one agent
  step is closest to the desired angle.
- Speed: maximum cornering speed ``sqrt(a_lat / κ)`` per sample, capped at top speed and at
  ``jump_speed`` around jumps, then a backward pass limiting deceleration to ``a_brake`` (the
  forward/backward speed-profile idea used in racing-line tools such as TUM's
  trajectory_planning_helpers, https://github.com/TUMFTM/trajectory_planning_helpers).
- Pedal: gas below the target speed, brake well above it, coast in between.
``a_lat`` and ``lookahead_gain`` are the two parameters tuned on validation tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..sim.controls import STEER_LEFT, STEER_NONE, STEER_RIGHT
from ..sim.math3d import quat_conj, quat_rotate


@dataclass(frozen=True)
class ScriptedParams:
    a_lat: float = 10.0  # m/s² lateral acceleration the speed profile allows (tuned on validation)
    lookahead_gain: float = 0.45  # s: pure-pursuit look-ahead distance = gain × speed (tuned on validation)
    lookahead_min: float = 8.0  # m
    lookahead_max: float = 40.0  # m
    a_brake: float = 8.0  # m/s² deceleration the speed profile assumes
    jump_speed: float = 25.0  # m/s: speed cap from 30 m before a gap to its end
    brake_margin: float = 1.5  # m/s above target before braking
    speed_scale: float = 1.0  # certification runs at 0.9, 1.0 and 1.1


class ScriptedDriver:
    """Batched scripted driver bound to one environment's loaded tracks."""

    def __init__(self, env, params: ScriptedParams = ScriptedParams()):
        self.env = env
        self.p = params
        c = env.cfg
        self.wheelbase = c.connections[0][0] - c.connections[2][0]  # front axle x − rear axle x
        self.profile = self._speed_profile(env.tracks)

    def _speed_profile(self, tb) -> Tensor:
        """Target speed at every sample of every loaded track, ``[T, M]``."""
        p, ctl = self.p, self.env.cfg.controls
        c = tb.center
        d = c[:, 1:] - c[:, :-1]
        heading = torch.atan2(-d[..., 2], d[..., 0])  # horizontal direction of each segment
        dpsi = torch.remainder(heading[:, 1:] - heading[:, :-1] + torch.pi, 2 * torch.pi) - torch.pi
        seglen = torch.linalg.norm(d[..., [0, 2]], dim=-1).clamp(min=1e-6)
        kappa = (dpsi.abs() / seglen[:, 1:]).clamp(min=0)
        kappa = torch.nn.functional.pad(kappa, (1, 1), value=0.0)  # back to M samples
        kappa = torch.nn.functional.avg_pool1d(kappa.unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
        v = torch.sqrt(p.a_lat / kappa.clamp(min=1e-6)).clamp(max=ctl["top_speed"])
        # Jumps: cap the speed from 15 samples before each gap segment to the end of the gap.
        gap = torch.nn.functional.pad((~tb.seg_valid).float() * self._real_segments(tb), (0, 1))
        near_gap = torch.nn.functional.max_pool1d(torch.nn.functional.pad(gap, (0, 15)).unsqueeze(1), 16, stride=1).squeeze(1) > 0
        v = torch.where(near_gap, torch.clamp(v, max=p.jump_speed), v)
        # Backward pass: v[i] <= sqrt(v[i+1]² + 2 a_brake ds).
        ds = torch.nn.functional.pad(torch.linalg.norm(d, dim=-1), (0, 1))
        for i in range(v.shape[1] - 2, -1, -1):
            v[:, i] = torch.minimum(v[:, i], torch.sqrt(v[:, i + 1] ** 2 + 2 * p.a_brake * ds[:, i]))
        return v

    @staticmethod
    def _real_segments(tb) -> Tensor:
        M = tb.center.shape[1]
        return (torch.arange(M - 1, device=tb.center.device) < (tb.n - 1).unsqueeze(1)).float()

    def targets(self, env=None) -> tuple[Tensor, Tensor, Tensor]:
        """The driver's targets for every car: ``(target speed, target steering angle, forward speed)``.
        Used by the reward-hint hybrid (phase 4) as well as by ``__call__``."""
        env = env or self.env
        p, ctl, tb = self.p, env.cfg.controls, env.tracks
        tid, idx, car = env.track_id, env.rules.index, env.car
        n = tb.n[tid]
        qc = quat_conj(car.quat)
        speed = quat_rotate(qc, car.vel)[:, 0]
        dist = torch.clamp(p.lookahead_gain * speed.abs(), p.lookahead_min, p.lookahead_max)
        j = torch.minimum(idx + torch.round(dist / 2.0).long(), n - 1)
        rel = quat_rotate(qc, tb.center[tid, j] - car.pos)
        lat = -rel[:, 2]
        delta = torch.atan(self.wheelbase * 2 * lat / (rel[:, 0] ** 2 + lat ** 2).clamp(min=1e-6))
        ahead = torch.minimum(idx.unsqueeze(1) + torch.arange(0, 6, device=idx.device), (n - 1).unsqueeze(1))
        v_target = self.profile[tid.unsqueeze(1), ahead].min(1).values * p.speed_scale
        return v_target, delta, speed

    def __call__(self, env=None) -> Tensor:
        """Actions ``[N]`` for the environment's current state."""
        env = env or self.env
        p, ctl, tb = self.p, env.cfg.controls, env.tracks
        tid, idx = env.track_id, env.rules.index
        car = env.car
        n = tb.n[tid]
        qc = quat_conj(car.quat)
        v_local = quat_rotate(qc, car.vel)
        speed = v_local[:, 0]
        # Pure pursuit target on the centreline.
        dist = torch.clamp(p.lookahead_gain * speed.abs(), p.lookahead_min, p.lookahead_max)
        off = torch.round(dist / 2.0).long()
        j = torch.minimum(idx + off, n - 1)
        rel = quat_rotate(qc, tb.center[tid, j] - car.pos)
        lat = -rel[:, 2]  # left of the car is chassis −Z
        d2 = rel[:, 0] ** 2 + lat ** 2
        delta = torch.atan(self.wheelbase * 2 * lat / d2.clamp(min=1e-6))
        # Choose the command whose angle after one agent step is closest to delta.
        scale = torch.clamp(1 - ctl["steer_speed_falloff"] * speed.abs() / ctl["top_speed"], min=ctl["min_steer_fraction"])
        step = ctl["steer_rate"] * env.cfg.dt * env.cfg.ticks_per_action
        cands = []
        for direction in (1.0, 0.0, -1.0):  # left, none, right
            target = direction * ctl["max_steer"] * scale
            nxt = car.steer + torch.clamp(target - car.steer, -step, step)
            cands.append((nxt - delta).abs())
        steer_cmd = torch.stack(cands, 1).argmin(1)  # 0 left, 1 none, 2 right (same order as actions)
        # Pedal from the speed profile a little ahead.
        ahead = torch.minimum(idx.unsqueeze(1) + torch.arange(0, 6, device=idx.device), (n - 1).unsqueeze(1))
        v_target = self.profile[tid.unsqueeze(1), ahead].min(1).values * p.speed_scale
        pedal = torch.where(speed < v_target, 0, torch.where(speed > v_target + p.brake_margin, 2, 1))
        return steer_cmd * 3 + pedal


def assert_action_order():
    """The candidate loop above relies on the action order left/none/right = 0/1/2."""
    assert (STEER_LEFT, STEER_NONE, STEER_RIGHT) == (0, 1, 2)
