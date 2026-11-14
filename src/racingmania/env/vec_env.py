"""Vectorised racing environment on the GPU: N cars, each on its own track, stepping together.

Global context: this is the RL interface to the PyTorch port of the game physics (``sim/``). One
``step`` is one agent decision (5 physics ticks, docs/formats.md). The reward is the centreline
progress in metres (scaled), the only reward term in the preregistered recipe
(reports/preregistration.md, section 2). Episodes end by *termination* (finished, or out: fell,
airborne too long, stalled) or by *truncation* at a time limit; following Gymnasium's rule
(https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/), the trainer
bootstraps only on truncation, using the final observation returned in ``info["final_obs"]``
(same-step autoreset). Starts mix the official standing start with random points along the
centreline at random speeds (Song et al. 2021, 2023; Kaufmann et al. 2023).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..sim.math3d import cross, quat_from_basis
from ..sim.rules import RuleState, reset_rules
from ..sim.step import CarState, agent_step
from ..sim.track_tensors import TrackBatch
from .obs import WINDOW_K, WINDOW_STRIDE, observe, window_indices

OUT_REASONS = {0: "none", 1: "fell", 2: "airborne", 3: "stalled"}


@dataclass
class EnvConfig:
    """Environment settings that are part of the recipe (logged with every run)."""

    random_start_prob: float = 0.5  # share of resets at a random point along the track
    random_start_speed: tuple[float, float] = (0.0, 30.0)  # m/s
    random_start_lateral: float = 0.5  # max lateral offset as a fraction of the half width
    max_steps: int = 1500  # time limit in agent steps (75 s) for training episodes
    reward_scale: float = 0.1  # reward = progress (m) × reward_scale
    window_k: int = WINDOW_K
    window_stride: int = WINDOW_STRIDE
    blind: bool = False  # control: zero the upcoming-track points except the nearest one
    time_cost: float = 0.0  # ablation: subtracted from the reward every step
    fall_penalty: float = 0.0  # ablation: subtracted on termination by "out"
    hint_weight: float = 0.0  # phase-4 hybrid: reward for matching the scripted driver's speed and steering
    scripted_starts: bool = False  # phase-4 hybrid: random starts copied from cars driven by the scripted driver


class RacingVecEnv:
    """Batched environment. All tensors live on ``device``; ``dtype`` is the simulation precision."""

    def __init__(self, cfg, tracks: TrackBatch, num_envs: int, env_cfg: EnvConfig | None = None,
                 device="cuda", dtype=torch.float32, seed: int = 0, track_ids: Tensor | None = None,
                 assign: Tensor | None = None, auto_reset: bool = True, compile: bool = False):
        self.cfg = cfg  # VehicleConfig
        self.env_cfg = env_cfg or EnvConfig()
        self.device, self.dtype = torch.device(device), dtype
        self.num_envs = num_envs
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        # Evaluation mode: env i always drives track assign[i], and finished cars stay frozen.
        self.assign = None if assign is None else assign.to(self.device)
        self.auto_reset = auto_reset
        # torch.compile fuses the many small tensor operations of a physics tick; on Linux/WSL it
        # speeds the simulator up by about 10x (reports/plan.md, amendments).
        self._agent_step = torch.compile(agent_step) if compile else agent_step  # recompiles once with dynamic batch size, then reuses
        self.set_tracks(tracks, track_ids)
        self.hint_driver = None
        self.expert = None
        if self.env_cfg.scripted_starts:  # a second batch of cars driven by the scripted driver supplies start states
            from ..drivers.scripted import ScriptedDriver
            from ..tracks.build_sets import load_tuned
            ecfg = EnvConfig(**{**self.env_cfg.__dict__, "scripted_starts": False, "hint_weight": 0.0})
            self.expert = RacingVecEnv(cfg, tracks, min(num_envs, 4096), ecfg, device=device, dtype=dtype, seed=seed + 1,
                                       track_ids=track_ids, compile=compile)
            self.expert_driver = ScriptedDriver(self.expert, load_tuned())
        self.reset_all()

    # ------------------------------------------------------------------ tracks and resets
    def set_tracks(self, tracks: TrackBatch, track_ids: Tensor | None = None):
        """Use a new set of loaded tracks (episodes in progress must be reset by the caller).

        ``track_ids``: optional subset of track indices to sample from (training-set-size ablation).
        """
        self.tracks = tracks.to(self.device, self.dtype)
        T, M = self.tracks.center.shape[:2]
        self.allowed = (track_ids if track_ids is not None else torch.arange(T)).to(self.device)
        # Random-start candidates: road samples between the start and 10 samples before the finish
        # with no gap within ±5 samples.
        gap = ~self.tracks.seg_valid  # [T, M-1] True where a segment has no road (or is padding)
        gap_pad = torch.cat((gap, torch.ones(T, 1, dtype=torch.bool, device=self.device)), dim=1)
        near_gap = torch.nn.functional.max_pool1d(gap_pad.float().unsqueeze(1), 11, stride=1, padding=5).squeeze(1) > 0
        idx = torch.arange(M, device=self.device)
        finish = self.tracks.cp_s.gather(1, (self.tracks.n_cp - 1).unsqueeze(1))  # arc length of the finish
        before_finish = self.tracks.cum < (finish - 20.0)
        self.start_ok = ~near_gap & before_finish & (idx >= self.tracks.start_index.unsqueeze(1))
        if getattr(self, "expert", None) is not None:  # keep the scripted-start batch on the same tracks
            self.expert.set_tracks(tracks, track_ids)
            self.expert.reset_all()
            from ..drivers.scripted import ScriptedDriver
            self.expert_driver = ScriptedDriver(self.expert, self.expert_driver.p)

    def _sample_tracks(self, n: int) -> Tensor:
        pick = torch.randint(len(self.allowed), (n,), generator=self.gen, device=self.device)
        return self.allowed[pick]

    def _reset_cars(self, ids: Tensor):
        """Reset the cars ``ids`` [k]: new track, standing or random start."""
        k = ids.numel()
        if k == 0:
            return
        e = self.env_cfg
        tb = self.tracks
        tid = self.assign[ids] if self.assign is not None else self._sample_tracks(k)
        rnd = torch.rand(k, generator=self.gen, device=self.device) < e.random_start_prob
        # Random sample index among the allowed start samples of each track (inverse-CDF sampling).
        ok = self.start_ok[tid].float()
        cdf = ok.cumsum(1)
        u = torch.rand(k, generator=self.gen, device=self.device) * cdf[:, -1]
        i_rand = torch.searchsorted(cdf, u.unsqueeze(1).contiguous()).squeeze(1).clamp(max=tb.center.shape[1] - 2)
        i = torch.where(rnd, i_rand, tb.start_index[tid])
        c = tb.center[tid, i]
        nxt = tb.center[tid, i + 1]
        prv = tb.center[tid, (i - 1).clamp(min=0)]
        fwd = nxt - prv
        fwd = fwd / torch.linalg.norm(fwd, dim=1, keepdim=True)
        left = tb.left[tid, i]
        up = cross(fwd, left)
        up = up / torch.linalg.norm(up, dim=1, keepdim=True)
        lat = (torch.rand(k, generator=self.gen, device=self.device) * 2 - 1) * e.random_start_lateral * tb.half_width[tid]
        pos_rand = c + left * lat.unsqueeze(1) + up * self.cfg.spawn_height
        q_rand = quat_from_basis(fwd, up)
        speed = e.random_start_speed[0] + torch.rand(k, generator=self.gen, device=self.device) * (e.random_start_speed[1] - e.random_start_speed[0])
        r = rnd.unsqueeze(1)
        pos = torch.where(r, pos_rand, tb.start_pos[tid])
        quat = torch.where(r, q_rand, tb.start_quat[tid])
        vel = torch.where(r, fwd * speed.unsqueeze(1), torch.zeros_like(fwd))
        zero3 = torch.zeros_like(vel)
        self.track_id[ids] = tid
        self.car.pos[ids], self.car.quat[ids], self.car.vel[ids], self.car.angvel[ids] = pos, quat, vel, zero3
        self.car.steer[ids] = 0.0
        new_rules = reset_rules(tb, tid, pos, i)
        for f in RuleState.__dataclass_fields__:
            getattr(self.rules, f)[ids] = getattr(new_rules, f)
        self.prev_action[ids] = 1.0  # (none, coast)
        self.contacts[ids] = 0.0
        self.ep_steps[ids] = 0
        self.ep_return[ids] = 0.0
        self.ep_start_s[ids] = new_rules.s
        self.ep_standing[ids] = ~rnd
        if self.expert is not None:
            self._copy_expert_starts(ids[rnd])

    def _copy_expert_starts(self, ids: Tensor):
        """Overwrite cars ``ids`` with the states of random cars of the scripted-driver batch."""
        k = ids.numel()
        if k == 0:
            return
        src = torch.randint(self.expert.num_envs, (k,), generator=self.gen, device=self.device)
        self.track_id[ids] = self.expert.track_id[src]
        for f in ("pos", "quat", "vel", "angvel", "steer"):
            getattr(self.car, f)[ids] = getattr(self.expert.car, f)[src]
        for f in RuleState.__dataclass_fields__:
            getattr(self.rules, f)[ids] = getattr(self.expert.rules, f)[src]
        self.rules.ticks[ids] = 0
        self.ep_start_s[ids] = self.rules.s[ids]

    def reset_all(self) -> Tensor:
        """Reset every car and return the observations."""
        N, d, dt = self.num_envs, self.device, self.dtype
        self.track_id = torch.zeros(N, dtype=torch.long, device=d)
        z3 = torch.zeros(N, 3, device=d, dtype=dt)
        self.car = CarState(pos=z3.clone(), quat=torch.zeros(N, 4, device=d, dtype=dt), vel=z3.clone(), angvel=z3.clone(),
                            steer=torch.zeros(N, device=d, dtype=dt))
        zl = torch.zeros(N, dtype=torch.long, device=d)
        zb = torch.zeros(N, dtype=torch.bool, device=d)
        self.rules = RuleState(index=zl.clone(), s=torch.zeros(N, device=d, dtype=dt), anchor_s=torch.zeros(N, device=d, dtype=dt),
                               stall_ticks=zl.clone(), air_ticks=zl.clone(), next_cp=zl.clone(), ticks=zl.clone(),
                               finished=zb.clone(), out=zb.clone(), out_reason=zl.clone())
        self.prev_action = torch.ones(N, 2, device=d, dtype=dt)
        self.contacts = torch.zeros(N, 4, device=d, dtype=dt)
        self.ep_steps = zl.clone()
        self.ep_return = torch.zeros(N, device=d, dtype=dt)
        self.ep_start_s = torch.zeros(N, device=d, dtype=dt)
        self.ep_standing = zb.clone()
        self.reported = zb.clone()
        self._reset_cars(torch.arange(N, device=d))
        return self.observe()

    # ------------------------------------------------------------------ observations
    def raw_inputs(self) -> dict:
        """The raw policy inputs of docs/formats.md "Policy model" for every car."""
        tb, e = self.tracks, self.env_cfg
        tid = self.track_id
        win = window_indices(self.rules.index, tb.n[tid], e.window_k, e.window_stride)  # [N, K]
        t2 = tid.unsqueeze(1).expand_as(win)
        seg = win.clamp(max=tb.seg_valid.shape[1] - 1)
        gap = (~tb.seg_valid[t2, seg]).to(self.dtype) * (win < (tb.n[tid] - 1).unsqueeze(1)).to(self.dtype)
        win_center = tb.center[t2, win]
        if e.blind:  # blind control: every observed point collapses onto the nearest sample
            win_center = win_center[:, :1].expand_as(win_center)
            win_left = tb.left[t2[:, :1], win[:, :1]].expand_as(win_center)
            gap = gap[:, :1].expand_as(gap)
        else:
            win_left = tb.left[t2, win]
        return {"chassis": self.car.flat(), "steer": self.car.steer.unsqueeze(1), "contacts": self.contacts,
                "prev_action": self.prev_action, "win_center": win_center, "win_left": win_left, "win_gap": gap}

    def observe(self) -> Tensor:
        r = self.raw_inputs()
        return observe(r["chassis"], r["steer"], r["contacts"], r["prev_action"], r["win_center"], r["win_left"],
                       r["win_gap"], self.cfg.controls["max_steer"])

    # ------------------------------------------------------------------ step
    def step(self, action: Tensor):
        """Advance every car by one agent step with ``action [N]`` (integers 0..8).

        Returns ``obs, reward, terminated, truncated, info``. Cars whose episode ended are reset
        before returning (their ``obs`` is the new episode's first observation); ``info`` holds the
        final observation and the finished episodes' statistics.
        """
        e = self.env_cfg
        self.car, self.rules, gained, contact = self._agent_step(self.cfg, self.tracks, self.track_id, self.car, self.rules, action)
        self.contacts = contact.to(self.dtype)
        self.prev_action = torch.stack((torch.div(action, 3, rounding_mode="floor"), action % 3), dim=1).to(self.dtype)
        self.ep_steps += 1
        terminated = self.rules.finished | self.rules.out
        truncated = ~terminated & (self.ep_steps >= e.max_steps)
        reward = gained * e.reward_scale - e.time_cost - e.fall_penalty * self.rules.out.to(self.dtype)
        if e.hint_weight > 0:  # Evans et al. 2023-style hint: closeness to a classical driver's speed and steering
            if self.hint_driver is None:
                from ..drivers.scripted import ScriptedDriver
                from ..tracks.build_sets import load_tuned
                self.hint_driver = ScriptedDriver(self, load_tuned())
            v_t, d_t, v = self.hint_driver.targets(self)
            ctl = self.cfg.controls
            hint = -((v - v_t).abs() / ctl["top_speed"] + (self.car.steer - d_t).abs() / ctl["max_steer"])
            reward = reward + e.hint_weight * hint * (~terminated).to(self.dtype)
        if self.expert is not None:
            with torch.no_grad():
                self.expert.step(self.expert_driver(self.expert))
        self.ep_return += reward
        done = terminated | truncated
        info = {}
        if not self.auto_reset:  # evaluation: report each car's single run once, never reset
            done = done & ~self.reported
            self.reported |= done
        if bool(done.any()):
            ids = done.nonzero().squeeze(1)
            info["final_obs"] = self.observe()  # observation at the end of the step, before resets
            info["done_ids"] = ids
            info["episode"] = {
                "return": self.ep_return[ids].clone(), "steps": self.ep_steps[ids].clone(),
                "finished": self.rules.finished[ids].clone(), "out_reason": self.rules.out_reason[ids].clone(),
                "progress": (self.rules.s[ids] - self.ep_start_s[ids]).clone(),
                "standing_start": self.ep_standing[ids].clone(),
                "ticks": self.rules.ticks[ids].clone(), "track_id": self.track_id[ids].clone(),
            }
            if self.auto_reset:
                self._reset_cars(ids)
        return self.observe(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self) -> dict:
        """Everything needed to continue exactly where the environment stopped."""
        return {"track_id": self.track_id, "car": self.car.__dict__, "rules": self.rules.__dict__,
                "prev_action": self.prev_action, "contacts": self.contacts, "ep_steps": self.ep_steps,
                "ep_return": self.ep_return, "ep_start_s": self.ep_start_s, "ep_standing": self.ep_standing,
                "gen": self.gen.get_state()}

    def load_state_dict(self, sd: dict):
        self.track_id = sd["track_id"].to(self.device)
        self.car = CarState(**{k: v.to(self.device) for k, v in sd["car"].items()})
        self.rules = RuleState(**{k: v.to(self.device) for k, v in sd["rules"].items()})
        for k in ("prev_action", "contacts", "ep_steps", "ep_return", "ep_start_s", "ep_standing"):
            setattr(self, k, sd[k].to(self.device))
        self.gen.set_state(sd["gen"].cpu())
