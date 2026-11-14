"""Unit tests for tracks, rules and the environment (reports/plan.md, Verification: tracks and environment)."""
import json

import pytest
import torch

from racingmania.env.vec_env import EnvConfig, RacingVecEnv
from racingmania.sim.track_tensors import TrackBatch
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.campaign import campaign_track
from racingmania.tracks.generator import SPLITS, generate_track, split_of_seed
from racingmania.tracks.ribbon import crossing_ok
from racingmania.tracks.store import batch_from_pool, pack

CFG = load_vehicle_config()


def test_seed_reproducible():
    a, b = generate_track(5), generate_track(5)
    assert json.dumps(a["center"]) == json.dumps(b["center"]) and a["gap"] == b["gap"]


def test_splits_disjoint():
    ranges = sorted(SPLITS.values())
    assert all(ranges[i][1] <= ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert split_of_seed(0) == "train" and split_of_seed(2_000_000) == "test"
    with pytest.raises(ValueError):
        split_of_seed(60_000)


def test_crossing_clearance_and_structure():
    for s in range(8):
        t = generate_track(s)
        assert t is None or crossing_ok(t)
        if t:
            assert t["checkpoints"] == sorted(t["checkpoints"]) and len(t["gap"]) == len(t["center"]) - 1
            assert len(t["vertices"]) < 32768


def test_pool_batch_matches_json_batch():
    """Tracks packed into a pool give the same geometry as tracks loaded from JSON (to float32 precision)."""
    tracks = [campaign_track("campaign-01"), generate_track(3)]
    a = TrackBatch.from_tracks(tracks, dtype=torch.float64)
    b = batch_from_pool(pack(tracks), dtype=torch.float64)
    m = a.center.shape[1]
    assert torch.allclose(a.center, b.center[:, :m], atol=1e-4)
    assert torch.allclose(a.cum, b.cum[:, :m], atol=1e-2)
    assert torch.equal(a.seg_valid, b.seg_valid[:, : m - 1])


@pytest.fixture(scope="module")
def env():
    tracks = [campaign_track("campaign-01"), campaign_track("campaign-04")]
    tb = TrackBatch.from_tracks(tracks, dtype=torch.float32)
    return RacingVecEnv(CFG, tb, 64, EnvConfig(max_steps=40), device="cpu", seed=0)


def test_reward_telescopes_to_progress(env):
    """Sum of per-step progress rewards equals the change in progress over the episode."""
    env.reset_all()
    s0 = env.rules.s.clone()
    total = torch.zeros(env.num_envs)
    for _ in range(10):
        _, r, term, trunc, _ = env.step(torch.full((env.num_envs,), 3))
        assert not bool((term | trunc).any())
        total += r
    assert torch.allclose(total / env.env_cfg.reward_scale, env.rules.s - s0, atol=1e-3)


def test_truncation_flag_at_time_limit(env):
    env.reset_all()
    seen_trunc = False
    for _ in range(45):
        _, _, term, trunc, info = env.step(torch.full((env.num_envs,), 4))  # coasting straight
        assert not bool((term & trunc).any())
        seen_trunc |= bool(trunc.any())
        if "final_obs" in info:
            assert info["final_obs"].shape == (env.num_envs, env.observe().shape[1])
    assert seen_trunc


def test_obs_invariant_to_world_pose():
    """Rotating and translating the whole world (track and car) leaves the observation unchanged."""
    from racingmania.sim.math3d import quat_from_axis_angle, quat_mult, quat_rotate
    t = campaign_track("campaign-02")
    tb = TrackBatch.from_tracks([t], dtype=torch.float64)
    env = RacingVecEnv(CFG, tb, 8, EnvConfig(random_start_prob=1.0), device="cpu", dtype=torch.float64, seed=1)
    o1 = env.observe()
    ang = torch.tensor(0.7, dtype=torch.float64)
    q = quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), ang)
    shift = torch.tensor([100.0, 5.0, -40.0], dtype=torch.float64)
    rot = lambda v: quat_rotate(q.expand(*v.shape[:-1], 4), v)
    env.tracks.center = rot(env.tracks.center) + shift
    env.tracks.left = rot(env.tracks.left)
    env.car.pos = rot(env.car.pos) + shift
    env.car.vel = rot(env.car.vel)
    env.car.angvel = rot(env.car.angvel)
    env.car.quat = quat_mult(q.expand(8, 4), env.car.quat)
    assert torch.allclose(o1, env.observe(), atol=1e-9)
