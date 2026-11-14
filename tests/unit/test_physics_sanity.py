"""Physics sanity checks kept next to the parity tests (reports/plan.md, Verification)."""
import torch

from racingmania.sim.rules import reset_rules
from racingmania.sim.step import CarState, agent_step, physics_tick
from racingmania.sim.track_tensors import TrackBatch
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.campaign import campaign_track

CFG = load_vehicle_config()
TRACK = campaign_track("campaign-01")
TB = TrackBatch.from_tracks([TRACK], dtype=torch.float64)


def _start(n=1, lift=0.0):
    s = torch.tensor([TRACK["start"]["position"] + TRACK["start"]["quaternion"] + [0.0] * 6], dtype=torch.float64).repeat(n, 1)
    s[:, 1] += lift
    car = CarState.from_flat(s, torch.zeros(n, dtype=torch.float64))
    tid = torch.zeros(n, dtype=torch.long)
    return car, tid, reset_rules(TB, tid, car.pos, TB.start_index[tid])


def test_free_fall_matches_semi_implicit_euler():
    """High above the road no wheel touches: height follows the damped semi-implicit Euler free fall."""
    car, tid, rules = _start(lift=50.0)
    y, v = float(car.pos[0, 1]), 0.0
    one = torch.ones(1, dtype=torch.long)
    for _ in range(50):
        car, contacts, _ = physics_tick(CFG, TB, tid, rules.index, car, one, one)
        assert int(contacts[0]) == 0
        v = v * CFG.linear_damping_factor + CFG.mass * CFG.gravity[1] * (CFG.inv_mass * CFG.dt)
        y = y + v * CFG.dt
    assert abs(float(car.pos[0, 1]) - y) < 1e-9


def test_rest_height_is_spawn_height():
    """A car placed at the spawn height stays there (the suspension carries its weight)."""
    car, tid, rules = _start()
    y0 = float(car.pos[0, 1])
    for _ in range(40):
        car, rules, _, _ = agent_step(CFG, TB, tid, car, rules, torch.tensor([4]))
    assert abs(float(car.pos[0, 1]) - y0) < 0.02


def test_coasting_never_gains_energy_on_flat_ground():
    car, tid, rules = _start()
    car.vel[:, 0] = 20.0
    g = -CFG.gravity[1]
    energy = lambda c: float(0.5 * (c.vel ** 2).sum() + g * c.pos[0, 1])
    e_prev = energy(car)
    for _ in range(30):
        car, rules, _, _ = agent_step(CFG, TB, tid, car, rules, torch.tensor([4]))
        assert energy(car) <= e_prev + 1e-6
        e_prev = energy(car)


def test_batch_invariance_and_determinism():
    """A car's trajectory does not depend on the other cars in the batch."""
    car1, tid1, rules1 = _start(1)
    carN, tidN, rulesN = _start(5)
    carN.vel[1:, 0] = torch.tensor([5.0, 10.0, 15.0, 20.0], dtype=torch.float64)
    for _ in range(30):
        car1, rules1, _, _ = agent_step(CFG, TB, tid1, car1, rules1, torch.tensor([3]))
        carN, rulesN, _, _ = agent_step(CFG, TB, tidN, carN, rulesN, torch.tensor([3, 0, 2, 5, 3]))
    assert torch.equal(car1.flat()[0], carN.flat()[0])
