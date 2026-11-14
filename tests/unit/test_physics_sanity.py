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


def test_mechanical_energy_counts_rotation_in_the_chassis_frame():
    """The exploit detector's energy: ½v² + g·y + ½ωᵀIω/m, with ω taken in the chassis frame, so the
    same spin gives the same energy however the chassis is oriented (eval.evaluator.mechanical_energy)."""
    from racingmania.eval.evaluator import mechanical_energy
    from racingmania.sim.math3d import quat_from_axis_angle, quat_rotate
    w_local = torch.tensor([[1.5, -2.0, 0.7]], dtype=torch.float64)
    inertia = 1.0 / torch.tensor(CFG.inv_inertia, dtype=torch.float64)
    expected_rot = float(0.5 * (inertia * w_local[0] ** 2).sum() / CFG.mass)
    q = quat_from_axis_angle(torch.tensor([[0.3, 0.8, -0.5]], dtype=torch.float64) / (0.3**2 + 0.8**2 + 0.5**2) ** 0.5,
                             torch.tensor([1.1], dtype=torch.float64))
    car = CarState(pos=torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float64), quat=q,
                   vel=torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64), angvel=quat_rotate(q, w_local),
                   steer=torch.zeros(1, dtype=torch.float64))
    g = -CFG.gravity[1]
    assert abs(float(mechanical_energy(car, CFG, g)[0]) - (12.5 + 2.0 * g + expected_rot)) < 1e-9


def test_tumbling_in_the_air_never_gains_mechanical_energy():
    """In free flight with spin, damping only removes energy: the detector's energy never rises."""
    from racingmania.eval.evaluator import mechanical_energy
    car, tid, rules = _start(lift=50.0)
    car.vel[:, 0] = 20.0
    car.angvel[:] = torch.tensor([2.0, 3.0, -1.0], dtype=torch.float64)
    g = -CFG.gravity[1]
    one = torch.ones(1, dtype=torch.long)
    e_prev = float(mechanical_energy(car, CFG, g)[0])
    for _ in range(100):
        car, contacts, _ = physics_tick(CFG, TB, tid, rules.index, car, one, one)
        assert int(contacts[0]) == 0
        e = float(mechanical_energy(car, CFG, g)[0])
        assert e <= e_prev + 1e-9
        e_prev = e


def test_batch_invariance_and_determinism():
    """A car's trajectory does not depend on the other cars in the batch."""
    car1, tid1, rules1 = _start(1)
    carN, tidN, rulesN = _start(5)
    carN.vel[1:, 0] = torch.tensor([5.0, 10.0, 15.0, 20.0], dtype=torch.float64)
    for _ in range(30):
        car1, rules1, _, _ = agent_step(CFG, TB, tid1, car1, rules1, torch.tensor([3]))
        carN, rulesN, _, _ = agent_step(CFG, TB, tidN, carN, rulesN, torch.tensor([3, 0, 2, 5, 3]))
    assert torch.equal(car1.flat()[0], carN.flat()[0])


def test_hard_landing_does_not_tunnel():
    """Drops from up to 15 m (17 m/s impact, beyond the hardest generated jump landing) are absorbed:
    the wheel rays catch the road, the car never falls through and settles back at its rest height."""
    lifts = torch.tensor([2.0, 6.0, 10.0, 15.0], dtype=torch.float64)
    car, tid, rules = _start(len(lifts))
    car.pos[:, 1] += lifts
    car.vel[:, 0] = 5.0  # slow enough to stay on the 50 m start straight
    y_rest = float(_start()[0].pos[0, 1])
    for _ in range(80):
        car, rules, _, _ = agent_step(CFG, TB, tid, car, rules, torch.full((len(lifts),), 4))
    assert not bool(rules.out.any())
    assert torch.allclose(car.pos[:, 1], torch.full_like(car.pos[:, 1], y_rest), atol=0.02)
