"""Parity of the PyTorch port with the original cannon-es physics running in Node.js.

Global context: this is the main physics test (reports/plan.md, Verification). The browser game's
own code (web/js/core + vendored cannon-es) is the reference; the port in src/racingmania/sim must
reproduce one tick from random states to 1e-9 in float64, and whole driven runs closely.
"""

import json
import shutil

import numpy as np
import pytest

from racingmania.sim.reference import port_cases, port_sequence, random_cases, run_harness
from racingmania.tracks.campaign import campaign_track
from racingmania.utils.paths import WEB_ASSETS

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed to run the original physics")


def _track(name):
    p = WEB_ASSETS / "tracks" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else campaign_track(name)


@pytest.mark.parametrize("name", ["campaign-01", "campaign-03", "campaign-05"])
def test_single_tick_matches_cannon_es(name):
    track = _track(name)
    cases = random_cases(track, 2000, seed=7)
    js = run_harness({"track": track, "cases": cases})["cases"]
    py = port_cases(track, cases)
    contacts = [c["contacts"] for c in js]
    assert len(set(contacts)) == 5, "test inputs should cover 0..4 wheels in contact"
    for a, b in zip(js, py):
        assert np.max(np.abs(np.array(a["state"]) - np.array(b["state"]))) < 1e-9
        assert abs(a["steer"] - b["steer"]) < 1e-12
        assert (a["contacts"], a["index"], a["finished"], a["out"]) == (b["contacts"], b["index"], b["finished"], b["out"])
        assert abs(a["s"] - b["s"]) < 1e-9


def test_inverse_inertia_matches_cannon_es():
    from racingmania.sim.vehicle_config import load_vehicle_config
    track = _track("campaign-01")
    js = run_harness({"track": track, "cases": random_cases(track, 1)})["cases"][0]
    assert np.allclose(js["inv_inertia"], load_vehicle_config().inv_inertia, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_driven_sequence_matches_cannon_es(seed):
    """Mostly-throttle random driving from the start line: every tick agrees closely, and the run ends
    on the same tick for the same reason."""
    track = _track("campaign-01")
    rng = np.random.default_rng(seed)
    pedal = rng.choice([0, 1, 2], p=[0.8, 0.15, 0.05], size=600)
    steer = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3], size=600)
    actions = (steer * 3 + pedal).tolist()
    start = {"state": track["start"]["position"] + track["start"]["quaternion"] + [0.0] * 6, "steer": 0.0,
             "index": track["start"]["index"]}
    js = run_harness({"track": track, "sequences": [{**start, "actions": actions}]})["sequences"][0]
    py = port_sequence(track, start, actions)
    assert len(js) == len(py)
    assert (js[-1]["finished"], js[-1]["out"]) == (py[-1]["finished"], py[-1]["out"])
    worst = max(np.max(np.abs(np.array(a["state"]) - np.array(b["state"]))) for a, b in zip(js, py))
    assert worst < 1e-6
