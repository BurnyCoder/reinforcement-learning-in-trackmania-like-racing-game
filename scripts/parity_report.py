"""Measure parity between the PyTorch port and the original cannon-es physics and save it for the reports.

Usage: uv run python scripts/parity_report.py  → results/parity.json
Same procedure as tests/parity (random single ticks on three campaign tracks, plus driven sequences),
but records the measured differences instead of only asserting a bound, so the paper's numbers come
from a file.
"""
import json

import numpy as np

import _common  # noqa: F401

from racingmania.sim.reference import port_cases, port_sequence, random_cases, run_harness
from racingmania.utils.paths import RESULTS_DIR, WEB_ASSETS

out = {"single_tick": {}, "sequences": {}}
for name in ("campaign-01", "campaign-03", "campaign-05"):
    track = json.loads((WEB_ASSETS / "tracks" / f"{name}.json").read_text())
    cases = random_cases(track, 2000, seed=7)
    js = run_harness({"track": track, "cases": cases})["cases"]
    py = port_cases(track, cases)
    diffs = np.array([np.max(np.abs(np.array(a["state"]) - np.array(b["state"]))) for a, b in zip(js, py)])
    flags = sum((a["contacts"], a["index"], a["finished"], a["out"]) != (b["contacts"], b["index"], b["finished"], b["out"])
                for a, b in zip(js, py))
    out["single_tick"][name] = {"cases": len(cases), "max_abs_state_diff": float(diffs.max()),
                                "median_abs_state_diff": float(np.median(diffs)), "flag_mismatches": int(flags),
                                "contact_histogram": np.bincount([a["contacts"] for a in js], minlength=5).tolist()}
    rng = np.random.default_rng(0)
    actions = (rng.choice(3, p=[0.3, 0.4, 0.3], size=600) * 3 + rng.choice(3, p=[0.8, 0.15, 0.05], size=600)).tolist()
    start = {"state": track["start"]["position"] + track["start"]["quaternion"] + [0.0] * 6, "steer": 0.0,
             "index": track["start"]["index"]}
    jq = run_harness({"track": track, "sequences": [{**start, "actions": actions}]})["sequences"][0]
    pq = port_sequence(track, start, actions)
    worst = max(np.max(np.abs(np.array(a["state"]) - np.array(b["state"]))) for a, b in zip(jq, pq))
    out["sequences"][name] = {"ticks_js": len(jq), "ticks_port": len(pq), "max_abs_state_diff": float(worst),
                              "same_end": (jq[-1]["finished"], jq[-1]["out"]) == (pq[-1]["finished"], pq[-1]["out"])}
    print(name, out["single_tick"][name], out["sequences"][name], flush=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "parity.json").write_text(json.dumps(out, indent=2))
