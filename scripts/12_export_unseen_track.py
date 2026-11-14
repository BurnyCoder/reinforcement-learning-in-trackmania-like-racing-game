"""Put one held-out generated track into the browser game, to show a model on a track it never saw.

Usage (in WSL, GPU): uv run python scripts/12_export_unseen_track.py --run generalist/seed0 --split test

Global context: the game's menu lists the hand-made campaign tracks (web/assets/index.json); the
generalists were trained only on generated tracks from the training seed range and never on a test
seed (preregistration, section 3). This script picks a test track by a fixed rule — among the
run's finished test tracks with at least one jump, the one with the median score in its official
evaluation (``results/<run>/eval_test.csv``), so the clip shows a typical lap rather than the best
one — rebuilds it from its seed with the deterministic generator, checks that it is identical to the
track in the evaluated pool, certifies it with the frozen scripted driver to get the reference run
the game needs for medals (docs/formats.md, "Track file"), and adds it to the menu as kind "unseen".
Reading the test track is logged in the test-access log.
"""
import argparse
import csv
import json

import numpy as np

import _common  # noqa: F401

from racingmania.eval.evaluator import _log_access
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.build_sets import load_tuned
from racingmania.tracks.generator import generate_track
from racingmania.tracks.store import batch_from_pool, load_pool, pack, pool_path
from racingmania.tracks.validate import certify
from racingmania.utils.paths import RESULTS_DIR, WEB_ASSETS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", default="generalist/seed0")
ap.add_argument("--split", default="test", choices=["test", "val"])
ap.add_argument("--track", default=None, help="use this track instead of the selection rule")
a = ap.parse_args()

rows = {r["track"]: r for r in csv.DictReader(open(RESULTS_DIR / a.run / f"eval_{a.split}.csv", encoding="utf-8"))}
if a.track is None:  # selection rule: median score among finished tracks with at least one jump
    feats = {r["track"]: r for r in csv.DictReader(open(RESULTS_DIR / "tracks" / f"{a.split}_features.csv", encoding="utf-8"))}
    cand = sorted((float(r["score"]), t) for t, r in rows.items() if r["finished"] == "1" and int(feats[t]["jumps"]) >= 1)
    a.track = cand[len(cand) // 2][1]
    rule = (f"the median-scoring track among the {len(cand)} {a.split} tracks with a jump that {a.run} finished "
            "in its official evaluation")
else:
    rule = "chosen by name"
seed = int(a.track.rsplit("-", 1)[1])
if a.split == "test":
    _log_access("test", "export:unseen-track-demo", 1)

track = generate_track(seed)
pool = load_pool(pool_path(a.split))
i = pool["names"].index(a.track)
n = int(pool["n"][i])
same = np.allclose(np.asarray(track["center"], np.float32), pool["center"][i][:n], atol=1e-5) and len(track["center"]) == n
assert same, f"{a.track}: regenerated geometry differs from the evaluated pool"

cfg = load_vehicle_config()
passed, ref, seqs = certify(cfg, batch_from_pool(pack([track]), device="cuda"), load_tuned(), device="cuda")
pool_ref = int(pool["ref_ticks"][i]) if "ref_ticks" in pool else int(rows[a.track]["ref_ticks"])
ticks = pool_ref  # the reference of the official evaluation; the fresh run's actions are kept if it matches
track["reference"] = {"ticks": ticks, "time": round(ticks * cfg.dt, 2), "certified": bool(passed[0])}
if int(ref[0]) == pool_ref:
    track["reference"]["actions"] = seqs[0]
(WEB_ASSETS / "tracks" / f"{a.track}.json").write_text(json.dumps(track), encoding="utf-8")

index_path = WEB_ASSETS / "index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
entry = {"name": a.track, "title": f"Unseen test track {seed}", "kind": "unseen",
         "note": f"Held-out {a.split} seed {seed}, never used in training: {rule}."}
index["tracks"] = [t for t in index["tracks"] if t["name"] != a.track] + [entry]
index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
print(f"{a.track}: identical to the pool track; reference {ticks * cfg.dt:.2f} s (fresh certification {int(ref[0]) * cfg.dt:.2f} s, "
      f"passed {bool(passed[0])}); official {a.run} time {rows[a.track]['time']} s; added to the menu")
