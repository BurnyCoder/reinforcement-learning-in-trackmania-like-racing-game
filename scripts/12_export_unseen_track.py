"""Put held-out generated tracks into the browser game, to show a model on tracks it never saw.

Usage (in WSL, GPU): uv run python scripts/12_export_unseen_track.py --run generalist/seed0 --split test \
                         --rule best median_jump mid_fail

Global context: the game's menu lists the hand-made campaign tracks (web/assets/index.json); the
generalists were trained only on generated tracks from the training seed range and never on a test
seed (preregistration, section 3). This script picks test tracks by the fixed rules of
``racingmania.eval.clips`` over the run's official evaluation (``results/<run>/eval_test.csv``): the
best finished track, a typical one (median score among finished tracks with a jump) and the fall
closest to mid-track, so the README can show labelled best, typical and failure cases. Each is rebuilt
from its seed with the deterministic generator, checked identical to the track in the evaluated pool,
certified with the frozen scripted driver to get the reference run the game needs for medals
(docs/formats.md, "Track file"), and added to the menu as kind "unseen". Reading test tracks is logged
in the test-access log. With ``--ghosts scripted run`` the official runs of the scripted driver and of
the run itself on each chosen track (finished or not) are saved as ghosts, so the game's "Watch replay"
shows exactly the runs that were scored, including failures.
"""
import argparse
import csv
import json

import numpy as np

import _common  # noqa: F401

from racingmania.eval.clips import RULES, select_track
from racingmania.eval.evaluator import _log_access
from racingmania.sim.vehicle_config import load_vehicle_config
from racingmania.tracks.build_sets import load_tuned
from racingmania.tracks.generator import generate_track
from racingmania.tracks.store import batch_from_pool, load_pool, pack, pool_path
from racingmania.tracks.validate import certify
from racingmania.utils.paths import RESULTS_DIR, WEB_ASSETS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", default="generalist/seed0")
ap.add_argument("--split", default="test", choices=["test", "val", "oor"])
ap.add_argument("--variant", default=None, help="only out-of-range tracks of this variant (oor split), e.g. tight_turns")
ap.add_argument("--rule", nargs="+", default=["median_jump"], choices=sorted(RULES), help="selection rules (racingmania.eval.clips)")
ap.add_argument("--ghosts", nargs="*", default=[], choices=["scripted", "run"],
                help="also save official runs on these tracks as ghosts: the scripted driver's and/or the run's own")
a = ap.parse_args()

rows = {r["track"]: r for r in csv.DictReader(open(RESULTS_DIR / a.run / f"eval_{a.split}.csv", encoding="utf-8"))}
chosen = {}
for rule in a.rule:
    name, why = select_track(RESULTS_DIR, a.run, a.split, rule, variant=a.variant)
    chosen[name] = f"{why} in the official evaluation of {a.run}"
if a.split in ("test", "oor"):
    _log_access(a.split, "export:unseen-track-demo", len(chosen))
cfg = load_vehicle_config()
pool = load_pool(pool_path(a.split))
index_path = WEB_ASSETS / "index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))


def export_track(name: str, why: str):
    """Rebuild ``name`` from its seed, check it against the evaluated pool, certify it, add it to the menu."""
    seed = int(name.rsplit("-", 1)[1])
    track = generate_track(seed)
    i = pool["names"].index(name)
    n = int(pool["n"][i])
    same = len(track["center"]) == n and np.allclose(np.asarray(track["center"], np.float32), pool["center"][i][:n], atol=1e-5)
    assert same, f"{name}: regenerated geometry differs from the evaluated pool"
    passed, ref, seqs = certify(cfg, batch_from_pool(pack([track]), device="cuda"), load_tuned(), device="cuda")
    ticks = int(pool["ref_ticks"][i])  # the reference of the official evaluation
    track["reference"] = {"ticks": ticks, "time": round(ticks * cfg.dt, 2), "certified": bool(passed[0])}
    if int(ref[0]) == ticks:  # the fresh run's actions belong to this reference only if it reproduced it
        track["reference"]["actions"] = seqs[0]
    (WEB_ASSETS / "tracks" / f"{name}.json").write_text(json.dumps(track), encoding="utf-8")
    title = f"Unseen test track {seed}" if a.split != "oor" else f"Out-of-range track {seed} ({a.variant.replace('_', ' ')})" if a.variant         else f"Out-of-range track {seed}"  # short: the game's header shows it; the note gives the ranges
    entry = {"name": name, "title": title, "kind": "unseen",
             "note": f"Held-out {a.split} seed {seed}, never used in training; chosen as {why}."}
    index["tracks"] = [t for t in index["tracks"] if t["name"] != name] + [entry]
    print(f"{name}: identical to the pool track; reference {ticks * cfg.dt:.2f} s (fresh certification {int(ref[0]) * cfg.dt:.2f} s, "
          f"passed {bool(passed[0])}); official {a.run}: {rows[name]['end']} at {rows[name]['time']} s; added to the menu", flush=True)


for name, why in chosen.items():
    export_track(name, why)

def save_official_ghosts(policy, results_run: str, model: str, env_cfg):
    """Re-run ``policy``'s official evaluation on the whole set (one batch, fresh compile, as in 03_evaluate and
    04_baselines), check every chosen track ends on the same tick as in ``results/<results_run>/eval_<split>.csv``,
    and save those runs as ghosts (unfinished runs too, flagged, so a failure can be replayed exactly)."""
    from racingmania.eval.evaluator import evaluate
    from racingmania.runs import eval_tracks
    tb, protected = eval_tracks(a.split)
    replays = {r["track"]: r for r in evaluate(policy, cfg, tb, device="cuda", env_cfg=env_cfg, compile=True, record_replays=True,
                                               log_access=a.split if protected else None)}
    official = {r["track"]: r for r in csv.DictReader(open(RESULTS_DIR / results_run / f"eval_{a.split}.csv", encoding="utf-8"))}
    ghosts = index.setdefault("ghosts", {})
    for name in chosen:
        r = replays[name]
        assert r["ticks"] == int(official[name]["ticks"]), f"{name}: replay {r['ticks']} ticks, official {official[name]['ticks']}"
        rel = f"assets/ghosts/{name}/{model}.json"  # relative to web/, as in index.json
        (WEB_ASSETS.parent / rel).parent.mkdir(parents=True, exist_ok=True)
        (WEB_ASSETS.parent / rel).write_text(json.dumps({"model": model, "time": r["time"], "finished": bool(r["finished"]),
                                                         "end": r["end"], "states": r["replay"]}), encoding="utf-8")
        ghosts[name] = sorted(set(ghosts.get(name, [])) | {rel})
        print(f"{name}: {model} ghost ({r['end']} at {r['time']:.2f} s), identical to its official run", flush=True)


if "scripted" in a.ghosts:  # the scripted driver's official runs, for the game's "Watch replay"
    from racingmania.baselines import _Scripted
    from racingmania.env.vec_env import EnvConfig
    save_official_ghosts(_Scripted(load_tuned()), "baselines/scripted/seed0", "scripted-driver", EnvConfig())
if "run" in a.ghosts:  # the run's own official runs (finished or not)
    from racingmania.runs import load_policy
    experiment, seed = a.run.split("/")[0], int(a.run.rsplit("seed", 1)[1])
    policy, _, _, env_cfg = load_policy(experiment, seed, None)
    save_official_ghosts(policy, a.run, a.run.replace("/", "-"), env_cfg)
index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
