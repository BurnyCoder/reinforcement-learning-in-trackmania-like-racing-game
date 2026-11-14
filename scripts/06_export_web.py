"""Phase 6: export a trained run to the browser game: ONNX policy, AI ghosts, and the asset index.

Usage: uv run python scripts/06_export_web.py --run generalist/seed1   (or --run baselines/untrained/seed0 for the untrained control)
Writes web/assets/models/<run>.onnx (+ results/<run>/policy.onnx), web/assets/ghosts/<track>/<run>.json,
updates web/assets/index.json, and records the ONNX-vs-PyTorch action agreement in run.json.
"""
import argparse
import csv
import json
import shutil

import torch

import _common  # noqa: F401

from racingmania.drivers.simple import RandomDriver
from racingmania.env.vec_env import RacingVecEnv
from racingmania.export.onnx_export import export_policy
from racingmania.runs import campaign_pool, load_experiment, load_policy, run_dirs, update_run_json, vehicle_for
from racingmania.tracks.campaign import CAMPAIGN_PLANS
from racingmania.tracks.store import batch_from_pool
from racingmania.utils.paths import WEB_ASSETS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", required=True, help="run id, e.g. generalist/seed1 or specialist-campaign-01/seed0")
a = ap.parse_args()
rid = a.run
res_dir, _, _ = run_dirs(rid)
seed = int(rid.rsplit("seed", 1)[1])
if rid.startswith("baselines/untrained/"):  # the untrained control: same weights as in its evaluation (baselines.untrained_agent)
    from racingmania.baselines import untrained_agent
    from racingmania.env.vec_env import EnvConfig
    agent, norm = untrained_agent(seed, "cuda")
    env_cfg, cfg = EnvConfig(), vehicle_for({})
    policy = lambda env: agent.act(norm(env.observe().float()), deterministic=True)[1]  # noqa: E731
    flat, description = f"untrained-seed{seed}", "Untrained network: the standard PPO actor with random initial weights and no training (control)"
else:
    exp_name = rid.split("/")[0]
    track = None
    if exp_name.startswith("specialist") and "-campaign-" in exp_name:
        exp_name, track = exp_name.split("-campaign-")[0], "campaign-" + exp_name.split("-campaign-")[1]
    policy, agent, norm, env_cfg = load_policy(exp_name, seed, track, device="cuda")
    cfg = vehicle_for(load_experiment(exp_name))
    flat, description = rid.replace("/", "-"), load_experiment(exp_name).get("description", "")
names = sorted(CAMPAIGN_PLANS)
tb = batch_from_pool(campaign_pool(names), device="cuda")
# Collect varied inputs for the check: 512 cars on campaign tracks, random starts, a few hundred steps.
env = RacingVecEnv(cfg, tb, 512, env_cfg=env_cfg, device="cuda", seed=123)
mix = RandomDriver(seed=5)
with torch.no_grad():
    for t in range(200):
        act = policy(env) if t % 3 else mix(env)
        env.step(act)
onnx_path = WEB_ASSETS / "models" / f"{flat}.onnx"
check = export_policy(agent, norm, env, onnx_path)
shutil.copy2(onnx_path, res_dir / "policy.onnx")
update_run_json(res_dir, onnx_check=check, onnx_file=f"assets/models/{flat}.onnx")

index_path = WEB_ASSETS / "index.json"
index = json.loads(index_path.read_text())
index["models"] = [m for m in index.get("models", []) if m["name"] != flat]
index["models"].append({"name": flat, "file": f"assets/models/{flat}.onnx", "K": env_cfg.window_k, "stride": env_cfg.window_stride,
                        "description": description, "run": rid})
ghosts = index.setdefault("ghosts", {})
keep_unfinished = rid.startswith("baselines/")  # a control's runs are kept even though they fail: they are what the README shows
ends = ({r["track"]: r["end"] for r in csv.DictReader(open(res_dir / "eval_campaign.csv", encoding="utf-8"))}
        if (res_dir / "eval_campaign.csv").exists() else {})  # end reason of each official campaign run
for rep in sorted((res_dir / "replays").glob("*.json")) if (res_dir / "replays").exists() else []:
    d = json.loads(rep.read_text())
    if not d.get("finished") and not keep_unfinished:
        continue
    dst = WEB_ASSETS / "ghosts" / d["track"] / f"{flat}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"model": flat, "time": d["time"], "finished": bool(d.get("finished")),
                               "end": ends.get(d["track"], "finished" if d.get("finished") else "fell"), "states": d["states"]}))
    rel = f"assets/ghosts/{d['track']}/{flat}.json"
    ghosts[d["track"]] = sorted(set(ghosts.get(d["track"], [])) | {rel})
index_path.write_text(json.dumps(index, indent=2))
print("exported", flat, check)
