"""Hugging Face cards: one model card per run and the card of the results dataset.

Local purpose: build Markdown cards with YAML front matter from the run records and the analysis.
The model card follows the convention of Stable-Baselines3 cards on the Hub (``model-index`` with
task type ``reinforcement-learning`` and one metric list per evaluated dataset, ``verified:
false``); see https://huggingface.co/docs/hub/model-cards and
https://huggingface.co/docs/hub/datasets-cards for the metadata fields.

Global context: ``publish.publish_model`` uploads ``model_card(rid)`` as the README of each run's
model repository and ``publish.publish_results_dataset`` uploads ``results/HF_DATASET_CARD.md`` as
the README of the results dataset; both are in the project's Hugging Face collection.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import yaml

from ..eval import stats
from . import taxonomy as tx
from .render import HF_RESULTS_REPO, REPO_URL
from .tables import ci, md_table, num, pct


def _yaml(d: dict) -> str:
    return "---\n" + yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=1000) + "---\n"


def _clean(x):
    return None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x


def run_scores(res, rid: str) -> dict:
    """Per-set SWS and finish rate of one run, with track-only bootstrap intervals (one seed)."""
    out = {}
    for s, rows in res.evals.get(rid, {}).items():
        sc = [float(r["score"]) for r in rows]
        fi = [float(r["finished"]) for r in rows]
        a, b = stats.bootstrap_mean([sc]), stats.bootstrap_mean([fi])
        out[s] = {"sws": a.estimate, "sws_lo": a.lo, "sws_hi": a.hi, "finish": b.estimate, "finish_lo": b.lo,
                  "finish_hi": b.hi, "n_tracks": len(rows)}
    return out


def run_clips(repo: Path, rid: str | None = None, readme_only: bool = False) -> list[dict]:
    """Recorded video clips (scripts/11_record_videos.py): the README clips (docs/videos/videos.json), then the clips
    recorded for the Hugging Face cards (docs/videos/cards/videos.json); only run ``rid``'s if given, and only clips
    whose MP4 exists."""
    files = [repo / "docs" / "videos" / "videos.json"] + ([] if readme_only else [repo / "docs" / "videos" / "cards" / "videos.json"])
    clips = [c for f in files if f.exists() for c in json.loads(f.read_text(encoding="utf-8"))]
    return [c for c in clips if (rid is None or c.get("run") == rid) and c.get("mp4") and (repo / c["mp4"]).exists()]


def video_section(clips: list[dict], url) -> str:
    """Markdown for clips: an HTML5 <video> per clip (Hugging Face cards render it, as in the Hub's own examples, e.g.
    https://huggingface.co/mishig/test_vid) and the clip's caption parts from videos.json. ``url(clip)`` gives the
    MP4's address."""
    out = []
    for c in clips:
        cap = c.get("caption") or {"What you see": c.get("explain", "").strip()}
        out.append("\n".join([f"### {c['title']}", "",
                              f'<video src="{url(c)}" controls muted loop playsinline width="100%"></video>', "",
                              *[f"- **{k}:** {v}" for k, v in cap.items()]]))
    return "\n\n".join(out)


def model_card(res, rid: str, repo_id: str | None = None) -> str:
    """Hugging Face model card for run ``rid`` (works for planned, training, trained and evaluated runs). With the
    model repository's id, the card's videos point at the MP4s that publish.publish_model uploads to ``videos/``."""
    r = res.run(rid) or next((q for q in res.queued if q["rid"] == rid), None)
    if r is None:
        r = {"rid": rid, "experiment": rid.split("/")[0], "seed": None, "status": "unknown", "algo": None, "label": rid,
             "description": "", "architecture": "-", "objective": "-", "observation": "-", "training_data": "-", "start_states": "-"}
    algo = (r.get("algo") or "ppo").lower()
    scores = run_scores(res, rid)
    name = "racingmania-" + rid.replace("/", "-")
    results = []
    for s in tx.SETS:
        if s not in scores:
            continue
        v = scores[s]
        results.append({"task": {"type": "reinforcement-learning", "name": "reinforcement-learning"},
                        "dataset": {"name": f"RacingMania {tx.SET_LABELS[s].lower()} tracks", "type": f"racingmania-{s}"},
                        "metrics": [{"type": "success_weighted_speed", "value": round(v["sws"], 4), "name": "SWS", "verified": False},
                                    {"type": "finish_rate", "value": round(v["finish"], 4), "name": "Finish rate", "verified": False}]})
    meta = {"library_name": "pytorch", "license": "mit",
            "tags": ["reinforcement-learning", "racingmania", algo, "racing", "procedural-generation", "deep-reinforcement-learning"]}
    if results:
        meta["model-index"] = [{"name": name, "results": results}]
    rows = [[tx.SET_LABELS.get(s, s), v["n_tracks"], ci(v["sws"], v["sws_lo"], v["sws_hi"]), pct(v["finish"])] for s, v in scores.items()]
    ppo = r.get("ppo") or {}
    env = r.get("env") or {}
    cfg_rows = [[f"`{k}`", f"`{v}`"] for k, v in {**ppo, **{f"env.{k}": v for k, v in env.items()}}.items()]
    is_policy = algo in ("ppo", "pqn", "bc", "dagger")
    parts = [
        _yaml(meta),
        f"# {name}",
        f"**{r.get('label', rid)}**, seed {r.get('seed')}: {r.get('description') or 'a run of the RacingMania research project.'}",
        f"Status: **{r.get('status')}** ({tx.STATUS_TEXT.get(r.get('status'), '')}); physics version {r.get('physics') or 'v1'}; "
        f"card generated {date.today().isoformat()} from the result files.",
        "RacingMania is an original 3D time-attack racing game for the browser (three.js + cannon-es `RaycastVehicle`) with "
        "reinforcement-learning drivers trained on a batched PyTorch port of the same car physics. Code, paper and full "
        f"results: <{REPO_URL}>.",
    ]
    clips = run_clips(res.paths.repo, rid)
    if clips:
        base = f"https://huggingface.co/{repo_id}/resolve/main/videos/" if repo_id else "videos/"
        parts += ["## Videos", "Official evaluation runs of this model replayed in the browser game (\"Watch replay\"): the car "
                  "states the evaluator recorded while scoring the run, so each clip is exactly the run its caption describes. "
                  f"More clips, including the controls: <{REPO_URL}#videos>.",
                  video_section(clips, lambda c: base + Path(c["mp4"]).name)]
    parts += [
        "## Model",
        md_table(["Item", "Value"], [["Algorithm", algo.upper()], ["Architecture", r.get("architecture", "-")],
                                     ["Objective", r.get("objective", "-")], ["Observation", r.get("observation", "-")],
                                     ["Training data", r.get("training_data", "-")], ["Start states", r.get("start_states", "-")],
                                     ["Agent steps trained", num(r.get("global_step")) if r.get("global_step") else "-"],
                                     ["Wall-clock hours", num(r.get("wall_hours"), 2) if r.get("wall_hours") else "-"],
                                     ["Git commit", f"`{r.get('git_commit')}`" if r.get("git_commit") else "-"]]),
        "## Evaluation",
        "Success-weighted speed (SWS): mean over tracks of finished x (scripted driver's time / agent's time); 1.0 means "
        "finishing every track as fast as the scripted reference driver. One deterministic run per track from a standing "
        "start with the most probable action; intervals are 95 % percentile bootstrap intervals over tracks (one seed).",
        md_table(["Track set", "Tracks", "SWS [95 % CI]", "Finished"], rows, "_Not evaluated yet._"),
    ]
    if is_policy:
        parts += [
            "## How to use",
            "Files: `final.pt` (PyTorch checkpoint: `agent` and `norm` state dicts, `ppo` and `env_cfg` settings, `obs_dim`), "
            "`policy.onnx` (observation function + frozen normaliser + actor in one graph, as run by the browser game), "
            "`run.json`, training and validation curves, and per-track evaluation CSVs under `eval/`.",
            "PyTorch, inside a clone of the repository (`uv sync`):",
            "```python\nimport torch\nfrom racingmania.env.obs import obs_size\nfrom racingmania.rl.networks import ActorCritic\n"
            "from racingmania.rl.normalizer import RunningNorm\n\nsd = torch.load(\"final.pt\", map_location=\"cpu\", weights_only=False)\n"
            "k = sd[\"env_cfg\"][\"window_k\"]\nagent = ActorCritic(obs_size(k), tuple(sd[\"ppo\"][\"hidden\"]), sd[\"ppo\"][\"action_mode\"])\n"
            "agent.load_state_dict(sd[\"agent\"]); agent.eval()\nnorm = RunningNorm(obs_size(k)); norm.load_state_dict(sd[\"norm\"]); norm.frozen = True\n"
            "# action = agent.act(norm(env.observe().float()), deterministic=True)[1]   (see racingmania.runs.load_policy)\n```"
            if algo != "pqn" else
            "PQN checkpoints hold a Q-network; load them with `racingmania.rl.pqn` (see the repository).",
            "ONNX (inputs and outputs are specified in `docs/formats.md`, section \"Policy model\"):",
            "```python\nimport numpy as np, onnxruntime as ort\nsess = ort.InferenceSession(\"policy.onnx\")\n"
            "inputs = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in sess.get_inputs()}\n"
            "steer_logits, pedal_logits = sess.run(None, inputs)\n"
            "steer, pedal = int(steer_logits.argmax()), int(pedal_logits.argmax())  # 0/1/2 = left/none/right, gas/coast/brake\n```",
            "To watch it drive, open the game (`web/index.html`, or the GitHub Pages site), choose it in a track's AI drivers "
            "table and press \"Watch AI\" (live in the browser) or \"Watch replay\" (its official evaluation run).",
        ]
    parts += ["## Training configuration (from run.json)", md_table(["Setting", "Value"], cfg_rows, "_Not logged yet._"),
              "## Licence and citation", f"MIT. Author: Libor \"Burny\" Burian. Repository: <{REPO_URL}>."]
    return "\n\n".join(parts) + "\n"


def dataset_card(res, an) -> str:
    n_rows = sum(len(rows) for d in res.evals.values() for rows in d.values())
    size = "n<1K" if n_rows < 1_000 else "1K<n<10K" if n_rows < 10_000 else "10K<n<100K" if n_rows < 100_000 else "100K<n<1M"
    meta = {"license": "mit", "pretty_name": "RacingMania reinforcement-learning results", "language": ["en"],
            "task_categories": ["reinforcement-learning"], "size_categories": [size],
            "tags": ["reinforcement-learning", "racingmania", "racing", "procedural-generation", "generalization", "ppo", "pqn"]}
    evaluated = sum(1 for r in res.runs if r["status"] == "evaluated")
    rows = []
    for c in an.configs:
        d = an.summaries.get(c["key"], {})
        s = an.primary_set
        if s in d:
            rows.append([c["label"], d[s]["n_seeds"], ci(d[s]["sws"], d[s]["sws_lo"], d[s]["sws_hi"]), pct(d[s]["finish"])])
    parts = [
        _yaml(meta), "# RacingMania results",
        "Every result file of the RacingMania project: per-run `run.json` (status, configuration, hardware, physics version), "
        "per-track evaluation CSVs (`eval_<set>.csv`: track, finished, ticks, time, reference ticks, score, end reason, "
        "progress, maximum speed, exploit flags), training and validation curves, track-set summaries, the test-access log, "
        "and the generated reports and paper. Column definitions: `docs/formats.md` in the repository.",
        f"Snapshot of {date.today().isoformat()}: {len(res.runs)} registered runs ({evaluated} evaluated), {n_rows:,} per-track "
        f"evaluation rows.",
        f"## Headline ({tx.SET_LABELS.get(an.primary_set, 'no evaluations yet') if an.primary_set else 'no evaluations yet'})",
        md_table(["Configuration", "Seeds", "SWS [95 % CI]", "Finished"], rows, "_No evaluation yet._"),
        "## Layout", "- `results/<experiment>[-<track>]/seed<k>/`: one trained run\n- `results/baselines/<name>/seed<k>/`: "
        "training-free baselines\n- `results/tracks/`: track-set summaries and scripted-driver tuning\n- `reports/`, `paper/`: "
        "generated reports, figures and the paper",
        f"## Source\n\nCode: <{REPO_URL}>. Author: Libor \"Burny\" Burian. Licence: MIT.",
    ]
    clips = run_clips(res.paths.repo, readme_only=True)  # the README's clips, controls included; MP4s under videos/
    if clips:
        parts[-1:-1] = ["## Videos", "The README's clips: official evaluation runs replayed in the browser game, chosen by fixed "
                        "rules (best, typical and failure cases, then the controls). Each model's own clips are on its model card.",
                        video_section(clips, lambda c: f"https://huggingface.co/datasets/{HF_RESULTS_REPO}/resolve/main/videos/"
                                                       f"{Path(c['mp4']).name}")]
    return "\n\n".join(_clean(p) for p in parts) + "\n"
