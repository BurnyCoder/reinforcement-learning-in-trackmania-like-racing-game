"""Record the README video clips in the browser game and write the README video section.

Usage (Windows; the game served locally):
  uv run python -m http.server 8123 --directory web
  uv run python scripts/11_record_videos.py [--clips docs/videos/clips.yaml]
  # Hugging Face card clips (one per model without a README clip; MP4 only, README untouched):
  uv run python scripts/11_record_videos.py --clips docs/videos/cards/clips.yaml --out docs/videos/cards --no-readme --no-gif

Global context: the clips are listed in docs/videos/clips.yaml. Each names a model from
web/assets/index.json and either a fixed track or a selection rule over the model's official
evaluation (``results/<run>/eval_<set>.csv``; rules in ``racingmania.eval.clips``), so the README shows
labelled best, typical and failure cases. Each clip is recorded in the game's "Watch AI" mode, i.e. the
exported ONNX policy driving the browser's own cannon-es physics (docs/formats.md, "Policy model"),
the same run a visitor sees on GitHub Pages. Playwright records the page (``record_video_dir``,
https://playwright.dev/python/docs/videos); ffmpeg trims it and writes an H.264 MP4 and a
palette-optimised GIF (https://ffmpeg.org/ffmpeg-filters.html#palettegen). The browser's lap or fall
time is compared with the official one, and the README section between the VIDEOS markers is rewritten
with one full-width block per clip (GitHub READMEs do not play videos stored in the repository, so
each GIF links to its MP4; a table would shrink the clips to thumbnails on phones). ``videos.json`` in the
output folder keeps each clip's files and caption, which the Hugging Face cards reuse (report/cards.py).
"""
import argparse
import csv
from contextlib import nullcontext as _nullcontext
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml
from playwright.sync_api import sync_playwright

from racingmania.eval.clips import select_track, variant_text
from racingmania.report.features import geometry_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DT = 0.01  # physics tick (s), web/assets/vehicle.json
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--url", default="http://127.0.0.1:8123/")
ap.add_argument("--clips", default=str(ROOT / "docs" / "videos" / "clips.yaml"))
ap.add_argument("--out", default=str(ROOT / "docs" / "videos"))
ap.add_argument("--headed", action="store_true", help="show the browser window instead of rendering headless")
ap.add_argument("--gif-width", type=int, default=400)
ap.add_argument("--gif-fps", type=int, default=8)
ap.add_argument("--captions-only", action="store_true", help="rewrite the README section from the last recording (videos.json)")
ap.add_argument("--record", type=int, nargs="*", default=None, help="record only these clips (1-based); reuse the others from videos.json")
ap.add_argument("--no-readme", action="store_true", help="do not rewrite the README video section (clips for the Hugging Face cards)")
ap.add_argument("--no-gif", action="store_true", help="write only the MP4 (the Hugging Face cards play MP4 in a <video> tag)")
a = ap.parse_args()
out_dir = Path(a.out).resolve()  # absolute, so file paths can be stored relative to the repository root
out_dir.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
index = json.loads((ROOT / "web" / "assets" / "index.json").read_text(encoding="utf-8"))
menu = {t["name"]: t for t in index["tracks"]}
SET_NAMES = {"campaign": "hand-made campaign track", "test": "held-out generated test track", "val": "generated validation track",
             "oor": "held-out out-of-range test track"}


def model_text(run: str) -> str:
    """One sentence on what the model is and how it was trained, from its run.json."""
    if run.startswith("demos/untrained-sampled/"):  # a demonstration: demo.json instead of run.json (not in the report)
        d = json.loads((RESULTS / run / "demo.json").read_text(encoding="utf-8"))
        return (f"`{run}`: the untrained control network (random initial weights, seed {d['network_seed']}) with each action "
                f"sampled from its probabilities (sampling seed {d['sample_seed']}) instead of the most probable one")
    info = json.loads((RESULTS / run / "run.json").read_text(encoding="utf-8"))
    if run.startswith("baselines/scripted/"):
        tuned = json.loads((RESULTS / "tracks" / "scripted_tuning.json").read_text(encoding="utf-8")).get("chosen", {})
        return (f"`{run}`: the scripted driver, a hand-written controller (pure-pursuit steering and a curvature-based speed profile, "
                f"see [The scripted driver](#the-scripted-driver)) with its 2 parameters tuned on validation tracks before any training"
                + (f" (a_lat = {tuned['a_lat']} m/s², look-ahead gain = {tuned['lookahead_gain']} s)" if tuned else "")
                + "; nothing is learned, and its time on each track is the reference that scores 1.0")
    if run.startswith("baselines/random/"):
        return (f"`{run}`: random actions, a uniformly random key combination every agent step from a seeded generator "
                f"(seed {info['seed']}); no network and no training; a control")
    if run.startswith("baselines/untrained/"):
        return (f"`{run}`: the same network as the trained agents (2 × 256 units) with random initial weights "
                f"(seed {info['seed']}) and no training; a control")
    trained_on = "1 track" if info.get("n_train_tracks") == 1 else f"{info['n_train_tracks']:,} generated training tracks"
    if info.get("track"):
        trained_on = f"{info['track']} only"
    exp_cfg = ROOT / "configs" / "experiments" / f"{info['experiment']}.yaml"  # warm starts are recorded in the config
    init = (yaml.safe_load(exp_cfg.read_text(encoding="utf-8")) or {}).get("init_from") if exp_cfg.exists() else None
    if init:
        trained_on += f", starting from the weights of `{init}`"
    steps = info["global_step"]
    amount = f"{round(steps / 1e9, 1):g} billion" if steps >= 1e9 else f"{steps / 1e6:,.0f} million"  # budgets end at the first rollout past them
    return (f"`{run}`: {info['algo'].upper()}, trained by reinforcement learning for {amount} agent steps on {trained_on} "
            f"(seed {info['seed']}, {info.get('wall_hours', 0):.1f} h on one GPU)")


def map_text(track: str, run: str, split: str) -> str:
    """One sentence on the track: name, kind, length, jumps, banking, and whether the model trained on it."""
    t = json.loads((ROOT / "web" / "assets" / "tracks" / f"{track}.json").read_text(encoding="utf-8"))
    f = geometry_features(t["center"], t["left"], t["gap"])
    seen = track in run  # a specialist's run id names its training track; generalists never train on campaign or test tracks
    jumps = "no jumps" if f["jumps"] == 0 else f"{f['jumps']} jump{'' if f['jumps'] == 1 else 's'}"
    if run.startswith("baselines/scripted/"):
        relation = ("the scripted driver is not trained; it drove this track before at three speed settings to certify it, "
                    "and its run at the normal setting set the reference time")
    elif run.startswith(("baselines/untrained/", "demos/untrained-sampled/")):
        relation = "this network was never trained on any track"
    elif run.startswith("baselines/random/"):
        relation = "random actions ignore the track entirely"
    else:
        relation = "the track this model was trained on" if seen else "this model never saw it during training"
    title = menu.get(track, {}).get("title", track).split(" (")[0]  # out-of-range titles carry the variant: stated below instead
    variant = ""
    if split == "oor":  # one generator factor beyond the training range (tracks/generator.py OOR_VARIANTS)
        feats = {r["track"]: r for r in csv.DictReader(open(RESULTS / "tracks" / "oor_features.csv", encoding="utf-8"))}
        variant = f" with one factor beyond the training range, {variant_text(feats[track]['variant'])},"
    return (f"{title} (`{track}`), a {SET_NAMES[split]}{variant or ','} {f['length_m']:,.0f} m long with "
            f"{jumps} and up to {f['max_bank_deg']:.0f}° of banking; {relation}")


def clip_info(c: dict) -> dict:
    """Resolve a clip from clips.yaml: track (fixed or by rule), official result and caption parts."""
    run = c["run"]
    rule_run = c.get("rule_run", run)  # whose official evaluation the selection rule reads
    who = f"`{rule_run}`" if c.get("rule_run") else "it"
    track, why = (c["track"], "") if c.get("track") else select_track(RESULTS, rule_run, c["split"], c["rule"], who, c.get("variant"))
    if c.get("rule_run"):
        why = f"the same track as the {rule_run} clip: {why}"
    c = {**c, "model": c["replay"], "mode": "replay"}
    row = next(r for r in csv.DictReader(open(RESULTS / run / f"eval_{c['split']}.csv", encoding="utf-8")) if r["track"] == track)
    return {**c, "until": c.get("until", "finish"), "run": run, "track": track, "why": why, "row": row,
            "official_s": int(row["ticks"]) * DT, "reference_s": int(row["ref_ticks"]) * DT}


def record(new_page, model: str, track: str, until: str, mode: str = "watch") -> tuple[Path, float, float, dict]:
    """Drive one Watch-AI run of ``model``, or (mode "replay") play its recorded run on ``track``; returns (raw video,
    run start s, clip end s, game state at the finish or where an unfinished run ended)."""
    ctx, page, t0 = new_page()
    page.goto(a.url)
    page.wait_for_selector("#track-list [data-track]")
    page.click(f'#track-list [data-track="{track}"]')
    # The chosen row of the menu's AI drivers table drives "Watch AI" (its network) and "Watch replay" (its recorded run).
    pick = f'#track-detail input[name="driver"][value="{model}"]'
    page.wait_for_selector(pick, state="attached")
    page.eval_on_selector(pick, "el => el.click()")  # a JS click: the key-hint footer can cover the row
    page.eval_on_selector(f'button[data-mode="{mode}"]', "el => el.click()")
    page.wait_for_selector("#hud", state="visible")
    start = time.time() - t0
    cond = "window.racingmania.state.finished === true" if until == "finish" else "window.racingmania.state.respawns >= 1"
    page.wait_for_function(cond, timeout=180_000, polling=50)
    if until == "out":
        final = page.evaluate("window.racingmania.state")  # read where the run ended: a replay's clock stops there
    page.wait_for_timeout(3000 if until == "finish" else 1500)  # finish screen, or the end-of-run notice
    if until == "finish":  # after the finish screen the race clock has reached the finish and the time is frozen
        final = page.evaluate("window.racingmania.state")
    end = time.time() - t0
    video = Path(page.video.path())
    ctx.close()  # finalises the video file
    return video, start, end, final


clips = [clip_info(c) for c in yaml.safe_load(Path(a.clips).read_text(encoding="utf-8"))["clips"]]
todo = set(range(1, len(clips) + 1)) if a.record is None and not a.captions_only else set(a.record or [])
if len(todo) < len(clips):  # reuse the other clips and the browser times measured when they were recorded
    last = {(v["model"], v["track"]): v for v in json.loads((out_dir / "videos.json").read_text(encoding="utf-8"))}
    for k, c in enumerate(clips, 1):
        if k not in todo:
            c.update({f: last[(c["model"], c["track"])].get(f) for f in ("game_s", "game_finished", "mp4", "gif")})
with sync_playwright() if todo else _nullcontext() as p:
    browser = p.chromium.launch(headless=not a.headed, args=["--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=d3d11"]) if p else None
    tmp = Path(tempfile.mkdtemp())

    def new_page():
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
        return ctx, ctx.new_page(), time.time()

    for c in [c for k, c in enumerate(clips, 1) if k in todo] if p else []:
        raw, start, end, final = record(new_page, c["model"], c["track"], c["until"], c["mode"])
        stem = out_dir / f"{c['model']}_{c['track']}"
        trim = ["-ss", f"{max(0.0, start - 0.5):.2f}", "-to", f"{end:.2f}", "-i", str(raw)]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *trim, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                        "-preset", "slow", "-movflags", "+faststart", "-an", f"{stem}.mp4"], check=True)
        gif_filter = (f"fps={a.gif_fps},scale={a.gif_width}:-1:flags=lanczos,split[s0][s1];"
                      "[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4")
        if not a.no_gif:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *trim, "-vf", gif_filter, f"{stem}.gif"], check=True)
        c.update(game_s=round(final["time"], 2), game_finished=bool(final["finished"]),
                 mp4=f"{stem.relative_to(ROOT).as_posix()}.mp4", gif=None if a.no_gif else f"{stem.relative_to(ROOT).as_posix()}.gif")
        print(f"{c['title']}: {c['model']} on {c['track']}: browser {c['game_s']:.2f} s ({'finished' if c['game_finished'] else 'out'}), "
              f"official {c['row']['end']} at {c['official_s']:.2f} s", flush=True)
    if browser:
        browser.close()
    shutil.rmtree(tmp, ignore_errors=True)


def compare_text(c: dict) -> str:
    """Sentence from a random-vs-sampled comparison file: both drivers' distributions, the test, and this clip's place."""
    d = json.loads((RESULTS / c["compare"]).read_text(encoding="utf-8"))
    r, u, tests = d["drivers"]["random_actions"], d["drivers"]["untrained_sampled"], d["tests"]
    me = d["drivers"][c["compare_as"]]
    p10, p50, p90 = me["time_p10_p50_p90_s"]
    t = c["official_s"]
    if t <= 2.0 + 1e-9:
        place = f"an unlucky one: {me['share_out_at_2s']:.0%} of its runs end at 2.0 s like this one"
    elif t >= p90:
        place = "a lucky one, above the 90th percentile of its runs"
    elif t <= p10:
        place = "an unlucky one, in the bottom tenth of its runs"
    else:
        place = f"an ordinary one (the middle half of its runs is out after {p50:.1f} s or so)"
    return (f"in {d['runs']} runs each on this track, random actions are out after {r['time_mean_s']:.1f} s on average "
            f"(median {r['time_p10_p50_p90_s'][1]:.1f} s, 10–90 % range {r['time_p10_p50_p90_s'][0]:.1f}–{r['time_p10_p50_p90_s'][2]:.1f} s) and "
            f"the untrained network with sampled actions after {u['time_mean_s']:.1f} s (median {u['time_p10_p50_p90_s'][1]:.1f} s, "
            f"{u['time_p10_p50_p90_s'][0]:.1f}–{u['time_p10_p50_p90_s'][2]:.1f} s), with no detectable difference between them (Mann–Whitney "
            f"p = {tests['time_mannwhitney_p']:.2f} for the time, {tests['progress_mannwhitney_p']:.2f} for the progress). The gap between "
            f"the two random clips is therefore consistent with luck, and this run is {place} "
            f"(`results/{c['compare']}`, `scripts/14_random_vs_sampled.py`).")


def eval_text(c: dict) -> str:
    """The official result, the scripted driver's reference and what the browser showed."""
    r = c["row"]
    how = ("deterministic controller" if c["run"].startswith("baselines/scripted/") else
           "seeded random keys" if c["run"].startswith("baselines/random/") else
           "most probable action, untrained weights" if c["run"].startswith("baselines/untrained/") else
           "most probable action, final checkpoint")
    src = (f"official evaluation in `results/{c['run']}/eval_{c['split']}.csv` (standing start, {how})"
           if not c["run"].startswith("demos/") else
           f"demonstration run through the same evaluator, recorded in `results/{c['run']}/eval_{c['split']}.csv` (standing start, "
           f"sampled actions; not scored)")
    if r["finished"] == "1" and c["run"].startswith("baselines/scripted/"):
        res = (f"finished in {c['official_s']:.2f} s (the track's reference time, from the scripted driver's separate "
               f"certification run, is {c['reference_s']:.2f} s)")
    elif r["finished"] == "1":
        res = (f"finished in {c['official_s']:.2f} s; the scripted driver needs {c['reference_s']:.2f} s, so this is "
               f"{c['reference_s'] / c['official_s']:.2f}× its speed")
    else:
        # progress is the arc-length position on the centreline; every start line is at sample 3, about the 6 m mark
        res = (f"did not finish: {r['end']} after {c['official_s']:.2f} s at the {float(r['progress']):,.0f} m mark of "
               f"the road, whose start line is at about 6 m; the scripted driver finishes in {c['reference_s']:.2f} s")
    for other in c.get("also", []):  # another run's official result on the same track, for comparison
        o = next(x for x in csv.DictReader(open(RESULTS / other / f"eval_{c['split']}.csv", encoding="utf-8")) if x["track"] == c["track"])
        res += (f"; on the same track `{other}` " + (f"finished in {int(o['ticks']) * DT:.2f} s" if o["finished"] == "1"
                                                      else f"did not finish ({o['end']} after {int(o['ticks']) * DT:.2f} s)"))
    check = "" if abs(c["game_s"] - c["official_s"]) < 0.005 else f" (in the browser the replay ended at {c['game_s']:.2f} s)"
    return f"{src}: {res}. The clip replays this run{check}"


blocks, group = [], None
for k, c in enumerate(clips, 1):
    if c.get("group") and c["group"] != group:  # section heading when a new group starts (positive, negative, controls)
        group = c["group"]
        blocks.append(f"### {group}")
    # The caption's parts, kept in videos.json for the Hugging Face cards (report/cards.py) as well as the README.
    c["caption"] = {"What you see": c["explain"].strip(), "Model": f"{model_text(c['run'])}.",
                    "Map": f"{map_text(c['track'], c['run'], c['split'])}.", "Evaluation": f"{eval_text(c)}."}
    if c["why"]:
        c["caption"]["Why this clip"] = f"chosen by a fixed rule as {c['why']} in its official evaluation."
    if c.get("compare"):  # many-run comparison (scripts/14_random_vs_sampled.py): where this single run falls
        c["caption"]["Over many runs"] = compare_text(c)
    lines = [f"#### {k}. {c['title']}", "", f"[![{c['title']}: {c['model']} on {c['track']}]({c['gif']})]({c['mp4']})", "",
             *[f"- **{part}:** {text}" for part, text in c["caption"].items()]]
    blocks.append("\n".join(lines))
(out_dir / "videos.json").write_text(json.dumps([{k: v for k, v in c.items() if k != "row"} | {"official": c["row"]} for c in clips],
                                               indent=2), encoding="utf-8")
section = ("<!-- VIDEOS:START -->\n## Videos\n\n"
           "Each clip replays an official evaluation run in the browser game: the car states the evaluator recorded while scoring "
           "the run, shown by the game's renderer (\"Watch replay\"). Each clip is chosen by a fixed rule from the official "
           "evaluation and labelled with its result; click a clip for the full-resolution MP4. In the game the trained policies also "
           "drive live (\"Watch AI\", \"Race vs AI\"), where the browser's 64-bit physics can make a long lap differ slightly from the "
           "official 32-bit GPU run. Generated by `scripts/11_record_videos.py` from `docs/videos/clips.yaml` and "
           "`results/<run>/eval_*.csv`.\n\n"
           + "\n\n".join(blocks) + "\n<!-- VIDEOS:END -->")
if not a.no_readme:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "<!-- VIDEOS:START -->" in text, "README has no VIDEOS markers"
    text = re.sub(r"<!-- VIDEOS:START -->.*?<!-- VIDEOS:END -->", lambda _: section, text, flags=re.S)
    readme.write_text(text, encoding="utf-8", newline="\n")
    print("README video section updated")
print(f"{out_dir / 'videos.json'} written ({len(clips)} clips)")
