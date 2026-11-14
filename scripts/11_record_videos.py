"""Record each model's best lap in the browser game as MP4 and GIF, and write the README video section.

Usage (Windows, game served locally):
  uv run python -m http.server 8123 --directory web
  uv run python scripts/11_record_videos.py --model specialist-campaign-01-seed0 --model generalist-seed0 \
      --lap generalist-seed0:test-2000386

Global context: ``--model`` records the model's best lap: its finished campaign track with the highest
score (scripted driver's reference time ÷ the model's time) in its official evaluation,
``results/<run>/eval_campaign.csv``. ``--lap MODEL:TRACK`` records a given track, e.g. a held-out test
track added to the menu by scripts/12_export_unseen_track.py (its official time is read from
``eval_test.csv``, and the menu entry's note, which states how the track was chosen, goes into the
caption). Each lap is recorded in the game's "Watch AI" mode, i.e. the
exported ONNX policy driving the browser's own cannon-es physics (docs/formats.md, "Policy model"),
the same run a visitor sees on GitHub Pages. Playwright records the page (``record_video_dir``,
https://playwright.dev/python/docs/videos); ffmpeg trims the recording to the lap plus the finish
screen and writes an H.264 MP4 and a palette-optimised GIF
(https://ffmpeg.org/ffmpeg-filters.html#palettegen). The game's lap time is checked against the
official one, and the README section between the VIDEOS markers is rewritten from these numbers
(GitHub READMEs do not play videos stored in the repository, so each GIF links to its MP4).
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DT = 0.01  # physics tick (s), web/assets/vehicle.json
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--url", default="http://127.0.0.1:8123/")
ap.add_argument("--model", action="append", default=[], help="model name in web/assets/index.json: record its best campaign lap")
ap.add_argument("--lap", action="append", default=[], help="MODEL:TRACK: record this track")
ap.add_argument("--out", default=str(ROOT / "docs" / "videos"))
ap.add_argument("--headed", action="store_true", help="show the browser window instead of rendering headless")
ap.add_argument("--gif-width", type=int, default=420)
ap.add_argument("--gif-fps", type=int, default=10)
a = ap.parse_args()
out_dir = Path(a.out)
out_dir.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
index = json.loads((ROOT / "web" / "assets" / "index.json").read_text(encoding="utf-8"))
models = {m["name"]: m for m in index["models"]}
menu = {t["name"]: t for t in index["tracks"]}


def lap_info(model: str, track: str | None = None) -> dict:
    """Official result of ``model`` on ``track`` (default: its highest-scoring finished campaign track)."""
    run = models[model]["run"]
    split = "campaign" if track is None or track.startswith("campaign") else track.rsplit("-", 1)[0]
    rows = [r for r in csv.DictReader(open(ROOT / "results" / run / f"eval_{split}.csv", encoding="utf-8")) if r["finished"] == "1"]
    r = max(rows, key=lambda r: float(r["score"])) if track is None else next(r for r in rows if r["track"] == track)
    entry = menu.get(r["track"], {})
    seen = r["track"] in run  # a specialist's run id names its training track; generalists never train on campaign or test tracks
    where = "on its training track" if seen else ("on a hand-made track it never saw in training" if r["track"].startswith("campaign")
                                                  else "on a held-out generated track it never saw in training")
    title = (f"{model}: best lap, " if track is None else f"{model} ") + f"{where} ({entry.get('title', r['track'])})"
    return {"model": model, "run": run, "track": r["track"], "official_s": int(r["ticks"]) * DT, "reference_s": int(r["ref_ticks"]) * DT,
            "title": title, "description": models[model].get("description", ""), "note": entry.get("note", "")}


def record(new_page, model: str, track: str) -> tuple[Path, float, float, dict]:
    """Drive one Watch-AI lap; returns (raw video, lap start s, clip end s, final game state)."""
    ctx, page, t0 = new_page()
    page.goto(a.url)
    page.wait_for_selector("#track-list [data-track]")
    page.click(f'#track-list [data-track="{track}"]')
    page.wait_for_selector("#model-select")
    page.select_option("#model-select", model)
    page.click('button[data-mode="watch"]')
    page.wait_for_selector("#hud", state="visible")
    start = time.time() - t0
    page.wait_for_function("window.racingmania.state.finished === true", timeout=180_000, polling=100)
    page.wait_for_timeout(3000)  # keep the finish screen with the lap time and splits in the clip
    end = time.time() - t0
    final = page.evaluate("window.racingmania.state")
    video = Path(page.video.path())
    ctx.close()  # finalises the video file
    return video, start, end, final


laps = [lap_info(m) for m in a.model] + [lap_info(*l.split(':')) for l in a.lap]
with sync_playwright() as p:
    browser = p.chromium.launch(headless=not a.headed, args=["--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=d3d11"])
    tmp = Path(tempfile.mkdtemp())

    def new_page():
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
        return ctx, ctx.new_page(), time.time()

    for lap in laps:
        raw, start, end, final = record(new_page, lap["model"], lap["track"])
        stem = out_dir / f"{lap['model']}_{lap['track']}"
        trim = ["-ss", f"{max(0.0, start - 0.5):.2f}", "-to", f"{end:.2f}", "-i", str(raw)]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *trim, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                        "-preset", "slow", "-movflags", "+faststart", "-an", f"{stem}.mp4"], check=True)
        gif_filter = (f"fps={a.gif_fps},scale={a.gif_width}:-1:flags=lanczos,split[s0][s1];"
                      "[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *trim, "-vf", gif_filter, f"{stem}.gif"], check=True)
        lap.update(game_s=round(final["time"], 2), finished=bool(final["finished"]), respawns=final["respawns"],
                   mp4=f"{stem.relative_to(ROOT).as_posix()}.mp4", gif=f"{stem.relative_to(ROOT).as_posix()}.gif")
        print(f"{lap['model']} on {lap['track']}: game {lap['game_s']:.2f} s, official {lap['official_s']:.2f} s, "
              f"scripted driver {lap['reference_s']:.2f} s; GIF {Path(lap['gif']).name}", flush=True)
    browser.close()
    shutil.rmtree(tmp, ignore_errors=True)
(out_dir / "videos.json").write_text(json.dumps(laps, indent=2), encoding="utf-8")

# README section: one row per lap, GIF linked to its MP4, description with the numbers above.
def caption(l: dict) -> str:
    text = (f"**{l['title']}.** {l['description']}. {l['official_s']:.2f} s, against {l['reference_s']:.2f} s for the scripted driver "
            f"({l['reference_s'] / l['official_s']:.2f}× its speed).")
    if l["note"]:
        text += f" {l['note']}"
    if abs(l["game_s"] - l["official_s"]) >= 0.005:  # the browser should reproduce the official lap; say so if it did not
        text += f" The browser lap took {l['game_s']:.2f} s."
    return text


rows_md = "\n".join(f"| [![{l['title']}]({l['gif']})]({l['mp4']}) | {caption(l)} |" for l in laps)
section = ("<!-- VIDEOS:START -->\n"
           "Laps driven live in the browser game by the exported policies (click a clip for the full-resolution MP4). The times are "
           "those of the official evaluation, which the browser reproduces. Generated by `scripts/11_record_videos.py` from "
           "`results/<run>/eval_*.csv`.\n\n"
           f"| Clip | What it shows |\n|---|---|\n{rows_md}\n<!-- VIDEOS:END -->")
readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8")
if "<!-- VIDEOS:START -->" in text:
    text = re.sub(r"<!-- VIDEOS:START -->.*?<!-- VIDEOS:END -->", lambda _: section, text, flags=re.S)
else:  # first run: place the section right after the game screenshot near the top
    anchor = "![RacingMania in the browser](docs/images/game.png)\n"
    text = text.replace(anchor, anchor + "\n" + section + "\n", 1)
readme.write_text(text, encoding="utf-8", newline="\n")
print("README video section updated")
