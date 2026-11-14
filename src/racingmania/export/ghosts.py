"""Recorded runs into the browser game as AI ghosts.

Local purpose: copy a run's recorded runs (``results/<run>/replays/<track>.json``, written by the evaluator,
runs.save_replays) to ``web/assets/ghosts/<track>/<name>.json`` and list them in ``web/assets/index.json``.

Global context: ghosts are the replay format of docs/formats.md; the game shows them next to the player
and plays them with "Watch replay" (web/js/replay.js). Finished runs feed the Author medal; unfinished runs
(controls, demonstrations, the README's failure clips) are kept only when asked and are flagged
``finished: false`` with their end reason, so the game ends them where the recorded run ended.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..utils.paths import WEB_ASSETS


def export_ghosts(index: dict, res_dir: Path, name: str, keep_unfinished: bool) -> list[str]:
    """Write ``res_dir``'s recorded campaign runs as ghosts called ``name`` and register them in ``index``.
    Returns the ghost paths (relative to web/, as in index.json)."""
    ghosts = index.setdefault("ghosts", {})
    ends = ({r["track"]: r["end"] for r in csv.DictReader(open(res_dir / "eval_campaign.csv", encoding="utf-8"))}
            if (res_dir / "eval_campaign.csv").exists() else {})  # end reason of each recorded campaign run
    written = []
    for rep in sorted((res_dir / "replays").glob("*.json")) if (res_dir / "replays").exists() else []:
        d = json.loads(rep.read_text(encoding="utf-8"))
        if not d.get("finished") and not keep_unfinished:
            continue
        rel = f"assets/ghosts/{d['track']}/{name}.json"
        dst = WEB_ASSETS.parent / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"model": name, "time": d["time"], "finished": bool(d.get("finished")),
                                   "end": ends.get(d["track"], "finished" if d.get("finished") else "fell"), "states": d["states"]}),
                       encoding="utf-8")
        ghosts[d["track"]] = sorted(set(ghosts.get(d["track"], [])) | {rel})
        written.append(rel)
    return written
