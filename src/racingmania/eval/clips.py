"""Which lap to show: fixed selection rules over a run's official evaluation, for the README videos.

Local purpose: choose the track of a video clip from ``results/<run>/eval_<set>.csv`` by a rule that is
written down before looking at the clip, so the README shows labelled best, typical and failure cases
instead of hand-picked ones.

Global context: used by scripts/12_export_unseen_track.py (puts the chosen held-out track into the
game) and scripts/11_record_videos.py (records the clips listed in docs/videos/clips.yaml). Track
lengths and jump counts come from ``results/tracks/<set>_features.csv`` (report/features.py).
"""

from __future__ import annotations

import csv
from pathlib import Path

RULES = {
    "best": "the highest-scoring finished track",
    "median_jump": "the median-scoring finished track among those with at least one jump",
    "mid_fail": "the fall whose position is closest to the middle of its track",
}


def _rows(results: Path, run: str, split: str) -> dict[str, dict]:
    return {r["track"]: r for r in csv.DictReader(open(results / run / f"eval_{split}.csv", encoding="utf-8"))}


def _features(results: Path, split: str) -> dict[str, dict]:
    path = results / "tracks" / f"{split}_features.csv"
    return {r["track"]: r for r in csv.DictReader(open(path, encoding="utf-8"))} if path.exists() else {}


def select_track(results: Path, run: str, split: str, rule: str, who: str = "it") -> tuple[str, str]:
    """Track name chosen by ``rule`` from the run's official evaluation on ``split``, and a sentence
    stating the rule with its population size (for the caption; ``who`` names the run in it)."""
    rows = _rows(results, run, split)
    finished = [(float(r["score"]), t) for t, r in rows.items() if r["finished"] == "1"]
    whose = "its" if who == "it" else f"{who}'s"
    if rule == "best":
        _, track = max(finished)
        if len(finished) == 1:
            return track, f"{whose} only finished {split} track"
        return track, f"{RULES[rule]} of the {len(finished)} {split} tracks {who} finished"
    feats = _features(results, split)
    if rule == "median_jump":
        cand = sorted((s, t) for s, t in finished if int(feats[t]["jumps"]) >= 1)
        return cand[len(cand) // 2][1], f"{RULES[rule]} ({len(cand)} {split} tracks with a jump that {who} finished)"
    if rule == "mid_fail":
        falls = [(abs(float(r["progress"]) / float(feats[t]["length_m"]) - 0.5), t) for t, r in rows.items() if r["end"] == "fell"]
        return min(falls)[1], f"{RULES[rule]} among {whose} {len(falls)} falls on {split} tracks"
    raise ValueError(f"unknown rule {rule!r}; known: {sorted(RULES)}")
