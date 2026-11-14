"""End-to-end tests of the report pipeline (``racingmania.report.build``) on synthetic result files.

Global context: ``scripts/07_make_report.py`` turns ``results/`` into every table, figure, report,
the README block, the paper and the website data; no number may be typed by hand. These tests
generate plausible fake results in a temporary folder (two generalist seeds, a pilot, a blind
control, a single-track ablation, specialists and the training-free baselines), build everything
there, and check that the files exist, that the numbers printed in the README block equal the
means of the CSVs, that the hypothesis table follows the preregistered tests, and that the build
also works with no results at all. ``make_fixture`` is reusable for manual inspection.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from racingmania.report import build
from racingmania.report.pdf import build_pdf

EVAL_COLS = ["track", "finished", "ticks", "time", "ref_ticks", "score", "end", "progress", "max_speed", "flag_overspeed",
             "max_coast_energy_gain", "flag_energy_gain"]
SETS = {"test": ("test", 2_000_000, 60), "val": ("val", 1_000_000, 30), "train_sample": ("train", 0, 40),
        "oor": ("oor", 3_000_000, 20)}
CAMPAIGN = [f"campaign-0{i}" for i in range(1, 6)]
# (finish probability, mean speed factor = ref/agent time) per experiment
QUALITY = {"generalist": (0.8, 1.05), "blind": (0.5, 0.9), "size-1": (0.25, 0.85), "specialist": (0.3, 0.9),
           "baseline-scripted": (1.0, 1.0), "baseline-random": (0.0, 0.0), "baseline-untrained": (0.0, 0.0)}


def _tracks(s: str) -> list[tuple[str, int]]:
    if s == "campaign":
        return [(n, 2500 + 300 * i) for i, n in enumerate(CAMPAIGN)]
    prefix, lo, n = SETS[s]
    rng = np.random.default_rng(lo + 7)  # reference ticks are a property of the track, shared by all agents
    return [(f"{prefix}-{lo + i}", int(rng.integers(2000, 4000))) for i in range(n)]


def _write_eval(path: Path, experiment: str, seed: int, s: str, own_track: str | None = None):
    rng = np.random.default_rng([hash(experiment) % 2**31, seed, len(s)])
    p, speed = QUALITY[experiment]
    rows = []
    for name, ref in _tracks(s):
        pf, sp = (1.0, 1.15) if name == own_track else (p, speed)
        fin = experiment == "baseline-scripted" or rng.random() < pf
        ticks = ref if experiment == "baseline-scripted" else (int(round(ref / max(0.3, sp + rng.normal(0, 0.05)))) if fin else int(ref * 3))
        rows.append({"track": name, "finished": int(fin), "ticks": ticks, "time": round(ticks * 0.01, 2), "ref_ticks": ref,
                     "score": (ref / ticks) if fin else 0.0, "end": "finished" if fin else "fell", "progress": 500.0,
                     "max_speed": 40.0, "flag_overspeed": 0, "max_coast_energy_gain": 0.5, "flag_energy_gain": 0})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EVAL_COLS)
        w.writeheader()
        w.writerows(rows)


def _write_curves(d: Path, seed: int, final: float):
    steps = np.linspace(5e6, 2e8, 40)
    with open(d / "train_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "global_step", "wall_hours", "sps", "lr", "explained_variance", "kl", "clipfrac", "pg_loss",
                    "v_loss", "entropy", "episodes", "ep_return", "ep_finish_rate", "ep_finish_rate_standing", "ep_progress",
                    "ep_fell", "ep_stalled", "ep_steps"])
        for i, st in enumerate(steps):
            f = final * (1 - np.exp(-3 * i / 40))
            w.writerow([i, int(st), st / 1.5e5 / 3600, 150000, 1e-3, 0.9, 0.01, 0.1, -0.01, 0.5, 1.5, 100, 50 * f + seed,
                        f, f, 400 * f, 1 - f, 0.0, 800])
    with open(d / "val_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sws", "finish_rate", "speed_ratio_finished", "n_tracks", "flags_overspeed", "flags_energy_gain",
                    "iteration", "global_step", "wall_hours"])
        for i, st in enumerate(steps[::5]):
            f = final * (1 - np.exp(-3 * i / 8))
            w.writerow([f, f, 1.0, 30, 0, 0, i * 5, int(st), st / 1.5e5 / 3600])


def _run(results: Path, rid: str, experiment: str, seed: int, sets: list[str], phase: str, track: str | None = None,
         algo: str = "ppo", n_train: int | None = 50000, curves: bool = True):
    d = results / rid
    for s in sets:
        _write_eval(d / f"eval_{s}.csv", experiment, seed, s, own_track=track)
    if curves:
        _write_curves(d, seed, QUALITY[experiment][0])
    run = {"experiment": experiment, "seed": seed, "track": track, "status": "evaluated", "algo": algo, "phase": phase,
           "description": f"synthetic {experiment}", "git_commit": "0" * 40, "physics": "v1",
           "hardware": {"gpu": "test"}, "global_step": 200_000_000, "wall_hours": 0.5, "n_train_tracks": n_train}
    if algo == "ppo":
        run["ppo"] = {"total_steps": 200_000_000, "num_envs": 16384, "num_steps": 32, "lr": 0.001, "gamma": 0.99,
                      "hidden": [256, 256], "action_mode": "factored", "seed": seed}
        run["env"] = {"window_k": 20, "blind": experiment == "blind", "time_cost": 0.0, "fall_penalty": 0.0, "random_start_prob": 0.5}
    (d / "run.json").write_text(json.dumps(run), encoding="utf-8")


def make_fixture(results: Path):
    """Write synthetic results: the layout and columns of runs.py, baselines.py, evaluator.py and ppo.py."""
    full = ["val", "train_sample", "campaign", "test", "oor"]
    _run(results, "generalist/seed0", "generalist", 0, ["val", "train_sample", "campaign"], "1-first-models")
    _run(results, "generalist/seed1", "generalist", 1, full, "2-single-seed")
    _run(results, "generalist/seed2", "generalist", 2, full, "6-seed-sensitivity")
    _run(results, "blind/seed1", "blind", 1, full, "2-single-seed")
    _run(results, "size-1/seed1", "size-1", 1, ["test", "val"], "2-single-seed", n_train=1)
    _run(results, "specialist-campaign-01/seed0", "specialist", 0, ["campaign", "val"], "1-first-models", track="campaign-01", n_train=1)
    _run(results, "specialist-campaign-01/seed1", "specialist", 1, ["campaign", "test"], "2-single-seed", track="campaign-01", n_train=1)
    for name, seeds in (("scripted", [0]), ("random", [0, 1]), ("untrained", [0, 1])):
        for k in seeds:
            _run(results, f"baselines/{name}/seed{k}", f"baseline-{name}", k, ["val", "test", "campaign"], "baselines",
                 algo=name, n_train=None, curves=False)
    t = results / "tracks"
    t.mkdir(parents=True, exist_ok=True)
    for s, (prefix, lo, n) in SETS.items():
        split = "train" if s == "train_sample" else s
        (t / f"{split}_summary.json").write_text(json.dumps({"split": split, "seed_range": [lo, lo + n], "generated": n + 5,
                                                             "geometric_rejects": 2, "certification_rejects": 3, "kept": n,
                                                             "mean_length_m": 900.0, "mean_ref_time_s": 30.0}), encoding="utf-8")
    rng = np.random.default_rng(3)
    with open(t / "test_features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "jumps", "max_bank_deg", "height_range_m", "length_m", "variant"])
        for name, _ in _tracks("test"):
            w.writerow([name, int(rng.integers(0, 3)), round(float(rng.uniform(0, 14)), 2), round(float(rng.uniform(0, 30)), 2), 900.0, "train_ranges"])
    (results / "bench").mkdir(parents=True, exist_ok=True)
    (results / "bench" / "linux.json").write_text(json.dumps({"platform": "linux", "torch": "2.15.0", "gpu": "test-gpu", "compile": True,
        "runs": [{"envs": 4096, "agent_steps_per_s": 123456.7, "ms_per_step": 33.2, "warmup_s": 40.0}]}), encoding="utf-8")
    (results / "test_access_log.jsonl").write_text(json.dumps({"time": "2026-11-14T00:00:00", "split": "test",
                                                               "policy": "generalist/seed1", "tracks": 60}) + "\n", encoding="utf-8")


def _csv_mean(path: Path, col: str) -> float:
    with open(path, newline="", encoding="utf-8") as fh:
        return float(np.mean([float(r[col]) for r in csv.DictReader(fh)]))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    results, out = root / "results", root / "out"
    make_fixture(results)
    out.mkdir()
    (out / "README.md").write_text("# Project\n\nIntro.\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n\nMore.\n", encoding="utf-8")
    summary = build.build_all(results_dir=results, out_root=out, pdf=False)
    return results, out, summary


def test_all_outputs_exist(built):
    results, out, summary = built
    for rel in ["reports/summary.md", "reports/readme_results.md", "paper/paper.md", "web/assets/data/results.json",
                "reports/figures/sws_by_method.png", "reports/figures/sws_by_method.svg", "reports/figures/curves_generalist.png",
                "reports/figures/training_set_size.png", "reports/figures/finish_by_feature.png",
                "reports/figures/specialists_campaign.png", "reports/figures/seed_spread.png"]:
        assert (out / rel).exists(), rel
    assert len(list((out / "reports" / "experiments").glob("*.md"))) == 12
    assert (results / "HF_DATASET_CARD.md").read_text(encoding="utf-8").startswith("---\nlicense: mit")
    paper = (out / "paper" / "paper.md").read_text(encoding="utf-8")
    assert 'Libor "Burny" Burian' in paper
    assert paper.rstrip().endswith("a human provided the exact specification that was executed.")
    json.loads((out / "web" / "assets" / "data" / "results.json").read_text(encoding="utf-8"))
    summary_md = (out / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "| linux | 2.15.0 | test-gpu | True | 4,096 | 123457 |" in summary_md  # benchmark file -> summary table


def test_readme_block_numbers_match_csvs(built):
    results, out, _ = built
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "old" not in readme.split("<!-- RESULTS:START -->")[1] and readme.endswith("More.\n")
    block = (out / "reports" / "readme_results.md").read_text(encoding="utf-8")
    assert block.strip() in readme
    sws = np.mean([_csv_mean(results / f"generalist/seed{k}/eval_test.csv", "score") for k in (1, 2)])
    fin = np.mean([_csv_mean(results / f"generalist/seed{k}/eval_test.csv", "finished") for k in (1, 2)])
    row = next(l for l in block.splitlines() if l.startswith("| Generalist (PPO) |"))
    assert f"{sws:.3f} [" in row and f"{100 * fin:.1f} %" in row
    scripted = next(l for l in block.splitlines() if l.startswith("| Scripted driver |"))
    assert "1.000 [1.000, 1.000]" in scripted  # the reference scores exactly 1 by construction
    assert f"finished {100 * fin:.1f} % of the 60 unseen test tracks" in block


def test_hypotheses_follow_preregistration(built):
    results, out, _ = built
    data = json.loads((out / "web" / "assets" / "data" / "results.json").read_text(encoding="utf-8"))
    h = {x["id"]: x for x in data["hypotheses"]}
    per_seed = [_csv_mean(results / f"generalist/seed{k}/eval_test.csv", "score") for k in (1, 2)]
    t = sps.ttest_1samp(per_seed, 1.0, alternative="greater")
    assert h["H1"]["estimate"] == pytest.approx(np.mean(per_seed))
    assert h["H1"]["statistic"] == pytest.approx(t.statistic) and h["H1"]["p"] == pytest.approx(t.pvalue)
    assert h["H1"]["lo"] <= h["H1"]["estimate"] <= h["H1"]["hi"]
    blind = _csv_mean(results / "blind/seed1/eval_test.csv", "score")
    assert h["H3"]["estimate"] == pytest.approx(np.mean(per_seed) - blind)
    assert h["H3"]["p"] is None  # Welch's test needs two seeds per group; the blind control has one
    assert all(x["decision"] == "not yet tested" for x in h.values())  # 2 of 10 generalist seeds
    md = (out / "reports" / "summary.md").read_text(encoding="utf-8")
    assert re.search(r"\| H1 \|.*\(preliminary\)", md)
    # Preliminary readings (not tests): the reading follows where the interval lies against the threshold.
    for x in h.values():
        side = ("above" if x["lo"] > x["threshold"] else "below" if x["hi"] < x["threshold"] else "includes")
        assert x["reading"].startswith("preliminary (") and side in x["reading"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "Preliminary readings from the seeds available, not tests" in readme and "Not done yet:" in readme


def test_model_card(built):
    results, _, _ = built
    card = build.model_card("generalist/seed1", results_dir=results)
    assert card.startswith("---\nlibrary_name: pytorch\nlicense: mit\n")
    assert "type: reinforcement-learning" in card and "ppo" in card.split("---")[1]
    sws = _csv_mean(results / "generalist/seed1/eval_test.csv", "score")
    assert f"value: {round(sws, 4)}" in card
    assert "reinforcement-learning-in-trackmania-like-racing-game" in card


def test_build_is_idempotent(built):
    results, out, _ = built
    before = {p: p.read_bytes() for p in [out / "reports" / "summary.md", out / "web" / "assets" / "data" / "results.json",
                                          out / "paper" / "paper.md", out / "reports" / "figures" / "sws_by_method.svg"]}
    build.build_all(results_dir=results, out_root=out, pdf=False)
    for p, b in before.items():
        assert p.read_bytes() == b, p


def test_build_with_no_results(tmp_path):
    out = build.build_all(results_dir=tmp_path / "results", out_root=tmp_path / "out", pdf=False)
    summary = (tmp_path / "out" / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "No model has been evaluated yet" in summary and "Not yet available" in summary
    assert set(out["hypotheses"].values()) == {"not yet tested"}
    assert "results are not yet available" in (tmp_path / "out" / "paper" / "paper.md").read_text(encoding="utf-8").lower()
    assert out["figures"] == []


def test_pdf_step_never_raises(tmp_path, monkeypatch):
    import racingmania.report.pdf as pdf
    md = tmp_path / "paper.md"
    md.write_text("# x\n", encoding="utf-8")
    monkeypatch.setattr(pdf, "find_pandoc", lambda: None)
    assert build_pdf(md, tmp_path) is None
