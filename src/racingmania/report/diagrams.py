"""Paper-style diagrams of the method (matplotlib, PNG + SVG), with every number read from the code, configs or results.

Local purpose: four figures for the README and the paper.
- ``architecture``: the actor-critic network, ML-paper style: the observation vector as a stacked block (height
  proportional to its size), the running normaliser, the actor and critic MLPs (block heights proportional to
  layer widths), the two categorical action heads, and the parameter counts.
- ``system``: the pipeline from the seeded track generator to the batched simulator and PPO, the evaluator, the
  report, and the browser game that runs the same physics and the exported policy.
- ``protocol``: the track sets (seed ranges, generated and certified counts), what each set is used for, and the
  primary metric.
- ``run_status``: runs evaluated, in progress and postponed, per phase of the preregistered run matrix.

Global context: called by ``build.build_all`` next to figures.py (same palette and ``_save``), so one command
(``scripts/07_make_report.py``) regenerates every table, figure and diagram. Sizes come from ``env.obs`` (observation
layout), ``rl.ppo.PPOConfig`` (network and PPO settings), ``env.vec_env.EnvConfig`` (episodes), the vehicle file
(tick and action rate) and ``results/tracks/*_summary.json`` (track counts); the status chart from the run statuses.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ..env import obs as O
from ..env.vec_env import EnvConfig
from ..rl.ppo import PPOConfig
from ..utils.logging import get_logger
from .figures import INK, INK2, MUTED, SLOTS, SURFACE, _save

log = get_logger(__name__)
LIGHT = {"blue": "#dbe8f8", "orange": "#fbe3d7", "green": "#d6f1e6", "yellow": "#fbefcc", "grey": "#efeeea", "purple": "#e3dff8"}


def _box(ax, x, y, w, h, text, fc, ec=INK2, size=8.5, weight="normal", color=INK, ls="-", lw=1.0):
    """Rounded box centred at (x, y) with centred text (axes in data units)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=color, linespacing=1.25)


def _arrow(ax, p0, p1, text=None, color=INK2, ls="-", rad=0.0, size=7.5, both=False, off=(0, 0.12)):
    """Arrow from p0 to p1 with an optional label at its midpoint."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=10, color=color, lw=1.0,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=size, color=INK2)


def _canvas(w, h, figsize, y0=0.0):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w); ax.set_ylim(y0, h); ax.axis("off"); ax.grid(False)
    return fig, ax


def _params(sizes: list[int]) -> int:
    """Weights + biases of an MLP with layer sizes [in, h1, ..., out]."""
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


# ----------------------------------------------------------------------------- network architecture
def architecture(out: Path) -> dict:
    """The actor-critic network as in ML papers: blocks scaled by size, layer types and parameter counts."""
    k, hidden = O.WINDOW_K, list(PPOConfig().hidden)
    n_obs, n_act = O.obs_size(k), 6  # two 3-way categorical heads: steer (left, none, right) and pedal (gas, coast, brake)
    groups = [("velocity", 3), ("spin", 3), ("gravity direction", 3), ("steering angle", 1), ("wheel contacts", 4),
              ("previous action (one-hot)", 6), (f"{k} centreline points ahead (x, y, z)", 3 * k),
              (f"road-left vector at each point", 3 * k), ("gap flag at each point", k)]
    assert sum(n for _, n in groups) == n_obs, "observation layout out of sync with env/obs.py"
    actor, critic = [n_obs, *hidden, n_act], [n_obs, *hidden, 1]
    fig, ax = _canvas(16.6, 9, (12.9, 7.4), y0=-0.7)
    ONNX = SLOTS[2]  # green outline: part of the ONNX graph the browser game runs
    scale = 6.2 / n_obs  # data units per observation element (the observation block is 6.2 units tall)
    # Observation vector: a stacked block, each group's height proportional to its length. The six small car-state
    # groups share one label (their blocks are too thin for one label each).
    x0, y = 2.15, 7.75
    cols = [LIGHT["blue"]] * 3 + [LIGHT["grey"], LIGHT["grey"], LIGHT["grey"]] + [LIGHT["green"], LIGHT["yellow"], LIGHT["orange"]]
    n_state = sum(n for _, n in groups[:6])
    for i, ((name, n), fc) in enumerate(zip(groups, cols)):
        h = n * scale
        ax.add_patch(Rectangle((x0 - 0.35, y - h), 0.7, h, fc=fc, ec=INK2, lw=0.7))
        if i >= 6:
            ax.text(x0 - 0.5, y - h / 2, f"{name}  {n}", ha="right", va="center", fontsize=7.3, color=INK)
        y -= h
    ax.text(x0 - 0.5, 7.75 - n_state * scale / 2, f"car state  {n_state}\n"
            + " · ".join(f"{name} {n}" for name, n in groups[:3]) + "\n" + " · ".join(f"{name} {n}" for name, n in groups[3:6]),
            ha="right", va="center", fontsize=6.9, color=INK, linespacing=1.3)
    ax.add_patch(Rectangle((x0 - 0.35, 7.75 - n_obs * scale), 0.7, n_obs * scale, fc="none", ec=ONNX, lw=1.8))
    ax.text(x0, 8.05, f"Observation\n{n_obs} numbers, car frame", ha="center", va="bottom", fontsize=8.5, weight="bold")
    ax.text(x0, 1.2, f"look-ahead: one point every {O.WINDOW_STRIDE * 2} m,\n{(k - 1) * O.WINDOW_STRIDE * 2} m of road ahead",
            ha="center", va="top", fontsize=7.3, color=INK2)
    # Running normaliser.
    _box(ax, 3.9, 4.65, 1.1, 2.6, "running\nmean / var\nnormaliser\n(clip ±10)", LIGHT["purple"], size=7.6, ec=ONNX, lw=1.8)
    _arrow(ax, (2.55, 4.65), (3.33, 4.65))

    def mlp(sizes, y_mid, color, light, name, xs, ec=INK2, lw=0.9):
        """Draw layers as vertical blocks (height ∝ width) at x positions xs; return the last block's right edge."""
        top = max(sizes[1:])
        prev = (4.47, 4.65)
        for i, (x, n) in enumerate(zip(xs, sizes[1:])):
            h = 0.25 + 2.3 * n / top
            ax.add_patch(FancyBboxPatch((x - 0.22, y_mid - h / 2), 0.44, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                        fc=light if i < len(sizes) - 2 else color, ec=ec, lw=lw))
            act = "tanh" if i < len(sizes) - 2 else ("logits" if n > 1 else "value")
            ax.text(x, y_mid + h / 2 + 0.08, f"{n}", ha="center", va="bottom", fontsize=8, weight="bold")
            ax.text(x, y_mid - h / 2 - 0.08, f"Linear\n{sizes[i]}→{n}\n{act}", ha="center", va="top", fontsize=6.8, color=INK2)
            _arrow(ax, prev, (x - 0.24, y_mid))
            prev = (x + 0.24, y_mid)
        ax.text(xs[0] - 0.55, y_mid + 1.75, f"{name}: {_params(sizes):,} parameters", ha="left", va="bottom", fontsize=8.5,
                weight="bold", color=color)
        return prev

    xs = [6.0, 7.6, 9.2]
    end_a = mlp(actor, 6.55, SLOTS[0], LIGHT["blue"], "Actor (policy)", xs, ec=ONNX, lw=1.8)
    end_c = mlp(critic, 2.15, SLOTS[1], LIGHT["orange"], "Critic (value)", xs)
    # Action heads and keys.
    _box(ax, 11.4, 7.25, 2.2, 0.75, "steer: softmax over\nleft · none · right", LIGHT["blue"], size=7.6, ec=ONNX, lw=1.8)
    _box(ax, 11.4, 5.95, 2.2, 0.75, "pedal: softmax over\ngas · coast · brake", LIGHT["blue"], size=7.6, ec=ONNX, lw=1.8)
    _arrow(ax, end_a, (10.28, 7.25)); _arrow(ax, end_a, (10.28, 5.95))
    tpa = json.loads((Path(__file__).resolve().parents[3] / "web" / "assets" / "vehicle.json").read_text(encoding="utf-8"))
    hold = tpa["ticks_per_action"] * tpa["dt"]
    _box(ax, 14.5, 6.6, 3.2, 1.3, f"one of 9 key combinations,\nheld {hold:.2f} s ({tpa['ticks_per_action']} physics ticks);\n"
         f"training samples it; evaluation\ntakes the most probable one", LIGHT["green"], size=7.3)
    _arrow(ax, (12.52, 7.25), (12.88, 6.75)); _arrow(ax, (12.52, 5.95), (12.88, 6.45))
    _box(ax, 11.4, 2.15, 2.2, 0.9, "V(s): expected\ndiscounted return\n(training only)", LIGHT["orange"], size=7.6)
    _arrow(ax, end_c, (10.28, 2.15))
    _box(ax, 14.5, 2.15, 3.2, 1.6, f"PPO loss: clipped surrogate\n(clip {PPOConfig().clip_coef}) + value error\n"
         f"− {PPOConfig().ent_coef} × entropy;\nGAE (γ {PPOConfig().gamma}, λ {PPOConfig().gae_lambda})", LIGHT["grey"], size=7.3)
    _arrow(ax, (12.52, 2.15), (12.88, 2.15))
    # What the browser runs: one ONNX graph (observation code + normaliser + actor), outlined in green.
    ax.add_patch(Rectangle((10.6, 8.45), 0.45, 0.22, fc="none", ec=ONNX, lw=1.8))
    ax.text(11.15, 8.56, "green outline: exported to ONNX and run live by the browser game", ha="left", va="center",
            fontsize=7.4, color=ONNX)
    ax.text(0.2, -0.55, f"Separate actor and critic MLPs (no shared layers), orthogonal initialisation (CleanRL); "
            f"{_params(actor) + _params(critic):,} parameters in total. Sizes from env/obs.py and rl/ppo.py.",
            fontsize=7.3, color=MUTED, ha="left", va="bottom")
    return _save(fig, out, "diagram_architecture",
                 f"Network architecture: the {n_obs}-number observation (block heights proportional to size) is normalised and "
                 f"fed to separate actor and critic MLPs ({' → '.join(map(str, actor))} and {' → '.join(map(str, critic))}, "
                 f"tanh hidden units). The actor's 6 logits form two 3-way categorical heads, steering and pedal; the critic's "
                 f"value is used only in training.")


# ----------------------------------------------------------------------------- system pipeline
def system(res, out: Path) -> dict:
    """From the track generator to training, evaluation, the report and the browser game."""
    ppo, env = PPOConfig(), EnvConfig()
    sets = res.track_sets or {}
    kept = lambda s: f"{sets[s]['kept']:,}" if s in sets else "?"
    veh = json.loads((Path(__file__).resolve().parents[3] / "web" / "assets" / "vehicle.json").read_text(encoding="utf-8"))
    hz, act_hz = round(1 / veh["dt"]), round(1 / (veh["dt"] * veh["ticks_per_action"]))
    fig, ax = _canvas(16, 7.4, (12.5, 6.6), y0=-1.05)
    ax.text(0.2, 7.15, "Training", fontsize=9.5, weight="bold", color=SLOTS[0])
    ax.text(0.2, 3.55, "Evaluation, report and game", fontsize=9.5, weight="bold", color=SLOTS[1])
    _box(ax, 1.85, 5.6, 3.0, 1.75, "Seeded track generator\nstraights and arcs with hills,\nbanking, ramps and gap jumps;\n"
         "each track certified by the\nscripted driver (reference time)", LIGHT["grey"], size=7.4)
    _box(ax, 5.55, 5.6, 3.0, 1.75, f"Track sets (disjoint seeds)\ntrain pool {kept('train')} · val {kept('val')}\n"
         f"test {kept('test')} · out-of-range {kept('oor')}\n5 hand-made campaign tracks", LIGHT["yellow"], size=7.4)
    _box(ax, 9.45, 5.6, 3.3, 1.75, f"Batched PyTorch simulator\nline-by-line port of cannon-es\nRaycastVehicle; {hz} Hz physics;\n"
         f"{ppo.num_envs:,} cars, each on its own track;\nreward = 0.1 × metres of progress", LIGHT["blue"], size=7.4)
    _box(ax, 13.55, 5.6, 3.3, 1.75, f"PPO trainer\nactions at {act_hz} Hz; {ppo.num_steps}-step rollouts;\n"
         f"{ppo.update_epochs} epochs × {ppo.num_minibatches} minibatches;\nKL-adaptive learning rate;\n"
         f"checkpoint every 10 min (resumable)", LIGHT["blue"], size=7.4)
    _arrow(ax, (3.37, 5.6), (4.03, 5.6)); _arrow(ax, (7.07, 5.6), (7.78, 5.6))
    _arrow(ax, (11.12, 5.85), (11.88, 5.85), "obs, reward", size=7)
    _arrow(ax, (11.88, 5.3), (11.12, 5.3), "actions", size=7, off=(0, -0.42))
    _box(ax, 1.85, 1.75, 3.0, 1.9, "Evaluator (same simulator)\nfinal checkpoint, standing start,\nmost probable action,\n"
         "one run per track;\nSWS against the reference time", LIGHT["green"], size=7.4)
    _box(ax, 5.55, 1.75, 3.0, 1.9, "results/\nper-track CSVs, run.json,\nreplays; every test-set\nread is logged", LIGHT["yellow"], size=7.4)
    _box(ax, 9.45, 1.75, 3.3, 1.9, "scripts/07_make_report.py\ntables, figures, diagrams,\nreports, README block,\npaper, website data", LIGHT["grey"], size=7.4)
    _box(ax, 13.55, 1.75, 3.3, 1.9, "Browser game (three.js +\nunmodified cannon-es)\nWatch AI and Race vs AI run the\n"
         "ONNX policy live; replays of\nofficial runs; medals, ghosts", LIGHT["orange"], size=7.4)
    _arrow(ax, (13.55, 4.72), (2.6, 2.72), "final checkpoint", rad=0.08, size=7, off=(0, 0.25))
    _arrow(ax, (3.37, 1.75), (4.03, 1.75)); _arrow(ax, (7.07, 1.75), (7.78, 1.75))
    _arrow(ax, (14.6, 4.72), (14.6, 2.72), "ONNX export", size=7, off=(0.6, -0.1))
    _arrow(ax, (5.55, 0.8), (13.0, 0.8), None, rad=0.18)  # ghosts: the evaluator's recorded runs, exported to the game
    ax.text(9.45, -0.42, "ghosts and replays: the official evaluation runs (scripts/06_export_web.py)", ha="center",
            va="bottom", fontsize=7, color=INK2)
    _arrow(ax, (11.1, 4.72), (12.6, 2.72), "parity tests\n(tick by tick, Node.js)", ls="--", color=SLOTS[2], size=7,
           both=True, off=(-0.55, -0.15))
    ax.text(0.2, -1.0, "Numbers from rl/ppo.py, env/vec_env.py, web/assets/vehicle.json and results/tracks/*_summary.json.",
            fontsize=7.2, color=MUTED, ha="left", va="bottom")
    return _save(fig, out, "diagram_system",
                 f"System overview: seeded, certified tracks feed a batched PyTorch port of the game's car physics, where PPO "
                 f"trains {ppo.num_envs:,} cars in parallel; the evaluator scores the final checkpoint against the scripted "
                 f"driver's reference times; one script turns the result files into every table, figure and report; the browser "
                 f"game runs the same physics (checked tick by tick) and the exported policy.")


# ----------------------------------------------------------------------------- track sets and protocol
def protocol(res, out: Path) -> dict | None:
    """Track sets with seed ranges and certified counts, what each set is for, and the metric."""
    sets = res.track_sets or {}
    if "test" not in sets:
        return None
    rows = [("train", "Training pool", "generalists and ablations train here (fixed 1,000-track sample scored for the train-test gap)", LIGHT["blue"]),
            ("val", "Validation", "pilot decisions, the learning gate and training dashboards; never reported as a result", LIGHT["grey"]),
            ("test", "Test", "every reported score (primary metric); read only by the evaluator, each read logged", LIGHT["green"]),
            ("test_unfiltered", "Unfiltered test", "secondary check: tracks kept even when the scripted driver fails", LIGHT["green"]),
            ("oor", "Out-of-range test", "one generator factor beyond training (tighter turns, steeper hills, longer jumps or tracks)", LIGHT["orange"])]
    fig, ax = _canvas(16, 7.6, (12.5, 7.0), y0=-1.6)
    ax.text(0.2, 7.3, "Track sets: disjoint seed ranges of one generator, plus 5 hand-made tracks", fontsize=9.5, weight="bold")
    y = 6.5
    for key, name, use, fc in rows:
        if key not in sets:
            continue
        d = sets[key]
        a, b = d["seed_range"]
        _box(ax, 1.7, y, 3.0, 0.82, f"{name}\nseeds {a:,}–{b - 1:,}", fc, size=7.6, weight="bold")
        _box(ax, 4.75, y, 2.7, 0.82, f"{d['generated']:,} generated\n{d['kept']:,} certified", "white", size=7.6)
        ax.text(6.35, y, use, fontsize=7.6, va="center", color=INK)
        y -= 1.02
    _box(ax, 1.7, y, 3.0, 0.82, "Campaign\n5 hand-made tracks", LIGHT["yellow"], size=7.6, weight="bold")
    ax.text(6.35, y, "specialists train on one each; every model is also scored on all five\n"
            "(hand-made, never part of the generalists' training)", fontsize=7.6, va="center")
    _box(ax, 8.0, -0.5, 7.4, 1.35,"Primary metric: success-weighted speed (SWS)\nSWS = mean over tracks of  finished × (reference time / agent time)\n"
         "1.0 = finishes every track as fast as the scripted driver;\nnot capped at 1 (the reference is not the optimum)",
         LIGHT["purple"], size=7.6)
    ax.text(0.2, -1.55, "Counts from results/tracks/*_summary.json (certified = the scripted driver finishes the track in the simulator).",
            fontsize=7.2, color=MUTED, ha="left", va="bottom")
    return _save(fig, out, "diagram_protocol",
                 "Evaluation protocol: training, validation and test tracks come from disjoint seed ranges of one generator and "
                 "are kept only if the scripted driver finishes them (its time is the reference); an unfiltered and an "
                 "out-of-range test set check the filter and extrapolation; the five hand-made campaign tracks are a separate "
                 "out-of-distribution set.")


# ----------------------------------------------------------------------------- run status per phase
def run_status(an, out: Path) -> dict | None:
    """Horizontal stacked bars: runs evaluated, trained or training, and not yet run (postponed), per phase."""
    phases = [(p, d) for p, d in an.status_counts.items() if p != "baselines" and sum(d.values())]
    if not phases:
        return None
    labels = {"1-first-models": "1. First two models", "2-single-seed": "2. One seed of every experiment",
              "3-patched-physics": "3. Patched physics", "4-imitation-hybrids": "4. Imitation and hybrids",
              "5-hyperparameters": "5. Hyperparameter sensitivity", "6-seed-sensitivity": "6. Seed sensitivity"}
    parts = [("evaluated", ("evaluated",), SLOTS[2]), ("trained or training", ("trained", "training"), SLOTS[3]),
             ("postponed" if an.paused else "not started", ("postponed", "not started", "planned"), "#c9c8c1")]
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(phases) + 1.3))
    for i, (p, d) in enumerate(reversed(phases)):
        left = 0
        for name, keys, color in parts:
            n = sum(d.get(k, 0) for k in keys)
            if n:
                ax.barh(i, n, left=left, color=color, edgecolor=SURFACE, height=0.62)
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=8, color=INK)
                left += n
    ax.set_yticks(range(len(phases)), [labels.get(p, p) for p, _ in reversed(phases)])
    ax.set_xlabel("runs")
    ax.set_xlim(0, None)
    ax.grid(axis="y", visible=False)
    used = [(name, color) for name, keys, color in parts if any(d.get(k, 0) for _, d in phases for k in keys)]
    ax.legend([Rectangle((0, 0), 1, 1, fc=c) for _, c in used], [n for n, _ in used], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=len(used), fontsize=8)  # below the axis, in status order
    total = sum(sum(d.values()) for _, d in phases)
    done = sum(d.get("evaluated", 0) for _, d in phases)
    title = f"Run matrix: {done} of {total} runs evaluated"
    if an.paused:
        title += f"; training paused on {an.paused['date']}"
    ax.set_title(title, loc="left")
    return _save(fig, out, "diagram_run_status",
                 f"Runs per phase of the preregistered run matrix by status ({done} of {total} evaluated)"
                 + (f"; training was paused on {an.paused['date']} and the remaining runs are postponed." if an.paused else "."))


def make_all(res, an, out: Path) -> dict:
    """Draw every diagram; a failing one is logged and skipped, never stopping the report."""
    out_figs = {}
    for name, fn in [("diagram_architecture", lambda: architecture(out)), ("diagram_system", lambda: system(res, out)),
                     ("diagram_protocol", lambda: protocol(res, out)), ("diagram_run_status", lambda: run_status(an, out))]:
        try:
            f = fn()
        except Exception:
            log.exception("diagram %s failed", name)
            f = None
        if f:
            out_figs[name] = f
    return out_figs
