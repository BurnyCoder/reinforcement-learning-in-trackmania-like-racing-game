"""Names, groups and plain descriptions of experiments, track sets and run statuses.

Local purpose: one place that maps an experiment name (``configs/experiments/<name>.yaml``) and a
run record to the report's vocabulary: its experiment group (one report file each), its role in
comparisons (method, baseline, control, ablation, ...), a readable label, and short descriptions
of architecture, objective, training data and start states built from the run's own config.

Global context: ``collect.py`` attaches these fields to every run; tables, figures, the paper and
the website then group and order runs by them. Text here is vocabulary, never a result.
"""

from __future__ import annotations

import re

# Evaluation sets in display order (reports/preregistration.md, section 3). "train_own" is an
# optional set: a run's own training tracks, needed for a true train-test gap of the size ablation.
SETS = ["test", "val", "train_sample", "train_own", "oor", "campaign", "test_unfiltered"]
SET_LABELS = {
    "test": "Test (unseen generated)",
    "val": "Validation",
    "train_sample": "Training sample",
    "train_own": "Own training tracks",
    "oor": "Out-of-range test",
    "campaign": "Campaign (hand-made)",
    "test_unfiltered": "Unfiltered test",
}
SET_SHORT = {"test": "Test", "val": "Val", "train_sample": "Train", "train_own": "Own train", "oor": "OOR",
             "campaign": "Campaign", "test_unfiltered": "Unfiltered"}

STATUSES = ["postponed", "not started", "planned", "training", "trained", "evaluated"]
STATUS_TEXT = {
    "postponed": "in the run matrix; training was paused before it started (configs/run_matrix.yaml, paused)",
    "not started": "in the run matrix, not yet registered",
    "planned": "registered; code and config pushed before training",
    "training": "training in progress (or interrupted and resumable)",
    "trained": "trained to its step budget, not yet evaluated",
    "evaluated": "evaluated; scores available",
}

# Experiment groups: (slug, title, one-line question). One report file per group.
GROUPS = [
    ("first-models", "First models (phase 1)", "Does the pipeline learn one track (specialist) and many tracks (generalist pilot)?"),
    ("baselines", "Training-free baselines and controls", "Floor (random actions, untrained network) and reference (scripted driver)."),
    ("generalist", "Generalist", "Can one policy trained on the generated pool drive unseen tracks (H1)?"),
    ("blind-control", "Blind control", "Does the agent use the look-ahead (H3)?"),
    ("training-set-size", "Training-set size", "Is track diversity what generalises (H2)?"),
    ("specialists", "Specialists", "How much time does a generalist lose against single-track specialists?"),
    ("pqn", "Baseline algorithm: PQN", "Does the result depend on PPO?"),
    ("ablations", "Ablations: inputs, actions and reward", "Which design choices matter (exploratory)?"),
    ("imitation-hybrids", "Imitation and hybrids (phase 4)", "What does help from the scripted driver change (exploratory)?"),
    ("hyperparameters", "Hyperparameter sensitivity (phase 5)", "How far does the test score move when one setting changes?"),
    ("patched-physics", "Patched physics (phase 3)", "How much did an exploit contribute?"),
    ("seed-sensitivity", "Seed sensitivity (phase 6)", "How much do results vary between seeds, and do the single-seed results hold?"),
]
GROUP_TITLES = {g[0]: g[1] for g in GROUPS}

# Role in comparisons; order is the display order of tables and figures.
CATEGORIES = ["reference", "control", "method", "baseline", "ablation", "imitation", "hyperparameter", "specialist", "other"]
CATEGORY_LABELS = {"reference": "Reference", "control": "Control", "method": "Main method", "baseline": "Baseline algorithm",
                   "ablation": "Ablation", "imitation": "Imitation / hybrid", "hyperparameter": "Hyperparameter",
                   "specialist": "Specialist", "other": "Other"}

_LABELS = {
    "generalist": "Generalist (PPO)",
    "blind": "Blind control",
    "pqn": "PQN",
    "baseline-scripted": "Scripted driver",
    "baseline-random": "Random actions",
    "baseline-untrained": "Untrained network",
    "baseline-cem": "CEM planner",
    "ablation-continuous": "Ablation: continuous actions",
    "ablation-lookahead5": "Ablation: 5 look-ahead points",
    "ablation-lookahead40": "Ablation: 40 look-ahead points",
    "ablation-timecost": "Ablation: time cost",
    "ablation-10hz": "Ablation: 10 Hz actions",
    "ablation-fallpenalty": "Ablation: fall penalty",
    "ablation-joint9": "Ablation: one 9-way head",
    "imitation-bc": "Behaviour cloning",
    "imitation-dagger": "DAgger",
    "hybrid-bc-ppo": "BC then PPO",
    "hybrid-reward-hints": "PPO + reward hints",
    "hybrid-scripted-starts": "PPO + scripted start states",
    "hp-lr-div3": "HP: learning rate / 3",
    "hp-gamma995": "HP: discount 0.995",
    "hp-net3x512": "HP: 3x512 networks",
    "hp-wd1e-5": "HP: weight decay 1e-5",
    "specialist": "Specialist",
    "specialist-finetune": "Fine-tuned specialist",
}


def group_of(experiment: str, pilot: bool, physics: str | None) -> str:
    """Report group of a run."""
    if pilot:
        return "first-models"
    if physics and physics != "v1":
        return "patched-physics"
    if experiment.startswith("baseline-"):
        return "baselines"
    if experiment == "generalist":
        return "generalist"
    if experiment == "blind":
        return "blind-control"
    if experiment.startswith("size-"):
        return "training-set-size"
    if experiment.startswith("specialist"):
        return "specialists"
    if experiment == "pqn":
        return "pqn"
    if experiment.startswith("ablation-"):
        return "ablations"
    if experiment.startswith(("imitation-", "hybrid-")):
        return "imitation-hybrids"
    if experiment.startswith("hp-"):
        return "hyperparameters"
    return "other"


def category_of(experiment: str) -> str:
    """Role of an experiment in comparisons."""
    if experiment in ("baseline-scripted", "baseline-cem"):
        return "reference"
    if experiment in ("baseline-random", "baseline-untrained", "blind"):
        return "control"
    if experiment == "generalist":
        return "method"
    if experiment == "pqn":
        return "baseline"
    if experiment.startswith(("size-", "ablation-")):
        return "ablation"
    if experiment.startswith(("imitation-", "hybrid-")):
        return "imitation"
    if experiment.startswith("hp-"):
        return "hyperparameter"
    if experiment.startswith("specialist"):
        return "specialist"
    return "other"


def config_key(experiment: str, track: str | None, pilot: bool, physics: str | None) -> str:
    """Identifier of a configuration; its seeds are aggregated together.

    Seed-0 phase-1 models are kept apart (preregistration, section 7: "the two seed-0 models are
    reported separately"), and so are runs on a patched physics version (section 8).
    """
    key = experiment + (f"-{track}" if track else "")
    if pilot:
        key += "-pilot"
    if physics and physics != "v1":
        key += f"@{physics}"
    return key


def label_of(experiment: str, track: str | None, pilot: bool, physics: str | None, description: str = "") -> str:
    """Readable label of a configuration."""
    m = re.fullmatch(r"size-(\d+)", experiment)
    base = f"Trained on {int(m.group(1)):,} track{'' if int(m.group(1)) == 1 else 's'}" if m else _LABELS.get(experiment, description or experiment)
    if track:
        base += f" ({track})"
    if pilot:
        base += " [pilot, seed 0]"
    if physics and physics != "v1":
        base += f" [physics {physics}]"
    return base


def size_of(experiment: str) -> int | None:
    """Number of training tracks of a training-set-size ablation, from its name (size-N)."""
    m = re.fullmatch(r"size-(\d+)", experiment)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------- descriptions
def architecture(run: dict) -> str:
    """Network description from the run's logged config (falls back to the experiment file)."""
    algo = (run.get("algo") or "").lower()
    if algo == "scripted":
        return "pure pursuit + curvature speed profile (no network)"
    if algo == "random":
        return "none (uniform random actions)"
    if algo == "cem":
        return "none (planner in the simulator)"
    cfg = {**run.get("exp_cfg", {}).get("ppo", {}), **run.get("exp_cfg", {}).get("pqn", {}), **(run.get("ppo") or {})}
    hidden = cfg.get("hidden") or [256, 256]
    layers = "x".join(str(h) for h in hidden) if len(set(hidden)) > 1 else f"{len(hidden)}x{hidden[0]}"
    if algo == "pqn":
        return f"Q-network {layers}, LayerNorm, one 9-way head"
    head = {"factored": "two 3-way heads", "joint": "one 9-way head", "continuous": "Gaussian (steer, pedal)"}.get(
        cfg.get("action_mode", "factored"), cfg.get("action_mode", "factored"))
    who = "actor" if algo in ("bc", "dagger") else "actor and critic"
    return f"MLP {layers} tanh ({who}), {head}"


def _env(run: dict) -> dict:
    return {**run.get("exp_cfg", {}).get("env", {}), **(run.get("env") or {})}


def objective(run: dict) -> str:
    """Training objective: the reward terms (or the supervised loss) of the run."""
    algo = (run.get("algo") or "").lower()
    if algo in ("scripted", "random", "untrained", "cem"):
        return "none (not trained)"
    if algo in ("bc", "dagger") or run.get("experiment", "").startswith("imitation-"):
        return "copy the scripted driver's actions (cross-entropy)"
    env = _env(run)
    text = "centreline progress"
    if env.get("time_cost"):
        text += f" - time cost {env['time_cost']}/step"
    if env.get("fall_penalty"):
        text += f" - penalty {env['fall_penalty']} when out"
    if env.get("hint_weight"):
        text += f" + scripted-driver hint x{env['hint_weight']}"
    return text


def observation(run: dict) -> str:
    env = _env(run)
    if env.get("blind"):
        return "car state + nearest centreline point only (blind)"
    k = env.get("window_k")
    return f"car state + {k} look-ahead points" if k else "car state + look-ahead points (default)"


def training_data(run: dict) -> str:
    if run.get("track"):
        return f"one campaign track ({run['track']})"
    algo = (run.get("algo") or "").lower()
    if algo in ("scripted", "random", "untrained", "cem"):
        return "none"
    n = run.get("n_train_tracks")
    return f"{int(n):,} generated tracks" if n else "generated training pool"


def start_states(run: dict) -> str:
    algo = (run.get("algo") or "").lower()
    if algo in ("scripted", "random", "untrained", "cem"):
        return "-"
    env = _env(run)
    if env.get("scripted_starts"):
        return "standing + scripted-driver states"
    p = env.get("random_start_prob")
    return f"standing + random centreline (p={p})" if p is not None else "standing + random centreline"
