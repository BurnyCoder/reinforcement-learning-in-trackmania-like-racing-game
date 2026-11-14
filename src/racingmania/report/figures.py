"""Figures of the report (matplotlib, PNG + SVG) drawn from the analysis, never from typed numbers.

Local purpose: training curves per configuration with seeds overlaid; SWS and finish rate by
configuration with 95 % bootstrap intervals; the training-set-size curve (log x-axis) with the
train-test gap in its own panel; finish rate by track feature; specialist vs generalist times on
the campaign tracks with the scripted reference; and the spread over seeds. Each function returns
None when its data do not exist yet, so the report says "not yet available" instead of drawing
an empty plot.

Global context: called by ``build.build_all``; files go to ``reports/figures/`` and are linked from
the reports, README block, paper and website.
Chart rules followed (no graph crimes): every score axis starts at 0 and is labelled with its
unit; intervals are drawn wherever they exist; one y-scale per panel (no dual axes); colour
follows the entity (fixed slot per role, never by rank) and is paired with a marker shape, so
identity never rests on colour alone; captions state seeds and tracks. Palette: the validated
reference palette of the data-viz guideline used for this project (categorical light-mode slots,
checked for colour-vision-deficiency separation with its validator); sub-3:1-contrast colours are
always accompanied by labels or a table.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (https://matplotlib.org/stable/users/explain/figure/backends.html)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils.logging import get_logger  # noqa: E402
from . import taxonomy as tx  # noqa: E402

log = get_logger(__name__)

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT_STYLE = {  # role -> (colour slot, marker); fixed per role so a filter never repaints survivors
    "method": (SLOTS[0], "o"), "control": (SLOTS[1], "s"), "reference": (SLOTS[2], "D"), "baseline": (SLOTS[3], "^"),
    "ablation": (SLOTS[4], "v"), "imitation": (SLOTS[5], "P"), "hyperparameter": (SLOTS[6], "X"),
    "specialist": (SLOTS[7], "*"), "other": (MUTED, "."),
}
SET_NAME = {"test": "test set", "val": "validation set (development set; preliminary)", "train_sample": "training sample",
            "oor": "out-of-range set", "campaign": "campaign tracks"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "svg.fonttype": "path",
    "svg.hashsalt": "racingmania",  # fixed salt -> deterministic SVG element ids, so rebuilds are byte-identical
})


def _save(fig, out: Path, name: str, caption: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})  # no timestamp: rebuilds are identical
    svg = out / f"{name}.svg"
    svg.write_bytes(svg.read_bytes().replace(b"\r\n", b"\n"))  # matplotlib writes text-mode CRLF on Windows; keep LF everywhere
    plt.close(fig)
    return {"name": name, "png": f"{name}.png", "svg": f"{name}.svg", "caption": caption}


def _err(est, lo, hi):
    """Asymmetric error-bar lengths; NaN bounds become zero-length bars."""
    lo = est if lo != lo else lo
    hi = est if hi != hi else hi
    return [[max(0.0, est - lo)], [max(0.0, hi - est)]]


# ----------------------------------------------------------------------------- training curves
def training_curves(res, an, out: Path) -> dict:
    """One figure per configuration with curves: episode return, finish rate, validation SWS vs steps."""
    figs = {}
    for c in an.configs:
        rids = [r for r in c["rids"] if r in res.train_curves or r in res.val_curves]
        if not rids:
            continue
        colour, _ = CAT_STYLE.get(c["category"], CAT_STYLE["other"])
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
        panels = [("train", "ep_return", "Mean episode return", "return (reward units)"),
                  ("train", "ep_finish_rate", "Training episodes finished", "share finished"),
                  ("val", "sws", "Validation SWS", "SWS (1.0 = scripted driver)")]
        for ax, (src, col, title, ylabel) in zip(axes, panels):
            drawn = 0
            for rid in rids:
                d = (res.train_curves if src == "train" else res.val_curves).get(rid, {})
                if col not in d or "global_step" not in d:
                    continue
                x, y = d["global_step"] / 1e6, d[col]
                ok = ~np.isnan(y)
                if not ok.any():
                    continue
                ax.plot(x[ok], y[ok], color=colour, lw=1.4, alpha=0.85 if len(rids) < 4 else 0.55)
                ax.annotate(f"s{res.run(rid)['seed']}", (x[ok][-1], y[ok][-1]), xytext=(3, 0), textcoords="offset points",
                            fontsize=7, color=INK2, va="center")
                drawn += 1
            ax.set_title(title, loc="left")
            ax.set_xlabel("agent steps (millions)")
            ax.set_ylabel(ylabel)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(0.0, lo), max(hi, 1.05) if col != "ep_return" else hi)
            if col == "sws":
                ax.axhline(1.0, color=INK2, lw=0.8)
                ax.annotate("scripted driver = 1.0", (0.01, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                            textcoords="offset points", fontsize=7, color=INK2)
            if not drawn:
                ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        fig.suptitle(c["label"], x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        seeds = ", ".join(str(res.run(r)["seed"]) for r in rids)
        name = "curves_" + c["key"].replace("@", "_at_").replace("/", "_")
        figs[c["key"]] = _save(fig, out, name, f"Training curves of {c['label']}; one line per seed (seeds {seeds}). "
                               "Return and finish rate are over training episodes (standing and random starts); "
                               "validation SWS is the evaluator's score on the validation tracks.")
    return figs


# ----------------------------------------------------------------------------- method comparison
def sws_by_method(an, out: Path) -> dict | None:
    s = an.primary_set
    rows = [c for c in an.configs if an.summary(c["key"], s)] if s else []
    if not rows:
        return None
    rows = rows[::-1]  # matplotlib draws the first row at the bottom
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(rows) + 1.4), sharey=True)
    for i, c in enumerate(rows):
        v = an.summary(c["key"], s)
        colour, marker = CAT_STYLE.get(c["category"], CAT_STYLE["other"])
        for ax, (k, lo, hi) in zip(axes, [("sws", "sws_lo", "sws_hi"), ("finish", "finish_lo", "finish_hi")]):
            ax.errorbar(v[k], i, xerr=_err(v[k], v[lo], v[hi]), fmt=marker, color=colour, ms=6, capsize=2.5, lw=1.2)
        axes[1].annotate(f"n={v['n_seeds']}", (1.0, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                         textcoords="offset points", fontsize=7, color=INK2, va="center")
    axes[0].set_yticks(range(len(rows)), [c["label"] for c in rows])
    axes[0].axvline(1.0, color=INK2, lw=0.8)
    axes[0].set_title("Success-weighted speed", loc="left")
    axes[0].set_xlabel("SWS (1.0 = scripted driver's speed on every track)")
    axes[1].set_title("Finish rate", loc="left")
    axes[1].set_xlabel("share of tracks finished")
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(0, 1.02)
    handles = [plt.Line2D([], [], color=CAT_STYLE[k][0], marker=CAT_STYLE[k][1], ls="", label=tx.CATEGORY_LABELS[k])
               for k in tx.CATEGORIES if any(c["category"] == k for c in rows)]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), bbox_to_anchor=(0.5, -0.02 - 0.6 / len(rows)))
    fig.tight_layout()
    ns = sorted({an.summary(c["key"], s)["n_tracks"] for c in rows})
    return _save(fig, out, "sws_by_method", f"SWS and finish rate on the {SET_NAME.get(s, s)} ({', '.join(f'{n:,}' for n in ns)} tracks), "
                 "mean over seeds with 95 % two-level bootstrap intervals (10,000 resamples of seeds and tracks); "
                 "n = number of seeds. With one seed the interval reflects track sampling only.")


# ----------------------------------------------------------------------------- training-set size
def size_curve(an, out: Path) -> dict | None:
    pts = an.size_curve
    if not pts:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True)
    x = np.array([max(1, p["n_train_tracks"]) for p in pts], float)
    for p, xi in zip(pts, x):
        colour, marker = CAT_STYLE["method" if p["key"] == "generalist" else "ablation"]
        axes[0].errorbar(xi, p["sws"], yerr=_err(p["sws"], p["lo"], p["hi"]), fmt=marker, color=colour, ms=6, capsize=2.5)
        axes[0].annotate(f"n={p['n_seeds']}", (xi, p["sws"]), xytext=(5, -10), textcoords="offset points", fontsize=7, color=INK2)
        g = p.get("gap")
        if g:
            axes[1].errorbar(xi, g["estimate"], yerr=_err(g["estimate"], g["lo"], g["hi"]), fmt=marker, color=colour, ms=6, capsize=2.5)
    axes[0].plot(x, [p["sws"] for p in pts], color=AXIS, lw=1, zorder=0)
    axes[0].set_ylim(bottom=0)
    axes[0].axhline(1.0, color=INK2, lw=0.8)
    axes[0].set_ylabel("SWS")
    axes[0].set_title(f"Score on the {SET_NAME.get(pts[0]['set'], pts[0]['set'])}", loc="left")
    axes[1].axhline(0.0, color=INK2, lw=0.8)
    axes[1].set_ylabel("SWS(own training tracks) - SWS(test)")
    axes[1].set_title("Train - test gap", loc="left")
    if not any(p.get("gap") for p in pts):
        axes[1].text(0.5, 0.5, "not yet available: needs scores on each run's own training tracks", ha="center",
                     va="center", transform=axes[1].transAxes, color=MUTED, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of training tracks (log scale)")
    fig.tight_layout()
    return _save(fig, out, "training_set_size", "Training-set-size ablation: score (top) and train-test gap (bottom) against the "
                 "number of training tracks; the largest value is the generalist trained on the whole pool. Mean over seeds "
                 "(n per point) with 95 % two-level bootstrap intervals.")


# ----------------------------------------------------------------------------- track features
def feature_figure(an, out: Path) -> dict | None:
    fr = an.feature_rates or {}
    feats = fr.get("features") or {}
    if not feats:
        return None
    panels = [(f, t) for f, t in [("jumps", "Number of jumps"), ("max_bank_deg", "Steepest banking (deg)"),
                                  ("height_range_m", "Height range (m)")] if f in feats]
    if fr.get("oor_variants"):
        panels.append(("oor", "Out-of-range variant"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=True, squeeze=False)
    methods = fr["methods"]
    styles = [(SLOTS[i], m) for i, m in zip(range(len(methods)), ["o", "s", "^", "D"])]
    for ax, (f, title) in zip(axes[0], panels):
        data = fr["oor_variants"] if f == "oor" else {k: v["bins"] for k, v in feats[f].items()}
        order = sorted({b for d in data.values() for b in d}) if f == "oor" else next(iter(feats[f].values()))["order"]
        for j, m in enumerate(methods):
            d = data.get(m["key"])
            if not d:
                continue
            colour, marker = styles[j]
            xs = [i + (j - (len(methods) - 1) / 2) * 0.12 for i, b in enumerate(order) if b in d]
            for xi, b in zip(xs, [b for b in order if b in d]):
                v = d[b]
                ax.errorbar(xi, v["rate"], yerr=_err(v["rate"], v["lo"], v["hi"]), fmt=marker, color=colour, ms=5, capsize=2)
        ax.set_xticks(range(len(order)), [o.replace(" (", "\n(") for o in order], fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("share of tracks finished")
    handles = [plt.Line2D([], [], color=styles[j][0], marker=styles[j][1], ls="", label=m["label"]) for j, m in enumerate(methods)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _save(fig, out, "finish_by_feature", f"Finish rate by track feature on the {SET_NAME.get(fr['set'], fr['set'])}: "
                 "jumps counted per track, banking and height range split into tertiles of that set (ranges in the labels). "
                 "Mean over seeds and tracks in each bin with 95 % two-level bootstrap intervals. Secondary, descriptive analysis.")


# ----------------------------------------------------------------------------- specialists
SPEC_STYLE = {"generalist": (SLOTS[0], "o"), "specialist": (SLOTS[1], "s"), "specialist-finetune": (SLOTS[2], "^"),
              "generalist-pilot": (SLOTS[6], "X"), "specialist-pilot": (SLOTS[4], "v"), "baseline-cem": (SLOTS[3], "D")}


def _spec_kind(key: str) -> str | None:
    if key in ("generalist", "generalist-pilot", "baseline-cem"):
        return key
    if key.startswith("specialist-finetune-"):
        return "specialist-finetune"
    if key.startswith("specialist-") and key.endswith("-pilot"):
        return "specialist-pilot"
    if key.startswith("specialist-"):
        return "specialist"
    return None


def specialist_figure(an, out: Path) -> dict | None:
    rows = an.specialists
    if not rows or not any(_spec_kind(e["key"]) for t in rows for e in t["entries"]):
        return None
    kinds = [k for k in SPEC_STYLE if any(_spec_kind(e["key"]) == k for t in rows for e in t["entries"])]
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2.5, 3.8))
    top = max([x["time"] for t in rows for e in t["entries"] for x in e["seeds"] if x["finished"]]  # DNF times are caps, not times
              + [t["ref_time"] for t in rows if t["ref_time"] == t["ref_time"]] + [1.0])
    dnf_y = top * 1.08
    for i, t in enumerate(rows):
        if t["ref_time"] == t["ref_time"]:
            ax.hlines(t["ref_time"], i - 0.4, i + 0.4, color=INK, lw=1.6)
        for e in t["entries"]:
            k = _spec_kind(e["key"])
            if k is None:
                continue
            j = kinds.index(k)
            colour, marker = SPEC_STYLE[k]
            xc = i + (j - (len(kinds) - 1) / 2) * 0.16
            fin = [x["time"] for x in e["seeds"] if x["finished"]]
            ax.scatter([xc] * len(fin), fin, color=colour, marker=marker, s=22, alpha=0.6, linewidths=0)
            if fin:
                ax.scatter([xc], [np.mean(fin)], color=colour, marker=marker, s=60, edgecolors=INK, linewidths=0.6, zorder=3)
            ndnf = e["n"] - len(fin)
            if ndnf:
                ax.scatter([xc], [dnf_y], color=colour, marker="x", s=30)
                ax.annotate(f"{ndnf} DNF", (xc, dnf_y), xytext=(0, 5), textcoords="offset points", fontsize=6, ha="center", color=INK2)
    ax.set_xticks(range(len(rows)), [t["track"] for t in rows])
    ax.set_ylim(0, dnf_y * 1.1)
    ax.set_ylabel("finishing time (s)")
    ax.set_title("Campaign tracks: time per agent (lower is faster)", loc="left")
    handles = [plt.Line2D([], [], color=INK, lw=1.6, label="scripted driver (reference)")]
    labels = {"generalist": "generalist (zero-shot)", "specialist": "specialist (own track)", "specialist-finetune": "fine-tuned specialist",
              "generalist-pilot": "generalist pilot (seed 0)", "specialist-pilot": "specialist pilot (seed 0)", "baseline-cem": "CEM planner"}
    handles += [plt.Line2D([], [], color=SPEC_STYLE[k][0], marker=SPEC_STYLE[k][1], ls="", label=labels[k]) for k in kinds]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "specialists_campaign", "Finishing time on each campaign track: small marks are single seeds, large "
                 "marks the mean over finishing seeds, x marks runs that did not finish (DNF); the black bar is the scripted "
                 "driver's reference time. Specialists are scored on their own track; the generalist zero-shot.")


# ----------------------------------------------------------------------------- seeds
def seed_spread(an, out: Path) -> dict | None:
    rows = an.seed_sensitivity
    if not rows:
        return None
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(rows) + 1.3))
    for i, r in enumerate(rows):
        c = an.config(r["key"])
        colour, marker = CAT_STYLE.get(c["category"], CAT_STYLE["other"])
        per = [p["sws"] for p in an.summary(r["key"], "test")["per_seed"]]
        ax.scatter(per, [i] * len(per), color=colour, marker=marker, s=18, alpha=0.6, linewidths=0)
        v = an.summary(r["key"], "test")
        ax.errorbar(v["sws"], i, xerr=_err(v["sws"], v["sws_lo"], v["sws_hi"]), fmt="|", color=INK, ms=12, capsize=3, lw=1)
    ax.set_yticks(range(len(rows)), [f"{r['label']} (n={r['n']})" for r in rows])
    ax.set_xlim(left=0)
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_xlabel("test SWS per seed (dots); mean with 95 % bootstrap interval (bar)")
    ax.set_title("Spread over seeds", loc="left")
    fig.tight_layout()
    return _save(fig, out, "seed_spread", "Seed sensitivity: test SWS of every seed (dots) and the mean with its 95 % two-level "
                 "bootstrap interval, for every configuration with at least two seeds.")


def make_all(res, an, out: Path) -> dict:
    """Draw every figure whose data exist; returns name -> {png, svg, caption} (curves under 'curves')."""
    figs = {"curves": training_curves(res, an, out)}
    for name, fn in [("sws_by_method", sws_by_method), ("training_set_size", size_curve), ("finish_by_feature", feature_figure),
                     ("specialists_campaign", specialist_figure), ("seed_spread", seed_spread)]:
        try:
            f = fn(an, out)
        except Exception:  # a broken figure must not stop the rest of the report
            log.exception("figure %s failed", name)
            f = None
        if f:
            figs[name] = f
    log.info("figures: %d training-curve figures, %s", len(figs["curves"]), ", ".join(k for k in figs if k != "curves") or "no others")
    return figs
