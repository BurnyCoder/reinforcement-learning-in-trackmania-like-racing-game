"""Plain-language sentences generated from the numbers ("the generalist finished X % of unseen tracks").

Local purpose: every sentence here is a template filled from ``analysis.Analysis``; a sentence is
produced only when its numbers exist, and preliminary numbers (validation set, fewer seeds than
preregistered) are labelled as such. No result is ever typed by hand.

Global context: used for the README results block, the summary report and the paper's abstract
and results section, so the three always say the same thing.
"""

from __future__ import annotations

from .analysis import GROUP_NAMES
from .tables import ci, pct


def _seeds(n: int) -> str:
    return f"{n} seed" + ("" if n == 1 else "s")


def _lc(label: str) -> str:
    """A run label inside a sentence: first letter lowercased unless the first word is an acronym (PQN, CEM);
    a label that is a participle phrase ("Trained on 1 track") gets a noun ("agent trained on 1 track")."""
    s = label if label[:2].isupper() else label[:1].lower() + label[1:]
    return "agent " + s if s.startswith("trained on") else s


def _set_phrase(s: str, n_tracks: int) -> str:
    if s == "test":
        return f"the {n_tracks:,} unseen test tracks"
    return (f"the {n_tracks:,} validation tracks (the development set used for decisions, so these numbers are "
            "preliminary; test tracks are read only after the recipe is frozen)")


def headline(an, detail: bool = True) -> list[str]:
    """Sentences on the main agent against its controls and baselines, and on the hypotheses."""
    s = an.primary_set
    if not s:
        return ["No model has been evaluated yet, so no result is available; every table below says so."]
    out = []
    key = "generalist" if an.summary("generalist", s) else ("generalist-pilot" if an.summary("generalist-pilot", s) else None)
    if key:
        g, lab = an.summary(key, s), an.config(key)["label"]
        out.append(f"The {_lc(lab)} ({_seeds(g['n_seeds'])}) finished {pct(g['finish'])} of {_set_phrase(s, g['n_tracks'])} "
                   f"(95 % CI {pct(g['finish_lo'])} to {pct(g['finish_hi'])}), with a success-weighted speed (SWS) of "
                   f"{ci(g['sws'], g['sws_lo'], g['sws_hi'])}; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.")
        r = g["time_ratio_finished"]
        if r == r:
            out.append(f"On the tracks it finished, it needed {r:.2f} times the scripted driver's time "
                       f"({abs(r - 1) * 100:.1f} % {'more' if r > 1 else 'less'} time on average).")
        comps = []
        for other in ("blind", "size-1", "pqn", "baseline-untrained", "baseline-random"):
            o = an.summary(other, s)
            if not o:
                continue
            d = 100 * (g["finish"] - o["finish"])
            comps.append(f"{abs(d):.1f} percentage points {'more' if d >= 0 else 'fewer'} than the "
                         f"{_lc(an.config(other)['label'])} ({pct(o['finish'])}, {_seeds(o['n_seeds'])})")
        if comps:
            out.append("Its finish rate was " + "; ".join(comps) + ".")
    else:
        for other in ("baseline-scripted", "baseline-random", "baseline-untrained"):
            o = an.summary(other, s)
            if o:
                out.append(f"The {_lc(an.config(other)['label'])} finished {pct(o['finish'])} of {_set_phrase(s, o['n_tracks'])} "
                           f"(SWS {ci(o['sws'], o['sws_lo'], o['sws_hi'])}).")
        out.append("No trained agent has been evaluated yet.")
    tested = [h for h in an.hypotheses if h["tested"]]
    if tested:
        for h in tested:
            out.append(f"{h['id']}: {h['statement']}. Decision: **{h['decision']}**, estimate "
                       f"{ci(h['estimate'], h['lo'], h['hi'])} against a threshold of {h['threshold']}, Holm-adjusted p = "
                       f"{h['p_holm']:.4f}.")
    else:
        out += preliminary_hypotheses(an) if detail else hypotheses_brief(an)
    for t in an.specialists:
        p = t.get("generalist_slower_pct")
        if p == p and p is not None:
            n = {e["key"]: e["n_finished"] for e in t["entries"]}  # seeds averaged on each side (finished runs only)
            ng, ns = n.get("generalist", 0), n.get("specialist-" + t["track"], 0)
            single = "; a single run on each side, so no interval" if ng == 1 and ns == 1 else ""
            out.append(f"On campaign track {t['track']} the generalist (zero-shot, {_seeds(ng)}) was {abs(p):.1f} % "
                       f"{'slower' if p > 0 else 'faster'} than the specialist trained on that track ({_seeds(ns)}{single}).")
    return out


def _not_done(an) -> str:
    """Lead-in for the list of unfinished phases: says when training was paused, if it is."""
    return f"Training was paused on {an.paused['date']}; not done yet and postponed" if an.paused else "Not done yet"


def hypotheses_brief(an) -> list[str]:
    """One sentence on H1-H3 before their tests, for the paper's abstract and discussion; the Hypotheses
    section carries the full preliminary_hypotheses block."""
    if not an.hypotheses:
        return []
    groups = {"consistent with": [], "against": [], "includes": [], "not estimable": []}
    for h in an.hypotheses:
        groups[next(k for k in groups if k in h["reading"])].append(h["id"])
    words = {"consistent with": "consistent with {}", "against": "against {}",
             "includes": "inconclusive for {}", "not estimable": "{} not estimable yet"}
    parts = [words[k].format(" and ".join(v)) for k, v in groups.items() if v]
    done = (f" {_not_done(an)}: " + ", ".join(x.split(":")[0] for x in an.outstanding) + ".") if an.outstanding else ""
    return [f"Preliminary readings from the seeds so far, not the preregistered tests: {'; '.join(parts)}.{done}"]


def preliminary_hypotheses(an) -> list[str]:
    """Sentences on H1-H3 before their preregistered tests: what the seeds so far show, labelled preliminary,
    and what is not done yet (used by the README results block and the paper)."""
    if not an.hypotheses:
        return []
    have = {k: v for h in an.hypotheses for k, v in h["seeds"].items()}
    so_far = ", ".join(f"{GROUP_NAMES.get(k, k)} {v}" for k, v in have.items())
    out = [f"The preregistered tests of H1-H3 are not done yet: they need test results from 10 generalist seeds and 5 seeds each "
           f"of the blind control and the single-track ablation (so far: {so_far}). Preliminary readings from the seeds available, "
           f"not tests:"]
    for h in an.hypotheses:
        what = "SWS" if h["id"] == "H1" else "SWS difference, paired on the same tracks,"
        if h["estimate"] == h["estimate"]:
            out.append(f"{h['id']}: {h['statement']}. {what} {ci(h['estimate'], h['lo'], h['hi'])}; {h['reading']}.")
        else:
            out.append(f"{h['id']}: {h['statement']}. {h['reading'][:1].upper()}{h['reading'][1:]}.")
    r = an.h2_related
    if r:
        out.append(f"Exploratory and not H2 itself: specialists trained on one hand-made track each finished "
                   f"{pct(r['spec_finish_min'])} to {pct(r['spec_finish_max'])} of the unseen test tracks ({r['n_specialists']} runs, seed 1), "
                   f"against {pct(r['generalist_finish'])} for the generalist.")
    caveat = ("With one seed per configuration these intervals reflect only which tracks were sampled; seed-to-seed and "
              "checkpoint-to-checkpoint variation are not in them yet, so they can be too narrow.")
    if an.outstanding:
        caveat += f" {_not_done(an)}: " + "; ".join(an.outstanding) + "."
    out.append(caveat)
    return out
