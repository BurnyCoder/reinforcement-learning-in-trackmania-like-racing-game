"""Plain-language sentences generated from the numbers ("the generalist finished X % of unseen tracks").

Local purpose: every sentence here is a template filled from ``analysis.Analysis``; a sentence is
produced only when its numbers exist, and preliminary numbers (validation set, fewer seeds than
preregistered) are labelled as such. No result is ever typed by hand.

Global context: used for the README results block, the summary report and the paper's abstract
and results section, so the three always say the same thing.
"""

from __future__ import annotations

from .tables import ci, pct


def _seeds(n: int) -> str:
    return f"{n} seed" + ("" if n == 1 else "s")


def _set_phrase(s: str, n_tracks: int) -> str:
    if s == "test":
        return f"the {n_tracks:,} unseen test tracks"
    return (f"the {n_tracks:,} validation tracks (the development set used for decisions, so these numbers are "
            "preliminary; test tracks are read only after the recipe is frozen)")


def headline(an) -> list[str]:
    """Sentences on the main agent against its controls and baselines, and on the hypotheses."""
    s = an.primary_set
    if not s:
        return ["No model has been evaluated yet, so no result is available; every table below says so."]
    out = []
    key = "generalist" if an.summary("generalist", s) else ("generalist-pilot" if an.summary("generalist-pilot", s) else None)
    if key:
        g, lab = an.summary(key, s), an.config(key)["label"]
        out.append(f"The {lab} ({_seeds(g['n_seeds'])}) finished {pct(g['finish'])} of {_set_phrase(s, g['n_tracks'])} "
                   f"(95 % CI {pct(g['finish_lo'])} to {pct(g['finish_hi'])}), with a success-weighted speed (SWS) of "
                   f"{ci(g['sws'], g['sws_lo'], g['sws_hi'])}; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.")
        r = g["time_ratio_finished"]
        if r == r:
            out.append(f"On the tracks it finished, it needed {r:.2f} times the scripted driver's time "
                       f"({abs(r - 1) * 100:.1f} % {'slower' if r > 1 else 'faster'} on average).")
        comps = []
        for other in ("blind", "size-1", "pqn", "baseline-untrained", "baseline-random"):
            o = an.summary(other, s)
            if not o:
                continue
            d = 100 * (g["finish"] - o["finish"])
            comps.append(f"{abs(d):.1f} percentage points {'more' if d >= 0 else 'fewer'} than the "
                         f"{an.config(other)['label'].lower()} ({pct(o['finish'])}, {_seeds(o['n_seeds'])})")
        if comps:
            out.append("Its finish rate was " + "; ".join(comps) + ".")
    else:
        for other in ("baseline-scripted", "baseline-random", "baseline-untrained"):
            o = an.summary(other, s)
            if o:
                out.append(f"The {an.config(other)['label'].lower()} finished {pct(o['finish'])} of {_set_phrase(s, o['n_tracks'])} "
                           f"(SWS {ci(o['sws'], o['sws_lo'], o['sws_hi'])}).")
        out.append("No trained agent has been evaluated yet.")
    tested = [h for h in an.hypotheses if h["tested"]]
    if tested:
        for h in tested:
            out.append(f"{h['id']} ({h['statement'].lower()}) is **{h['decision']}**: estimate "
                       f"{ci(h['estimate'], h['lo'], h['hi'])} against a threshold of {h['threshold']}, Holm-adjusted p = "
                       f"{h['p_holm']:.4f}.")
    else:
        have = an.hypotheses[0]["seeds"].get("generalist", 0) if an.hypotheses else 0
        out.append(f"The preregistered hypotheses H1-H3 are not tested yet: they need test results from 10 generalist seeds "
                   f"and 5 seeds each of the blind control and the single-track ablation ({have} generalist seeds so far).")
    for t in an.specialists:
        p = t.get("generalist_slower_pct")
        if p == p and p is not None:
            out.append(f"On campaign track {t['track']} the generalist (zero-shot) was {abs(p):.1f} % "
                       f"{'slower' if p > 0 else 'faster'} than the specialist trained on that track.")
    return out
