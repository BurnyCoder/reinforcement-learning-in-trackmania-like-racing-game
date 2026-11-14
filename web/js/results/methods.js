// methods.js — configuration comparison, hypothesis table and the training-set-size view.
// Global context: numbers come from results.json "summaries" (per configuration and track set:
// SWS and finish rate with 95 % two-level bootstrap intervals), "hypotheses" (H1-H3 computed as
// preregistered) and "size_curve"; nothing is recomputed in the browser.

import { CATEGORY_SLOT, dotIntervalChart, lineChart, slotColor } from "./chart.js";
import { $, ci, esc, fillSelect, isNum, num, pct, pval, table } from "./format.js";

const colorOf = (cfg) => slotColor(CATEGORY_SLOT[cfg?.category] ?? 7);

/** Track sets that have at least one summary, in the report's order. */
export function availableSets(data) {
  const have = new Set(Object.values(data.summaries || {}).flatMap((d) => Object.keys(d)));
  return Object.keys(data.sets || {}).filter((s) => have.has(s));
}

export function renderHypotheses(data, container) {
  const rows = (data.hypotheses || []).map((h) => {
    const seeds = Object.entries(h.seeds || {}).map(([k, v]) => `${esc(k)} ${v}/${h.required[k]}`).join(", ");
    const est = ci(h.estimate, h.lo, h.hi) + (h.preliminary ? " (preliminary)" : "");
    const stat = isNum(h.statistic) ? `t = ${h.statistic.toFixed(3)}, df = ${num(h.df, 1)}` : "—";
    return [esc(h.id), esc(h.statement), `${h.id === "H1" ? "SWS" : "SWS difference"} &gt; ${h.threshold}`, seeds, est, stat,
      pval(h.p), pval(h.p_holm), `<span class="decision">${esc(h.decision)}</span>`];
  });
  container.innerHTML = table(["", "Hypothesis", "Tested quantity", "Seeds (have/required)", "Estimate [95 % CI]", "Test statistic",
    "p (one-sided)", "p (Holm)", "Decision"], rows, [4, 6, 7]);
}

export function initMethods(data) {
  const setSel = $("methods-set"), metSel = $("methods-metric");
  const sets = availableSets(data);
  fillSelect(setSel, sets.map((s) => [s, data.sets[s]]), data.primary_set);
  const draw = () => {
    const s = setSel.value, metric = metSel.value;
    const items = (data.configs || []).filter((c) => data.summaries?.[c.key]?.[s]).map((c) => {
      const v = data.summaries[c.key][s];
      const [val, lo, hi] = metric === "sws" ? [v.sws, v.sws_lo, v.sws_hi] : [v.finish, v.finish_lo, v.finish_hi];
      return {
        label: c.label, value: val, lo, hi, color: colorOf(c), note: `n=${v.n_seeds}`,
        tip: `<b>${esc(c.label)}</b> (${esc(data.categories[c.category] || c.category)})<br>SWS ${ci(v.sws, v.sws_lo, v.sws_hi)}` +
          `<br>finished ${pct(v.finish)} [${pct(v.finish_lo)}, ${pct(v.finish_hi)}]<br>${v.n_seeds} seed(s), ${v.n_tracks.toLocaleString("en")} tracks`,
      };
    });
    dotIntervalChart($("methods-chart"), items, {
      title: `${metric === "sws" ? "Success-weighted speed" : "Finish rate"} on the ${data.sets[s] || s} set`,
      xLabel: metric === "sws" ? "SWS (1.0 = scripted driver's speed on every track)" : "share of tracks finished",
      xMax: metric === "finish" ? 1.02 : undefined,
      refLine: metric === "sws" ? { value: 1, label: "scripted driver = 1.0" } : undefined,
    });
    const cats = [...new Set(items.length ? (data.configs || []).filter((c) => data.summaries?.[c.key]?.[s]).map((c) => c.category) : [])];
    $("methods-note").innerHTML = items.length
      ? `Colour = role: ${cats.map((k) => `<span class="key"><span class="swatch" style="background:${slotColor(CATEGORY_SLOT[k] ?? 7)}"></span> ${esc(data.categories[k] || k)}</span>`).join(" ")}. ` +
        "Dots are means over seeds, bars 95 % two-level bootstrap intervals; n = seeds. With one seed the interval reflects track sampling only."
      : "";
    const rows = (data.configs || []).filter((c) => data.summaries?.[c.key]?.[s]).map((c) => {
      const v = data.summaries[c.key][s];
      return [esc(c.label), esc(data.categories[c.category] || c.category), v.n_seeds, v.n_tracks.toLocaleString("en"),
        ci(v.sws, v.sws_lo, v.sws_hi), pct(v.finish), num(v.iqm), num(v.time_ratio_finished), esc(c.architecture), esc(c.training_data)];
    });
    $("methods-table").innerHTML = table(["Configuration", "Role", "Seeds", "Tracks", "SWS [95 % CI]", "Finished", "IQM",
      "Time ratio (finished)", "Architecture", "Training data"], rows, [2, 3, 4, 5, 6, 7]);
  };
  setSel.onchange = draw;
  metSel.onchange = draw;
  return draw;
}

export function renderSize(data) {
  const pts = data.size_curve || [];
  const series = [{
    label: "SWS", color: slotColor(4), x: pts.map((p) => Math.max(1, p.n_train_tracks)), y: pts.map((p) => p.sws),
    lo: pts.map((p) => p.lo), hi: pts.map((p) => p.hi),
  }];
  const setName = pts.length ? data.sets[pts[0].set] || pts[0].set : "";
  const xs = pts.map((p) => Math.max(1, p.n_train_tracks));
  const xDomain = xs.length ? [Math.max(0.5, Math.min(...xs) / 2), Math.max(...xs) * 2] : undefined; // same x-axis in both charts
  lineChart($("size-chart"), series, {
    title: `Score on the ${setName} set`, xLabel: "training tracks (log scale)", yLabel: "SWS", logX: true, points: true, xDomain,
    refLine: { value: 1, label: "scripted driver = 1.0" }, emptyText: "Not yet available: no training-set-size run has results.",
  });
  const g = pts.filter((p) => p.gap);
  lineChart($("gap-chart"), [{
    label: "train − test gap", color: slotColor(4), x: g.map((p) => Math.max(1, p.n_train_tracks)), y: g.map((p) => p.gap.estimate),
    lo: g.map((p) => p.gap.lo), hi: g.map((p) => p.gap.hi),
  }], {
    title: "Train − test gap", xLabel: "training tracks (log scale)", yLabel: "SWS(own training tracks) − SWS(test)", logX: true, points: true, xDomain,
    yMin: Math.min(0, ...g.map((p) => p.gap.lo).filter(isNum)), height: 260,
    emptyText: "Not yet available: needs scores on each run's own training tracks.",
  });
  $("size-table").innerHTML = table(["Training tracks", "Configuration", "Seeds", "SWS [95 % CI]", "Train − test gap [95 % CI]", "Training score from"],
    pts.map((p) => [p.n_train_tracks.toLocaleString("en"), esc(p.label), p.n_seeds, ci(p.sws, p.lo, p.hi),
      p.gap ? ci(p.gap.estimate, p.gap.lo, p.gap.hi) : "—", esc(p.gap?.own_set || "—")]), [0, 2, 3, 4]);
}
