// tracks.js — sortable, filterable per-track table for one track set and one configuration.
// Global context: results.json "per_track" is column-oriented per set: track names, reference
// times, geometry features, and for every configuration the seed-averaged score, finished share and
// mean finishing time. The scripted driver's reference time is the yardstick of the SWS metric.

import { $, esc, fillSelect, isNum, num, pct } from "./format.js";
import { availableSets } from "./methods.js";

const PAGE = 200;

export function initTracks(data) {
  const pt = data.per_track || {};
  const setSel = $("tracks-set"), mSel = $("tracks-method"), filt = $("tracks-filter"), unf = $("tracks-unfinished"), more = $("tracks-more");
  const sets = availableSets(data).filter((s) => pt[s]);
  fillSelect(setSel, sets.map((s) => [s, data.sets[s]]), data.primary_set);
  let sortKey = "track", sortDir = 1, shown = PAGE;
  const labels = Object.fromEntries((data.configs || []).map((c) => [c.key, c.label]));
  const fillMethods = () => {
    const ms = Object.keys(pt[setSel.value]?.methods || {});
    fillSelect(mSel, ms.map((k) => [k, labels[k] || k]), ms.includes("generalist") ? "generalist" : ms[0]);
  };
  const draw = () => {
    const d = pt[setSel.value];
    if (!d) { $("tracks-table").innerHTML = `<p class="muted">Not yet available: no per-track results.</p>`; more.hidden = true; return; }
    const m = d.methods[mSel.value] || { score: [], finished: [], time: [], n_seeds: 0 };
    let rows = d.tracks.map((t, i) => ({
      track: t, ref: d.ref_time[i], score: m.score[i], finished: m.finished[i], time: m.time[i],
      ratio: isNum(m.time[i]) && isNum(d.ref_time[i]) ? m.time[i] / d.ref_time[i] : null,
      jumps: d.features[i]?.jumps, bank: d.features[i]?.max_bank_deg, height: d.features[i]?.height_range_m, variant: d.features[i]?.variant,
    }));
    const q = filt.value.trim().toLowerCase();
    if (q) rows = rows.filter((r) => r.track.toLowerCase().includes(q));
    if (unf.checked) rows = rows.filter((r) => isNum(r.finished) && r.finished < 1);
    rows.sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
    });
    const cols = [["track", "Track"], ["ref", "Reference time (s)"], ["score", "Score"], ["finished", "Finished (share of seeds)"],
      ["time", "Mean time (s)"], ["ratio", "Time / reference"], ["jumps", "Jumps"], ["bank", "Max bank (°)"], ["height", "Height range (m)"], ["variant", "Variant"]];
    const head = cols.map(([k, l]) => `<th class="sortable${k === sortKey ? (sortDir > 0 ? " sorted-asc" : " sorted-desc") : ""}" data-key="${k}">${esc(l)}</th>`).join("");
    const body = rows.slice(0, shown).map((r) => `<tr><td>${esc(r.track)}</td><td class="num">${num(r.ref, 2)}</td><td class="num">${num(r.score)}</td>` +
      `<td class="num">${pct(r.finished, 0)}</td><td class="num">${num(r.time, 2)}</td><td class="num">${num(r.ratio, 3)}</td>` +
      `<td class="num">${r.jumps ?? "—"}</td><td class="num">${num(r.bank, 1)}</td><td class="num">${num(r.height, 1)}</td><td>${esc(r.variant || "—")}</td></tr>`).join("");
    $("tracks-table").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    $("tracks-note").textContent = `${rows.length.toLocaleString("en")} tracks; ${labels[mSel.value] || mSel.value}, averaged over ${m.n_seeds} seed(s). Click a column to sort.`;
    more.hidden = rows.length <= shown;
    for (const th of $("tracks-table").querySelectorAll("th.sortable")) {
      th.onclick = () => { sortDir = th.dataset.key === sortKey ? -sortDir : 1; sortKey = th.dataset.key; draw(); };
    }
  };
  setSel.onchange = () => { fillMethods(); shown = PAGE; draw(); };
  mSel.onchange = () => { shown = PAGE; draw(); };
  filt.oninput = () => { shown = PAGE; draw(); };
  unf.onchange = () => { shown = PAGE; draw(); };
  more.onclick = () => { shown += PAGE; draw(); };
  fillMethods();
  return draw;
}
