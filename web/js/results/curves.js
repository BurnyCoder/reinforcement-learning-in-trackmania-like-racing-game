// curves.js — training curves of selected runs (episode return, finish rate, validation SWS, …).
// Global context: results.json "curves" holds each run's train_metrics.csv and val_metrics.csv,
// down-sampled to at most 300 points by the report script. Colours are given to runs when they are
// selected and kept while selected (colour follows the run, never its position); at most 8 runs.

import { lineChart, slotColor } from "./chart.js";
import { $, esc, fillSelect } from "./format.js";

const METRICS = [
  ["val:sws", "Validation SWS", "SWS"], ["val:finish_rate", "Validation finish rate", "share finished"],
  ["train:ep_return", "Mean episode return", "return"], ["train:ep_finish_rate", "Training episodes finished", "share finished"],
  ["train:ep_progress", "Mean episode progress", "metres"], ["train:entropy", "Policy entropy", "nats"], ["train:sps", "Throughput", "agent steps / s"],
];
const MAX_RUNS = 8;

export function initCurves(data) {
  const curves = data.curves || {};
  const rids = Object.keys(curves);
  const slots = new Map(); // rid -> colour slot, kept while selected
  const list = $("curves-runs"), metSel = $("curves-metric");
  fillSelect(metSel, METRICS.map(([v, l]) => [v, l]), "val:sws");
  if (!rids.length) {
    list.innerHTML = `<p class="muted">No training curves yet.</p>`;
    $("curves-chart").innerHTML = `<div class="empty">Not yet available: no run has logged training metrics.</div>`;
    return () => {};
  }
  const groups = {};
  for (const rid of rids) (groups[curves[rid].label] ||= []).push(rid);
  list.innerHTML = Object.entries(groups).map(([label, rs]) => `<h4>${esc(label)}</h4>` + rs.map((rid) =>
    `<label><input type="checkbox" value="${esc(rid)}"><span class="swatch" data-rid="${esc(rid)}"></span>seed ${curves[rid].seed}</label>`).join("")).join("");
  const boxes = [...list.querySelectorAll("input[type=checkbox]")];
  const initial = rids.filter((r) => curves[r].key === "generalist").slice(0, MAX_RUNS);
  for (const b of boxes) b.checked = (initial.length ? initial : rids.slice(0, 1)).includes(b.value);
  const draw = () => {
    const chosen = boxes.filter((b) => b.checked).map((b) => b.value);
    for (const rid of [...slots.keys()]) if (!chosen.includes(rid)) slots.delete(rid);
    for (const rid of chosen) if (!slots.has(rid)) slots.set(rid, [...Array(MAX_RUNS).keys()].find((i) => ![...slots.values()].includes(i)));
    for (const b of boxes) b.disabled = !b.checked && chosen.length >= MAX_RUNS;
    for (const sw of list.querySelectorAll(".swatch")) sw.style.background = slots.has(sw.dataset.rid) ? slotColor(slots.get(sw.dataset.rid)) : "transparent";
    const [src, col] = metSel.value.split(":");
    const meta = METRICS.find((m) => m[0] === metSel.value);
    const series = chosen.map((rid) => {
      const d = curves[rid][src] || { step: [] };
      return { label: `${curves[rid].label} s${curves[rid].seed}`, color: slotColor(slots.get(rid)),
        x: d.step.map((v) => (v == null ? null : v / 1e6)), y: d[col] || [] };
    });
    lineChart($("curves-chart"), series, {
      title: meta[1], xLabel: "agent steps (millions)", yLabel: meta[2],
      refLine: col === "sws" ? { value: 1, label: "scripted driver = 1.0" } : undefined,
      emptyText: "No data for this metric and selection.",
    });
  };
  list.onchange = draw;
  metSel.onchange = draw;
  return draw;
}
