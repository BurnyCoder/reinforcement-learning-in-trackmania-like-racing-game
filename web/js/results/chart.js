// chart.js — small hand-written SVG charts for the results explorer (no libraries, no CDN).
// Local purpose: linear and log scales, "nice" ticks, a dot-with-interval chart (one row per
// configuration), a multi-line chart, and one shared hover tooltip.
// Global context: used by methods.js, curves.js and the size view; colours come from CSS custom
// properties (--s1..--s8, the validated categorical palette), so light and dark mode both work.
// Rules kept: zero-based value axes, one y-scale per chart, solid hairline grid, intervals drawn
// wherever they exist, legend for two or more series, hover tooltip on every mark.

import { esc, isNum } from "./format.js";

const NS = "http://www.w3.org/2000/svg";
export const SLOT_VARS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
export const CATEGORY_SLOT = { method: 0, control: 1, reference: 2, baseline: 3, ablation: 4, imitation: 5, hyperparameter: 6, specialist: 7 };
export const slotColor = (i) => `var(${SLOT_VARS[i % SLOT_VARS.length]})`;

/** Create an SVG element with attributes. */
export function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Linear scale [d0, d1] -> [r0, r1]. */
export const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

/** Log10 scale for positive domains. */
export const log10 = (d0, d1, r0, r1) => (v) => r0 + ((Math.log10(v) - Math.log10(d0)) / (Math.log10(d1) - Math.log10(d0) || 1)) * (r1 - r0);

/** About n "nice" ticks (1, 2, 5 × 10^k steps) covering [lo, hi]. */
export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

// ------------------------------------------------------------------------------- tooltip
const tip = () => document.getElementById("tooltip");
export function bindTip(node, html) {
  node.addEventListener("mousemove", (ev) => {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8);
    t.style.left = `${x}px`;
    t.style.top = `${ev.clientY + 14}px`;
  });
  node.addEventListener("mouseleave", () => { tip().hidden = true; });
}

/** Message instead of a chart. */
export function empty(container, text) {
  container.innerHTML = `<div class="empty">${esc(text)}</div>`;
}

// ------------------------------------------------------------------------------- dot + interval rows
/**
 * Horizontal dot-and-interval chart: one row per item {label, value, lo, hi, color, tip}.
 * opts: {xMax, xLabel, refLine, title}. The value axis always starts at 0.
 */
export function dotIntervalChart(container, items, opts = {}) {
  if (!items.length) return empty(container, "Not yet available: no results for this selection.");
  const rowH = 26, left = Math.min(320, 12 + 6.4 * Math.max(...items.map((d) => d.label.length))), right = 60, top = 30, bottom = 44;
  const width = Math.max(560, Math.min(1100, container.clientWidth || 900));
  const height = top + bottom + rowH * items.length;
  const xMax = opts.xMax ?? Math.max(1.05, ...items.map((d) => (isNum(d.hi) ? d.hi : d.value) || 0)) * 1.04;
  const x = linear(0, xMax, left, width - right);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": opts.title || "chart" });
  if (opts.title) svg.append(el("text", { x: left, y: 16, class: "title" }, opts.title));
  const grid = el("g", { class: "grid" });
  for (const t of niceTicks(0, xMax)) {
    grid.append(el("line", { x1: x(t), x2: x(t), y1: top - 6, y2: height - bottom }));
    svg.append(el("text", { x: x(t), y: height - bottom + 16, "text-anchor": "middle" }, String(t)));
  }
  svg.prepend(grid);
  svg.append(el("text", { x: (left + width - right) / 2, y: height - 8, "text-anchor": "middle" }, opts.xLabel || ""));
  if (opts.refLine !== undefined) {
    svg.append(el("line", { x1: x(opts.refLine.value), x2: x(opts.refLine.value), y1: top - 6, y2: height - bottom, class: "ref" }));
    svg.append(el("text", { x: x(opts.refLine.value) + 4, y: top - 10 }, opts.refLine.label));
  }
  items.forEach((d, i) => {
    const y = top + rowH * i + rowH / 2;
    svg.append(el("text", { x: left - 8, y: y + 4, "text-anchor": "end" }, d.label));
    if (!isNum(d.value)) return;
    if (isNum(d.lo) && isNum(d.hi)) {
      svg.append(el("line", { x1: x(d.lo), x2: x(d.hi), y1: y, y2: y, stroke: d.color, "stroke-width": 2 }));
      for (const v of [d.lo, d.hi]) svg.append(el("line", { x1: x(v), x2: x(v), y1: y - 5, y2: y + 5, stroke: d.color, "stroke-width": 2 }));
    }
    svg.append(el("circle", { cx: x(d.value), cy: y, r: 5, fill: d.color, class: "mark" }));
    if (d.note) svg.append(el("text", { x: width - right + 6, y: y + 4 }, d.note));
    const hit = el("rect", { x: left, y: y - rowH / 2, width: width - right - left, height: rowH, class: "hit" });
    bindTip(hit, d.tip || esc(d.label));
    svg.append(hit);
  });
  container.innerHTML = "";
  container.append(svg);
}

// ------------------------------------------------------------------------------- lines
/**
 * Multi-line chart. series: [{label, color, x: [], y: [], lo?: [], hi?: []}] (null gaps skipped).
 * opts: {xLabel, yLabel, title, logX, xDomain, yMin (default 0), refLine, points (draw markers), height, emptyText}.
 */
export function lineChart(container, series, opts = {}) {
  const pts = series.flatMap((s) => s.x.map((xv, i) => [xv, s.y[i], s.lo?.[i], s.hi?.[i]])).filter(([a, b]) => isNum(a) && isNum(b));
  if (!pts.length) return empty(container, opts.emptyText || "Not yet available: no data for this selection.");
  const width = Math.max(520, Math.min(1000, container.clientWidth || 800)), height = opts.height || 340;
  const m = { l: 64, r: 110, t: 30, b: 44 };
  let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  if (opts.logX) { x0 = Math.max(0.5, x0 / 2); x1 = x1 * 2; } else if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (opts.xDomain) [x0, x1] = opts.xDomain; // shared x-range when two charts are read together
  const ys = pts.flatMap((p) => [p[1], p[2], p[3]]).filter(isNum);
  let y0 = Math.min(opts.yMin ?? 0, ...ys), y1 = Math.max(...ys, opts.refLine?.value ?? -Infinity);
  if (y0 === y1) y1 = y0 + 1;
  y1 += (y1 - y0) * 0.06;
  const x = (opts.logX ? log10 : linear)(x0, x1, m.l, width - m.r);
  const y = linear(y0, y1, height - m.b, m.t);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": opts.title || "chart" });
  const grid = el("g", { class: "grid" });
  for (const t of niceTicks(y0, y1)) {
    grid.append(el("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }));
    svg.append(el("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, String(t)));
  }
  const xt = opts.logX ? Array.from({ length: 8 }, (_, k) => 10 ** k).filter((v) => v >= x0 && v <= x1) : niceTicks(x0, x1, 6);
  for (const t of xt) svg.append(el("text", { x: x(t), y: height - m.b + 16, "text-anchor": "middle" }, opts.logX ? t.toLocaleString("en") : String(t)));
  svg.prepend(grid);
  if (y0 < 0) svg.append(el("line", { x1: m.l, x2: width - m.r, y1: y(0), y2: y(0), class: "ref" }));
  if (opts.refLine) {
    svg.append(el("line", { x1: m.l, x2: width - m.r, y1: y(opts.refLine.value), y2: y(opts.refLine.value), class: "ref" }));
    svg.append(el("text", { x: m.l + 4, y: y(opts.refLine.value) - 4 }, opts.refLine.label));
  }
  if (opts.title) svg.append(el("text", { x: m.l, y: 16, class: "title" }, opts.title));
  svg.append(el("text", { x: (m.l + width - m.r) / 2, y: height - 8, "text-anchor": "middle" }, opts.xLabel || ""));
  svg.append(el("text", { x: 14, y: (m.t + height - m.b) / 2, transform: `rotate(-90 14 ${(m.t + height - m.b) / 2})`, "text-anchor": "middle" }, opts.yLabel || ""));
  for (const s of series) {
    const p = s.x.map((xv, i) => [xv, s.y[i], s.lo?.[i], s.hi?.[i]]).filter(([a, b]) => isNum(a) && isNum(b));
    if (!p.length) continue;
    if (!opts.points) {
      svg.append(el("path", { d: p.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(""), fill: "none", stroke: s.color, "stroke-width": 2 }));
    }
    for (const [a, b, lo, hi] of p) {
      if (isNum(lo) && isNum(hi)) svg.append(el("line", { x1: x(a), x2: x(a), y1: y(lo), y2: y(hi), stroke: s.color, "stroke-width": 2 }));
      if (opts.points) svg.append(el("circle", { cx: x(a), cy: y(b), r: 5, fill: s.color, class: "mark" }));
      const hit = el("circle", { cx: x(a), cy: y(b), r: 7, class: "hit" });
      bindTip(hit, `<b>${esc(s.label)}</b><br>${esc(opts.xLabel || "x")}: ${a.toLocaleString("en")}<br>${esc(opts.yLabel || "y")}: ${b.toFixed(3)}` +
        (isNum(lo) ? `<br>95 % interval: ${lo.toFixed(3)} to ${hi.toFixed(3)}` : ""));
      svg.append(hit);
    }
    const [la, lb] = p[p.length - 1];
    if (series.length <= 4) svg.append(el("text", { x: x(la) + 6, y: y(lb) + 4 }, s.label.length > 18 ? s.label.slice(0, 17) + "…" : s.label));
  }
  container.innerHTML = "";
  container.append(svg);
  if (series.length > 1) {
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = series.map((s) => `<span class="key"><span class="swatch" style="background:${s.color}"></span>${esc(s.label)}</span>`).join("");
    container.append(lg);
  }
}
