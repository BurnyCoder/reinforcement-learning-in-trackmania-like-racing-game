// format.js — number formatting, HTML escaping and small DOM helpers for the results explorer.
// Global context: every view of web/results.html prints numbers through these functions, so a
// missing value always shows as an em dash and intervals always read "est [lo, hi]".

export const NA = "—";

/** Is v a finite number (null, undefined and NaN are "missing")? */
export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** Fixed-decimal number or an em dash. */
export const num = (v, d = 3) => (isNum(v) ? v.toFixed(d) : NA);

/** Share in [0, 1] as a percentage string. */
export const pct = (v, d = 1) => (isNum(v) ? `${(100 * v).toFixed(d)} %` : NA);

/** Estimate with its interval, e.g. "0.812 [0.701, 0.905]". */
export const ci = (est, lo, hi, d = 3) => (!isNum(est) ? NA : isNum(lo) ? `${est.toFixed(d)} [${lo.toFixed(d)}, ${hi.toFixed(d)}]` : est.toFixed(d));

/** p-value with a floor. */
export const pval = (p) => (!isNum(p) ? NA : p < 1e-4 ? "< 0.0001" : p.toFixed(4));

/** Escape text for innerHTML. */
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** Markdown-style **bold** in generated sentences -> <strong> (input is escaped first). */
export const boldMd = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

/** Element by id. */
export const $ = (id) => document.getElementById(id);

/** Fill a <select> with [value, label] pairs, keeping the current value if it is still offered. */
export function fillSelect(sel, options, preferred) {
  const keep = sel.value || preferred;
  sel.innerHTML = options.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
  if (options.some(([v]) => v === keep)) sel.value = keep;
}

/** Simple HTML table from headers and rows of cell HTML; numeric cells get class "num". */
export function table(headers, rows, numericCols = []) {
  if (!rows.length) return `<p class="muted">Not yet available: the runs behind this table have no results yet.</p>`;
  const th = headers.map((h) => `<th>${esc(h)}</th>`).join("");
  const body = rows.map((r) => `<tr>${r.map((c, i) => `<td${numericCols.includes(i) ? ' class="num"' : ""}>${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`;
}
