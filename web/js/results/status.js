// status.js — the run-status board: every run of the run matrix as a chip, grouped by phase.
// Global context: statuses come from each run's run.json (planned → training → trained → evaluated),
// and run-matrix entries without a run.json are "not started" (reports/plan.md, publishing rule).
// Status colours are reserved for states and always come with an icon and the status word.

import { esc } from "./format.js";

const ICON = { evaluated: "✓", trained: "●", training: "▶", planned: "○", "not started": "·" };

export function renderStatus(data, board, legend) {
  const runs = data.runs || [];
  const counts = Object.fromEntries((data.statuses || []).map((s) => [s, runs.filter((r) => r.status === s).length]));
  legend.innerHTML = (data.statuses || []).map((s) =>
    `<span class="key"><span class="chip ${s.replace(" ", "-")}"><span class="icon">${ICON[s] || ""}</span>${esc(s)}</span>` +
    `${counts[s]} — ${esc(data.status_text?.[s] || "")}</span>`).join("");
  const phases = [...new Set(runs.map((r) => r.phase || "unassigned"))];
  board.innerHTML = phases.length ? phases.map((ph) => {
    const rs = runs.filter((r) => (r.phase || "unassigned") === ph);
    const done = rs.filter((r) => r.status === "evaluated").length;
    const chips = rs.map((r) => {
      const title = `${r.rid} — ${r.status}${r.global_step ? `, ${Number(r.global_step).toLocaleString("en")} steps` : ""}`;
      return `<span class="chip ${r.status.replace(" ", "-")}" title="${esc(title)}"><span class="icon">${ICON[r.status] || ""}</span>${esc(r.rid)}</span>`;
    }).join("");
    return `<div class="phase"><h3>${esc(ph)} <span class="muted small">${done}/${rs.length} evaluated</span></h3><div class="chips">${chips}</div></div>`;
  }).join("") : `<p class="muted">No runs are registered or planned.</p>`;
}
