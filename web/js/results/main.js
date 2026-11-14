// main.js — entry point of the results explorer: load the data, wire the tabs, draw each view.
// Global context: assets/data/results.json is written by scripts/07_make_report.py from results/;
// the URL is relative to results.html, so the page works locally and under a GitHub Pages sub-path.
// Views are drawn lazily the first time their tab is opened (charts need a visible container to
// measure its width) and redrawn on window resize.

import { initCurves } from "./curves.js";
import { $, boldMd, esc } from "./format.js";
import { initMethods, renderHypotheses, renderSize } from "./methods.js";
import { renderStatus } from "./status.js";
import { initTracks } from "./tracks.js";

const DATA_URL = "assets/data/results.json";

async function load() {
  const res = await fetch(DATA_URL, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${DATA_URL}: HTTP ${res.status}`);
  return res.json();
}

function setupTabs(drawers) {
  const tabs = [...document.querySelectorAll(".tabs button")];
  const drawn = new Set();
  const show = (name) => {
    for (const t of tabs) t.setAttribute("aria-selected", String(t.dataset.view === name));
    for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== `view-${name}`;
    if (!drawn.has(name) && drawers[name]) { drawers[name](); drawn.add(name); }
    history.replaceState(null, "", `#${name}`);
  };
  for (const t of tabs) t.onclick = () => show(t.dataset.view);
  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const current = tabs.find((t) => t.getAttribute("aria-selected") === "true")?.dataset.view;
      if (current && drawers[current]) drawers[current]();
    }, 200);
  });
  const fromHash = () => {
    const name = location.hash.slice(1);
    return tabs.some((t) => t.dataset.view === name) ? name : "overview";
  };
  window.addEventListener("hashchange", () => show(fromHash())); // links such as results.html#curves open that view
  show(fromHash());
}

async function main() {
  let data;
  try {
    data = await load();
  } catch (e) {
    $("generated").textContent = `Could not load the results (${e.message}). Run scripts/07_make_report.py to generate them.`;
    return;
  }
  const total = data.runs?.length || 0, evaluated = (data.runs || []).filter((r) => r.status === "evaluated").length;
  $("generated").textContent = `Generated ${data.generated} from the result files · ${evaluated} of ${total} runs evaluated` +
    (data.paused ? ` · training paused on ${data.paused.date}, the remaining runs are postponed` : "") +
    (data.primary_set ? ` · main set: ${data.sets[data.primary_set]}` : "");
  $("repo-link").href = data.repo;
  $("headline").innerHTML = (data.headline || []).map((s) => `<li>${boldMd(s)}</li>`).join("") || `<li>${esc("No results yet.")}</li>`;
  renderHypotheses(data, $("hypotheses"));
  setupTabs({
    status: () => renderStatus(data, $("status-board"), $("status-legend")),
    methods: initMethods(data),
    curves: initCurves(data),
    size: () => renderSize(data),
    tracks: initTracks(data),
  });
}

main();
