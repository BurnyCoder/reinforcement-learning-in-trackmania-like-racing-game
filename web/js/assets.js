// assets.js — loads the game's data files (index, vehicle, tracks, ghosts) over HTTP.
//
// Global context: everything the game reads lives under web/assets (docs/formats.md). URLs are
// resolved relative to this module, not to the page, so the same files work from the local server
// (http://localhost:8123/) and from a GitHub Pages sub-path (https://<user>.github.io/<repo>/).

export const WEB_ROOT = new URL('../', import.meta.url); // the web/ folder, wherever it is served from

// Absolute URL of a path given relative to web/ (paths in index.json are written this way).
export const webUrl = (path) => new URL(path, WEB_ROOT).href;

const cache = new Map(); // path -> Promise of parsed JSON, so each file is fetched once per page load

// Fetch and parse one JSON file. A failed request is dropped from the cache so it can be retried.
export function loadJson(path) {
  if (!cache.has(path)) {
    const p = fetch(webUrl(path)).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    });
    p.catch(() => cache.delete(path)); // do not keep a rejected promise around
    cache.set(path, p);
  }
  return cache.get(path);
}

export const loadIndex = () => loadJson('assets/index.json'); // track list, models, ghosts
export const loadVehicle = () => loadJson('assets/vehicle.json'); // physics parameters
export const loadTrack = (name) => loadJson(`assets/tracks/${name}.json`); // one track file

// A model's "file" may be a path relative to web/ ("assets/models/x.onnx") or a bare file name,
// which is looked up in web/assets/models/ (the folder docs/formats.md names for policy models).
export const modelUrl = (file) => webUrl(file.includes('/') ? file : `assets/models/${file}`);

// Load every AI ghost listed for a track in index.json ("ghosts": {"<track>": [paths]}).
// Files that fail to load are skipped with a console error rather than breaking the menu.
export async function loadAiGhosts(index, trackName) {
  const paths = index.ghosts?.[trackName] ?? [];
  const loaded = await Promise.all(paths.map((path) => loadJson(path).then(
    (g) => ({ path, model: g.model ?? path.split('/').pop().replace(/\.json$/, ''), time: g.time, states: g.states,
      finished: g.finished !== false, end: g.end }), // unfinished runs (finished: false) end at `time` with reason `end`
    (err) => { console.error(`ghost ${path} failed to load:`, err); return null; },
  )));
  return loaded.filter((g) => g && Array.isArray(g.states) && g.states.length > 0);
}
