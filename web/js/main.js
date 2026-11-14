// main.js — entry point of the browser game: loads the data, owns the app state (menu or race) and
// wires the loop, input, renderer, HUD, menu, race session, ghosts and the AI driver together.
//
// Global context: web/index.html loads this module. The game core (physics, controls, rules) lives
// in web/js/core and is shared with the Node.js parity tests; this file and its siblings are the
// presentation layer around it (docs/formats.md describes the data files read here).

import * as THREE from '../vendor/three/three.module.js';
import { loadIndex, loadVehicle, loadTrack, loadAiGhosts, modelUrl } from './assets.js';
import { FixedLoop } from './loop.js';
import { Keyboard } from './input.js';
import { Renderer } from './render.js';
import { createCarMesh } from './carmesh.js';
import { Hud } from './hud.js';
import { Menu } from './menu.js';
import { RaceSession } from './race.js';
import { loadBest, saveBest, loadSettings, saveSettings, samplePose } from './ghost.js';
import { ReplayCar } from './replay.js';
import { formatTime, formatDelta, medalTargets, medalFor } from './timing.js';

const MODE_LABELS = { race: 'Race', 'race-ai': 'Race vs AI', watch: 'Watch AI', replay: 'Replay' };
const COLORS = { player: 0xff6b35, ai: 0x7b61ff, bestGhost: 0xffffff, aiGhost: 0x7fd3ff };

const app = {
  state: 'loading', // 'loading' | 'menu' | 'race'
  tracks: new Map(), // name -> track file
  aiGhosts: new Map(), // name -> [{ path, model, time, states }]
  policies: new Map(), // model name -> Promise<Policy>
  settings: loadSettings(), // { track, model, ghosts: { key: bool } }
  session: null,
  cars: [], // [{ car, mesh }] for the session's cars
  ghostMeshes: [], // [{ states, mesh }]
  snap: true, // next camera update jumps instead of easing
  menuTime: 0, // seconds spent in the menu (drives the orbit camera)
};

const $ = (id) => document.getElementById(id);
const setStatus = (text) => { $('loading').textContent = text ?? ''; $('loading').hidden = !text; };
const showError = (text) => { $('error').textContent = text; $('error').hidden = false; };

// ---- data ---------------------------------------------------------------------------------------

async function ensureTrack(name) {
  if (!app.tracks.has(name)) {
    const [track, ghosts] = await Promise.all([loadTrack(name), loadAiGhosts(app.index, name)]);
    app.tracks.set(name, track);
    app.aiGhosts.set(name, ghosts);
  }
  return app.tracks.get(name);
}

const meta = (name) => app.index.tracks.find((t) => t.name === name);
const authorTime = (name) => {
  const times = (app.aiGhosts.get(name) ?? []).filter((g) => g.finished).map((g) => g.time).filter((t) => t > 0); // finished runs only
  return times.length ? Math.min(...times) : null;
};
const targetsOf = (name) => (app.tracks.has(name) ? medalTargets(app.tracks.get(name), authorTime(name)) : null);
const ghostOn = (key) => app.settings.ghosts?.[key] ?? key === 'best'; // your best: on by default; AI ghosts: off
const currentModel = () => app.index.models.find((m) => m.name === app.settings.model) ?? app.index.models[0];
const persist = () => saveSettings(app.settings);

function loadPolicy(model) {
  if (!app.policies.has(model.name)) {
    const p = import('./policy.js').then(({ Policy }) => Policy.load(model, modelUrl(model.file))); // ONNX runtime loads only now
    p.catch(() => app.policies.delete(model.name));
    app.policies.set(model.name, p);
  }
  return app.policies.get(model.name);
}

// ---- menu ---------------------------------------------------------------------------------------

function renderList() {
  app.menu.renderList(app.index.tracks.map((t) => {
    const best = loadBest(t.name)?.time ?? null;
    return { name: t.name, title: t.title ?? t.name, best, medal: medalFor(best, targetsOf(t.name)) };
  }), app.selected);
}

function renderDetail() {
  const name = app.selected, track = app.tracks.get(name);
  if (!track) return;
  const best = loadBest(name), targets = targetsOf(name);
  const jumps = track.gap.filter((g, i) => g && !track.gap[i - 1]).length; // runs of gap segments
  const cps = track.checkpoints.length - 1;
  app.menu.renderDetail({
    title: meta(name).title ?? name, name,
    info: `${Math.round(track.length)} m · ${cps} checkpoint${cps === 1 ? '' : 's'} · ${jumps} jump${jumps === 1 ? '' : 's'}`,
    best: best?.time ?? null, medal: medalFor(best?.time ?? null, targets), targets,
    ghosts: [
      { key: 'best', label: best ? `Your best (${formatTime(best.time)})` : 'Your best (none yet)', on: Boolean(best) && ghostOn('best'), disabled: !best },
      ...(app.aiGhosts.get(name) ?? []).map((g) => ({ key: g.path, on: ghostOn(g.path),
        label: g.finished ? `AI · ${g.model} (${formatTime(g.time)})` : `AI · ${g.model} (did not finish: ${g.end ?? 'out'} at ${formatTime(g.time)})` })),
    ],
    hasModels: app.hasModels, models: app.index.models, model: app.hasModels ? currentModel().name : null,
    hasReplay: (app.aiGhosts.get(name) ?? []).length > 0,
  });
}

async function selectTrack(name) {
  app.selected = name;
  app.settings.track = name;
  persist();
  renderList();
  const track = await ensureTrack(name);
  if (app.selected !== name) return; // a newer selection won the race
  app.renderer.setTrack(track);
  renderDetail();
}

function moveSelection(delta) {
  const names = app.index.tracks.map((t) => t.name);
  const i = Math.min(Math.max(names.indexOf(app.selected) + delta, 0), names.length - 1);
  if (names[i] !== app.selected) selectTrack(names[i]);
}

function onGhostToggle(key, on) { app.settings.ghosts = { ...app.settings.ghosts, [key]: on }; persist(); }
function onModel(name) { app.settings.model = name; persist(); }

// ---- race ---------------------------------------------------------------------------------------

async function startRace(mode) {
  if (app.state !== 'menu' || app.starting) return;
  if ((mode === 'race-ai' || mode === 'watch') && !app.hasModels) return;
  app.starting = true;
  try {
    const name = app.selected;
    const track = await ensureTrack(name);
    let policy = null, replay = null;
    if (mode === 'race-ai' || mode === 'watch') {
      setStatus('Loading AI driver…');
      policy = await loadPolicy(currentModel());
    } else if (mode === 'replay') { // the first ticked AI ghost of this track, else its first AI ghost
      const ghosts = app.aiGhosts.get(name) ?? [];
      const ghost = ghosts.find((g) => ghostOn(g.path)) ?? ghosts[0];
      if (!ghost) return;
      replay = new ReplayCar(app.cfg, track, ghost);
    }
    document.activeElement?.blur(); // Enter must not re-click the button that started the race
    app.renderer.setTrack(track);
    app.session = new RaceSession({ cfg: app.cfg, track, mode, policy, replay, best: loadBest(name) });
    app.targets = targetsOf(name);
    createCars();
    buildGhosts();
    app.menu.hide();
    app.hud.start(hudArgs());
    app.keyboard.active = true;
    app.keyboard.clear();
    app.state = 'race';
    app.snap = true;
    app.loop.resetClock();
    app.loop.running = true;
  } catch (e) {
    console.error(e);
    showError(`Could not start the race: ${e.message}`);
  } finally {
    setStatus(null);
    app.starting = false;
  }
}

const hudArgs = () => ({
  mode: app.session.mode,
  modeLabel: `${MODE_LABELS[app.session.mode]} · ${meta(app.selected).title ?? app.selected}`,
  targets: app.targets, bestTime: app.session.best?.time ?? null, cpTotal: app.session.track.checkpoints.length,
});

function createCars() {
  for (const c of app.cars) c.mesh.dispose();
  const s = app.session;
  app.cars = [[s.player, COLORS.player], [s.ai, COLORS.ai]].filter(([car]) => car).map(([car, body]) => {
    const mesh = createCarMesh(app.cfg, { body });
    app.renderer.scene.add(mesh.group);
    return { car, mesh };
  });
}

function buildGhosts() {
  for (const g of app.ghostMeshes) g.mesh.dispose();
  const list = [];
  const best = app.session.best;
  if (best?.states?.length && ghostOn('best')) list.push({ states: best.states, body: COLORS.bestGhost });
  const replayed = app.session?.ai?.states; // in replay mode the replayed run is drawn as the AI car, not as a ghost
  for (const g of app.aiGhosts.get(app.selected) ?? []) if (ghostOn(g.path) && g.states !== replayed) list.push({ states: g.states, body: COLORS.aiGhost });
  app.ghostMeshes = list.map(({ states, body }) => {
    const mesh = createCarMesh(app.cfg, { body, ghost: true });
    app.renderer.scene.add(mesh.group);
    return { states, mesh };
  });
}

function restart() {
  const s = app.session;
  s.best = loadBest(app.selected); // a run that just finished may be the new best
  s.restart();
  buildGhosts();
  app.hud.start(hudArgs());
  app.keyboard.clear();
  app.snap = true;
  app.loop.resetClock();
}

function toMenu() {
  app.loop.running = false;
  app.state = 'menu';
  for (const c of [...app.cars, ...app.ghostMeshes]) c.mesh.dispose();
  app.cars = [];
  app.ghostMeshes = [];
  app.session = null;
  app.keyboard.active = false;
  app.hud.hide();
  renderList();
  renderDetail();
  app.menu.show();
}

// Race events: split deltas, respawn notices, finish screens.
function handleEvents(events) {
  const s = app.session, dt = app.cfg.dt, last = s.track.checkpoints.length - 1;
  const focusCar = s.focus === s.player ? 'player' : 'ai';
  for (const e of events) {
    const isFocus = e.car === focusCar;
    if (e.type === 'checkpoint' && isFocus) {
      const bestTicks = s.best?.splits?.[e.k];
      const label = e.k === last ? 'Finish' : `CP ${e.k + 1}  ${formatTime(e.ticks * dt)}`;
      app.hud.showSplit(label, bestTicks != null ? (e.ticks - bestTicks) * dt : null);
    } else if (e.type === 'respawn' && isFocus) {
      const notices = { manual: 'Back to the last checkpoint', fell: 'Fell off: back to the last checkpoint', airborne: 'Airborne too long: back to the last checkpoint', stalled: 'No progress for 2 s: back to the last checkpoint',
        dnf: `Out (${e.end ?? 'fell'}): the recorded run ends here, did not finish` }; // dnf: end of an unfinished replay (replay.js)
      if (e.moved > 1) app.hud.message(notices[e.reason] ?? `Out (${e.reason})`); // waiting at the spawn point is not news
      app.snap = true;
    } else if (e.type === 'finish') {
      if (e.car === 'player') onPlayerFinish();
      else if (s.mode === 'watch' || s.mode === 'replay') onAiFinishWatched();
      else app.hud.message(`AI finished in ${formatTime(e.ticks * dt)}`, 4000);
    }
  }
}

function onPlayerFinish() {
  const s = app.session, p = s.player, dt = app.cfg.dt, name = app.selected;
  const time = p.finishTicks * dt, prev = s.best;
  const isBest = !prev || p.finishTicks < prev.ticks;
  let note = isBest
    ? (prev ? `New personal best (${formatDelta(time - prev.time)})` : 'First finish: saved as your best')
    : `${formatDelta(time - prev.time)} against your best ${formatTime(prev.time)}`;
  if (isBest && !saveBest(name, { time, ticks: p.finishTicks, splits: p.splits.slice(), states: p.ghost, respawns: p.respawns, date: new Date().toISOString() })) {
    note += ' (not saved: browser storage unavailable)';
  }
  const ai = s.ai ? (s.ai.finished && s.ai.finishTicks <= s.clock ? `AI: ${formatTime(s.ai.finishTicks * dt)}` : 'AI: still racing') : '';
  app.hud.showFinish({
    title: 'Finished', time, splits: p.splits.map((t) => t * dt), bestSplits: prev?.splits?.map((t) => t * dt),
    medal: medalFor(time, app.targets), note,
    extra: [p.respawns ? `${p.respawns} respawn${p.respawns === 1 ? '' : 's'}` : '', ai].filter(Boolean).join(' · '),
  });
  renderList();
}

function onAiFinishWatched() {
  const s = app.session, ai = s.ai, dt = app.cfg.dt, time = ai.finishTicks * dt;
  app.hud.showFinish({
    title: `AI finished · ${ai.policy.name}`, time, splits: ai.splits.map((t) => t * dt), bestSplits: s.best?.splits?.map((t) => t * dt),
    medal: medalFor(time, app.targets), note: s.best ? `${formatDelta(time - s.best.time)} against your best` : '',
    extra: [ai.respawns ? `${ai.respawns} respawn${ai.respawns === 1 ? '' : 's'}` : '', ai.late ? `${ai.late} late inference step(s)` : ''].filter(Boolean).join(' · '),
  });
}

function aiStatus(s) {
  const ai = s.ai;
  if (!ai) return null;
  if (ai.failed) return 'AI stopped: inference error (see console)';
  if (ai.stopped) return `AI ${ai.policy.name} did not finish (${ai.end})`;
  const late = ai.late ? ` · ${ai.late} late step${ai.late === 1 ? '' : 's'}` : '';
  if (ai.finished && ai.finishTicks <= s.clock) return `AI ${ai.policy.name} finished ${formatTime(ai.finishTicks * app.cfg.dt)}${late}`;
  const total = s.track.checkpoints.length - 1;
  return `AI ${ai.policy.name} · checkpoint ${Math.min(ai.stateAt(s.clock).cp, total)} / ${total}${late}`;
}

// ---- drawing ------------------------------------------------------------------------------------

const tmpPos = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpQuat = new THREE.Quaternion();
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
const ghostPose = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), steer: 0 };

// Interpolate between two tick states (alpha = 0 -> a, 1 -> b) into tmpPos / tmpQuat.
function lerpFrames(a, b, alpha) {
  tmpPos.set(...a.p).lerp(tmpB.set(...b.p), alpha);
  tmpQuat.slerpQuaternions(qa.set(...a.q), qb.set(...b.q), alpha);
}

function drawRace(alpha, elapsed) {
  const s = app.session, a = s.running ? alpha : 1; // a finished race shows its final tick
  handleEvents(s.drainEvents());
  for (const { car, mesh } of app.cars) {
    const [f0, f1] = s.framesOf(car);
    lerpFrames(f0, f1, a);
    mesh.setPose(tmpPos, tmpQuat);
    mesh.setWheels(f1.susp, f1.steer, f1.spin);
  }
  const step = Math.max(s.clock - 1 + a, 0) / app.cfg.ticks_per_action; // ghost entries are one per agent step
  for (const g of app.ghostMeshes) {
    samplePose(g.states, step, ghostPose);
    g.mesh.setPose(ghostPose.position, ghostPose.quaternion);
    g.mesh.setWheels(null, ghostPose.steer, null);
  }
  const [f0, f1] = s.framesOf(s.focus);
  lerpFrames(f0, f1, a);
  app.renderer.follow(tmpPos, tmpQuat, f1.speed, elapsed, app.snap);
  app.snap = false;
  app.renderer.setNextCheckpoint(f1.cp);
  app.hud.update({ time: s.timeOf(s.focus), speedKmh: f1.speed * 3.6, cp: f1.cp, ai: aiStatus(s) });
}

function frame(alpha, elapsed) {
  if (app.state === 'race') drawRace(alpha, elapsed);
  else { app.menuTime += elapsed; app.renderer.orbit(app.menuTime); }
  app.renderer.render();
}

// ---- keys ---------------------------------------------------------------------------------------

function onKey(e) {
  if (app.state === 'menu') {
    const tag = e.target.tagName;
    if ((tag === 'BUTTON' && e.code === 'Enter') || (tag === 'SELECT' && /Arrow|Enter/.test(e.code))) return; // native behaviour
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp') { e.preventDefault(); moveSelection(e.code === 'ArrowDown' ? 1 : -1); }
    else if (e.code === 'Enter' || e.code === 'NumpadEnter') { e.preventDefault(); startRace('race'); }
  } else if (app.state === 'race') {
    if (['Enter', 'NumpadEnter', 'KeyR', 'Backspace', 'Escape'].includes(e.code)) e.preventDefault();
    if (e.repeat) return;
    if (e.code === 'Enter' || e.code === 'NumpadEnter' || e.code === 'KeyR') restart();
    else if (e.code === 'Backspace') app.session.player?.respawnNow();
    else if (e.code === 'Escape') toMenu();
  }
}

// ---- start --------------------------------------------------------------------------------------

async function init() {
  setStatus('Loading…');
  const [index, cfg] = await Promise.all([loadIndex(), loadVehicle()]);
  index.models ??= [];
  Object.assign(app, { index, cfg, hasModels: index.models.length > 0 });
  app.renderer = new Renderer($('view'));
  app.hud = new Hud();
  app.keyboard = new Keyboard();
  app.menu = new Menu({ onSelect: selectTrack, onStart: startRace, onGhostToggle, onModel });
  app.loop = new FixedLoop({ dt: cfg.dt, maxTicks: 25, tick: () => app.session?.tick(app.keyboard), frame });
  window.addEventListener('keydown', onKey);
  $('finish').addEventListener('click', (e) => {
    const act = e.target.closest('button')?.dataset.act;
    if (act === 'retry') restart();
    if (act === 'menu') toMenu();
  });
  const names = index.tracks.map((t) => t.name);
  await selectTrack(names.includes(app.settings.track) ? app.settings.track : names[0]);
  app.state = 'menu';
  app.menu.show();
  setStatus(null);
  app.loop.start();
  for (const name of names) ensureTrack(name).then(renderList, (err) => console.error(`track ${name}:`, err)); // medals need every track
}

// Read-only view of the game for debugging and the browser tests (tests/web/test_game.py).
window.racingmania = {
  get state() {
    const s = app.session;
    if (!s) return { screen: app.state, selected: app.selected };
    const f = s.framesOf(s.focus)[1];
    return {
      screen: app.state, mode: s.mode, time: s.timeOf(s.focus), clock: s.clock, tick: f.ticks, speedKmh: f.speed * 3.6,
      position: f.p, quaternion: f.q, checkpoint: f.cp, finished: s.focus.finished, respawns: s.focus.respawns, aiLate: s.ai?.late ?? 0,
    };
  },
};

init().catch((e) => { console.error(e); setStatus(null); showError(`Could not load the game data: ${e.message}`); });
