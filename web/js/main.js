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
  countdown: 0, // seconds left of the start countdown (beginCountdown); the race clock waits while > 0
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
const finishedRun = (g) => g.finished !== false; // ghosts without the flag (older files or a cached older assets.js) are finished runs
const authorTime = (name) => {
  const times = (app.aiGhosts.get(name) ?? []).filter(finishedRun).map((g) => g.time).filter((t) => t > 0); // finished runs only
  return times.length ? Math.min(...times) : null;
};
const targetsOf = (name) => (app.tracks.has(name) ? medalTargets(app.tracks.get(name), authorTime(name)) : null);
const ghostOn = (key) => app.settings.ghosts?.[key] ?? key === 'best'; // your best: on by default; AI ghosts: off

// The AI drivers of a track: every exported network (index.json "models") and every recorded run on the track
// (its AI ghosts, e.g. the scripted driver or a control), merged by name. Order: the ghosts' order (best run
// first, assets.js byBestRun), then networks with no recorded run here, in index order.
function driversOf(name) {
  const ghosts = app.aiGhosts.get(name) ?? [];
  const runOf = new Map(ghosts.map((g) => [g.model, g]));
  const rows = app.index.models.map((m) => ({ name: m.name, model: m, ghost: runOf.get(m.name) ?? null }));
  const networks = new Set(rows.map((r) => r.name));
  for (const g of ghosts) if (!networks.has(g.model)) rows.push({ name: g.model, model: null, ghost: g }); // recorded run only
  const rank = (r) => (r.ghost ? ghosts.indexOf(r.ghost) : ghosts.length);
  return rows.sort((a, b) => rank(a) - rank(b)); // stable: networks without a run keep index order
}

// The chosen AI driver (Race vs AI, Watch AI, Watch replay): the saved choice if this track lists it, else the
// best driver with a live network, else the best recorded run.
const chosenDriver = (name) => {
  const drivers = driversOf(name);
  return drivers.find((d) => d.name === app.settings.model) ?? drivers.find((d) => d.model) ?? drivers[0] ?? null;
};
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
  const best = loadBest(name), targets = targetsOf(name), chosen = chosenDriver(name);
  const drivers = driversOf(name).map((d) => {
    const g = d.ghost, run = d.model?.run ?? '';
    const learner = d.model && !run.startsWith('baselines/'); // a trained network (not a control)
    return {
      name: d.name, selected: d.name === chosen?.name, live: Boolean(d.model),
      description: d.model?.description ?? 'Recorded run only: no network to drive live in the game',
      // Specialists are named after their track (run "specialist-campaign-01/seed0"); every other network
      // trained on generated pool tracks, which the menu never lists, so for them every menu track is new.
      tag: learner ? (run.split('/')[0].endsWith(name) ? 'trained on this track' : 'never trained on this track') : '',
      finished: Boolean(g && finishedRun(g)),
      timeText: !g ? 'no run here' : finishedRun(g) ? formatTime(g.time) : `out (${g.end ?? 'fell'}) at ${formatTime(g.time)}`,
      medal: g && finishedRun(g) ? medalFor(g.time, targets) : null,
      ghostKey: g?.path ?? null, ghostOn: Boolean(g) && ghostOn(g.path),
    };
  });
  const jumps = track.gap.filter((g, i) => g && !track.gap[i - 1]).length; // runs of gap segments
  const cps = track.checkpoints.length - 1;
  app.menu.renderDetail({
    title: meta(name).title ?? name, name,
    info: `${Math.round(track.length)} m · ${cps} checkpoint${cps === 1 ? '' : 's'} · ${jumps} jump${jumps === 1 ? '' : 's'}`,
    best: best?.time ?? null, medal: medalFor(best?.time ?? null, targets), targets,
    bestGhost: { on: Boolean(best) && ghostOn('best'), disabled: !best },
    drivers, canLive: Boolean(chosen?.model), canReplay: Boolean(chosen?.ghost),
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
function onModel(name) { app.settings.model = name; persist(); renderDetail(); } // the chosen AI driver

// ---- race ---------------------------------------------------------------------------------------

async function startRace(mode) {
  if (app.state !== 'menu' || app.starting) return;
  const driver = chosenDriver(app.selected);
  if ((mode === 'race-ai' || mode === 'watch') && !driver?.model) return;
  if (mode === 'replay' && !driver?.ghost) return;
  app.starting = true;
  try {
    const name = app.selected;
    const track = await ensureTrack(name);
    let policy = null, replay = null;
    if (mode === 'race-ai' || mode === 'watch') {
      setStatus('Loading AI driver…');
      policy = await loadPolicy(driver.model);
    } else if (mode === 'replay') { // the chosen driver's recorded official run on this track
      replay = new ReplayCar(app.cfg, track, driver.ghost);
    }
    document.activeElement?.blur(); // Enter must not re-click the button that started the race
    app.renderer.setTrack(track);
    app.session = new RaceSession({ cfg: app.cfg, track, mode, policy, replay, best: loadBest(name) });
    app.session.driverName = mode === 'race' ? null : driver.name; // shown in the HUD's mode label
    app.targets = targetsOf(name);
    createCars();
    buildGhosts();
    app.menu.hide();
    app.hud.start(hudArgs());
    app.keyboard.active = true;
    app.keyboard.forgetTaps(); // keys still held count from the first tick; taps made in the menu do not
    app.state = 'race';
    app.snap = true;
    beginCountdown();
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
  modeLabel: [MODE_LABELS[app.session.mode], app.session.driverName, meta(app.selected).title ?? app.selected].filter(Boolean).join(' · '),
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
  app.keyboard.forgetTaps(); // a key held through R keeps acting (no new keydown comes for it)
  app.snap = true;
  beginCountdown();
}

// Start countdown, in the style of TrackMania's 3-2-1: while a human drives, the race clock waits at 0 for
// COUNTDOWN_S and the keyboard is still read, so a key held at GO, or held through a restart, acts from the
// very first physics tick. Spectator modes (watch, replay) start at once.
const COUNTDOWN_S = 1.5; // "3", "2", "1" for 0.5 s each: short, so restarting stays quick
function beginCountdown() {
  app.countdown = app.session.player ? COUNTDOWN_S : 0;
  app.hud.countdown(app.countdown ? '3' : '');
  app.loop.resetClock();
  app.loop.running = app.countdown === 0;
}

// Once per frame during a race: count down in real time, then let the fixed-tick loop run.
function stepCountdown(elapsed) {
  if (!(app.countdown > 0)) return;
  app.countdown -= elapsed;
  if (app.countdown > 0) { app.hud.countdown(String(Math.ceil(app.countdown / (COUNTDOWN_S / 3)))); return; }
  app.countdown = 0;
  app.keyboard.forgetTaps(); // a key tapped and released during the countdown does not count; held keys do
  app.hud.countdown('GO!');
  app.loop.running = true; // nothing was banked while stopped, so the first tick comes one dt after GO
}

function toMenu() {
  app.loop.running = false;
  app.countdown = 0;
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
  if (app.state === 'race') { stepCountdown(elapsed); drawRace(alpha, elapsed); }
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
  Object.assign(app, { index, cfg });
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
      screen: app.state, mode: s.mode, countdown: app.countdown, time: s.timeOf(s.focus), clock: s.clock, tick: f.ticks, speedKmh: f.speed * 3.6,
      position: f.p, quaternion: f.q, checkpoint: f.cp, finished: s.focus.finished, respawns: s.focus.respawns, aiLate: s.ai?.late ?? 0,
    };
  },
};

init().catch((e) => { console.error(e); setStatus(null); showError(`Could not load the game data: ${e.message}`); });
