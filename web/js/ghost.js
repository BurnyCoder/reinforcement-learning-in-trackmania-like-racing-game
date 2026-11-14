// ghost.js — ghost runs: recording one state per agent step, playing them back, and keeping the
// player's best run per track in localStorage.
//
// Global context: a ghost is the replay format of docs/formats.md, one [x, y, z, qx, qy, qz, qw, steer]
// per agent step (0.05 s). Entry k is the state k agent steps after the start (entry 0 = the start).
// The Python evaluator writes AI ghosts in the same format (web/assets/ghosts/<track>/<model>.json).

import * as THREE from '../vendor/three/three.module.js';

const STORAGE_PREFIX = 'racingmania:v1:'; // bump the version if the stored layout changes

// One ghost entry from the simulator, rounded (mm, 1e-5) to keep localStorage small.
export function ghostState(sim) {
  const s = sim.state(); // [px,py,pz, qx,qy,qz,qw, ...velocities]
  const r = (x, k) => Math.round(x * k) / k;
  return [r(s[0], 1e3), r(s[1], 1e3), r(s[2], 1e3), r(s[3], 1e5), r(s[4], 1e5), r(s[5], 1e5), r(s[6], 1e5), r(sim.steerAngle, 1e4)];
}

// Pose of a ghost at a fractional agent step: position and steer linearly interpolated, rotation
// spherically interpolated. Before the start / after the end the first / last entry is held.
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
export function samplePose(states, step, out) {
  const last = states.length - 1;
  const t = Math.min(Math.max(step, 0), last);
  const i = Math.min(Math.floor(t), Math.max(last - 1, 0)), f = last > 0 ? t - i : 0;
  const a = states[i], b = states[Math.min(i + 1, last)];
  out.position.set(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f);
  qa.set(a[3], a[4], a[5], a[6]);
  qb.set(b[3], b[4], b[5], b[6]);
  out.quaternion.slerpQuaternions(qa, qb, f);
  out.steer = a[7] + (b[7] - a[7]) * f;
  return out;
}

// localStorage can be unavailable (privacy mode) or full; the game then simply does not persist.
function readJson(key) {
  try { return JSON.parse(localStorage.getItem(STORAGE_PREFIX + key)); } catch { return null; }
}
function writeJson(key, value) {
  try { localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value)); return true; } catch { return false; }
}

// Best run of a track: { time, ticks, splits: [ticks at each checkpoint], states: [...], date }.
export const loadBest = (track) => readJson(`best:${track}`);
export const saveBest = (track, run) => writeJson(`best:${track}`, run);

// Menu settings (ghost toggles, chosen model).
export const loadSettings = () => readJson('settings') ?? {};
export const saveSettings = (s) => writeJson('settings', s);
