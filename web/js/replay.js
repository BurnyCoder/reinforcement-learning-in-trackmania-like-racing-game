// replay.js — a recorded run (an AI ghost) played back as if it were a car, for the "Watch replay" mode.
//
// Global context: ghosts store one pose per agent step, [x, y, z, qx, qy, qz, qw, steer] (docs/formats.md,
// "Replay"), recorded by the Python evaluator, so a replay shows exactly the official run. Drivers that
// are not neural networks (the scripted driver) can only be shown in the game this way. ReplayCar offers
// the interface that RaceSession and the HUD use for the live AI car (aicar.js: advance, stateAt, events,
// splits, finishTicks, policy.name, late, failed, respawns): poses are interpolated between agent steps
// (ghost.js samplePose), the speed comes from successive positions, checkpoints are detected from the
// nearest centreline sample (track.checkpoints are sample indices, docs/formats.md), and the finish tick
// is the recorded time. A run that did not finish (ghost.finished === false, e.g. an official run that
// fell off) ends at its recorded time with a "did not finish" notice instead of a finish.

import * as THREE from '../vendor/three/three.module.js';
import { samplePose } from './ghost.js';

export class ReplayCar {
  constructor(cfg, track, ghost) {
    this.cfg = cfg;
    this.track = track;
    this.states = ghost.states;
    this.policy = { name: ghost.model }; // shown by the HUD like a live AI driver's name
    this.tpa = cfg.ticks_per_action; // ticks per ghost entry
    this.endTicks = Math.round(ghost.time / cfg.dt); // recorded finish tick, or the tick the run went out
    this.completes = ghost.finished !== false; // AI ghosts exported for medals are finished runs
    this.late = 0; // interface of AiCar: a replay is never late,
    this.failed = false; // never fails,
    this.end = ghost.end ?? 'fell'; // why an unfinished run ended (evaluator end reasons: fell, airborne, stalled)
    this.pose = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), steer: 0 };
    this.restart();
  }

  get finished() { return this.finishTicks !== null; }
  get done() { return this.finished || this.stopped; }

  restart() {
    this.events = []; // { type: 'checkpoint' | 'finish', ticks, ... }, released by RaceSession at the race clock
    this.splits = []; // tick of each checkpoint passed (last entry = finish), as in carrun.js
    this.finishTicks = null;
    this.stopped = false; // an unfinished run has reached its end
    this.respawns = 0; // becomes 1 when an unfinished run ends (the game would respawn a live car there)
    this.cp = 0; // next checkpoint
    this.index = this.track.start?.index ?? 0; // nearest centreline sample, followed along the track
    this.ticks = 0;
  }

  // Release checkpoint and finish events up to the race clock (same events as a live car's).
  advance(clock) {
    const cps = this.track.checkpoints, last = cps.length - 1;
    while (!this.done && this.ticks < clock) {
      this.ticks += 1;
      if (this.ticks >= this.endTicks && !this.completes) {
        this.stopped = true;
        this.respawns = 1;
        this.events.push({ type: 'respawn', reason: 'dnf', end: this.end, moved: 2, ticks: this.endTicks });
        break;
      }
      if (this.ticks >= this.endTicks) {
        this.splits[last] = this.endTicks;
        this.finishTicks = this.endTicks;
        this.events.push({ type: 'checkpoint', k: last, ticks: this.endTicks }, { type: 'finish', ticks: this.endTicks });
        break;
      }
      if (this.ticks % this.tpa !== 0) continue; // a new ghost entry every agent step
      this.index = this.nearestIndex(this.states[Math.min(this.ticks / this.tpa, this.states.length - 1)]);
      while (this.cp < last && this.index >= cps[this.cp]) {
        this.splits[this.cp] = this.ticks;
        this.events.push({ type: 'checkpoint', k: this.cp, ticks: this.ticks });
        this.cp += 1;
      }
    }
  }

  // Nearest centreline sample to a ghost entry, searched a little behind and ahead of the last one.
  nearestIndex(s) {
    const c = this.track.center;
    let best = this.index, bestD = Infinity;
    for (let i = Math.max(this.index - 5, 0); i < Math.min(this.index + 40, c.length); i++) {
      const d = (c[i][0] - s[0]) ** 2 + (c[i][1] - s[1]) ** 2 + (c[i][2] - s[2]) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // Render state at a race-clock tick, in the shape of carrun.js snapshot() (no suspension or wheel spin).
  stateAt(tick) {
    const t = Math.min(Math.max(tick, 0), this.endTicks), step = t / this.tpa;
    samplePose(this.states, step, this.pose);
    const i = Math.max(Math.min(Math.floor(step), this.states.length - 2), 0);
    const a = this.states[i], b = this.states[i + 1] ?? a;
    const speed = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / (this.tpa * this.cfg.dt); // m/s over that agent step
    const p = this.pose.position, q = this.pose.quaternion;
    return { ticks: Math.round(t), p: [p.x, p.y, p.z], q: [q.x, q.y, q.z, q.w], speed, susp: null, spin: null,
      steer: this.pose.steer, cp: this.splits.filter((k) => k <= t).length };
  }
}
