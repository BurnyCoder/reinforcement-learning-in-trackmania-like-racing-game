// carrun.js — one car's run on a track as the game sees it: the core RaceSim plus checkpoint splits,
// the finish time and respawning at the last passed checkpoint with the clock still running.
//
// Global context: physics and rules come unchanged from web/js/core (RaceSim). The core ends a run
// when the car is "out"; the game instead respawns the car (docs/formats.md, "Rules"), so this module
// re-places the chassis and restores the rule fields that must survive the respawn (clock and
// checkpoint count). PlayerCar (race.js) and AiCar (aicar.js) build on this class.

import { RaceSim } from './core/sim.js';
import { roadFrame } from './roadframe.js';

// Chassis state at rest above centreline sample i in the road frame (forward = road direction,
// up = forward x left): position center[i] + up * spawn_height, zero velocity.
export function spawnState(track, cfg, i) {
  const { p, u, q } = roadFrame(track, i);
  p.addScaledVector(u, cfg.spawn_height);
  return [p.x, p.y, p.z, q.x, q.y, q.z, q.w, 0, 0, 0, 0, 0, 0];
}

export class CarRun {
  constructor(cfg, track) {
    this.cfg = cfg;
    this.track = track;
    this.sim = new RaceSim(cfg, track); // own cannon-es world: cars never collide with each other
    this.restart();
  }

  // Back to the standing start with the clock at zero.
  restart() {
    this.sim.reset();
    this.phase = 0; // tick within the current agent step (0 = a new action is due)
    this.action = [1, 1]; // held (steer, pedal); (none, coast) before the first latch
    this.splits = []; // rules.ticks when each checkpoint was passed (last entry = finish)
    this.finishTicks = null;
    this.respawns = 0;
    this.events = []; // { type: 'checkpoint' | 'finish' | 'respawn', ticks, ... } for the HUD
  }

  get finished() { return this.finishTicks !== null; }
  get ticks() { return this.sim.rules.ticks; } // race clock of this car, in ticks

  // One physics tick with the held action, then splits, finish and respawn bookkeeping.
  tickOnce() {
    const rules = this.sim.rules;
    const before = rules.nextCheckpoint;
    this.sim.tick(this.action[0], this.action[1]);
    this.phase = (this.phase + 1) % this.cfg.ticks_per_action;
    for (let k = before; k < rules.nextCheckpoint; k++) { // several checkpoints can be passed in one tick
      this.splits[k] = rules.ticks;
      this.events.push({ type: 'checkpoint', k, ticks: rules.ticks });
    }
    if (rules.finished) {
      this.finishTicks = rules.ticks;
      this.events.push({ type: 'finish', ticks: rules.ticks });
    } else if (rules.out) {
      const reason = rules.outReason;
      const moved = this.respawn();
      this.events.push({ type: 'respawn', reason, moved, ticks: rules.ticks });
    }
  }

  // Put the car back at the last passed checkpoint (or the start) without stopping the clock.
  // Returns how far (m) the car was moved, so a car waiting at its spawn point is not announced.
  respawn() {
    const before = this.sim.car.chassis.position.clone();
    const rules = this.sim.rules;
    const ticks = rules.ticks, next = rules.nextCheckpoint; // survive sim.reset()
    if (next > 0) {
      const i = this.track.checkpoints[next - 1];
      this.sim.reset(spawnState(this.track, this.cfg, i), 0, i);
    } else {
      this.sim.reset(); // nothing passed yet: the standing start
    }
    rules.ticks = ticks; // the clock keeps running
    rules.nextCheckpoint = next; // passed checkpoints stay passed
    this.respawns += 1;
    return before.distanceTo(this.sim.car.chassis.position);
  }

  // Everything the renderer needs for this tick (positions are copied, not referenced).
  snapshot() {
    const b = this.sim.car.chassis, w = this.sim.car.vehicle.wheelInfos;
    return {
      ticks: this.sim.rules.ticks,
      p: [b.position.x, b.position.y, b.position.z],
      q: [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w],
      speed: b.velocity.length(), // m/s
      susp: w.map((x) => x.suspensionLength),
      spin: w.map((x) => x.rotation),
      steer: this.sim.steerAngle,
      cp: this.sim.rules.nextCheckpoint,
    };
  }
}
