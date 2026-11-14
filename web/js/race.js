// race.js — one race session: the race clock, the player's car, the optional AI car and the ghosts.
//
// Global context: main.js creates a RaceSession when a track and a mode are chosen and calls tick()
// from the fixed-timestep loop (loop.js). Modes: 'race' (human), 'race-ai' (human plus the live AI,
// no collisions: each car has its own physics world) and 'watch' (AI only). Nothing here touches the
// DOM; the HUD reads the session's state and drains its events.

import { CarRun } from './carrun.js';
import { AiCar } from './aicar.js';
import { ghostState } from './ghost.js';

// The human's car: keyboard latched on the agent's 20 Hz grid, one ghost entry per agent step.
export class PlayerCar extends CarRun {
  restart() {
    super.restart();
    this.ghost = [ghostState(this.sim)]; // entry 0 = the start
    this.curr = this.snapshot(); // state after the latest tick
    this.prev = this.curr; // state one tick earlier, for interpolated rendering
  }

  tick(keyboard) {
    if (this.finished) return;
    if (this.phase === 0) this.action = keyboard.latch(); // new action every ticks_per_action ticks, held in between
    const respawnsBefore = this.respawns;
    this.tickOnce();
    if (this.phase === 0) this.ghost.push(ghostState(this.sim)); // one entry per completed agent step
    this.prev = this.curr;
    this.curr = this.snapshot();
    if (this.respawns !== respawnsBefore) this.prev = this.curr; // do not interpolate across a teleport
  }

  // Manual respawn (Backspace).
  respawnNow() {
    if (this.finished) return;
    const moved = this.respawn();
    this.events.push({ type: 'respawn', reason: 'manual', moved, ticks: this.ticks });
    this.curr = this.prev = this.snapshot();
  }
}

export class RaceSession {
  // policy: a loaded Policy for the AI modes (null otherwise); ghosts: [{ label, kind, states, time }].
  constructor({ cfg, track, mode, policy = null, replay = null, ghosts = [], best = null }) {
    this.cfg = cfg;
    this.track = track;
    this.mode = mode;
    this.player = mode === 'watch' || mode === 'replay' ? null : new PlayerCar(cfg, track);
    // The second car: a live AI driver, or in "replay" mode a recorded run (replay.js) with the same interface.
    this.ai = replay ?? (policy && mode !== 'race' ? new AiCar(cfg, track, policy) : null);
    this.focus = this.player ?? this.ai; // the car the camera and HUD follow
    this.ghosts = ghosts;
    this.best = best; // stored best run of this track, for split deltas
    this.restart();
  }

  restart() {
    this.clock = 0; // race clock in ticks, shared by every car and ghost
    this.player?.restart();
    this.ai?.restart();
    this.ai?.advance(0); // start the first inference right away
    this.aiEventsShown = 0; // AI events are released when the race clock reaches them
  }

  // The race continues while some car is still driving at the race clock (the AI may have finished
  // ahead of the clock, which must still reach its finish tick).
  get running() {
    const aiDriving = this.ai && !this.ai.failed && !this.ai.stopped && !(this.ai.finished && this.ai.finishTicks <= this.clock);
    return Boolean((this.player && !this.player.finished) || aiDriving);
  }

  // One 0.01 s tick of the race clock.
  tick(keyboard) {
    if (!this.running) return;
    this.player?.tick(keyboard);
    this.clock += 1;
    this.ai?.advance(this.clock);
  }

  // Events of the focus car up to the race clock (the AI simulates ahead, so its events wait).
  drainEvents() {
    const out = [];
    if (this.player) { out.push(...this.player.events.map((e) => ({ ...e, car: 'player' }))); this.player.events.length = 0; }
    if (this.ai) {
      const ev = this.ai.events;
      while (ev.length && ev[0].ticks <= this.clock) out.push({ ...ev.shift(), car: 'ai' });
    }
    return out;
  }

  // Race time (s) of a car as seen at the race clock: frozen at its finish.
  timeOf(car) {
    const t = car.finished && car.finishTicks <= this.clock ? car.finishTicks : this.clock;
    return t * this.cfg.dt;
  }

  // Two states to interpolate for rendering a car at race-clock tick clock - 1 + alpha.
  framesOf(car) {
    if (car === this.player) return [car.prev, car.curr];
    return [car.stateAt(this.clock - 1), car.stateAt(this.clock)];
  }
}
