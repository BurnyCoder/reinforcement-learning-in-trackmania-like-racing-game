// aicar.js — a car driven live by the ONNX policy, scheduled so inference never blocks the page.
//
// Global context: the policy was trained (and is evaluated in Python) with no action delay: the action
// for agent step k is computed from the state at the start of step k and held for ticks_per_action
// ticks. Inference in the browser is asynchronous, so this car's simulation runs up to LEAD ticks
// ahead of the race clock and waits at each step boundary until its action is ready; as soon as an
// inference finishes, the car continues on its own (from the promise callback), so it is not limited
// to one step per frame. The renderer shows it at the race clock from a short history of tick states.
// The closed loop is therefore the same as in training. Only if an inference has been pending for
// LATE_MS while the car is more than MAX_LAG ticks behind the race clock does the car keep its previous
// action for a step (counted in `late`), so a slow machine degrades gracefully instead of stalling.

import { CarRun } from './carrun.js';

const LEAD = 10; // ticks the AI may simulate ahead of the race clock (2 agent steps)
const MAX_LAG = 10; // ticks behind the race clock ...
const LATE_MS = 100; // ... and real milliseconds of waiting after which a late action is skipped
const HISTORY = 64; // tick states kept for rendering

export class AiCar extends CarRun {
  constructor(cfg, track, policy) {
    super(cfg, track);
    this.policy = policy;
    this.late = 0; // steps that reused the previous action because inference was late
    this.failed = false; // inference threw: the car stops being driven
  }

  restart() {
    super.restart();
    this.prevAction = [1, 1]; // docs/formats.md: (1, 1) at the start
    this.step = 0; // agent steps decided so far in this run
    this.req = null; // { step, done, action, error, started }: the inference in flight or just finished
    this.clock = 0; // latest race clock seen
    this.history = [this.snapshot()]; // history[i] is the state at tick historyBase + i
    this.historyBase = 0;
  }

  get done() { return this.finished || this.failed; }

  // Simulate up to LEAD ticks past the race clock, stopping at a step boundary whose action is not ready.
  advance(clock) {
    this.clock = Math.max(this.clock, clock);
    while (!this.done && this.ticks < this.clock + LEAD) {
      if (this.phase === 0 && !this.decide()) break;
      const respawnsBefore = this.respawns;
      this.tickOnce();
      if (this.respawns !== respawnsBefore) { this.prevAction = [1, 1]; this.phase = 0; } // a fresh start after a respawn
      this.history.push(this.snapshot());
      if (this.history.length > HISTORY) { this.history.shift(); this.historyBase += 1; }
    }
  }

  // Set this.action for the step starting now; false means "wait for the inference".
  decide() {
    const r = this.req;
    if (r && r.step === this.step && r.done) {
      if (r.error) { this.failed = true; return false; }
      this.use(r.action);
      return true;
    }
    if (!r || r.done) this.req = this.request(this.step); // one inference at a time, from the current state
    const waited = performance.now() - this.req.started;
    if (this.ticks < this.clock - MAX_LAG && waited > LATE_MS) { // inference too slow: keep the previous action
      this.late += 1;
      if (this.late === 1 || this.late % 100 === 0) console.warn(`AI inference late ${this.late} time(s); previous action reused`);
      this.use(this.action);
      return true;
    }
    return false;
  }

  use(action) { this.action = action; this.prevAction = action; this.step += 1; }

  request(step) {
    const req = { step, done: false, action: null, error: null, started: performance.now() };
    const restartedAt = this.history; // a restart replaces the history array: then this result is stale
    this.policy.act(this.sim, this.prevAction).then(
      (a) => { req.action = a; req.done = true; if (this.history === restartedAt) this.advance(this.clock); },
      (e) => { req.error = e; req.done = true; console.error('AI inference failed:', e); },
    );
    return req;
  }

  // State at a race-clock tick for rendering (clamped to what has been simulated).
  stateAt(tick) {
    const i = Math.min(Math.max(Math.round(tick) - this.historyBase, 0), this.history.length - 1);
    return this.history[i];
  }
}
