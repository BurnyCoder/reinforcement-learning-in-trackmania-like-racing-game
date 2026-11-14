// loop.js — fixed-timestep game loop: physics ticks at a constant 0.01 s, rendering once per frame.
//
// Global context: the physics core (web/js/core) must advance in exact 0.01 s ticks so a run in the
// browser is the same deterministic trajectory the training simulator produces. Real time is
// accumulated and spent in whole ticks; the leftover fraction is passed to the renderer, which
// interpolates between the previous and current tick (the "fix your timestep" pattern,
// https://gafferongames.com/post/fix_your_timestep/).

export class FixedLoop {
  // dt: tick length (s); maxTicks: catch-up cap per frame; tick(): one simulation tick;
  // frame(alpha, elapsed): draw, alpha in [0, 1) = fraction of the next tick already elapsed.
  constructor({ dt, maxTicks = 25, tick, frame }) {
    this.dt = dt;
    this.maxTicks = maxTicks;
    this.tick = tick;
    this.frame = frame;
    this.running = false; // false: no ticks (menu), frames are still drawn
    this.acc = 0; // real time not yet spent on ticks (s)
    this.last = null; // timestamp of the previous frame (ms), null after a pause
    this.onFrame = this.onFrame.bind(this);
    // A hidden tab must not bank time: forget the last timestamp so nothing is caught up on return.
    document.addEventListener('visibilitychange', () => { this.last = null; this.acc = 0; });
  }

  start() { requestAnimationFrame(this.onFrame); }

  // Drop banked time, e.g. when a run restarts, so the first tick happens one dt after now.
  resetClock() { this.acc = 0; this.last = null; }

  onFrame(now) {
    requestAnimationFrame(this.onFrame);
    if (document.hidden) { this.last = null; return; } // paused while hidden
    const elapsed = this.last === null ? 0 : (now - this.last) / 1000; // seconds since the previous frame
    this.last = now;
    if (this.running) {
      this.acc += elapsed;
      let n = 0;
      while (this.acc >= this.dt && n < this.maxTicks) { this.tick(); this.acc -= this.dt; n++; }
      if (n === this.maxTicks && this.acc >= this.dt) this.acc = 0; // too slow: drop the backlog instead of spiralling
    } else {
      this.acc = 0;
    }
    this.frame(Math.min(this.acc / this.dt, 1), elapsed);
  }
}
