// input.js — keyboard state for driving, turned into a (steer, pedal) action on the 20 Hz grid.
//
// Global context: humans and the AI act on the same grid. The race code calls latch() once every
// ticks_per_action ticks (docs/formats.md) and holds the result, exactly like the agent's action.
// Menu and race commands (Enter, R, Backspace, Esc) are handled in main.js.

import { STEER_LEFT, STEER_NONE, STEER_RIGHT, PEDAL_GAS, PEDAL_COAST, PEDAL_BRAKE } from './core/controls.js';

// Physical key codes (KeyboardEvent.code), so WASD works on any keyboard layout.
const BINDINGS = {
  ArrowUp: 'gas', KeyW: 'gas',
  ArrowDown: 'brake', KeyS: 'brake',
  ArrowLeft: 'left', KeyA: 'left',
  ArrowRight: 'right', KeyD: 'right',
};

export class Keyboard {
  constructor(target = window) {
    this.down = new Set(); // controls whose key is held now
    this.tapped = new Set(); // controls pressed since the last latch (a tap shorter than 50 ms still counts once)
    this.active = false; // true while racing: then arrow keys do not scroll the page
    target.addEventListener('keydown', (e) => {
      const c = BINDINGS[e.code];
      if (!c) return;
      if (this.active) e.preventDefault(); // keep arrows from scrolling or moving focus during a race
      this.down.add(c);
      this.tapped.add(c);
    });
    target.addEventListener('keyup', (e) => { const c = BINDINGS[e.code]; if (c) this.down.delete(c); });
    window.addEventListener('blur', () => this.clear()); // keys released while the window was unfocused never send keyup
  }

  clear() { this.down.clear(); this.tapped.clear(); }

  // Forget taps but keep keys that are still held. Used when a run starts or restarts: a key held through
  // R must keep acting, since the browser auto-repeats only the newest key and no new keydown would come.
  forgetTaps() { this.tapped.clear(); }

  // The action for the next agent step. Opposite keys together, or none, mean "none"/"coast".
  latch() {
    const on = (c) => this.down.has(c) || this.tapped.has(c);
    const left = on('left'), right = on('right'), gas = on('gas'), brake = on('brake');
    this.tapped.clear();
    const steer = left === right ? STEER_NONE : left ? STEER_LEFT : STEER_RIGHT;
    const pedal = gas === brake ? PEDAL_COAST : gas ? PEDAL_GAS : PEDAL_BRAKE;
    return [steer, pedal];
  }
}
