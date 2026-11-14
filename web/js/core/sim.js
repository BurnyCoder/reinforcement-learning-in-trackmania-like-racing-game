// sim.js — one car on one track: physics, controls and rules advanced together, one tick at a time.
//
// Global context: the browser game drives a RaceSim from the keyboard or from the ONNX policy, and
// the Node.js parity harness drives the same class to produce reference transitions for the
// PyTorch port. Nothing here touches the DOM.

import { createWorld, createTrackBody, createCar, setChassisState, getChassisState, forwardSpeed, wheelsInContact } from './physics.js';
import { wheelInputs, applyWheelInputs, decodeAction } from './controls.js';
import { RaceRules } from './rules.js';

export class RaceSim {
  constructor(cfg, track) {
    this.cfg = cfg; // parsed vehicle.json
    this.track = track; // parsed track file
    this.world = createWorld(cfg);
    this.trackBody = createTrackBody(track);
    this.world.addBody(this.trackBody);
    this.car = createCar(this.world, cfg);
    this.rules = new RaceRules(track);
    this.reset();
  }

  // Put the car at a state (default: the track's start, at rest) and restart the rules.
  reset(state = null, steerAngle = 0, index = null) {
    const st = state ?? [...this.track.start.position, ...this.track.start.quaternion, 0, 0, 0, 0, 0, 0];
    setChassisState(this.car, st);
    this.steerAngle = steerAngle; // front-wheel angle, rate-limited state of the controls
    this.rules.reset(st.slice(0, 3), index ?? this.track.start.index);
  }

  // Advance one physics tick with a (steer, pedal) action. Returns the progress gained (m).
  tick(steer, pedal) {
    const speed = forwardSpeed(this.car); // measured before the tick, as in the port
    const inputs = wheelInputs(this.cfg.controls, this.cfg.dt, this.steerAngle, steer, pedal, speed);
    this.steerAngle = inputs.steerAngle;
    applyWheelInputs(this.car, this.cfg, inputs);
    this.world.step(this.cfg.dt); // fixed step: exactly one World.internalStep(dt)
    const p = this.car.chassis.position;
    return this.rules.update([p.x, p.y, p.z], wheelsInContact(this.car));
  }

  // Advance one agent step (ticks_per_action ticks with the same action) and stop early if the run
  // ends. Returns the progress gained over the step.
  step(action) {
    const [steer, pedal] = decodeAction(action);
    let gained = 0;
    for (let i = 0; i < this.cfg.ticks_per_action; i++) {
      gained += this.tick(steer, pedal);
      if (this.done) break;
    }
    return gained;
  }

  get done() { return this.rules.finished || this.rules.out; }
  get time() { return this.rules.ticks * this.cfg.dt; } // race clock (s)
  state() { return getChassisState(this.car); }
}
