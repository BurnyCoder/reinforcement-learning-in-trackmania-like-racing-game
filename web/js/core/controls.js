// controls.js — turns one keyboard-style action into steering angle, engine force and brake impulse.
//
// Global context: the same mapping is used by human players (keyboard), by the AI in the browser,
// and, line for line, by the PyTorch port (src/racingmania/sim/controls.py). It is part of the
// physics contract in docs/formats.md ("How an action becomes wheel inputs"), because cannon-es
// itself has no engine model, no rolling resistance and no top speed (docs/physics.md, quirks).

// Action indices: steer 0=left, 1=none, 2=right; pedal 0=gas, 1=coast, 2=brake (docs/formats.md).
export const STEER_LEFT = 0, STEER_NONE = 1, STEER_RIGHT = 2;
export const PEDAL_GAS = 0, PEDAL_COAST = 1, PEDAL_BRAKE = 2;

// Combine the two choices into one integer 0..8 and split it back (used for stored action sequences).
export const encodeAction = (steer, pedal) => steer * 3 + pedal;
export const decodeAction = (a) => [Math.floor(a / 3), a % 3];

// Compute the wheel inputs for one tick.
// steerAngle: current front-wheel angle (rad), part of the car state because it is rate-limited.
// speed: chassis velocity projected on the chassis forward axis (physics.forwardSpeed), before the tick.
// Returns { steerAngle, engine, brake }: the new steering angle, the engine force per drive wheel
// and the brake impulse per wheel. Every arithmetic step is written in the same order as the port.
export function wheelInputs(ctl, dt, steerAngle, steer, pedal, speed) {
  const absSpeed = Math.abs(speed);
  // Steering authority shrinks with speed so that full lock at top speed does not spin the car.
  const steerScale = Math.max(ctl.min_steer_fraction, 1 - ctl.steer_speed_falloff * absSpeed / ctl.top_speed);
  const dir = steer === STEER_LEFT ? 1 : steer === STEER_RIGHT ? -1 : 0; // positive angle turns left (+X toward -Z)
  const target = dir * ctl.max_steer * steerScale; // desired front-wheel angle
  const maxDelta = ctl.steer_rate * dt; // rate limit per tick: a keyboard tap cannot snap the wheels
  const delta = Math.min(Math.max(target - steerAngle, -maxDelta), maxDelta); // clamp the change
  const newSteer = steerAngle + delta;

  let engine = 0; // engine force per drive wheel (N); positive drives toward chassis +X
  let brake = 0; // brake impulse per wheel per tick (N s); cannon-es uses it as an impulse cap
  if (pedal === PEDAL_GAS) {
    engine = ctl.engine_force * Math.max(0, 1 - speed / ctl.top_speed); // force fades to zero at top speed
  } else if (pedal === PEDAL_BRAKE) {
    if (speed > ctl.reverse_speed_threshold) {
      brake = ctl.brake_impulse; // moving forward: brake
    } else {
      engine = -ctl.reverse_force * Math.max(0, 1 + speed / ctl.reverse_top_speed); // stopped: reverse gear
    }
  } else {
    brake = ctl.coast_brake_impulse; // coasting: small rolling resistance
  }
  return { steerAngle: newSteer, engine, brake };
}

// Write the inputs into the cannon-es vehicle (steering on front wheels, engine on drive wheels,
// brake on all wheels). Must be called before world.step(dt).
export function applyWheelInputs(car, cfg, inputs) {
  const v = car.vehicle;
  for (let i = 0; i < v.wheelInfos.length; i++) {
    v.setSteeringValue(cfg.front_wheels.includes(i) ? inputs.steerAngle : 0, i);
    v.applyEngineForce(cfg.drive_wheels.includes(i) ? inputs.engine : 0, i);
    v.setBrake(inputs.brake, i);
  }
}
