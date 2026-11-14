// parity_harness.mjs — runs the browser game's own physics (cannon-es + core modules) headless and
// writes reference transitions for the PyTorch port's parity tests.
//
// Global context: tests/parity/test_physics_parity.py writes a request file, runs
//   node tests/js/parity_harness.mjs <request.json> <response.json>
// and compares the port's results with this file's. The request names a track and lists
//   cases:     single ticks from a given chassis state, steering angle, nearest index and action
//   sequences: many ticks from a start state with one action per agent step
// Nothing here re-implements physics: it only calls RaceSim from web/js/core/sim.js.

import { readFileSync, writeFileSync } from 'node:fs';
import { RaceSim } from '../../web/js/core/sim.js';
import { decodeAction } from '../../web/js/core/controls.js';
import { wheelsInContact } from '../../web/js/core/physics.js';
import { loadVehicle, loadJson } from './load.mjs';

const [, , requestPath, responsePath] = process.argv;
const req = loadJson(requestPath);
const cfg = req.vehicle ?? loadVehicle(); // the test may pass a modified vehicle definition
const sim = new RaceSim(cfg, req.track); // the track dictionary is embedded in the request

// Snapshot of everything the port must reproduce after a tick.
const snapshot = () => ({
  state: sim.state(),
  steer: sim.steerAngle,
  contacts: wheelsInContact(sim.car),
  index: sim.rules.index,
  s: sim.rules.s,
  finished: sim.rules.finished,
  out: sim.rules.out,
  reason: sim.rules.outReason,
  inv_inertia: [sim.car.chassis.invInertia.x, sim.car.chassis.invInertia.y, sim.car.chassis.invInertia.z],
});

const cases = (req.cases ?? []).map((c) => {
  sim.reset(c.state, c.steer, c.index);
  const [steer, pedal] = decodeAction(c.action);
  sim.tick(steer, pedal);
  return snapshot();
});

const sequences = (req.sequences ?? []).map((q) => {
  sim.reset(q.state, q.steer, q.index);
  const ticks = [];
  for (const a of q.actions) {
    const [steer, pedal] = decodeAction(a);
    for (let i = 0; i < cfg.ticks_per_action && !sim.done; i++) {
      sim.tick(steer, pedal);
      ticks.push(snapshot());
    }
    if (sim.done) break;
  }
  return ticks;
});

writeFileSync(responsePath, JSON.stringify({ cases, sequences }));
