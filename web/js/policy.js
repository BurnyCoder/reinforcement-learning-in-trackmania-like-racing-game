// policy.js — the AI driver: runs a trained policy (one ONNX graph) with onnxruntime-web.
//
// Global context: the exported graph contains the observation function, the frozen normaliser and
// the actor (docs/formats.md, "Policy model"), so the browser only gathers raw inputs, runs the graph
// once per agent step and takes the argmax of each head. This module is imported lazily by main.js,
// only when an AI mode starts, so the WASM runtime is never downloaded otherwise.
// ONNX Runtime Web API: https://onnxruntime.ai/docs/api/js/ (InferenceSession, Tensor, env.wasm).

import * as ort from '../vendor/onnxruntime-web/ort.wasm.min.mjs';

// The runtime fetches its .mjs/.wasm helpers from this folder (resolved relative to this module).
ort.env.wasm.wasmPaths = new URL('../vendor/onnxruntime-web/', import.meta.url).href;
ort.env.wasm.numThreads = 1; // GitHub Pages cannot send the cross-origin-isolation headers threads need

const argmax = (a) => a.reduce((best, x, i) => (x > a[best] ? i : best), 0);

export class Policy {
  // model: an entry of index.json "models": { name, file, K, stride, description }.
  static async load(model, url) {
    const session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
    const policy = new Policy(session, model);
    await policy.warmup();
    return policy;
  }

  constructor(session, model) {
    this.session = session;
    this.name = model.name;
    this.K = model.K; // number of look-ahead samples
    this.stride = model.stride ?? 3; // samples between look-ahead points (3 x 2 m = 6 m)
    this.queue = Promise.resolve(); // runs are chained so at most one is inside the session at a time
  }

  // One throw-away run on a neutral input, so the first real step does not pay the one-time setup cost.
  async warmup() {
    const zeros = (dims) => new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
    const chassis = zeros([1, 13]);
    chassis.data[6] = 1; // identity quaternion (qw = 1)
    const K = this.K;
    await this.session.run({ chassis, steer: zeros([1, 1]), contacts: zeros([1, 4]), prev_action: zeros([1, 2]),
      win_center: zeros([1, K, 3]), win_left: zeros([1, K, 3]), win_gap: zeros([1, K]) });
  }

  // Raw inputs for the current simulator state, exactly as listed in docs/formats.md.
  feeds(sim, prevAction) {
    const track = sim.track, M = track.center.length, K = this.K;
    const f32 = (data, dims) => new ort.Tensor('float32', Float32Array.from(data), dims);
    const winCenter = new Float32Array(K * 3), winLeft = new Float32Array(K * 3), winGap = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      const j = Math.min(sim.rules.index + this.stride * k, M - 1); // rules' nearest sample + k strides
      winCenter.set(track.center[j], 3 * k);
      winLeft.set(track.left[j], 3 * k);
      winGap[k] = j < M - 1 ? track.gap[j] : 0; // gap of the segment starting at j (none after the last sample)
    }
    return {
      chassis: f32(sim.state(), [1, 13]), // [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
      steer: f32([sim.steerAngle], [1, 1]), // current front-wheel angle (rad)
      contacts: f32(sim.car.vehicle.wheelInfos.map((w) => (w.isInContact ? 1 : 0)), [1, 4]),
      prev_action: f32(prevAction, [1, 2]), // previous (steer, pedal) indices
      win_center: new ort.Tensor('float32', winCenter, [1, K, 3]),
      win_left: new ort.Tensor('float32', winLeft, [1, K, 3]),
      win_gap: new ort.Tensor('float32', winGap, [1, K]),
    };
  }

  // Inputs are read now, synchronously, so later ticks cannot change them; the run itself waits for
  // any earlier run to finish.
  act(sim, prevAction) {
    const feeds = this.feeds(sim, prevAction);
    const run = this.queue.then(() => this.session.run(feeds));
    this.queue = run.catch(() => {}); // a failed run must not block the next one
    return run.then((out) => [argmax(out.steer_logits.data), argmax(out.pedal_logits.data)]);
  }
}
