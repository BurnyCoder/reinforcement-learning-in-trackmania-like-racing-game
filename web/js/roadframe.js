// roadframe.js — the local frame of the road at a centreline sample (forward, up, right, rotation).
//
// Global context: used to respawn the car on the road (carrun.js) and to stand the checkpoint gates
// on it (trackmesh.js). Track arrays are defined in docs/formats.md: center[i] is the centreline,
// left[i] the unit vector to the left edge (banking included). Up = forward x left.

import * as THREE from '../vendor/three/three.module.js';

export function roadFrame(track, i) {
  const M = track.center.length;
  const k = Math.min(i, M - 2); // the segment starting at i (the one before it at the last sample)
  const f = new THREE.Vector3(...track.center[k + 1]).sub(new THREE.Vector3(...track.center[k])).normalize(); // forward
  const u = new THREE.Vector3().crossVectors(f, new THREE.Vector3(...track.left[i])).normalize(); // road normal
  const r = new THREE.Vector3().crossVectors(f, u); // right; (f, u, r) is right-handed like (+X, +Y, +Z)
  const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(f, u, r)); // maps chassis axes onto road axes
  return { p: new THREE.Vector3(...track.center[i]), f, u, r, q };
}
