# Third-party notices

RacingMania's own code is released under the MIT licence (see `LICENSE`). It includes or derives from the following third-party work. Each entry says what is used, where it lives in this repository, and under which licence.

## Vendored unchanged (in `web/vendor/`)

| Library | Version | Licence | Files | Source |
|---|---|---|---|---|
| cannon-es | 0.20.0 | MIT, Copyright (c) 2015 cannon.js Authors | `web/vendor/cannon-es/cannon-es.js`, `LICENSE` | https://github.com/pmndrs/cannon-es |
| three.js | 0.187.0 | MIT, Copyright © 2010-2026 three.js authors | `web/vendor/three/three.module.js`, `three.core.js`, `LICENSE` | https://github.com/mrdoob/three.js |
| ONNX Runtime Web | 1.31.1 | MIT, Copyright (c) Microsoft Corporation | `web/vendor/onnxruntime-web/ort.wasm.min.mjs`, `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm` | https://github.com/microsoft/onnxruntime |

The files are byte-for-byte copies of the npm packages pinned in `package.json`.

## Ported

| Work | Licence | What was ported | Where |
|---|---|---|---|
| cannon-es `RaycastVehicle`, `WheelInfo`, `Body.integrate`, `Body.applyImpulse`, `World.internalStep` order, `Quaternion.integrate`, `Ray._intersectTrimesh`, `Ray.pointInTriangle`, `Trimesh.computeNormal` | MIT, Copyright (c) 2015 cannon.js Authors | The per-tick car physics, rewritten as batched PyTorch tensor code | `src/racingmania/sim/` |

cannon-es's vehicle is itself a port of Bullet's `btRaycastVehicle` (zlib licence, Copyright (c) 2005 Erwin Coumans, https://github.com/bulletphysics/bullet3).

## Adapted

| Work | Licence | What was adapted | Where |
|---|---|---|---|
| CleanRL `ppo_continuous_action_isaacgym.py` | MIT, Copyright (c) 2019 CleanRL developers; the file also carries an NVIDIA BSD-3-Clause notice | The PPO training loop for GPU-vectorised environments | `src/racingmania/rl/ppo.py` |
| CleanRL `pqn.py` | MIT, Copyright (c) 2019 CleanRL developers | The PQN baseline | `src/racingmania/rl/pqn.py` |
| pmndrs/racing-game | MIT | Starting values for the vehicle parameters | `web/assets/vehicle.json` |

Entries are added here in the same commit that adds the code they refer to.
