# Source ledger, part C: car physics, tracks, baselines, exploits, tooling (literature review sections 8–12)

**Scope:** every factual claim in sections 8–12 of [`../literature-review.md`](../literature-review.md), checked against the primary source on 2026-11-14. The review itself was not edited.

**How each source was read.**

- *Read as raw text by the checker (no summarising model in between):* all source code (cannon-es, pmndrs/racing-game, use-cannon, Bullet, Rapier, Jolt, Godot, f1tenth_gym, Gymnasium, PyTorch v2.15.0, trajectory_planning_helpers, open-beni) fetched from `raw.githubusercontent.com` and read with line numbers; licences and repository status from the GitHub REST API (`spdx_id`, `pushed_at`, commit dates); versions and dates from the npm and PyPI registry JSON, the PyTorch wheel index and the GitHub releases API; DOIs from the Crossref API; documentation and arXiv HTML pages fetched and reduced to text with a tag-stripping script, then searched; PDFs (Coulter 1992, Baraff 1997, Catto 2015) converted with `pdftotext`.
- *cannon-es files read in full:* `src/objects/RaycastVehicle.ts` (673 lines), `src/objects/WheelInfo.ts` (429). *Read in the relevant parts:* `src/world/World.ts` (constructor, `rayTest`, `raycastClosest`, `step`, `internalStep`, `clearForces`), `src/objects/Body.ts` (constructor defaults, `updateInertiaWorld`, `applyImpulse`, `updateMassProperties`, `updateAABB`, `getVelocityAtWorldPoint`, `integrate`), `src/math/Quaternion.ts` (`setFromAxisAngle`, `mult`, `normalize`, `vmult`, `integrate`), `src/collision/Ray.ts` (everything except `_intersectSphere`, `_intersectConvex` and `_intersectHeightfield`), `src/collision/RaycastResult.ts`, `src/world/Narrowphase.ts` (collision-type table, getters, resolver lookup), `src/solver/GSSolver.ts` (`solve` head), `src/shapes/Trimesh.ts` (normals), `src/shapes/Box.ts` (`calculateInertia`), `src/math/Mat3.ts`, `src/math/Vec3.ts`, `src/math/Transform.ts`, `src/utils/Octree.ts` (`rayQuery`), `src/collision/NaiveBroadphase.ts` (`aabbQuery`), `src/utils/Utils.ts`. Line numbers refer to `master` on 2026-11-14 (last commit 2024-01-06; `package.json` version 0.20.0).
- *Abstract or metadata only (full text paywalled or not opened):* Kong et al. 2015, Perantoni & Limebeer 2015, Williams et al. 2016, Thrun et al. 2006, Pinneri et al. 2020.
- *Could not be opened:* the OpenAI hide-and-seek blog (HTTP 403); the Coulter PDF through WebFetch (certificate rejected; fetched directly instead). The Rapier issue with the step timing could not be located (the review gives no URL).
- *Secondary evidence only (web search results, marked `[2nd]`):* PolyTrack's source availability; GitHub Pages and cross-origin isolation headers.

**Tally:** 108 claims checked; 86 confirmed, 16 corrected (grouped into 13 numbered corrections below), 6 not found in source, 0 contradicted, 0 source unreachable. The six "not found" rows are: the kinematic-bicycle statement (paper not opened), the Rapier step timing (no issue found), PolyTrack's source status (secondary evidence only), two negative search claims, the unsourced remark about RL agents and TrackMania bugs, and the "publisher's terms" remark about the Hugging Face dataset.

## Ledger

| Review section | Claim (as written) | Source URL | Where in the source (file and function/line, section, table) | Supporting quote | Verdict |
|---|---|---|---|---|---|
| 8 Models | "The kinematic bicycle (Kong et al. 2015, DOI) has no grip limit." | [Crossref record for 10.1109/IVS.2015.7225830](https://api.crossref.org/works/10.1109/IVS.2015.7225830) | Crossref metadata (title, authors Kong, Pfeiffer, Schildbach, Borrelli; IV 2015) | "Kinematic and dynamic vehicle models for autonomous driving control design" | not found in source — the citation resolves, but the paper text is paywalled and was not opened; "no grip limit" follows from a kinematic model having no tyre forces and is not quoted |
| 8 Models | "The CommonRoad single-track model used by f1tenth_gym is linear in slip angle, so grip cannot be exceeded either" | [f1tenth_gym `dynamic_models.py`](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/dynamic_models.py) | module docstring L18–19; `vehicle_dynamics_st` L124–174 (terms linear in `x[2]`, `x[5]`, `x[6]` with constant `C_Sf`, `C_Sr`) | "Following the implementation of commanroad's Single Track Dynamics model" | confirmed (the "grip cannot be exceeded" part is the reviewer's inference from the linear tyre terms) |
| 8 Models | "Monster's "Car Physics for Games" clamps lateral force and is singular at zero speed" | [mirror](https://asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html) | section "High Speed Turning"; section on slip ratio (the only "Cap the force" sentence, about longitudinal force) | "For very small angles (below the peak) the lateral force can be approximated by a linear function: F lateral = C a * alpha" | corrected — see C6. The page caps the longitudinal traction force, not the lateral one; zero-speed singularity is implied by "take the arctangent of the lateral velocity divided by the longtitudinal velocity" but not stated |
| 8 Raycast vehicle | "Bullet `btRaycastVehicle`, zlib" | [bullet3 `LICENSE.txt`](https://raw.githubusercontent.com/bulletphysics/bullet3/master/LICENSE.txt) | first paragraph | "The files in this repository are licensed under the zlib license, except for the files under 'Extras' and examples/ThirdPartyLibs." | confirmed |
| 8 Raycast vehicle | "the spring-damper suspension force is applied as an impulse at the contact point" | [bullet3 `btRaycastVehicle.cpp`](https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/Vehicle/btRaycastVehicle.cpp) | `updateVehicle` L300–307; `updateSuspension` L425 | `getRigidBody()->applyImpulse(impulse, relpos);` | confirmed |
| 8 Raycast vehicle | "a side impulse cancels lateral velocity" | [bullet3 `btContactConstraint.cpp`](https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/ConstraintSolver/btContactConstraint.cpp) | `resolveSingleBilateral` L144–151 (called from `btRaycastVehicle.cpp` L538) | `btScalar contactDamping = btScalar(0.2);` | corrected — see C7. The impulse is 0.2 × the impulse that would cancel the relative lateral velocity |
| 8 Raycast vehicle | "an engine or brake impulse acts forward" | [bullet3 `btRaycastVehicle.cpp`](https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/Vehicle/btRaycastVehicle.cpp) | `updateFriction` L561–572 | `rollingFriction = wheelInfo.m_engineForce * timeStep;` | confirmed (in Bullet it is one or the other: the `else` branch computes the brake/rolling impulse) |
| 8 Raycast vehicle | "both are scaled down when they exceed friction-slip × suspension force" | same file | `updateFriction` L584, L608–621 | `btScalar maximp = wheelInfo.m_wheelsSuspensionForce * timeStep * wheelInfo.m_frictionSlip;` | confirmed |
| 8 Raycast vehicle | "PhysX's guide advises at least five integration steps per suspension spring period" | [PhysX vehicles guide](https://archive.docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/Vehicles.html) | tuning guide, `mSpringStrength` | "an alpha value greater than 5.0; that is, a minimum of five simulation updates per spring cycle" | confirmed |
| 8 Table, cannon-es | Licence "MIT" | [cannon-es `package.json`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/package.json) | L4; GitHub API `spdx_id` "MIT" | `"license": "MIT",` | confirmed |
| 8 Table, cannon-es | "about 390 lines plus `WheelInfo`" | [cannon-es `RaycastVehicle.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts) | whole file: 673 lines, 461 without blank and comment lines; `WheelInfo.ts` 429 lines | (line count) | corrected — see C1 |
| 8 Table, cannon-es | "only ray casts and impulses on one body" | same file | `updateFriction` L549–553; `Body.applyImpulse` L838 returns for non-dynamic bodies | `groundObject.applyImpulse(sideImp, rel_pos2)` | confirmed with a note: the side impulse's reaction is also applied to the ground body (no effect on a static ground) |
| 8 Table, cannon-es | "Pure JavaScript" | [GitHub API, pmndrs/cannon-es](https://api.github.com/repos/pmndrs/cannon-es) | repository description (sources are `.ts` files compiled to JavaScript; no native or WASM part) | "A lightweight 3D physics engine written in JavaScript." | confirmed |
| 8 Table, cannon-es | "no substeps" | [cannon-es `World.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/World.ts) | `step` L486–493 | `this.internalStep(dt)` | confirmed (with `timeSinceLastCalled` given, `step` runs up to `maxSubSteps = 10` whole fixed steps to catch up; these are full steps, not solver substeps) |
| 8 Table, cannon-es | "no gyroscopic term" | [cannon-es `Body.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/Body.ts) | `integrate` L964–971 (only `dt · I⁻¹ · torque`) | `angularVelo.x += dt * (e[0] * tx + e[1] * ty + e[2] * tz)` | confirmed |
| 8 Table, cannon-es | "frozen at 0.20.0 (npm, 2022-08-12)" | [npm registry, cannon-es](https://registry.npmjs.org/cannon-es) | `time["0.20.0"]`; `dist-tags.latest` = 0.20.0; last repository commit 2024-01-06 | `"0.20.0":"2022-08-12T16:46:00.929Z"` | confirmed |
| 8 Table, Rapier | Licence "Apache-2.0" | [GitHub API, dimforge/rapier](https://api.github.com/repos/dimforge/rapier) | `license.spdx_id` | `"spdx_id": "Apache-2.0"` | confirmed |
| 8 Table, Rapier | "about 780 lines of Rust" | [rapier `ray_cast_vehicle_controller.rs`](https://raw.githubusercontent.com/dimforge/rapier/master/src/control/ray_cast_vehicle_controller.rs) | whole file: 823 lines | (line count) | corrected — see C2 |
| 8 Table, Rapier | "a port of Bullet's" | same file | L1 (module doc) | "A vehicle controller based on ray-casting, ported and modified from Bullet’s `btRaycastVehicle`." | confirmed |
| 8 Table, Rapier | "Its world step uses substeps and a gyroscopic term" | [rapier `CHANGELOG.md`](https://raw.githubusercontent.com/dimforge/rapier/master/CHANGELOG.md) | v0.35.0-beta.0, "Modified", L293 | "Gyroscopic forces are applied per-substep in the principal inertia frame" | confirmed (changelog only; the integration code was not read) |
| 8 Table, Rapier | "it changes quickly" | same file | release headings L1–298 | "v0.36.1 (4 October 2026)" … "v0.34.0 (04 July 2026)" | confirmed (six releases in three months) |
| 8 Table, Rapier | "One `world.step()` measured about 8–11 µs in an issue, so CPU-only training would need several busy cores" | [GitHub issue search, dimforge/rapier and rapier.js](https://api.github.com/search/issues?q=repo:dimforge/rapier+microseconds+step) | four searches for "µs", "μs", "microseconds" with "step": 0 results | — | not found in source — the review gives no issue URL and none was found; the conclusion drawn from it is therefore unsupported |
| 8 Table, Jolt | "MIT"; "runs inside the constraint solver" | [Jolt `VehicleConstraint.h`](https://raw.githubusercontent.com/jrouwe/JoltPhysics/master/Jolt/Physics/Vehicle/VehicleConstraint.h) | L65; GitHub API `spdx_id` "MIT" | `class JPH_EXPORT VehicleConstraint : public Constraint, public PhysicsStepListener` | confirmed |
| 8 Table, ammo.js | "zlib" | [ammo.js `LICENSE`](https://raw.githubusercontent.com/kripken/ammo.js/main/LICENSE) | whole file (zlib licence text) | "This software is provided 'as-is', without any express or implied warranty." | confirmed |
| 8 Table, ammo.js | "same model as cannon-es" | [bullet3 `btRaycastVehicle.cpp`](https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/Vehicle/btRaycastVehicle.cpp) vs [cannon-es `RaycastVehicle.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts) | Bullet L561–572 (engine *or* brake) vs cannon-es L461–469 (brake *plus* engine); Bullet `btContactConstraint.cpp` L150 (`jacDiagABInv`) vs cannon-es L669–670 (linear mass only) | `rollingFriction += wheel.engineForce * timeStep` | corrected — see C3 |
| 8 Table, ammo.js | "Stale and heavy" | [GitHub API, kripken/ammo.js commits](https://api.github.com/repos/kripken/ammo.js/commits?per_page=6) | commit dates on `main`: 2026-09-22, 2026-09-08, 2026-06-16 (two), then 2023-11-20 | "Default to a 16MB heap that grows on demand (#449)" | corrected — see C3 ("stale" is out of date; "heavy" was not measured) |
| 8 Table, Godot | "MIT"; "Bullet-derived" | [godot `vehicle_body_3d.cpp`](https://raw.githubusercontent.com/godotengine/godot/master/scene/3d/physics/vehicle_body_3d.cpp) | L649; GitHub API `spdx_id` "MIT" | `// !BAS! We had this set to 0.4, in bullet its 0.2` | confirmed (the Bullet origin is shown only by this comment) |
| 8 Table, MuJoCo | Licence "Apache-2.0" (MJX, MuJoCo Warp) | [GitHub API, google-deepmind/mujoco](https://api.github.com/repos/google-deepmind/mujoco) and [mujoco_warp](https://api.github.com/repos/google-deepmind/mujoco_warp) | `license.spdx_id` of both | `"spdx_id": "Apache-2.0"` | confirmed |
| 8 Table, MuJoCo | "no car example" | [mujoco `model/car/car.xml`](https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/car/car.xml); [MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html); [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html) | repository folder `model/car` exists; neither docs page contains "car" or "vehicle" | `<geom type="cylinder" size=".03 .01" rgba=".5 .5 1 1"/>` | corrected — see C4 |
| 8 Table, MuJoCo | "MJX runs on GPU only under Linux or WSL2 (JAX lists WSL2 as experimental)" | [JAX installation](https://docs.jax.dev/en/latest/installation.html) | "Supported platforms" table, row NVIDIA GPU (columns Linux x86_64, Linux aarch64, Mac, Windows, Windows WSL2) | "NVIDIA GPU yes yes n/a no experimental" | corrected — see C4 (true for MJX-JAX on NVIDIA GPUs; the MJX page also lists AMD GPUs, Apple Silicon and TPUs) |
| 8 Table, MuJoCo | "per-world terrain is undocumented for MJX and costly for Warp" | [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html) | section "Per-world assets" | "MJWarp enables per-world asset functionality but does not provide utilities for construction of dependent per-world field variants." | corrected — see C4 (MJX page has no per-world asset section, which supports "undocumented"; "costly" is not in the source) |
| 8 Table, MuJoCo | "Warp's GPU runs are not deterministic" | same page | FAQ | "Is MJWarp on GPU deterministic? No." | confirmed |
| 8 Table, MuJoCo | "the browser would run the C engine instead" | [MuJoCo programming docs](https://mujoco.readthedocs.io/en/stable/programming/index.html) | "Getting started" | "live.mujoco.org , which runs the WebAssembly version of the MuJoCo Studio app" | confirmed |
| 8 Table, Brax | "Brax's physics is superseded by MJX" | [brax `README.md`](https://raw.githubusercontent.com/google/brax/main/README.md) | note at L4 | "If you want to use Brax for physics simulation, please use MJX" | confirmed |
| 8 Table, Isaac | "Isaac Sim needs 16 GB of VRAM" | [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) | system requirements table, minimum column, and footnote 2 | "GPUs with less than 16GB VRAM may be insufficient to run a complex scene rendering more than 16MP per frame." | confirmed |
| 8 Clones | "Stunt Rally is GPL-3" | [GitHub API, stuntrally/stuntrally3](https://api.github.com/repos/stuntrally/stuntrally3) | `license.spdx_id` | `"spdx_id": "GPL-3.0"` | confirmed |
| 8 Clones | "PolyTrack is closed source" | web search only | no licence or source repository published by the developer was found | — | not found in source — `[2nd]` only (shipped code is minified; a third-party deobfuscation repository exists) |
| 8 Clones | "No permissively licensed, browser-playable TrackMania clone with real car physics was found"; "No batched or GPU port of a raycast vehicle was found either" | — | negative search results; there is no source to check them against | — | not found in source — unverifiable by nature; see C5 for the wording next to SceneFactory-3D |
| 8 Clones | "SceneFactory-3D (arXiv 2610.02874)" | [arXiv 2610.02874](https://arxiv.org/abs/2610.02874) | abstract page, title | "SceneFactory-3D: Lifting 2D Traffic Scenes into 3D Physical Counterfactuals for Scalable Physically Grounded Safety Evaluation" | confirmed |
| 8 Clones | "describes the same design in an NVIDIA Warp kernel" | [arXiv 2610.02874v2, HTML](https://arxiv.org/html/2610.02874v2) | section 4.2 "Vehicle and road model" | "Each vehicle is a free rigid chassis with four raycast suspension legs and rotating wheels." | corrected — see C5 (related, not the same: forces applied as an external wrench, chassis integrated by Newton with XPBD, explicit tyre forces) |
| 8 Clones | "its code link is dead" | [github.com/SmallWorldLab/SceneFactory_3D](https://github.com/SmallWorldLab/SceneFactory_3D) (the link in the abstract) | HTTP 404 from both the web page and the GitHub API on 2026-11-14 | "Code: https://github.com/SmallWorldLab/SceneFactory_3D" | confirmed |
| 8 cannon-es facts | "add gravity to the force" (first in the step) | [cannon-es `World.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/World.ts) | `internalStep` L555–567 | `f.x += m * gx` | confirmed |
| 8 cannon-es facts | "collision detection and solver (nothing to do when the chassis has no contacts)" | same file; [`GSSolver.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/solver/GSSolver.ts) | `internalStep` L580, L620, L801; `GSSolver.solve` L53, L74 | `if (Neq !== 0) {` | confirmed |
| 8 cannon-es facts | "velocity damping `v *= pow(1 − damping, dt)` with both dampings defaulting to 0.01" | [`World.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/World.ts); [`Body.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/Body.ts) | `internalStep` L811–825; `Body` constructor L464, L512 | `const ld = pow(1.0 - bi.linearDamping, dt)` | confirmed |
| 8 cannon-es facts | "the `preStep` event, in which the vehicle updates and applies its impulses" | [`World.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/World.ts); [`RaycastVehicle.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts) | `internalStep` L827 (after damping, before integration); `addToWorld` L100–103 | `that.updateVehicle(world.dt)` | confirmed |
| 8 cannon-es facts | "then `Body.integrate` (`v += f/m·dt`, `x += v·dt`, quaternion update and normalisation)" | [`Body.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/Body.ts) | `integrate` L959–986 | `velo.x += force.x * iMdt * linearFactor.x` | confirmed (also `ω += dt·I⁻¹·τ` at L969–971, omitted in the review) |
| 8 cannon-es facts | "then the inertia update and force clearing" | [`Body.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/Body.ts); [`World.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/World.ts) | `integrate` L991 (inside `integrate`); `internalStep` L842, then `postStep` at L853 | `this.updateInertiaWorld()` | confirmed |
| 8 cannon-es facts | "`RaycastVehicle.castRay` goes through `World.rayTest`" | [`RaycastVehicle.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts) | `castRay` L273 | `this.world!.rayTest(source, target, raycastResult)` | confirmed |
| 8 cannon-es facts | "which returns the closest hit and skips back faces" | [`World.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/World.ts); [`Ray.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/collision/Ray.ts) | `rayTest` L311–314; `reportIntersection` L704, L718–724 | `this.raycastClosest(from, to, { skipBackfaces: true }, result)` | confirmed |
| 8 cannon-es facts | "A triangle mesh only collides with spheres and planes in cannon-es, so a box chassis never touches a mesh track; only the wheel rays do." | [`Narrowphase.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/world/Narrowphase.ts); [`Ray.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/collision/Ray.ts) | `COLLISION_TYPES` L50–51; getters L147–155 (`convexTrimesh` commented out); resolver lookup L383–385; `Ray` getter L171–173 | `// get [COLLISION_TYPES.convexTrimesh]() {` | confirmed |
| 8 cannon-es facts | "Suspension force is `stiffness × compression × clippedInvContactDotSuspension − damping × relative velocity`, multiplied by chassis mass and clamped to `[0, maxSuspensionForce]`." | [`RaycastVehicle.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts) | `updateSuspension` L213–232; `updateVehicle` L145–149; `updateFriction` L484 | `force = wheel.suspensionStiffness * length_diff * wheel.clippedInvContactDotSuspension` | corrected — see C8 (the upper clamp is applied only to the impulse, not to the stored force that sets the grip limit; damping coefficient depends on the sign of the velocity) |
| 8 cannon-es facts | "straight-line braking is not limited by grip (the skid clamp sits inside `if (sideImpulse !== 0)`" | same file | `updateFriction` L508–518 | `if (wheel.sideImpulse !== 0) {` | confirmed (the same holds for the engine impulse; the brake impulse is still capped at `wheel.brake`, L457 and L615–620) |
| 8 cannon-es facts | "Bullet shares it" | [bullet3 `btRaycastVehicle.cpp`](https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/Vehicle/btRaycastVehicle.cpp) | `updateFriction` L608–621 | `if (m_sideImpulse[wheel] != btScalar(0.))` | confirmed |
| 8 cannon-es facts | "Rapier fixed it in 0.35.0" | [rapier `CHANGELOG.md`](https://raw.githubusercontent.com/dimforge/rapier/master/CHANGELOG.md) | v0.35.0 (08 August 2026), "Fixed", L137–139 | "no longer has unlimited braking friction when a wheel's side impulse is exactly zero (e.g. braking in a straight line)" | confirmed |
| 8 cannon-es facts | "an open pull request (#215) reports the car launching when driving slowly over a raised body" | [GitHub API, cannon-es pull 215](https://api.github.com/repos/pmndrs/cannon-es/pulls/215) | title "RaycastVehicle improvements"; state open, not merged, created 2025-05-20; body | "raycast vehicle launching into the air when it passes slowly over something like a sidewalk body" | confirmed |
| 8 cannon-es facts | "each wheel is a single ray" | [`RaycastVehicle.ts`](https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts) | `updateVehicle` L134–136; `castRay` L273 (one `rayTest` per wheel) | `this.castRay(wheelInfos[i])` | confirmed |
| 8 Tuned values | "pmndrs/racing-game, MIT" | [racing-game `LICENSE.md`](https://raw.githubusercontent.com/pmndrs/racing-game/main/LICENSE.md) | L1; `package.json` L63; GitHub API `spdx_id` "MIT" | "# The MIT License" | confirmed |
| 8 Tuned values | "chassis mass 500" | [racing-game `Chassis.tsx`](https://raw.githubusercontent.com/pmndrs/racing-game/main/src/models/vehicle/Chassis.tsx) | L70 (component defaults), L89 (`useBox`) | `({ args = [2, 1.1, 4.7], mass = 500, children, ...props }, ref)` | confirmed |
| 8 Tuned values | "wheel radius 0.38, suspension stiffness 30, rest length 0.35, friction slip 1.5, roll influence 0" | [racing-game `store.ts`](https://raw.githubusercontent.com/pmndrs/racing-game/main/src/store.ts) | `wheelInfo` L50–61 | `suspensionRestLength: 0.35,` | confirmed (all five values read on their own lines) |
| 8 Tuned values | "engine force 1800 on the rear wheels" | [`store.ts`](https://raw.githubusercontent.com/pmndrs/racing-game/main/src/store.ts); [`Vehicle.tsx`](https://raw.githubusercontent.com/pmndrs/racing-game/main/src/models/vehicle/Vehicle.tsx) | `vehicleConfig` L27; `useFrame` L72–78 (wheels 2 and 3 are at `back`, L29) | `for (i = 2; i < 4; i++) api.applyEngineForce(speed < maxSpeed ? engineValue : 0, i)` | confirmed (driving forward uses `force * -1`, boost `-1.5`; cut to 0 at `maxSpeed: 88`) |
| 8 Tuned values | "maximum brake 65" | same files | `vehicleConfig` L28; `useFrame` L80 (rear wheels; `maxBrake / 1.5` while forward is held) | `maxBrake: 65,` | confirmed |
| 8 Tuned values | "steering 0.3" | same files | `vehicleConfig` L26; `useFrame` L77–79 (front wheels 0 and 1) | `steer: 0.3,` | confirmed |
| 8 Tuned values | (list of tuned values as a whole) | [`store.ts`](https://raw.githubusercontent.com/pmndrs/racing-game/main/src/store.ts) | `wheelInfo` L57 | `sideAcceleration: 3,` | corrected — see C9 (a dynamics-relevant non-default value is missing from the list) |
| 8 Tuned values | "nadevrix/micro-volt-rc" is MIT and uses "the same vehicle with ramps, jumps and respawn" | [micro-volt-rc `README.md`](https://raw.githubusercontent.com/nadevrix/micro-volt-rc/main/README.md) | L12 ("rampa"), L25, L26 ("Respawn"), L56 ("Salto"); GitHub API `spdx_id` "MIT" | "Sistema de suspensión independiente por raycast (`CANNON.RaycastVehicle`)" | confirmed |
| 8 Tuned values | "icurtis1/raycast-vehicle" is MIT and uses "the same vehicle with ramps, jumps and respawn" | [raycast-vehicle `README.md`](https://raw.githubusercontent.com/icurtis1/raycast-vehicle/main/README.md) | L8–9, L16, L37–38; GitHub API `spdx_id` "MIT" | "colliders for ramps, loops, and walls." | confirmed |
| 8 References | "Baraff 1997 (notes) for the state and box inertia" | [Baraff notes](https://www.cs.cmu.edu/~baraff/sigcourse/notesd1.pdf) | title page; section 1 (state vector `Y(t)`); section 5.1 | "5.1 Inertia Tensor of a Block" | confirmed |
| 8 References | "Catto, GDC 2015 (slides) for semi-implicit Euler" | [Catto slides](https://box2d.org/files/ErinCatto_NumericalMethods_GDC2015.pdf) | slide "Semi-implicit Euler: cheap and stable" | "If I use the current velocity in the position update, then explicit Euler becomes semi-implicit Euler." | confirmed |
| 8 References | "Fiedler (article) for fixed time steps" | [Fix Your Timestep!](https://gafferongames.com/post/fix_your_timestep/) | section "Fixed delta time" | "The simplest way to step forward is with fixed delta time, like 1/60th of a second" | confirmed |
| 9 Generators | "Gymnasium CarRacing walks between noisy polar checkpoints and has no self-intersection test" | [Gymnasium `car_racing.py`](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/box2d/car_racing.py) | `_create_track` L324–341; the word "intersect" does not occur in the file | `noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)` | confirmed |
| 9 Generators | "Maciel 2013 (random points → convex hull → spline, article) can self-intersect" | [Generating Procedural Racetracks](https://www.gamedeveloper.com/programming/generating-procedural-racetracks) | published 2013-12-29; requirement list and spline section | "It can have intersections, preferably, in a transverse way, so it will avoid the track going over itself" | confirmed |
| 9 Generators | "juangallostra/procedural-tracks (MIT) implements that method as closed 2D loops" | [procedural-tracks `README.md`](https://raw.githubusercontent.com/juangallostra/procedural-tracks/master/README.md) | L20–24; GitHub API `spdx_id` "MIT" | "Compute the points that, from the set of all points generated in step 1, form the convex Hull (red lines)" | confirmed |
| 9 Generators | "Barthet et al. 2024 (arXiv 2408.06346) use grid tiles with ramps, with no public code" | [arXiv 2408.06346, HTML](https://arxiv.org/html/2408.06346) | section IV (track generator); no repository link anywhere in the text | "whereas the loop and ramp are 3 tiles long" | confirmed (absence of a code link checked in the paper only) |
| 9 Schema | "TORCS describes a track as straight or arc segments with start and end height and start and end banking" | [TORCS track manual](https://torcs.sourceforge.net/api/track_manual.html) | sections 6 "Segment geometry" and 7 "Elevation and banking" | "type="lft" : left-hand turn; requires radius and arc (turn angle)." | confirmed (heights `z start`, `z end` or `grade`; "banking effects ( banking start/end )") |
| 9 Schema | "its generator code is GPL" | same page | section 2.1; licence header of the page source | "must be distributed under terms compatible with TORCS licensing (GPL version 2 or later)" | confirmed (TORCS licence as stated in the manual; the `trackgen` source header itself was not opened) |
| 9 Ribbon model | "Perantoni & Limebeer 2015 (DOI) describe a 3D track as a spine curve with orientation as a function of arc length and a lateral coordinate" | [Crossref record for 10.1115/1.4028253](https://api.crossref.org/works/10.1115/1.4028253) | abstract (full text paywalled) | "the track is specified in terms of three displacement-dependent curvatures and two edge variables" | confirmed in substance (abstract only; a wording closer to the source is given under "Optional precision edits") |
| 9 TrackMania | "commonly described as ticking at 100 Hz and being deterministic (donadigo); these are community statements" | [donadigo, TMX1](https://donadigo.com/tmx1) | sections 2 "Motivation" and the input-analysis section | "Because TrackMania’s physics update rate equals to 100 updates/second" | confirmed (also: "Trackmania’s physics engine is completely deterministic.") |
| 10 Pure pursuit | "Coulter 1992 (report): steering curvature 2x/l² toward a look-ahead point" | [CMU-RI-TR-92-01](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf) | section 2, after equations (2.1)–(2.2), p. 6 (the final equation is not legible in the scan's text layer; the sentence below states it in words) | "the gain is 2 times the inverse square of 1" (OCR renders the symbol *l* as "1") | confirmed |
| 10 Pure pursuit | "Stanley control (Thrun et al. 2006) is the common alternative" | [Crossref record for 10.1002/rob.20147](https://api.crossref.org/works/10.1002/rob.20147) | Crossref metadata | "Stanley: The robot that won the DARPA Grand Challenge" | confirmed (citation exists; "common alternative" is the reviewer's judgement) |
| 10 Speed profile | "A forward and backward pass over curvature with acceleration limits, as in TUM's `trajectory_planning_helpers` (LGPL-3.0, repo)" | [`calc_vel_profile.py`](https://raw.githubusercontent.com/TUMFTM/trajectory_planning_helpers/master/trajectory_planning_helpers/calc_vel_profile.py) | calls with `backwards=False` (L271) and `backwards=True` (L288); GitHub API `spdx_id` "LGPL-3.0" | "Calculates a velocity profile using the tire and motor limits as good as possible." | confirmed |
| 10 Speed profile | "it is 2D, so on hills and jumps it is approximate" | same file | docstring inputs: `kappa`, `el_lengths`, `ggv`, `mu`; no elevation or slope input | "curvature profile of given trajectory in rad/m (always unclosed)" | confirmed |
| 10 Reference | "MPPI (Williams et al. 2016, DOI)" | [Crossref record for 10.1109/ICRA.2016.7487277](https://api.crossref.org/works/10.1109/ICRA.2016.7487277) | Crossref metadata | "Aggressive driving with model predictive path integral control" | confirmed |
| 10 Reference | "CEM variants such as iCEM (Pinneri et al. 2020, arXiv 2008.06389)" | [arXiv 2008.06389](https://arxiv.org/abs/2008.06389) | abstract page, title | "Sample-efficient Cross-Entropy Method for Real-time Planning" | confirmed |
| 11 Hide-and-seek | "seekers learned "box surfing", possible because the movement action applies force even when the agent is off the ground" | [arXiv 1909.07528v2, HTML](https://arxiv.org/html/1909.07528v2) | section 3 (emergent phases) | "the agents’ movement action allows them to apply a force on themselves regardless of whether they are on the ground or not" | confirmed |
| 11 Hide-and-seek | "the accompanying blog lists further exploits, including seekers launching themselves off a ramp against a wall (confirmed through quotations in the specification-gaming list…)" | [specification-gaming list, CSV export](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pub?output=csv) | row "Hide-and-seek", column "Behavior" (the blog itself returned HTTP 403 again) | "Seekers learn that if they run at a wall with a ramp at the right angle, they can launch themselves upward." | confirmed |
| 11 Specification gaming | "(Krakovna et al. 2020, blog): a public list of about 90 examples" | [DeepMind blog](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/); list as above | blog introduction; the list has 90 titled rows on 2026-11-14 | "we have collected around 60 examples so far" | corrected — see C10 |
| 11 Specification gaming | "with fixed columns (intended goal, behaviour, and so on)" | list as above | header row | "Title, Description, Illustration, Type of system, Intended goal, Misspecified goal, Behavior, Authors, Source, Credit" | confirmed |
| 11 Specification gaming | "Lehman et al. 2018 (arXiv 1803.03453) collect physics-engine exploits from evolution" | [arXiv 1803.03453, HTML](https://arxiv.org/html/1803.03453) | anecdote titles; discussion | "Evolving Virtual Creatures Reveal Imperfectly Simulated Physics" | confirmed |
| 11 Specification gaming | "and advise visualising solutions regularly" | same page | discussion (lessons for practitioners) | "Practioners should regularly visualize their simulated solutions to test whether the proposed solutions are valid and reasonable." | confirmed |
| 11 TrackMania terms | "An uberbug is a sudden speed gain from a car–track collision during a bugslide" | [TAS Exchange FAQ](https://tmtas.exchange/faq) | "Techniques", "What is Uberbug?" | "Uberbug is a physics exploit that allows the car to gain massive speed through specific car and track interactions." | corrected — see C11 ("collision" and "sudden" are not in the source) |
| 11 TrackMania terms | "a noseboost chains nose-down slides to gain speed" | same page | "What is Noseboost?", "What is Nosebug?" | "Noseboost is a glitch where chaining multiple nosebugs can provide a massive speed boost." | confirmed (a nosebug is "performing bugslide inputs while the car is on its nose") |
| 11 TrackMania terms | "Claims that an RL agent learned such bugs come only from secondary sources." | — | the review cites no source and labels the item `[unverified]` | — | not found in source — nothing to check |
| 12 Policy in the browser | "onnxruntime-web 1.31.1" | [npm registry, onnxruntime-web](https://registry.npmjs.org/onnxruntime-web) | `dist-tags.latest`; `time["1.31.1"]` = 2026-10-26 | `"latest":"1.31.1"` | confirmed |
| 12 Policy in the browser | "the WASM backend supports all operators, WebGPU lacks several" | [ONNX Runtime Web tutorial](https://onnxruntime.ai/docs/tutorials/web/) | introduction, "Options for deployment" | "All ONNX operators are supported by WASM but only a subset are currently supported by WebGL, WebGPU and WebNN." | confirmed |
| 12 Policy in the browser | "multithreading needs cross-origin isolation" | [ONNX Runtime Web env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html) (not the page the review cites) | `env.wasm.numThreads` | "Only when the browser supports WebAssembly multi-threading and crossOriginIsolated mode is enabled, multi-threading will be enabled." | confirmed (the citation should point to this page) |
| 12 Policy in the browser | "that GitHub Pages cannot provide, so one thread is used" | web search only ([GitHub community discussion 13309](https://github.com/orgs/community/discussions/13309), not opened) | `[2nd]`: an open request to allow COOP and COEP headers on GitHub Pages; a service-worker shim (`coi-serviceworker`) emulates them | — | corrected — see C12 (headers cannot be set; isolation can still be emulated by a service worker) |
| 12 Policy in the browser | "`torch.onnx.export` uses the dynamo exporter by default" | [PyTorch v2.15.0 `torch/onnx/__init__.py`](https://raw.githubusercontent.com/pytorch/pytorch/v2.15.0/torch/onnx/__init__.py) | `export` signature L77–78; docs page for 2.15 says the same | `dynamo: bool = True,` | confirmed |
| 12 Policy in the browser | "default opset 20" | [PyTorch v2.15.0 `torch/onnx/_constants.py`](https://raw.githubusercontent.com/pytorch/pytorch/v2.15.0/torch/onnx/_constants.py) | constant; used in `_internal/exporter/_compat.py` L71–72 when `opset_version is None` | `ONNX_DEFAULT_OPSET = 20` | confirmed |
| 12 Precedents | "mjswan exports observation and termination terms to ONNX and runs them in the browser" | [mjswan, "How it works"](https://mjswan.readthedocs.io/en/latest/guides/how-it-works/) | section "The problem" | "mjswan's answer is to trace those functions to ONNX at build time and run them with ONNX Runtime Web" | confirmed |
| 12 Precedents | "Mondo-Robotics/open-beni (MIT, repo) is a browser racing game that runs PPO policies as ONNX in a web worker" | [open-beni `web/game/sim.worker.ts`](https://raw.githubusercontent.com/Mondo-Robotics/open-beni/main/web/game/sim.worker.ts); [`README.md`](https://raw.githubusercontent.com/Mondo-Robotics/open-beni/main/README.md) | worker L8, L178; README L16, L55, L171–193; GitHub API `spdx_id` "MIT" | `import * as ort from 'onnxruntime-web/wasm';` | confirmed (the worker also sets `ort.env.wasm.numThreads = 1`) |
| 12 Precedents | "its worker, inference loop and export script are the pattern we copy" | [GitHub API, open-beni tree](https://api.github.com/repos/Mondo-Robotics/open-beni/git/trees/main?recursive=1) | paths `web/game/sim.worker.ts`, `python/beni_rl/headless_export.py` | "ONNX/MNN exporter" (README file tree) | confirmed (the files exist) |
| 12 Hugging Face | "No TrackMania agent models exist on the Hub" | [Hub API, models search "trackmania"](https://huggingface.co/api/models?search=trackmania&limit=50) | 15 results, all `jeanmidev/trackmania-tmnf-*`, pipeline tag `text-generation` or none; card of `trackmania-tmnf-gpt2` | "This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on an unknown dataset." | corrected — see C13 |
| 12 Hugging Face | "RL model cards conventionally use `model-index` with task `reinforcement-learning` and a mean ± spread metric" | [sb3/ppo-CartPole-v1 card](https://huggingface.co/sb3/ppo-CartPole-v1/raw/main/README.md) | YAML front matter L7–20 | `value: 500.00 +/- 0.00` | confirmed (one representative card read) |
| 12 Hugging Face | "A dataset of community TrackMania tracks and replays exists" | [dataset card](https://huggingface.co/datasets/jeanmidev/trackmania-community-tracks-and-telemetry/raw/main/README.md) | card L33–61 | "Roughly **900,000 community-made [Trackmania](https://www.trackmania.com) tracks**" | confirmed |
| 12 Hugging Face | "but the layouts stay under the publisher's terms and are not used" | same card | front matter L2; L51 | `license: cc-by-4.0` | not found in source — the card declares CC-BY-4.0 and says nothing about publisher terms; see C13 |
| 12 Tooling | "PyTorch 2.15.0 with `cu132` wheels for Windows and Linux (Python 3.11–3.15)" | [PyPI JSON, torch](https://pypi.org/pypi/torch/json); [PyTorch cu132 index](https://download.pytorch.org/whl/cu132/torch/) | `info.version` 2.15.0, `requires_python` ">=3.11", uploaded 2026-10-28; index lists cp311–cp315 for `win_amd64` and `manylinux_2_28` | `torch-2.15.0+cu132-cp315-cp315-win_amd64.whl` | confirmed |
| 12 Tooling | "Gymnasium 1.4.0" | [PyPI JSON, gymnasium](https://pypi.org/pypi/gymnasium/json) | `info.version`; uploaded 2026-10-05 | `gymnasium-1.4.0-py3-none-any.whl` | confirmed |
| 12 Tooling | "TensorBoard 2.21" | [PyPI JSON, tensorboard](https://pypi.org/pypi/tensorboard/json) | `info.version` 2.21.0; uploaded 2026-06-29 | `tensorboard-2.21.0-py3-none-any.whl` | confirmed |
| 12 Tooling | "pandoc 3.14 with typst 0.15" | [GitHub API, jgm/pandoc latest release](https://api.github.com/repos/jgm/pandoc/releases/latest); [typst/typst latest release](https://api.github.com/repos/typst/typst/releases/latest) | `tag_name` "3.14" (2026-11-12) and "v0.15.1" (2026-07-17) | `"tag_name": "3.14"` | confirmed (typst's current patch release is 0.15.1) |

## Corrections to apply

Each item gives the exact text now in the review and the replacement.

1. **C1 — section 8, table row cannon-es, column "Car code".**
   Original: `about 390 lines plus `WheelInfo`; only ray casts and impulses on one body`
   Replacement: `673 lines (461 without blank and comment lines) plus `WheelInfo` (429 lines); only ray casts and impulses on the chassis (the side impulse's reaction also goes to the ground body, which does nothing when the ground is static)`
   Source: <https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts>

2. **C2 — section 8, table row Rapier.**
   Original: `about 780 lines of Rust, a port of Bullet's`
   Replacement: `823 lines of Rust (master, 2026-11-14), a port of Bullet's`
   Original: `One `world.step()` measured about 8–11 µs in an issue, so CPU-only training would need several busy cores`
   Replacement: delete the sentence, or add the URL of the issue; no such issue was found.
   Source: <https://raw.githubusercontent.com/dimforge/rapier/master/src/control/ray_cast_vehicle_controller.rs>

3. **C3 — section 8, table row "Bullet through ammo.js".**
   Original: `same model as cannon-es`
   Replacement: `the model cannon-es ported, with two differences: Bullet applies either the engine impulse or the brake impulse, cannon-es adds both; Bullet's side impulse uses the full effective mass at the contact, cannon-es only the linear mass`
   Original: `Stale and heavy`
   Replacement: `Lightly maintained (four commits in 2026, none between November 2023 and June 2026); an Emscripten build of the whole Bullet engine`
   Sources: <https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/Vehicle/btRaycastVehicle.cpp>, <https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/ConstraintSolver/btContactConstraint.cpp>, <https://api.github.com/repos/kripken/ammo.js/commits?per_page=6>

4. **C4 — section 8, table row "MuJoCo MJX / MuJoCo Warp".**
   Original: `no car example`
   Replacement: `no car or vehicle example in the MJX or MJWarp docs (the main repository ships one small toy model, `model/car/car.xml`)`
   Original: `MJX runs on GPU only under Linux or WSL2 (JAX lists WSL2 as experimental); per-world terrain is undocumented for MJX and costly for Warp`
   Replacement: `MJX-JAX runs on an NVIDIA GPU only under Linux or WSL2 (JAX lists WSL2 as experimental and native Windows as unsupported); per-world terrain is undocumented for MJX-JAX, and MJWarp supports per-world meshes and height fields but leaves building the per-world arrays to the user`
   Sources: <https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/car/car.xml>, <https://docs.jax.dev/en/latest/installation.html>, <https://mujoco.readthedocs.io/en/latest/mjwarp/index.html>

5. **C5 — section 8, paragraph after the table.**
   Original: `No batched or GPU port of a raycast vehicle was found either; SceneFactory-3D ([arXiv 2610.02874](https://arxiv.org/abs/2610.02874)) describes the same design in an NVIDIA Warp kernel but its code link is dead.`
   Replacement: `No public batched or GPU port of a game engine's raycast vehicle was found either. SceneFactory-3D ([arXiv 2610.02874](https://arxiv.org/abs/2610.02874)) describes a related design — a rigid chassis with four raycast suspension legs whose suspension, tyre, drive, brake and steering forces are computed in a custom NVIDIA Warp kernel and applied as an external wrench, while Newton (XPBD) integrates the chassis — but its code link returned HTTP 404 on 2026-11-14.`
   Source: <https://arxiv.org/html/2610.02874v2> (section 4.2)

6. **C6 — section 8, "Models".**
   Original: `Monster's "Car Physics for Games" clamps lateral force and is singular at zero speed`
   Replacement: `Monster's "Car Physics for Games" uses a lateral force linear in slip angle (`F = Ca·α`, given for small angles; the page caps only the longitudinal traction force) and computes slip angle from the arctangent of lateral over longitudinal velocity, which is undefined at zero speed`
   Source: <https://asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html>

7. **C7 — section 8, "The raycast vehicle".**
   Original: `a side impulse cancels lateral velocity`
   Replacement: `a side impulse removes part of the lateral velocity at the contact (factor 0.2 per wheel and step)`
   Source: <https://raw.githubusercontent.com/bulletphysics/bullet3/master/src/BulletDynamics/ConstraintSolver/btContactConstraint.cpp> (L144–151); cannon-es `RaycastVehicle.ts` L668–670

8. **C8 — section 8, "Facts about cannon-es", fourth bullet.**
   Original: `Suspension force is `stiffness × compression × clippedInvContactDotSuspension − damping × relative velocity`, multiplied by chassis mass and clamped to `[0, maxSuspensionForce]`.`
   Replacement: `Suspension force is `stiffness × (restLength − suspensionLength) × clippedInvContactDotSuspension − damping × suspensionRelativeVelocity`, multiplied by chassis mass and floored at 0; `damping` is `dampingCompression` when the relative velocity is negative and `dampingRelaxation` otherwise. The cap at `maxSuspensionForce` (default `Number.MAX_VALUE`) is applied only to the suspension impulse; the grip limit `suspensionForce × dt × frictionSlip` uses the uncapped value.`
   Source: <https://raw.githubusercontent.com/pmndrs/cannon-es/master/src/objects/RaycastVehicle.ts> (L145–149, L213–232, L484)

9. **C9 — section 8, "Tuned starting values".**
   Original: `friction slip 1.5, roll influence 0, engine force 1800 on the rear wheels, maximum brake 65, steering 0.3.`
   Replacement: `friction slip 1.5, side acceleration 3 (it divides the side impulse in the grip test), roll influence 0, engine force 1800 on the rear wheels (applied with a negative sign to drive forward, and set to 0 above speed 88), maximum brake 65 on the rear wheels (an impulse per step, not a force), steering 0.3 on the front wheels. Geometry: box `args` [2, 1.1, 4.7]; wheel connection points at x = ±0.85, y = −0.3, z = 1.35 (front) and −1.3 (rear); `directionLocal` (0, −1, 0); `axleLocal` (−1, 0, 0); use-cannon passes `indexRightAxis` 0, `indexForwardAxis` 2, `indexUpAxis` 1 unless told otherwise.`
   Sources: <https://raw.githubusercontent.com/pmndrs/racing-game/main/src/store.ts>, <https://raw.githubusercontent.com/pmndrs/racing-game/main/src/models/vehicle/Vehicle.tsx>, <https://raw.githubusercontent.com/pmndrs/use-cannon/master/packages/react-three-cannon/src/hooks.ts> (L697–699)

10. **C10 — section 11, "Specification gaming".**
    Original: `a public list of about 90 examples with fixed columns (intended goal, behaviour, and so on)`
    Replacement: `a public list ("around 60 examples" when the post was written; 90 entries on 2026-11-14) with fixed columns (type of system, intended goal, misspecified goal, behaviour, and so on)`
    Sources: <https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/>, the list's CSV export (URL in the ledger)

11. **C11 — section 11, "TrackMania terms".**
    Original: `An uberbug is a sudden speed gain from a car–track collision during a bugslide`
    Replacement: `An uberbug is a physics exploit that gives a massive speed gain through specific car–track interactions during a bugslide`
    Source: <https://tmtas.exchange/faq>

12. **C12 — section 12, "Policy in the browser".**
    Original: `multithreading needs cross-origin isolation that GitHub Pages cannot provide, so one thread is used ([docs](https://onnxruntime.ai/docs/tutorials/web/))`
    Replacement: `multithreading needs cross-origin isolation ([env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)), which GitHub Pages cannot switch on through response headers (only a service-worker shim can emulate it, `[2nd]`), so one thread is used ([docs](https://onnxruntime.ai/docs/tutorials/web/))`
    Source: <https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html>

13. **C13 — section 12, "Hugging Face".**
    Original: `No TrackMania agent models exist on the Hub`
    Replacement: `A Hub search for "trackmania" (2026-11-14) returns 15 models, all text-generation fine-tunes by one author with auto-generated cards; none is tagged `reinforcement-learning` or documents a driving policy`
    Original: `A dataset of community TrackMania tracks and replays exists, but the layouts stay under the publisher's terms and are not used.`
    Replacement: `A dataset of about 900,000 community TrackMania tracks with replays exists (`jeanmidev/trackmania-community-tracks-and-telemetry`, labelled CC-BY-4.0, collected from the Mania Exchange sites); it is not used here.`
    Sources: <https://huggingface.co/api/models?search=trackmania&limit=50>, <https://huggingface.co/datasets/jeanmidev/trackmania-community-tracks-and-telemetry>

### Optional precision edits (the original is not wrong)

- Section 8, first cannon-es bullet: after "`x += v·dt`" add "`ω += dt·I⁻¹·τ`", and end the bullet with "then the `postStep` event". The inertia update is the last line of `Body.integrate`, not a separate world phase.
- Section 8, last cannon-es bullet: "straight-line braking is not limited by grip" also applies to the engine impulse, and the brake impulse stays capped at the `brake` value itself.
- Section 9, ribbon model: closer to the abstract is "a ribbon in curvilinear coordinates: three curvatures as functions of distance along the track (a generalised Frenet–Serret frame) and two edge variables". Only the abstract was read.
- Section 12, tooling: typst's current release is 0.15.1.

## cannon-es port notes

All line numbers are for `master` of <https://github.com/pmndrs/cannon-es> (version 0.20.0). File names are relative to `src/`. Conventions that the code relies on: `a.cross(b)` is a × b (`math/Vec3.ts` L29–42); `q.mult(p)` is the Hamilton product q·p (`math/Quaternion.ts` L110–125); `q.vmult(v)` rotates v by q (L202–223); `Mat3` is row-major and `m.vmult(v)` is m·v (`math/Mat3.ts` L83–93); `Body.vectorToWorldFrame(v)` is `quaternion.vmult(v)` and `pointToWorldFrame(p)` adds `position` (`objects/Body.ts` L618–630); `Body.getVelocityAtWorldPoint(p)` is `velocity + angularVelocity × (p − position)` (L925–931).

### (a) Order of one world step

`World.step(dt)` with one argument calls `internalStep(dt)` once and then adds `dt` to `world.time` (`world/World.ts` L486–493). `internalStep` (L527–869) does, in this order:

1. `this.dt = dt` (L528). The vehicle reads this value.
2. **Gravity into the force accumulator**, dynamic bodies only (L556–567): `f.x += m * gx` (and y, z). Velocity is not touched here.
3. Subsystems, broadphase (L580), narrowphase (L620), contact and friction equations, user constraints.
4. **Solver**: `solver.solve(dt, this)` (L801). With no equations `GSSolver.solve` changes nothing: both the mass update and the velocity write-back sit inside `if (Neq !== 0) {` (`solver/GSSolver.ts` L53, L74). A box chassis on a `Trimesh` track never produces an equation (see (h)).
5. **Damping**, dynamic bodies only (L811–825), see (b).
6. **`preStep` event** (L827): `this.dispatchEvent(World_step_preStepEvent)`. `RaycastVehicle.addToWorld` registered `that.updateVehicle(world.dt)` on it (`objects/RaycastVehicle.ts` L100–103). All vehicle impulses change `velocity` and `angularVelocity` here, after damping and before gravity reaches the velocity.
7. **Integration** of every body (L839–841): `bodies[i].integrate(dt, quatNormalize, quatNormalizeFast)` with `quatNormalize = stepnumber % (this.quatNormalizeSkip + 1) === 0` (L836). Defaults `quatNormalizeSkip = 0`, `quatNormalizeFast = false` (L219–220), so the exact normalisation runs every step. See (c).
8. **Force clearing**: `this.clearForces()` (L842) sets `force` and `torque` of every body to zero (L944–955).
9. `this.stepnumber += 1` (L851), then the **`postStep` event** (L853).
10. Sleep bookkeeping, only if `world.allowSleep` (L857–867); `World` defaults it to false (`this.allowSleep = !!options.allowSleep`, L216). If it is enabled, a body whose own `allowSleep` is true (the `Body` default, L472) falls asleep after `sleepTimeLimit = 1` s below `sleepSpeedLimit = 0.1` (L474–475) and then skips integration; `applyImpulse` wakes it. The port should run with sleeping off (pmndrs/racing-game sets `allowSleep: false` on the chassis).

So per tick, for the chassis: `f = m·g` → damping of `v`, `ω` → vehicle impulses → `v += g·dt` (through `f`) → `x += v·dt` → quaternion update → world inertia update → forces cleared.

### (b) Damping

```ts
const ld = pow(1.0 - bi.linearDamping, dt)   // World.ts L816
v.scale(ld, v)                               // L818
const ad = pow(1.0 - bi.angularDamping, dt)  // L821
av.scale(ad, av)                             // L822
```

Defaults: `linearDamping` 0.01 (`Body.ts` L464) and `angularDamping` 0.01 (L512). With dt = 1/60 the factor is 0.99^(1/60) per step for both.

### (c) Body integration, impulses and inertia

`Body.integrate` (`objects/Body.ts` L939–992), for dynamic, awake bodies:

```ts
const iMdt = invMass * dt
velo.x += force.x * iMdt * linearFactor.x                    // L960 (y, z alike)
angularVelo.x += dt * (e[0] * tx + e[1] * ty + e[2] * tz)    // L969, e = invInertiaWorld, t = torque ∘ angularFactor
pos.x += velo.x * dt                                         // L974, uses the new velocity
quat.integrate(this.angularVelocity, dt, this.angularFactor, quat)   // L978
quat.normalize()                                             // L984
this.updateInertiaWorld()                                    // L991
```

There is no gyroscopic term. `linearFactor` and `angularFactor` default to (1, 1, 1) (L514, L520). `Quaternion.integrate` (`math/Quaternion.ts` L398–415) is called with `target` equal to the quaternion itself, so it adds ½·dt·(ω ⊗ q) to q in place, with a = ω ∘ angularFactor and b = the old q:

```ts
target.x += half_dt * (ax * bw + ay * bz - az * by)
target.y += half_dt * (ay * bw + az * bx - ax * bz)
target.z += half_dt * (az * bw + ax * by - ay * bx)
target.w += half_dt * (-ax * bx - ay * by - az * bz)
```

`Quaternion.normalize` divides by the Euclidean norm (L162–177).

`updateInertiaWorld` (L747–764) computes `invInertiaWorld = R · diag(invInertia) · Rᵀ` with R from `Mat3.setRotationFromQuaternion`: `m1.setRotationFromQuaternion(this.quaternion)`, `m1.transpose(m2)`, `m1.scale(I, m1)` (scales columns), `m1.mmult(m2, this.invInertiaWorld)`. It is skipped when the three `invInertia` components are equal and `force` is not set.

`Body.applyImpulse(impulse, relativePoint)` (L837–870): `velocity += impulse * invMass`; `angularVelocity += invInertiaWorld · (relativePoint × impulse)`. It returns immediately for non-dynamic bodies (L838) and does not apply `linearFactor` or `angularFactor`. Because `invInertiaWorld` is refreshed only at the end of `integrate`, every impulse of a tick uses the matrix that matches the orientation at the start of that tick.

Local inertia (`updateMassProperties`, L895–917, run by the constructor and by every `addShape`): `invMass = 1 / mass`; the inertia is that of a box whose half extents are **half the size of the body's world-space AABB at that moment** (`updateAABB` uses the current body quaternion, L709–742), through `Box.calculateInertia` (`shapes/Box.ts` L79–84): `I.x = (1/12)·m·((2·e.y)² + (2·e.z)²)`, cyclically. For a single unrotated box shape this is the box's own inertia; if the body is already rotated when a shape is added, it is not. A port must reproduce the numbers the JavaScript side actually ends up with (read `body.invInertia` once and copy it).

### (d) Axis defaults and `WheelInfo` defaults

`RaycastVehicle` constructor (`objects/RaycastVehicle.ts` L52–54):

```ts
this.indexRightAxis = typeof options.indexRightAxis !== 'undefined' ? options.indexRightAxis : 2
this.indexForwardAxis = typeof options.indexForwardAxis !== 'undefined' ? options.indexForwardAxis : 0
this.indexUpAxis = typeof options.indexUpAxis !== 'undefined' ? options.indexUpAxis : 1
```

So right = z, forward = x, up = y by default. (use-cannon overrides these to right = 0, forward = 2, up = 1.) Their uses: `indexRightAxis` picks the wheel-frame axis used as the axle in `updateFriction` (L421); `indexUpAxis` picks the component scaled by `rollInfluence` (L547) and the sign of the visual wheel spin (L168–172); `indexForwardAxis` is used only for the sign of `currentVehicleSpeedKmHour` (L127–131) and the visual wheel spin (L175). The driving direction in the dynamics is `hitNormal × axle`, not `indexForwardAxis`.

`WheelInfo` defaults (`objects/WheelInfo.ts` L329–360): `chassisConnectionPointLocal` (0,0,0), `directionLocal` (0,0,0), `axleLocal` (0,0,0) (all three must be supplied), `suspensionRestLength` 1, `suspensionMaxLength` 2, `radius` 1, `suspensionStiffness` 100, `dampingCompression` 10, `dampingRelaxation` 10, `frictionSlip` 10.5, `forwardAcceleration` 1, `sideAcceleration` 1, `rollInfluence` 0.01, `maxSuspensionForce` `Number.MAX_VALUE`, `isFrontWheel` true, `maxSuspensionTravel` 1, `useCustomSlidingRotationalSpeed` false, `customSlidingRotationalSpeed` −0.1. The constructor ignores any option given for the following and always starts them at these values (L381–399): `steering` 0, `rotation` 0, `deltaRotation` 0, `engineForce` 0, `brake` 0, `clippedInvContactDotSuspension` 1, `suspensionRelativeVelocity` 0, `suspensionForce` 0, `slipInfo` 0, `skidInfo` 0, `suspensionLength` 0, `sideImpulse` 0, `forwardImpulse` 0, `sliding` false, `isInContact` false. `suspensionMaxLength` and `isFrontWheel` are never read by `RaycastVehicle.ts`, and `WheelInfo.updateWheel` (L402–425) is never called by it.

### (e) `updateVehicle(dt)`: the order of every computation

`updateVehicle` (L115–198):

1. For each wheel, `updateWheelTransform(i)` (L338–372):
   - `updateWheelTransformWorld` (L325–331): `isInContact = false`; `chassisConnectionPointWorld = pointToWorldFrame(chassisConnectionPointLocal)`; `directionWorld = vectorToWorldFrame(directionLocal)`; `axleWorld = vectorToWorldFrame(axleLocal)`.
   - `up = −directionLocal`, `right = axleLocal` (normalised after use in a cross product), `steeringOrn = axisAngle(up, wheel.steering)`, `rotatingOrn = axisAngle(right, wheel.rotation)`, `worldTransform.quaternion = normalize(chassisQ · steeringOrn · rotatingOrn)` (L353–365). `Quaternion.setFromAxisAngle` does not normalise the axis (L54–61), so `directionLocal` must be a unit vector.
   - `worldTransform.position` is built from last tick's `suspensionLength` (L368–371); it is used only for drawing.
2. `currentVehicleSpeedKmHour` (L124–131); not used in the dynamics.
3. For each wheel, `castRay(wheel)` (see (f)).
4. `updateSuspension(dt)` (L200–237), for each wheel:

   ```ts
   const length_diff = susp_length - current_length      // suspensionRestLength - suspensionLength, L213–215
   force = wheel.suspensionStiffness * length_diff * wheel.clippedInvContactDotSuspension   // L217
   // L222–226: susp_damping = dampingCompression if suspensionRelativeVelocity < 0, else dampingRelaxation
   force -= susp_damping * projected_rel_vel             // L227
   wheel.suspensionForce = force * chassisMass           // L229
   if (wheel.suspensionForce < 0) { wheel.suspensionForce = 0 }   // L230–232
   ```

   and `suspensionForce = 0` when `isInContact` is false (L234). Stiffness and damping are therefore per unit of chassis mass.
5. For each wheel in index order, the suspension impulse (L142–153): a local copy of the force is capped (`if (suspensionForce > wheel.maxSuspensionForce)`, L146–148; the stored `wheel.suspensionForce` is not changed), then `impulse = hitNormalWorld * (suspensionForce * dt)` (L149), `relpos = hitPointWorld − chassis.position` (L151), `chassisBody.applyImpulse(impulse, relpos)` (L152). For a wheel without contact the impulse is the zero vector. No reaction is applied to the ground.
6. `updateFriction(dt)` (L381–556). It sees the velocities **after** all suspension impulses.
   - Reset (L393–409): `numWheelsOnGround` counts wheels with `raycastResult.body`; `sideImpulse = 0`, `forwardImpulse = 0` for every wheel.
   - Side impulse, for wheels with a ground body (L411–442):

     ```ts
     wheelTrans.vectorToWorldFrame(directions[this.indexRightAxis], axlei)   // L421, wheelTrans = worldTransform from step 1
     const proj = axlei.dot(surfNormalWS)                                    // L424
     surfNormalWS.scale(proj, surfNormalWS_scaled_proj)
     axlei.vsub(surfNormalWS_scaled_proj, axlei)                             // remove the normal component
     axlei.normalize()
     surfNormalWS.cross(axlei, forwardWS[i])                                 // L429, forward = normal × axle
     forwardWS[i].normalize()
     wheel.sideImpulse = resolveSingleBilateral(chassisBody, hitPointWorld, groundObject, hitPointWorld, axlei)   // L432–438
     wheel.sideImpulse *= sideFrictionStiffness2                             // L440, constant 1 (L577)
     ```

     with `resolveSingleBilateral` (L648–673):

     ```ts
     if (normalLenSqr > 1.1) { return 0 }
     const rel_vel = normal.dot(vel)                            // vel = v_chassis(p) − v_ground(p)
     const contactDamping = 0.2
     const massTerm = 1 / (body1.invMass + body2.invMass)
     const impulse = -contactDamping * rel_vel * massTerm
     ```

     On static ground this is `−0.2 · mass · (lateral velocity at the contact)` per wheel. `Vec3.normalize` of a zero vector returns the zero vector (`math/Vec3.ts` L106–123), so an axle parallel to the normal gives zero side and forward directions.
   - Forward impulse, for every wheel (L447–506); `this.sliding = false` first:

     ```ts
     const maxImpulse = wheel.brake ? wheel.brake : defaultRollingFrictionImpulse   // L457, default 0
     rollingFriction = calcRollingFriction(chassisBody, groundObject, hitPointWorld, forwardWS[i], maxImpulse)   // L461–467
     rollingFriction += wheel.engineForce * timeStep                                // L469
     ```

     with `calcRollingFriction` (L583–623): `vrel = forward · (v_chassis(p) − v_ground(p))`; `j1 = -vrel * jacDiagABInv` where `jacDiagABInv = relaxation / (denom0 + denom1)`, `relaxation = 1`; then `j1` is clamped to `[−maxImpulse, +maxImpulse]` (L615–620). `computeImpulseDenominator(body, pos, normal)` (L629–641) is `body.invMass + normal · ((invInertiaWorld · (r × normal)) × r)` with `r = pos − body.position`; it is 0 for a static ground body. So `brake` is the **largest braking impulse per step** (it is not multiplied by dt), the brake impulse opposes the current forward velocity up to that cap, and `engineForce` is a force (multiplied by dt). Both are added; with `brake = 0` the rolling part is 0.
   - Grip test, wheels with a ground body (L481–505):

     ```ts
     const maximp = wheel.suspensionForce * timeStep * wheel.frictionSlip     // L484, the stored, uncapped force
     wheel.forwardImpulse = rollingFriction                                    // L489
     const x = (wheel.forwardImpulse * fwdFactor) / wheel.forwardAcceleration // L491, fwdFactor = 0.5 (L445)
     const y = (wheel.sideImpulse * sideFactor) / wheel.sideAcceleration      // L492, sideFactor = 1 (L444)
     if (impulseSquared > maximpSquared) {                                    // L497, x² + y² > maximp²
       this.sliding = true
       wheel.sliding = true
       const factor = maximp / Math.sqrt(impulseSquared)
       wheel.skidInfo *= factor                                               // skidInfo was set to 1 at L479/L482
     }
     ```

     `fwdFactor`, `forwardAcceleration` and `sideAcceleration` enter only this test, not the applied impulses.
   - Sliding clamp (L508–518), only if **any** wheel slid this tick:

     ```ts
     if (wheel.sideImpulse !== 0) {
       if (wheel.skidInfo < 1) {
         wheel.forwardImpulse *= wheel.skidInfo
         wheel.sideImpulse *= wheel.skidInfo
       }
     }
     ```

     A wheel whose side impulse is exactly 0 keeps its full forward impulse (engine and brake) whatever the grip limit is.
   - Application, for each wheel in index order (L521–555): `rel_pos = hitPointWorld − chassis.position`; if `forwardImpulse !== 0`, `chassisBody.applyImpulse(forwardWS[i] * forwardImpulse, rel_pos)` (L529–533, at the contact point); then, if `sideImpulse !== 0`: `rel_pos` is taken to the chassis frame, its `indexUpAxis` component is multiplied by `rollInfluence`, it is taken back to the world frame (L546–548), `chassisBody.applyImpulse(axle[i] * sideImpulse, rel_pos)` (L549), and the opposite impulse is applied to the ground body at `hitPointWorld − ground.position` (L552–553; nothing happens for a static ground).
   - All side and forward impulses are computed from one velocity state and applied afterwards; `applyImpulse` is linear, so a batched port may sum them. The summation order (wheel 0 forward, wheel 0 side, wheel 1 forward, …) matters only for floating-point rounding.
7. Visual wheel spin (L157–197): updates `deltaRotation` and `rotation`. See (g).

### (f) `castRay`

`castRay(wheel)` (L249–323):

```ts
const raylen = wheel.suspensionRestLength + wheel.radius          // L258
wheel.directionWorld.scale(raylen, rayvector)
const source = wheel.chassisConnectionPointWorld
source.vadd(rayvector, target)                                    // L260–262
raycastResult.reset()                                             // L267: body null, hit point and normal zero, distance −1
chassisBody.collisionResponse = false                             // L270, so the ray ignores the chassis; restored at L274
this.world!.rayTest(source, target, raycastResult)                // L273
```

If a body was hit (L280–313):

```ts
wheel.isInContact = true
wheel.suspensionLength = hitDistance - wheel.radius               // L286
const minSuspensionLength = wheel.suspensionRestLength - wheel.maxSuspensionTravel
const maxSuspensionLength = wheel.suspensionRestLength + wheel.maxSuspensionTravel
// L291–297: clamp suspensionLength to [min, max]; on the upper clamp also wheel.raycastResult.reset()
const denominator = wheel.raycastResult.hitNormalWorld.dot(wheel.directionWorld)   // L299
chassisBody.getVelocityAtWorldPoint(wheel.raycastResult.hitPointWorld, chassis_velocity_at_contactPoint)
const projVel = wheel.raycastResult.hitNormalWorld.dot(chassis_velocity_at_contactPoint)   // L304, chassis velocity only
if (denominator >= -0.1) {
  wheel.suspensionRelativeVelocity = 0
  wheel.clippedInvContactDotSuspension = 1 / 0.1
} else {
  const inv = -1 / denominator
  wheel.suspensionRelativeVelocity = projVel * inv
  wheel.clippedInvContactDotSuspension = inv
}
```

Otherwise (L315–319): `suspensionLength = suspensionRestLength`, `suspensionRelativeVelocity = 0`, `hitNormalWorld = −directionWorld`, `clippedInvContactDotSuspension = 1`; `isInContact` stays false and `raycastResult.body` stays null.

Notes: `hitDistance` is the world distance from the ray origin to the hit point (`collision/Ray.ts` L700). Because the ray is only `restLength + radius` long, `suspensionLength ≤ restLength`, so the upper clamp (and its `reset()`) is unreachable for `maxSuspensionTravel ≥ 0`; the lower clamp limits compression to `maxSuspensionTravel`. The velocity of the ground body is ignored in `projVel`. The branch `denominator >= -0.1` is taken when the hit normal is tilted more than about 84.3° away from the reversed ray direction (cos⁻¹ 0.1); it sets the slope correction to 10 and the relative velocity to 0.

The ray itself (`World.rayTest` → `raycastClosest` with `{ skipBackfaces: true }`, `World.ts` L311–314 and L349–355; `Ray.intersectWorld`, `Ray.ts` L194–222):

- Bodies with `collisionResponse` false are skipped (L235); a shape is skipped when the distance from its position to the infinite ray line exceeds its `boundingSphereRadius` (L297–300).
- `_intersectTrimesh` (L589–692) works in the mesh's local frame. For each candidate triangle from the octree (an AABB query with the ray's box, `utils/Octree.ts` L166–174): `scalar = normal.dot(vector) / dot` with `vector = a − localFrom` and `dot = localDirection · normal` (L654–662); `if (scalar < 0) continue` (L665); the point `localFrom + localDirection * scalar` must pass `Ray.pointInTriangle(p, b, a, c) || Ray.pointInTriangle(p, a, b, c)` and lie within the ray length (`squaredDistance > fromToDistanceSquared` → skip, L679–684). There is no parallel-ray test (it is commented out, L656–659). `pointInTriangle` (L741–757) is the barycentric test `u >= 0 && v >= 0 && u + v < dot00 * dot11 - dot01 * dot01` without division.
- The face normal is the stored one, `normalize((b − a) × (c − a))` for indices (a, b, c) (`shapes/Trimesh.ts` L155–176 and L230–237), rotated to world space; it is **not** flipped towards the ray.
- `reportIntersection` (L697–735): `if (this.skipBackfaces && normal.dot(this.direction) > 0) { return }` (L704). In `CLOSEST` mode a hit replaces the stored one only if `distance < result.distance || !result.hasHit` (L720), so on an exact tie the first candidate in octree order wins.
- Consequence for the track mesh: a downward wheel ray only hits triangles whose stored normal points up, so triangles must be wound counter-clockwise seen from above.

### (g) State that survives from one tick to the next

- **Inputs**, which persist until set again: `wheel.steering`, `wheel.engineForce`, `wheel.brake`.
- **`wheel.rotation` and `wheel.deltaRotation`** (L182–196). `rotation` enters `worldTransform.quaternion` (step 1 of (e)), and that quaternion gives the axle used for friction (L421). The spin is a rotation about the normalised `axleLocal`, so it leaves `directions[indexRightAxis]` unchanged **only if `axleLocal` is parallel to that axis**. That holds in pmndrs/racing-game (`axleLocal` (−1, 0, 0) with `indexRightAxis` 0) but not with the cannon-es defaults if `axleLocal` is chosen along x while `indexRightAxis` stays 2. When it holds, the port can compute the axle as `chassisQ · steeringOrn · e_right` and drop the wheel spin from the dynamics.
- **`wheel.sliding`** is rewritten only for wheels on the ground (L496–499), so an airborne wheel keeps its last value; it is read only for the visual spin (L185).
- **`wheel.suspensionLength`** from the previous tick is read only for `worldTransform.position` (drawing).
- Everything else that the dynamics read is recomputed each tick before use: `isInContact`, `raycastResult`, `suspensionLength`, `suspensionRelativeVelocity`, `clippedInvContactDotSuspension`, `suspensionForce`, `sideImpulse`, `forwardImpulse`, `skidInfo`, `vehicle.sliding`, `numWheelsOnGround`.
- `forwardWS[i]` and `axle[i]` live in module-level arrays shared by all vehicles (L575–576); stale entries of airborne wheels are never used because their impulses are 0.

So with the axle condition above, the vehicle has no hidden dynamic state: the next state is a function of the chassis state, the three inputs per wheel and the track.

### (h) Which shape pairs involve `Trimesh`

`world/Narrowphase.ts` defines only `sphereTrimesh` and `planeTrimesh` (L50–51, getters L147–152); `convexTrimesh` is commented out (L153–155, L1907). The resolver is looked up by `(si.type | sj.type)` (L383–384), and a box (type 4) against a trimesh (type 256) has no entry, so no contact is generated. Rays do intersect a trimesh (`Ray.ts` L171–173).

### (i) Where cannon-es differs from Bullet's `btRaycastVehicle`

A port from Bullet's code or from descriptions of it would be wrong in these places:

- Engine and brake: Bullet uses the engine impulse when `m_engineForce != 0` and the brake/rolling impulse otherwise (`btRaycastVehicle.cpp` L561–572); cannon-es adds the two (L461–469).
- Side impulse: Bullet's `resolveSingleBilateral` uses `jacDiagABInv`, the effective mass including rotation (`btContactConstraint.cpp` L150–151); cannon-es uses `1 / (invMass1 + invMass2)` (L669).
- Both share the quirk that the sliding clamp is skipped for a wheel with `sideImpulse` exactly 0.

### (j) What differs from the review's description

1. The cap at `maxSuspensionForce` applies to the suspension impulse only; the grip limit uses the uncapped force (C8).
2. The side impulse is `−0.2 · mass · lateral contact velocity` per wheel, not a full cancellation (C7).
3. `brake` is an impulse cap per step, added to `engineForce · dt`; the grip test weights the forward impulse by 0.5 and divides by `forwardAcceleration` and `sideAcceleration` (racing-game sets `sideAcceleration` to 3, missing from the review, C9).
4. The friction axle comes from the wheel's `worldTransform` quaternion and `indexRightAxis`, so `wheel.rotation` is dynamic state unless `axleLocal` is parallel to that axis; the drive direction is `hitNormal × axle`, which is why racing-game drives forward with a negative engine force.
5. The chassis inertia is the box inertia of the body's world AABB at the time shapes are added, not necessarily of the box shape itself.
6. Trimesh hits depend on triangle winding (back faces are skipped using the stored, unflipped normal).
