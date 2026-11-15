# Car physics: what cannon-es does in one tick, and how the port mirrors it

The browser game runs the unmodified [cannon-es](https://github.com/pmndrs/cannon-es) 0.20.0 `RaycastVehicle` (MIT). The training simulator is a batched PyTorch port of the same code. This file is the specification both sides follow. Every statement was read from `node_modules/cannon-es/dist/cannon-es.js` (the file the game vendors); function names are given so each can be checked.

## Conventions

| Item | Value | Where it comes from |
|---|---|---|
| Up axis | +Y (`indexUpAxis = 1`) | `RaycastVehicle` constructor default |
| Forward axis | +X (`indexForwardAxis = 0`) | same |
| Right axis | +Z (`indexRightAxis = 2`) | same |
| Quaternion layout | `(x, y, z, w)` | `Quaternion` |
| Positive steering | turns the wheel's forward direction toward −Z, i.e. left | `updateWheelTransform`: rotation about `up = −directionLocal` |
| Tick | fixed `dt = 0.01 s`, called as `world.step(dt)` with one argument | `World.step` runs exactly one `internalStep(dt)` then |
| Wheel direction and axle | `directionLocal = (0, −1, 0)`, `axleLocal = (0, 0, 1)` | our vehicle setup |

With `axleLocal` parallel to the right axis, the wheel's spin angle (`wheel.rotation`) does not change the axle direction, so it is visual only. No wheel field carries dynamic state from one tick to the next: the dynamics depend only on the chassis state (position, quaternion, linear and angular velocity) and on the inputs (steering, engine force, brake).

## One tick, in order (`World.internalStep`)

1. **Gravity** is added to the force accumulator: `force += mass · gravity`. It reaches the velocity only in step 5.
2. **Collision detection and solver.** Nothing happens: the chassis has `collisionFilterMask = 0`, and cannon-es has no box–trimesh contact pair anyway (`Narrowphase` only pairs a trimesh with spheres and planes).
3. **Damping:** `velocity *= pow(1 − linearDamping, dt)` and `angularVelocity *= pow(1 − angularDamping, dt)`.
4. **`preStep` event → `RaycastVehicle.updateVehicle(dt)`** (details below). It changes the velocities by impulses.
5. **`Body.integrate`:** `velocity += force / mass · dt`; `angularVelocity += invInertiaWorld · torque · dt` (torque is zero); `position += velocity · dt`; quaternion update `q += dt/2 · (ω ⊗ q)` with the component formula of `Quaternion.integrate`; full normalisation every tick (`quatNormalizeSkip = 0`, `quatNormalizeFast = false`); then `updateInertiaWorld`.
6. Forces are cleared.

`invInertiaWorld = R · diag(invInertia) · Rᵀ` with `R` from the current quaternion (`Body.updateInertiaWorld`). `invInertia` is the inverse of a box inertia computed from the half extents of the body's bounding box when the shape is added (`Body.updateMassProperties`, `Box.calculateInertia`): `I_x = m/12 · (4e_y² + 4e_z²)` and cyclic.

## `updateVehicle(dt)`

For each wheel `w` (all quantities in world space unless noted):

**Ray cast (`castRay`).**
- `connection = position + R · connectionLocal`, `direction = R · directionLocal`.
- The ray runs from `connection` to `connection + direction · (restLength + radius)`.
- `World.rayTest` returns the **closest** hit and **skips back faces** (a hit is dropped when `normal · rayDirection > 0`).
- No hit: `suspensionLength = restLength`, `suspensionRelativeVelocity = 0`, `clippedInvContactDotSuspension = 1`, no forces.
- Hit: `suspensionLength = hitDistance − radius`, raised to `restLength − maxSuspensionTravel` if below it.
  - `denominator = hitNormal · direction`; `projVel = hitNormal · (velocity + ω × (hitPoint − position))`.
  - If `denominator >= −0.1`: `suspensionRelativeVelocity = 0`, `clippedInvContactDotSuspension = 10`.
  - Otherwise `inv = −1 / denominator`, `suspensionRelativeVelocity = projVel · inv`, `clippedInvContactDotSuspension = inv`.

**Suspension (`updateSuspension`).** For a wheel in contact:
`force = stiffness · (restLength − suspensionLength) · clippedInvContactDotSuspension − damping · suspensionRelativeVelocity`, where `damping = dampingCompression` if the relative velocity is negative, else `dampingRelaxation`. Then `suspensionForce = max(force · chassisMass, 0)`.

**Suspension impulses.** `impulse = hitNormal · min(suspensionForce, maxSuspensionForce) · dt`, applied at `hitPoint − position` with `Body.applyImpulse`: `velocity += impulse / mass`, `angularVelocity += invInertiaWorld · (r × impulse)`. All four are applied before friction is computed.

**Friction (`updateFriction`)**, using the velocities *after* the suspension impulses:
- Axle: `axle = R · (rotation about +Y by steering) · (0,0,1)`, then made perpendicular to the hit normal and normalised. `forward = hitNormal × axle`, normalised.
- Side impulse (`resolveSingleBilateral`): `side = −0.2 · (axle · contactVelocity) · chassisMass`. The 0.2 is a constant in the code; `sideFrictionStiffness2 = 1`.
- Forward impulse (`calcRollingFriction`): `j = −(forward · contactVelocity) / denom`, clamped to `[−brake, +brake]`, with `denom = 1/mass + forward · ((invInertiaWorld · (r × forward)) × r)`. Then `forwardImpulse = j + engineForce · dt`. With `brake = 0` there is no rolling resistance at all. `brake` is an impulse per tick, not a force.
- Friction limit: `maximp = suspensionForce · dt · frictionSlip` (the suspension force here is **not** capped by `maxSuspensionForce`). With `x = forwardImpulse · 0.5 / forwardAcceleration` and `y = side / sideAcceleration`, the wheel is sliding when `x² + y² > maximp²`, and then `skid = maximp / sqrt(x² + y²)`.
- If **any** wheel is sliding, every wheel with `side != 0` and `skid < 1` has both impulses multiplied by its `skid`.
- Impulses are applied at the hit point. For the side impulse the lever arm is first taken to the chassis frame, its up component multiplied by `rollInfluence`, and taken back.

## Grip limit

The friction limit above is an ellipse, not a circle: a wheel slides only when `(forwardImpulse · 0.5 / forwardAcceleration)² + (side / sideAcceleration)² > maximp²` (`web/vendor/cannon-es/cannon-es.js`, `updateFriction`). With `side_acceleration` = 3 and `friction_slip` = 1.5 (`web/assets/vehicle.json`), a wheel's side impulse can reach 3 × 1.5 = 4.5 times its suspension impulse before it slides, so on flat ground, with the wheels carrying the car's weight, the sideways grip allows up to about 4.5 g (≈ 44 m/s²). The side impulse itself removes only 20 % of the sideways contact velocity per tick (`resolveSingleBilateral` above), so this is an upper bound. The scripted driver plans corners for 6 m/s² (`results/tracks/scripted_tuning.json`), far below it, which leaves room for the faster cornering of the trained agents. Neither exploit detector looks at cornering grip.

## Triangle mesh details that affect parity

- `Trimesh` stores vertices in a `Float32Array`, indices in an `Int16Array` (at most 32,767 vertices per mesh) and face normals in a `Float32Array`. The port therefore rounds vertices and normals to 32-bit floats too, even in the 64-bit parity tests.
- The face normal of triangle `(A, B, C)` is `normalize((B − A) × (C − A))` (`Trimesh.updateNormals` calls `computeNormal(vb, va, vc)`). Track triangles are wound so that normals point up.
- Ray against triangle (`Ray._intersectTrimesh`): distance along the ray `scalar = normal · (A − from) / (direction · normal)`; skipped if negative; the point must pass `pointInTriangle(p, B, A, C) || pointInTriangle(p, A, B, C)` and lie within the ray length. `pointInTriangle` is the barycentric test without division.
- A ray only reaches a body's shape if it passes within the shape's bounding sphere and the body has `collisionResponse = true` and a matching filter mask; the track body keeps the defaults.

## Known quirks, left in on purpose (candidate exploits)

| Quirk | Cause |
|---|---|
| Straight-line braking is not limited by grip | The skid scaling runs only for wheels with `sideImpulse !== 0` |
| A wheel can sink into a ramp lip, and the suspension can bottom out on hard landings | One ray per wheel; `suspensionLength` is clamped at `restLength − maxSuspensionTravel` and the chassis itself touches nothing |
| No rolling resistance or engine braking under throttle | `maxImpulse = brake` is zero when the brake is off, so with gas held only the 1 % per second velocity damping slows the car. The game's controls apply `coast_brake_impulse` (1 N·s per wheel per tick) while coasting as a small rolling resistance (`web/js/core/controls.js`, `src/racingmania/sim/controls.py`) |
| One sliding wheel reduces grip on all wheels | The vehicle-wide `sliding` flag |
| Reported launch when driving slowly over a raised body | cannon-es pull request #215 (unmerged) |
| A wheel touching a surface at a steep angle pushes up to 10× harder, undamped | The spring force is multiplied by `clippedInvContactDotSuspension` = −1/(hit normal · ray direction); when that dot product is ≥ −0.1 (more than about 84° off) the factor is clamped to 10 and `suspensionRelativeVelocity` is set to 0, so no damping (`web/vendor/cannon-es/cannon-es.js`, `updateSuspension` and `castRay`; the same logic as Bullet's `btRaycastVehicle`). The three energy flags of the first generalist on training-sample tracks (2.5–7 J/kg) were replayed step by step: all were rebounds after a violent impact in runs that failed. Every flag so far is in a run that did not finish; later flags have not been replayed one by one (counts per run: `reports/summary.md`, "Evaluator flags") |

## Port structure (`src/racingmania/sim/`)

The port computes the same quantities for `N` cars at once as tensors, applies the impulses one wheel at a time in wheel order exactly as cannon-es does (each wheel's suspension impulse, then each wheel's forward and side friction impulses), because `Body.applyImpulse` changes the velocities at once and the floating-point sums must happen in the same order (`src/racingmania/sim/raycast_vehicle.py`), and casts each wheel ray against the triangles of the track segments near the car instead of the whole mesh. The parity tests in `tests/parity/` feed random chassis states and inputs to both implementations and require the next state to agree to 1e-9 in 64-bit floats.
