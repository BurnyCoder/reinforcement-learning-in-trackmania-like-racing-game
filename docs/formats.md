# Shared file formats and conventions

The browser game (JavaScript) and the training code (Python) read the same files. This document is the contract between them. If a format changes, both sides and this file change in the same commit.

## Coordinates and units

- Right-handed, **+Y up**, **+X forward** at yaw 0, **+Z right**. Metres, seconds, kilograms, radians.
- Quaternions are stored as `[x, y, z, w]`.
- Yaw is the rotation about +Y; positive yaw turns +X toward −Z (to the left).
- Physics tick `dt = 0.01 s`. The agent and the keyboard act once every `ticks_per_action = 5` ticks (20 actions per second).

## Actions

One action is a pair `(steer, pedal)`:

| Index | `steer` | `pedal` |
|---|---|---|
| 0 | left | gas |
| 1 | none | coast |
| 2 | right | brake |

A single integer `a = steer * 3 + pedal` (0–8) is used where one number is needed, for example in stored action sequences.

How an action becomes wheel inputs (identical in `web/js/core/controls.js` and `src/racingmania/sim/controls.py`; parameters in `vehicle.json` → `controls`):

1. `speed` = chassis velocity projected on the chassis forward axis.
2. Steering target = `{left: +1, none: 0, right: −1} × max_steer × steer_scale(speed)` with `steer_scale = max(min_steer_fraction, 1 − steer_speed_falloff × |speed| / top_speed)`.
3. The steering angle moves toward the target by at most `steer_rate × dt` per tick and is applied to the front wheels.
4. Gas: engine force `engine_force × max(0, 1 − speed / top_speed)` on each drive wheel, brake 0.
5. Brake: if `speed > reverse_speed_threshold`, brake impulse `brake_impulse` on every wheel and engine force 0; otherwise reverse: engine force `−reverse_force × max(0, 1 + speed / reverse_top_speed)` on each drive wheel.
6. Coast: engine force 0 and brake impulse `coast_brake_impulse` on every wheel.

## `web/assets/vehicle.json`

```json
{
  "version": 1,
  "dt": 0.01,
  "ticks_per_action": 5,
  "gravity": [0, -9.82, 0],
  "chassis": { "mass": 500, "half_extents": [2.35, 0.55, 1.0], "linear_damping": 0.01, "angular_damping": 0.01 },
  "wheel": {
    "radius": 0.38,
    "suspension_stiffness": 30,
    "suspension_rest_length": 0.35,
    "max_suspension_travel": 0.3,
    "damping_compression": 10,
    "damping_relaxation": 10,
    "friction_slip": 1.5,
    "roll_influence": 0.01,
    "max_suspension_force": 100000,
    "forward_acceleration": 1,
    "side_acceleration": 3,
    "direction": [0, -1, 0],
    "axle": [0, 0, 1]
  },
  "wheel_connections": [[1.35, -0.3, -0.85], [1.35, -0.3, 0.85], [-1.3, -0.3, -0.85], [-1.3, -0.3, 0.85]],
  "front_wheels": [0, 1],
  "drive_wheels": [2, 3],
  "controls": {
    "max_steer": 0.3,
    "steer_rate": 1.5,
    "steer_speed_falloff": 0.6,
    "min_steer_fraction": 0.35,
    "engine_force": 2000,
    "top_speed": 45,
    "brake_impulse": 15,
    "coast_brake_impulse": 1,
    "reverse_force": 1000,
    "reverse_top_speed": 10,
    "reverse_speed_threshold": 0.5
  },
  "spawn_height": 0.95
}
```

Wheel order is front-left, front-right, rear-left, rear-right. This is the file as committed (unchanged since its first commit); the simulator freeze in `reports/preregistration.md` (amendments) records its SHA-256 prefix, `37bc8a37f90ab12d`, as physics version `v1`.

## Track file (`web/assets/tracks/<name>.json`)

```json
{
  "format": 1,
  "name": "campaign-01",
  "seed": null,
  "spacing": 2.0,
  "half_width": 6.0,
  "center": [[x, y, z], ...],
  "left":   [[x, y, z], ...],
  "gap":    [0, 0, 1, ...],
  "floor":  [y, y, ...],
  "checkpoints": [40, 95, 160, 211],
  "vertices": [[x, y, z], ...],
  "triangles": [[a, b, c], ...],
  "triangle_segment": [0, 0, 1, 1, ...],
  "start": { "position": [x, y, z], "quaternion": [qx, qy, qz, qw], "index": 3 },
  "reference": { "ticks": 2345, "time": 23.45, "certified": true, "actions": [0, 0, 3, ...] },
  "length": 1234.5,
  "generator": { "version": 1, "split": "train", "variant": "train_ranges", "attempt": 0 },
  "plan": [ { "kind": "straight", "length": 40.0, "dh": 0.0 }, ... ]
}
```

- `center[i]` is the centreline sample at horizontal arc length `i × spacing` (progress, below, uses the 3D length); there are `M` samples. The centreline is defined through gaps too (it follows the intended flight path), so progress is always measurable.
- `left[i]` is the unit vector pointing to the left edge at sample `i`. It includes banking. The surface between samples `i` and `i+1` is the quad with corners `center ± half_width × left`.
- `gap[i] = 1` means the segment from sample `i` to `i+1` has no surface. `gap` has `M − 1` entries.
- `floor[i]`: a car whose chassis centre is below this height while its nearest sample is `i` is out. It is `center[i].y − 3` on road samples, and inside a gap `min(lip height, landing height) − 3`.
- `checkpoints` are centreline sample indices in driving order; the last one is the finish, 30 m before the end of the road.
- `vertices` has two entries per sample: `vertices[2i] = center[i] + half_width × left[i]` (left edge) and `vertices[2i+1] = center[i] − half_width × left[i]` (right edge). Every number is rounded to 32-bit float before it is written, because cannon-es stores mesh vertices as `Float32Array`.
- `triangles`: two per non-gap segment `i`, wound so that normals point up: `[2i, 2i+1, 2i+2]` and `[2i+1, 2i+3, 2i+2]`. `triangle_segment[k]` is the segment index of triangle `k`.
- `start.position` is the chassis centre at rest above sample `start.index` (`spawn_height` along the road normal); `start.quaternion` aligns chassis +X with the road direction and +Y with the road normal.
- `plan` is the segment list the track was built from (`src/racingmania/tracks/ribbon.py`), kept for reproducibility.
- `reference` is the frozen scripted driver's run from the standing start: its finishing time in ticks and seconds, `certified` (it finished at all three speed scales), and, for the campaign tracks, its action per agent step (integers 0–8); the held-out tracks exported to the game have no `actions`. Absent until the track is certified.
- A track has at most 16,000 samples, so the mesh stays within the 16-bit index limit of cannon-es.

The training pools are the same data packed into float32/integer tensors (`<data dir>/tracks/<set>.pt`, keys in `src/racingmania/tracks/store.py`): `center [T, M, 3]`, `left [T, M, 3]`, `floor [T, M]`, `gap [T, M−1]`, `n [T]` (samples per track), `checkpoints [T, C]`, `n_cp [T]`, `start_pos`, `start_quat`, `start_index`, `half_width [T]`, `ref_ticks [T]`, `seeds [T]`, plus `names` and `meta`, padded to a common `M`. The port derives the triangles from `center`, `left` and `half_width` with the same formulas, rounding to 32-bit floats first.

## Rules (identical on both sides)

Evaluated once per tick, after the physics step.

- **Progress.** `index` is the nearest centreline sample to the chassis centre (squared 3D distance), searched only in `[index − 2, index + 6]`. Arc-length progress is `s = cum[k] + t × (cum[k+1] − cum[k])`, where `cum` is the cumulative 3D length of the centreline, segment `k ∈ {index − 1, index}` is the one whose clamped projection of the chassis centre is nearer, and `t ∈ [0, 1]` is the position along it.
- **Checkpoints.** Checkpoint `k` is passed when `s >= cum[checkpoints[k]]` and all earlier ones are passed; several can be passed in one tick. The run is **finished** when the last one is passed; finishing is checked before the out conditions.
- **Out** (the run ends unfinished) when any of these holds:
  - the chassis centre is below `floor[index]`;
  - no wheel has touched the track for 500 consecutive ticks;
  - progress has not grown by at least 0.5 m during 200 consecutive ticks.
- A car that tips over has no wheel rays pointing at the road, falls through the ribbon and is out by the first rule, so there is no separate roll-over rule.
- In the game an out respawns the player at the last passed checkpoint with the clock running. In training and evaluation it ends the episode.

## Policy model (`web/assets/models/<name>.onnx`)

One ONNX graph per trained model contains the observation function, the frozen observation normaliser and the actor network (`src/racingmania/export/onnx_export.py`). The browser only gathers raw inputs, runs the graph once per agent step, and takes the argmax of each head. All inputs are float32 with batch size 1.

| Input | Shape | Content |
|---|---|---|
| `chassis` | `[1, 13]` | chassis state `[px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]` |
| `steer` | `[1, 1]` | current front-wheel angle (rad) |
| `contacts` | `[1, 4]` | 1 if the wheel's suspension ray hit the track on the last tick, else 0, in wheel order |
| `prev_action` | `[1, 2]` | previous `(steer, pedal)` indices as floats (`(1, 1)` at the start) |
| `win_center` | `[1, K, 3]` | `center[min(index + 3k, M−1)]` for `k = 0..K−1` (`index` = the rules' nearest sample) |
| `win_left` | `[1, K, 3]` | `left` at the same samples |
| `win_gap` | `[1, K]` | `gap` of the segment starting at the same samples (0 at the last sample) |

| Output | Shape | Content |
|---|---|---|
| `steer_logits` | `[1, 3]` | scores for left / none / right |
| `pedal_logits` | `[1, 3]` | scores for gas / coast / brake |

`K` and the stride (3 samples = 6 m) are stored with each model's entry in `web/assets/index.json` (`models: [{"name", "file", "K", "stride", "description", "run"}]`), so look-ahead ablations export correctly. The action is held for `ticks_per_action` ticks.

## Asset index (`web/assets/index.json`)

`tracks: [{"name", "title", "kind", "note"?}]` lists the tracks in the menu (kind `campaign`, or `unseen` for held-out test tracks added by `scripts/12_export_unseen_track.py`, whose `note` says how the track was chosen); `models` lists exported policies (above); `ghosts: {"<track>": ["assets/ghosts/<track>/<model>.json", ...]}` lists AI ghost files, each `{"model": "<model name>", "time": seconds, "finished"?: bool, "end"?: "<end reason>", "states": [[x,y,z,qx,qy,qz,qw,steer], ...]}` in the replay format below. Ghosts are official evaluation runs; a ghost with `"finished": false` (allowed for controls and for the README's failure clips) ends at `time` with its evaluator end reason (`fell`, `airborne`, `stalled`) instead of a finish. The game's menu merges `models` and a track's ghosts by model name into one AI drivers table (a ghost with no matching model is a recorded-only driver), and "Watch replay" plays the chosen driver's ghost (`web/js/replay.js`). `scripts/06_export_web.py` maintains `models` and `ghosts`.

## Results (`results/<experiment>/<run>/`)

| File | Content |
|---|---|
| `run.json` | Status (`planned`, `training`, `trained`, `evaluated`), config, seed, git commit, physics version, hardware, step budget, wall-clock time. Run-matrix entries with no `run.json` are reported as `not started`, or `postponed` while `configs/run_matrix.yaml` has a `paused` block (`{date, note}`; the run queue does not start while it is present) |
| `train_metrics.csv` | One row per PPO iteration: `iteration, global_step, wall_hours, sps, lr, explained_variance, kl, clipfrac, pg_loss, v_loss, entropy, episodes, ep_return, ep_finish_rate, ep_finish_rate_standing, ep_progress, ep_fell, ep_stalled, ep_steps` |
| `val_metrics.csv` | One row per validation evaluation (every 10 minutes): `sws, finish_rate, speed_ratio_finished, n_tracks, flags_overspeed, flags_energy_gain, iteration, global_step, wall_hours` |
| `eval_<set>.csv` | One row per track: `track` (name, e.g. `test-2000000`), `finished`, `ticks`, `time`, `ref_ticks`, `score`, `end` (`finished`, `fell`, `airborne`, `stalled`, `timeout`; the CEM planner writes `not_finished`), `progress` (arc-length position on the centreline in m; starts at the start sample, about 6 m), `max_speed`, `flag_overspeed`, `max_coast_energy_gain` (J/kg), `flag_energy_gain` |
| `failures_<set>.json` | `scripts/09_failure_analysis.py`: sharpest turn radius and jumps in the 60 m before each fall (`failures`) against random places on the same tracks (`random`) |
| `replays/<track>.json` | State trajectories for the game's ghosts: `states[0]` is the start state and `states[k]` the state after `k` agent steps, each `[x, y, z, qx, qy, qz, qw, steer]` |

Other result files: `results/test_access_log.jsonl` (one line per read of a protected set: `time`, `split`, `policy`, `tracks`), `results/parity.json` (port-versus-cannon-es differences from `scripts/parity_report.py`), and `results/tracks/` (per-set summaries, per-track features and the scripted driver's tuning grid).

Checkpoints and TensorBoard logs are large and live under `RACINGMANIA_DATA_DIR`. Each run's final checkpoint (`final.pt`) is uploaded to its Hugging Face model repository with `policy.onnx`, `run.json`, the metric CSVs and the evaluation CSVs; resume checkpoints and TensorBoard logs stay local.

## Video clips (`docs/videos/`)

`scripts/11_record_videos.py` records clips listed in a clips file (`docs/videos/clips.yaml` for the README, `docs/videos/cards/clips.yaml` for the Hugging Face model cards) and writes `videos.json` next to them: one entry per clip with its title, `replay` (the ghost), `run`, `split`, `track`, the official result (`official`: the run's row of `eval_<split>.csv`), the browser's time (`game_s`, `game_finished`), the files (`mp4`, and `gif` for README clips) and `caption` (`What you see`, `Model`, `Map`, `Evaluation`, and `Why this clip` or `Over many runs` where they apply). The README's video section and the Hugging Face cards are written from these entries (`report/cards.py`: model cards play a run's clips from the model repository's `videos/` folder; the results dataset card plays the README clips).

