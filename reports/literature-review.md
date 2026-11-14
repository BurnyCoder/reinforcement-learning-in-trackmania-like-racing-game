# Literature review

**Project:** RacingMania — an original 3D time-attack racing game for the browser, and reinforcement-learning (RL) drivers trained on a batched GPU port of its car physics.
**Author:** Libor "Burny" Burian. **Date:** 2026-11-14.
**Status:** first full pass, written from seven search-and-read passes over papers, repositories, documentation and model hubs. A separate fact-check pass re-reads every source and records the exact supporting quote in [`source-ledger.md`](source-ledger.md); corrections from that pass are listed in [section 15](#15-fact-check-log).

**How to read the evidence labels.** `[code]` = read in the repository's source. `[paper]` = read in the paper's text. `[docs]` = official documentation. `[2nd]` = secondary source only. `[unverified]` = not confirmed from a primary source; nothing in the design depends on such an item.

## 1. Questions this review answers

1. How have RL agents been trained for TrackMania and other racing games, and what reward, observation, action and start-state designs worked or failed?
2. What is known about training one policy on many tracks and testing it on unseen ones?
3. Which algorithm and hyperparameters suit thousands of GPU-parallel environments with a small discrete action space?
4. How should results be evaluated and reported (seeds, intervals, generalisation protocol, exploits)?
5. Which existing, tested car physics, track formats, baselines and browser tooling can be reused instead of written?

## 2. RL agents for TrackMania and Gran Turismo

| Work | Algorithm | Observation | Actions | Reward | Rate | Result |
|---|---|---|---|---|---|---|
| Linesight, TrackMania Nations Forever `[code]` ([repo](https://github.com/Linesight-RL/linesight), [docs](https://linesight-rl.github.io/linesight/build/html/)) | IQN with dueling heads | 160×120 grey image plus floats: last 5 actions, car-frame linear and angular velocity, map up-vector in the car frame, 40 upcoming reference-line points in the car frame, distance to finish | 12 discrete key combinations | −6/5000 per ms plus 5/500 per metre advanced along a reference line; hand-engineered bonuses all set to 0 | one action per 50 ms | Reported to beat 10 of 12 campaign world records |
| TMRL, TrackMania 2020 `[code]` ([repo](https://github.com/trackmania-rl/tmrl), MIT) | SAC, optionally REDQ | 19 lidar beams over screenshot pixels (history of 4) plus speed and 2 previous actions, or 4 stacked 64×64 frames | 3 analog axes (gas, brake, steer) | points passed on a demonstration trajectory per step; +100 at the end of the track | 20 Hz | About 45.5 s on its test track against a target of about 32 s |
| Neinders 2023, BSc thesis `[paper]` ([pdf](https://essay.utwente.nl/96153/1/Neinders_BA_EEMCS.pdf)) | SAC (TMRL) | TMRL lidar replaced by 15 upcoming track points in the car frame | as TMRL | progress only | 20 Hz | 44.631 s → 35.465 s; best human 33.798 s |
| Fuchs et al. 2021, Gran Turismo Sport `[paper]` ([arXiv 2008.07971](https://arxiv.org/abs/2008.07971)) | SAC | velocity, acceleration, heading to centreline, 13 range finders, previous steering, wall-contact flag, 10 curvature samples 1.0–2.8 s ahead | steering and one combined throttle/brake axis | centreline progress minus c·‖v‖² while touching a wall, c = 5e-4 | 10 Hz | Beat the fastest human in all three settings after 56–73 h |
| GT Sophy, Wurman et al. 2022 `[paper]` ([Nature](https://doi.org/10.1038/s41586-021-04357-7), [pdf](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf)) | QR-SAC | 60 equally spaced 3D points along each track edge and the centre line about 6 s ahead, car state, flags, last action | steering and combined throttle/brake | progress, masked off course; off-course and wall penalties proportional to speed squared | 10 Hz | Faster than top human drivers |

Details that matter for this project:

- **Linesight's horizon trick.** Each transition belongs to a "mini-race" of random length that ends at 7 s, which lets the agent be trained with a discount of 1 over a fixed 7-second horizon. Episodes are cut off after 2 s without progress along the reference line, and at 300 s. `[code]`
- **Linesight has no licence file.** The GitHub API reports `license: null`. Its ideas can be used; its code cannot be copied. `[code]`
- **What failed in Fuchs et al.** Without a wall penalty the agent ground along walls; a fixed penalty was either ignored or led to full braking and standing still; PPO needed much more data than SAC; a behaviour-cloning warm start did not help; on an unseen track the agent got stuck in a hairpin. `[paper]`
- **What GT Sophy's ablations show.** Raycast-plus-curvature inputs were clearly slower (117.11 s) than track points ahead (114.47 s); acting faster than 10 Hz gave no substantial gain. `[paper]`
- **Other TrackMania projects.** PedroAI (DQN, 100 ms steps, reference trajectory in the car frame, map changed every 10 rounds) warns that a finish bonus can make the agent take longer routes to build speed ([blog](https://www.trackmania.ai/blog/posts/pedroai-rl/pedroai-rl.html)). Yosh's widely watched agents have no published code or write-up; everything known about them is `[2nd]`.

## 3. RL for minimum-time racing in fast parallel simulators

Drone racing is the closest published setting to ours: PPO, a progress reward, a fast simulator.

| Work | Scale | Observation of the course | Reward | Start states | Notes |
|---|---|---|---|---|---|
| Song et al. 2021 `[paper]` ([arXiv 2103.08624](https://arxiv.org/abs/2103.08624)) | PPO, up to 100 parallel environments | next two gates; with only one gate the crash ratio was 23.0 % | progress along the gate-to-gate segment, plus a safety term and a body-rate penalty, minus a crash penalty | hover states around the centres of all segments, later sampled from previous trajectories | Random tracks from concatenated gate primitives with complexity adapted from the crash ratio; 97.4 % success on 1,000 random test tracks |
| Song et al. 2023 `[paper]` ([arXiv 2310.10943](https://arxiv.org/abs/2310.10943)) | PPO, "hundreds" of environments | four corners of the next two gates relative to the drone | r = ‖g−p_prev‖ − ‖g−p‖ − b‖ω‖, b = 0.01, −10 on collision, +10 on finishing | random hover states around segment centres, later from a buffer of successful states | Lap time alone is too sparse a reward; a policy trained to track a time-optimal trajectory was worse than the progress policy |
| Kaufmann et al. 2023, Swift `[paper]` ([Nature, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)) | PPO, 100 agents, 1e8 steps in 50 min | four corners of the next gate, plus the previous action | progress + perception + command terms − crash (5.0, ends the episode) | each agent starts at a random gate | The reward weights are in an image table and were not read `[unverified]` |
| Green et al. 2026, preprint `[paper]` ([arXiv 2606.27348](https://arxiv.org/abs/2606.27348)) | PPO, 96 environments, 5e8 steps, many B-spline tracks | corners of the next two gates | progress, gate-pass, crash, rate and command terms | fixed, 2 m behind the first gate | See below |

**Green et al. 2026 is the closest prior study.** A generalist trained on many generated tracks finished 69.8 % of 40 unseen tracks, against 0.0 % for single-track policies; it was 14.52 % slower than per-track specialists; fine-tuning it on one track converged in 23.19 % of a specialist's training iterations. Feasibility-informed track generation scored about 2.05× uniform sampling; weight decay 1e-5 helped and 1e-4 or more hurt; an LSTM was 13.4 % worse than an MLP. It is a preprint and its numbers have not been independently reproduced.

Ground vehicles:

- **Assetto Corsa Gym** (Remonda et al. 2024, [arXiv 2407.16680](https://arxiv.org/abs/2407.16680)) uses SAC and TD-MPC2, with other tracks only as pre-training. `[paper]`
- **AWS DeepRacer** (Balaji et al. 2019, [arXiv 1911.01562](https://arxiv.org/abs/1911.01562)) trains PPO with 10 discrete actions and reports that mixing several tracks in training did not improve performance in its small sim-to-real setting. `[paper]`
- **F1TENTH.** Evans et al. 2023 ([arXiv 2306.07003](https://arxiv.org/abs/2306.07003)) reward closeness to a classical planner's speed and steering (r = 1 − |v−v_ref| − |δ−δ_ref|), which is the model for our "reward hints" hybrid. Brunnbauer et al. 2022 ([arXiv 2103.04909](https://arxiv.org/abs/2103.04909)) report that model-free agents failed on complex maps and that only Dreamer generalised. `[paper]`
- **Learn-to-Race** ([arXiv 2103.11575](https://arxiv.org/abs/2103.11575)): the SAC baseline completed 31.1 % of the training track and 11.8 % of an unseen one. `[paper]`
- **Gymnasium CarRacing** ([docs](https://gymnasium.farama.org/environments/box2d/car_racing/)): a new random track every episode, reward −0.1 per frame plus 1000/N per new tile. Robust PLR was evaluated zero-shot from generated Bézier tracks to 20 real F1 layouts (see section 5). `[docs]`
- **OfflineMania** ([arXiv 2407.09415](https://arxiv.org/abs/2407.09415)) and the JAX project `viik2k/trace` (no licence) supply further reward and termination recipes; only the ideas are usable. `[paper]` `[code]`

## 4. What this implies for reward, observation, actions and start states

| Design choice | What the literature supports | Our choice |
|---|---|---|
| Reward | Dense per-step progress along a centre or reference line is used by every successful project (Linesight, TMRL, Fuchs, GT Sophy, Song, Kaufmann). Lap time alone is too sparse. | Metres of centreline progress per step, nothing else |
| Extra reward terms | A crash penalty is standard in drone racing; fixed penalties caused stalling in Gran Turismo; a per-step time cost is used by Linesight but makes ending the episode early attractive when progress is slow (our inference) | Both are ablations, not part of the main reward |
| Observation | Upcoming track geometry as points in the vehicle frame beats raycasts (GT Sophy, Neinders); at least two upcoming gates are needed (Song 2021) | About 20 upcoming centreline points in the car frame, plus car state and previous action |
| Actions | Linesight reaches record level with 12 discrete keyboard combinations; DeepRacer trains PPO with discrete actions; GT Sophy and drones use continuous axes. No head-to-head comparison was found. Seyde et al. 2021 ([arXiv 2111.02552](https://arxiv.org/abs/2111.02552)) show on-off control is competitive on DeepMind Control tasks, but they did not test driving | Keyboard actions: steer left/none/right × gas/coast/brake; continuous actions are an ablation |
| Start states | Random starts along the course, later from earlier trajectories (Song 2021 and 2023, Kaufmann 2023); staggered resets help massively parallel PPO (Bharthulwar et al. 2025, [arXiv 2511.21011](https://arxiv.org/abs/2511.21011)) | Standing starts mixed with random starts along the centreline |
| Action rate | 10 Hz (Fuchs, GT Sophy), 20 Hz (Linesight, TMRL) | 20 Hz; 10 Hz is an ablation |
| Known failure modes | wall grinding, standing still, reward farming from shaping bonuses, corner cutting where checkpoints are missing, detours for a finish bonus, exploiting simulator drift | No walls, no shaping bonuses, ordered checkpoints, no finish bonus |

## 5. Zero-shot generalisation: protocol and known effects

- **Kirk et al. 2023** ([arXiv 2111.09794](https://arxiv.org/abs/2111.09794)) define the generalisation gap as training return minus test return and treat it as an auxiliary number, because a random policy also has a gap of zero. For procedurally generated environments they recommend training on a set of seeds and testing on held-out seeds, reporting several test sets (one per kind of generalisation, including extrapolation along controllable factors) and several training-set sizes. They give no rule for test-set size. `[paper]`
- **Cobbe et al. 2019, CoinRun** ([arXiv 1812.02341](https://arxiv.org/abs/1812.02341)): training sets from 100 to 16,000 levels; overfitting is still visible at 16,000; an unbounded set is best. **Cobbe et al. 2020, Procgen** ([arXiv 1912.01588](https://arxiv.org/abs/1912.01588)): about 10,000 levels are needed to close the gap; larger networks generalise better. `[paper]`
- **Level replay.** PLR (Jiang et al. 2021, [arXiv 2010.03934](https://arxiv.org/abs/2010.03934)) and robust PLR ([arXiv 2110.02439](https://arxiv.org/abs/2110.02439)); on CarRacing, zero-shot returns on 20 F1 tracks were 341±22 for domain randomisation, 408±12 for PLR and 534±7 for robust PLR. `[paper]`
- **In racing specifically,** unseen-track results are rare and weak before Green et al. 2026: Fuchs' agent got stuck, Learn-to-Race dropped from 31.1 % to 11.8 %, only Dreamer transferred in Brunnbauer et al.

**Consequences for our protocol:** disjoint seed ranges for training, validation and test tracks; a second test set with one factor beyond the training range; hand-made tracks as a third; a training-set-size sweep (1, 10, 100, 1,000, 50,000); test performance reported before the gap.

## 6. Algorithms for thousands of parallel environments

- **PPO baseline code.** CleanRL's `ppo_continuous_action_isaacgym.py` (MIT, with an NVIDIA BSD-3 header) defaults to 4,096 environments × 16 steps, learning rate 0.0026, γ 0.99, λ 0.95, 2 minibatches, 4 epochs, clip 0.2, value coefficient 2, gradient norm 1, separate 2×256 tanh actor and critic. It has no observation normalisation, checkpointing or time-limit bootstrapping ([source](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py)). `[code]`
- **Implementation details.** Huang et al. 2022 ([blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)): observation normalisation with clipping, reward scaling, independent action components for multi-discrete actions. `[paper]`
- **Massively parallel settings.** Rudin et al. 2021 ([arXiv 2109.11978](https://arxiv.org/abs/2109.11978)): 4,096 environments × 24 steps, minibatch 24,576, 5 epochs, adaptive learning rate from a KL target, and the critic's own prediction used to bootstrap on time-outs; fewer than about 25 steps per environment hurt. Makoviychuk et al. 2021 ([arXiv 2108.10470](https://arxiv.org/abs/2108.10470)): 4,096–16,384 environments, horizons 8–32. `[paper]`
- **Time limits.** Gymnasium's rule: bootstrap on truncation, not on termination ([docs](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)). `[docs]`
- **Libraries.** Stable-Baselines3 needs NumPy environments and PufferLib has no path for PyTorch environments, so neither fits a GPU simulator; rsl_rl supports only continuous distributions; skrl supports categorical ones. A single-file PPO adapted from CleanRL is the lowest-risk choice. `[docs]`
- **PQN** (Gallici et al., ICLR 2025, [arXiv 2407.04811](https://arxiv.org/abs/2407.04811)): parallel Q-learning with layer normalisation and λ-returns, no target network or large replay buffer; it beats PPO on Atari-10 in sample efficiency and wall-clock. CleanRL ships `pqn.py` (MIT), for discrete actions. `[paper]` `[code]`
- **Other recent methods** need continuous actions: Parallel Q-Learning (Li et al. 2023, [arXiv 2307.12983](https://arxiv.org/abs/2307.12983)), FastTD3 ([arXiv 2505.22642](https://arxiv.org/abs/2505.22642)), FastSAC ([arXiv 2512.01996](https://arxiv.org/abs/2512.01996)). Hamed et al. 2026 ([arXiv 2607.17326](https://arxiv.org/abs/2607.17326)) find that PPO with 2,048 environments often wins on wall-clock. `[paper]`

**Consequence:** PPO is the main algorithm, because the closest prior work uses it with a progress reward; PQN is the single alternative-algorithm baseline.

## 7. Evaluation and statistics

- Henderson et al. 2018 ([arXiv 1709.06560](https://arxiv.org/abs/1709.06560)): two groups of five seeds with identical settings can differ significantly. `[paper]`
- Colas et al. 2018 and 2019 ([arXiv 1806.08295](https://arxiv.org/abs/1806.08295), [arXiv 1904.06979](https://arxiv.org/abs/1904.06979)): prefer Welch's t-test; about 20 seeds give power 0.8 for an effect size of 1. `[paper]`
- Agarwal et al. 2021 ([arXiv 2108.13264](https://arxiv.org/abs/2108.13264)): interquartile mean, stratified-bootstrap 95 % intervals, performance profiles, probability of improvement; implemented in `rliable` (Apache-2.0; the repository was archived in October 2025 but version 1.2.0 installs). `[paper]` `[code]`
- Patterson et al. 2024 ([arXiv 2304.01315](https://arxiv.org/abs/2304.01315)): five runs are too few for strong claims; default to bootstrap intervals; give every algorithm the same tuning budget and re-run the chosen configuration on fresh seeds. `[paper]`
- **Success-weighted metric.** SPL (Anderson et al. 2018, [arXiv 1807.06757](https://arxiv.org/abs/1807.06757), Eq. 1) is the mean of success × shortest-path length ÷ max(path taken, shortest path). Green et al. 2026 use a success × score metric for racing. Our success-weighted speed replaces path length with time and does not cap the ratio at 1, because our reference is a scripted driver, not the optimum. `[paper]`

## 8. Car physics: models, engines, and what is reused

**Models.** The kinematic bicycle (Kong et al. 2015, [DOI](https://doi.org/10.1109/IVS.2015.7225830)) has no grip limit. The CommonRoad single-track model used by f1tenth_gym is linear in slip angle, so grip cannot be exceeded either ([source](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models)). Monster's "Car Physics for Games" clamps lateral force and is singular at zero speed ([mirror](https://asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html)). These are planar; a 3D game with jumps needs a rigid body with suspension. `[paper]` `[code]`

**The raycast vehicle.** Games use a chassis rigid body with four suspension rays (Bullet `btRaycastVehicle`, zlib, [source](https://github.com/bulletphysics/bullet3)): the spring-damper suspension force is applied as an impulse at the contact point, a side impulse cancels lateral velocity, an engine or brake impulse acts forward, and both are scaled down when they exceed friction-slip × suspension force. PhysX's guide advises at least five integration steps per suspension spring period ([guide](https://archive.docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/Vehicles.html)). `[code]` `[docs]`

**Candidates for reuse.**

| Candidate | Licence | Car code | Verdict |
|---|---|---|---|
| cannon-es `RaycastVehicle` ([repo](https://github.com/pmndrs/cannon-es)) | MIT | about 390 lines plus `WheelInfo`; only ray casts and impulses on one body | **Chosen.** Pure JavaScript, no substeps, no gyroscopic term, frozen at 0.20.0 (npm, 2022-08-12) |
| Rapier `DynamicRayCastVehicleController` ([repo](https://github.com/dimforge/rapier)) | Apache-2.0 | about 780 lines of Rust, a port of Bullet's | Its world step uses substeps and a gyroscopic term, and it changes quickly; hard to port exactly. One `world.step()` measured about 8–11 µs in an issue, so CPU-only training would need several busy cores |
| Jolt `WheeledVehicleController` | MIT | runs inside the constraint solver | More detailed than needed |
| Bullet through ammo.js | zlib | same model as cannon-es | Stale and heavy |
| Godot `VehicleBody3D` | MIT | Bullet-derived | Would require adopting the whole engine |
| MuJoCo MJX / MuJoCo Warp | Apache-2.0 | no car example | MJX runs on GPU only under Linux or WSL2 (JAX lists WSL2 as experimental); per-world terrain is undocumented for MJX and costly for Warp; Warp's GPU runs are not deterministic; the browser would run the C engine instead ([MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html), [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)) |
| Brax, Isaac Lab | — | — | Brax's physics is superseded by MJX; Isaac Sim needs 16 GB of VRAM |

No permissively licensed, browser-playable TrackMania clone with real car physics was found: Stunt Rally is GPL-3, PolyTrack is closed source. No batched or GPU port of a raycast vehicle was found either; SceneFactory-3D ([arXiv 2610.02874](https://arxiv.org/abs/2610.02874)) describes the same design in an NVIDIA Warp kernel but its code link is dead.

**Facts about cannon-es that an exact port must mirror** `[code]`:

- One world step runs in this order: add gravity to the force; collision detection and solver (nothing to do when the chassis has no contacts); velocity damping `v *= pow(1 − damping, dt)` with both dampings defaulting to 0.01; the `preStep` event, in which the vehicle updates and applies its impulses; then `Body.integrate` (`v += f/m·dt`, `x += v·dt`, quaternion update and normalisation); then the inertia update and force clearing.
- `RaycastVehicle.castRay` goes through `World.rayTest`, which returns the closest hit and skips back faces.
- A triangle mesh only collides with spheres and planes in cannon-es, so a box chassis never touches a mesh track; only the wheel rays do.
- Suspension force is `stiffness × compression × clippedInvContactDotSuspension − damping × relative velocity`, multiplied by chassis mass and clamped to `[0, maxSuspensionForce]`.
- Known quirks: straight-line braking is not limited by grip (the skid clamp sits inside `if (sideImpulse !== 0)`; Bullet shares it; Rapier fixed it in 0.35.0); an open pull request (#215) reports the car launching when driving slowly over a raised body; each wheel is a single ray.

**Tuned starting values** (pmndrs/racing-game, MIT, [repo](https://github.com/pmndrs/racing-game)): chassis mass 500, wheel radius 0.38, suspension stiffness 30, rest length 0.35, friction slip 1.5, roll influence 0, engine force 1800 on the rear wheels, maximum brake 65, steering 0.3. Two further MIT projects use the same vehicle with ramps, jumps and respawn: [nadevrix/micro-volt-rc](https://github.com/nadevrix/micro-volt-rc) and [icurtis1/raycast-vehicle](https://github.com/icurtis1/raycast-vehicle).

**Rigid-body references.** Baraff 1997 ([notes](https://www.cs.cmu.edu/~baraff/sigcourse/notesd1.pdf)) for the state and box inertia; Catto, GDC 2015 ([slides](https://box2d.org/files/ErinCatto_NumericalMethods_GDC2015.pdf)) for semi-implicit Euler; Fiedler ([article](https://gafferongames.com/post/fix_your_timestep/)) for fixed time steps.

## 9. Tracks

- **Generators.** Gymnasium CarRacing walks between noisy polar checkpoints and has no self-intersection test; Maciel 2013 (random points → convex hull → spline, [article](https://www.gamedeveloper.com/programming/generating-procedural-racetracks)) can self-intersect; juangallostra/procedural-tracks (MIT) implements that method as closed 2D loops; Barthet et al. 2024 ([arXiv 2408.06346](https://arxiv.org/abs/2408.06346)) use grid tiles with ramps, with no public code. `[code]` `[paper]`
- **Schema reused.** TORCS describes a track as straight or arc segments with start and end height and start and end banking ([manual](https://torcs.sourceforge.net/api/track_manual.html)); its generator code is GPL, so only the schema is reused. `[docs]`
- **Ribbon model.** Perantoni & Limebeer 2015 ([DOI](https://doi.org/10.1115/1.4028253)) describe a 3D track as a spine curve with orientation as a function of arc length and a lateral coordinate. `[paper]`
- **TrackMania itself** is commonly described as ticking at 100 Hz and being deterministic ([donadigo](https://donadigo.com/tmx1)); these are community statements, not official ones. `[2nd]`

## 10. Non-learning baselines

- **Pure pursuit** (Coulter 1992, [report](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)): steering curvature 2x/l² toward a look-ahead point. Stanley control (Thrun et al. 2006) is the common alternative. `[paper]`
- **Speed profile.** A forward and backward pass over curvature with acceleration limits, as in TUM's `trajectory_planning_helpers` (LGPL-3.0, [repo](https://github.com/TUMFTM/trajectory_planning_helpers)); it is 2D, so on hills and jumps it is approximate. `[code]`
- **Near-optimal reference.** Sampling-based optimisation on the true simulator: MPPI (Williams et al. 2016, [DOI](https://doi.org/10.1109/ICRA.2016.7487277)) and CEM variants such as iCEM (Pinneri et al. 2020, [arXiv 2008.06389](https://arxiv.org/abs/2008.06389)). `[paper]`

## 11. Physics exploits and specification gaming

- **Hide-and-seek** (Baker et al. 2020, [arXiv 1909.07528](https://arxiv.org/abs/1909.07528)): seekers learned "box surfing", possible because the movement action applies force even when the agent is off the ground; the accompanying blog lists further exploits, including seekers launching themselves off a ramp against a wall (confirmed through quotations in the specification-gaming list, since the blog itself could not be fetched). `[paper]` `[2nd]`
- **Specification gaming** (Krakovna et al. 2020, [blog](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)): a public list of about 90 examples with fixed columns (intended goal, behaviour, and so on). Lehman et al. 2018 ([arXiv 1803.03453](https://arxiv.org/abs/1803.03453)) collect physics-engine exploits from evolution and advise visualising solutions regularly. `[paper]`
- **TrackMania terms.** An uberbug is a sudden speed gain from a car–track collision during a bugslide; a noseboost chains nose-down slides to gain speed ([TAS Exchange FAQ](https://tmtas.exchange/faq)). Claims that an RL agent learned such bugs come only from secondary sources. `[2nd]` `[unverified]`

**Consequence:** exploits found by our agents are kept and reported in the specification-gaming list's format, with first training step, number of seeds and the result after patching.

## 12. Browser deployment and tooling

- **Policy in the browser.** onnxruntime-web 1.31.1; the WASM backend supports all operators, WebGPU lacks several, and multithreading needs cross-origin isolation that GitHub Pages cannot provide, so one thread is used ([docs](https://onnxruntime.ai/docs/tutorials/web/)). `torch.onnx.export` uses the dynamo exporter by default, default opset 20. `[docs]`
- **Precedents.** mjswan exports observation and termination terms to ONNX and runs them in the browser ([docs](https://mjswan.readthedocs.io/en/latest/guides/how-it-works/)). Mondo-Robotics/open-beni (MIT, [repo](https://github.com/Mondo-Robotics/open-beni)) is a browser racing game that runs PPO policies as ONNX in a web worker; its worker, inference loop and export script are the pattern we copy. `[code]`
- **Hugging Face.** No TrackMania agent models exist on the Hub; RL model cards conventionally use `model-index` with task `reinforcement-learning` and a mean ± spread metric. A dataset of community TrackMania tracks and replays exists, but the layouts stay under the publisher's terms and are not used. `[docs]`
- **Tooling.** PyTorch 2.15.0 with `cu132` wheels for Windows and Linux (Python 3.11–3.15); Gymnasium 1.4.0; TensorBoard 2.21 for the live dashboard; pandoc 3.14 with typst 0.15 for the PDF. `[docs]`

## 13. Research-practice sources followed

Henderson et al. 2018, Agarwal et al. 2021 and Patterson et al. 2024 (section 7); Lones, "How to avoid machine learning pitfalls" ([arXiv 2108.02497](https://arxiv.org/abs/2108.02497)); Nanda's research-process posts ([sequence](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL)); Schulman's guide ([post](http://joschu.net/blog/opinionated-guide-ml-research.html)); Lipton & Steinhardt, "Troubling Trends" ([arXiv 1807.03341](https://arxiv.org/abs/1807.03341)); the NeurIPS paper checklist ([guide](https://neurips.cc/public/guides/PaperChecklist)). The practices taken from them are: a preregistered primary metric and falsification criteria, baselines and controls, held-out evaluation with an access log, many seeds with intervals, all negative results reported, and no claim stronger than its evidence.

## 14. Gaps, and what is new here

- No published RL agent drives **unseen** TrackMania-like car tracks with keyboard actions; the multi-track evidence is for drones (Green et al. 2026) or is negative (Fuchs, Learn-to-Race).
- No batched GPU port of a game's raycast vehicle is public, so training normally runs against the game itself in real time (Linesight, TMRL) or on a different simulator from the one humans play.
- Here the same car code runs in the browser game and, as a tested port, in training; the generalist is tested on 1,000 unseen generated tracks with many seeds; and exploits are reported systematically rather than removed.

## 15. Fact-check log

Filled in by the verification pass: each row is a claim that was corrected, weakened or removed, with the reason. Unverified items that remain: Swift's reward weights; whether any RL agent has used a TrackMania physics bug; the exact date of the MuJoCo Warp release; Rapier's substep integration code (read only from its changelog).

| Section | Original claim | Outcome |
|---|---|---|
| — | — | (pending) |
