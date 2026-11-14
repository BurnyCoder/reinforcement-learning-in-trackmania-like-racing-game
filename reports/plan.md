# Plan: "RacingMania" — an original 3D time-attack racer and an RL driver for it

> This is the plan as approved on 2026-11-14, copied in full. Every later change is listed, with its date and reason, under [Amendments](#amendments) at the end of this file. The preregistration (`reports/preregistration.md`) is the binding document for hypotheses, metric and analysis.

## Context

The project folder is empty (no git repo). The request is a simplified TrackMania-style game with simplified physics and graphics, plus a reinforcement-learning agent trained to play it, carried out to the research standard in the global CLAUDE.md (literature review → preregistration → tested implementation → first run with baselines and controls → ablations and seeds → paper, reports, website, Hugging Face).

The game is an **original** time-attack racer in the TrackMania genre: checkpoints, medals, ghosts, instant restart, deterministic fixed-tick physics. It does not copy TrackMania's tracks, art or look. The game is called "RacingMania" (name chosen by the user).

**Decisions taken with the user**

| Topic | Decision |
|---|---|
| Physics | Full 3D with hills, banked turns, ramps and jumps |
| Platform | Browser game, working from a local server and on GitHub Pages |
| Headline experiments | Generalisation to unseen generated tracks (primary); per-track specialists on fixed tracks (secondary) |
| Visibility | Public GitHub repo, Pages site and Hugging Face models |
| First 18 hours (soft target) | One single-seed specialist trained on one track (first model), then one single-seed generalist trained on many tracks (second model); everything else afterwards |
| If the first models do not learn properly | Give them more training time if that is what they need; otherwise stop and keep fixing them until they do, even past 18 hours. The aim is at least one interesting good result; negative results alone are also acceptable |
| Run order | 1 first two models; 2 one seed of every experiment; 3 patched physics, if an exploit is found; 4 imitation and hybrids; 5 hyperparameter sensitivity; 6 seed sensitivity |
| Game first | The playable browser game is pushed before any training starts |
| Physics source | Reuse existing tested car physics instead of writing it from scratch |
| Order of work | Literature review first; then write all code, the README and the preregistration and push them; then test; then run the long training |
| Physics exploits | Kept and reported as findings; training runs on patched physics follow later if any are found |
| Learning on its own first | No imitation of the scripted driver and no reward hints from it in the first experiments; imitation and hybrids are a later phase |
| Research rule | Search first, reuse before writing, read key papers in full, cite everything |
| Plan in the repo | The full plan is written to `reports/plan.md` and kept up to date |
| Publishing | Everything is pushed to GitHub and Hugging Face before each model's training, after its training and after its evaluation, with the updated paper, results, reports, README and website |

**Machine (checked):** RTX 5070 Laptop GPU with 8 GB VRAM, 16 logical CPUs, 32 GB RAM, Windows 11, WSL2 Ubuntu, `uv` 0.13.6, Node 24, `gh` and `hf` logged in. `C:` has 39 GB free, so checkpoints and logs go to `D:\AI\racingmania` (set in `.env`).

## How the pieces fit

```
seeded track generator ─► track files (ribbon, triangle mesh, scripted-driver witness, reference time)
        │                                   │
        │                 web/ game: three.js + cannon-es car physics (the tested original)
        │                                   │  tick-by-tick parity tests (Node.js runs the original)
        ▼                                   ▼
   batched PyTorch port of the same car physics on the GPU (100 Hz tick, thousands of cars, one track per car)
                                │
 PPO trainer (acts at 20 Hz) ─► checkpoints ─► one evaluator for PPO, scripted, CEM and random drivers
                                │                              │
   ONNX export: observation + normaliser + policy            results/*.csv
                                │                              │
   "watch the AI drive" and ghosts in the game     make_report ─► tables, figures, reports, README, paper.pdf, site data
```

## Sources and reuse

Standing rule for the whole project: search first, reuse before writing, cite everything.
- **Search first.** Before building any component, and whenever something is stuck (a test fails for unclear reasons, PPO does not learn, parity breaks), I search papers, GitHub, Hugging Face, Kaggle, docs and forums for how others solved it, and use their solution where the licence allows.
- **Use the original unless there is a reason to change it.** Every change to a reused source is listed with its reason.
- **Read in full, then cite.** The important papers are read in full, not from abstracts or summaries. Each cited claim goes into `reports/source-ledger.md` with its link, the exact supporting quote and where it is used. Anything a research agent reported is re-checked against the source before it is cited.
- **Sources in the code.** Every function that implements something from a source carries the link in its comments.
- **The design below is provisional until the literature review confirms it.** If the review turns up a better existing solution, the plan's choice is replaced and the change is recorded.

| Component | Reused from | Licence | As-is or changed |
|---|---|---|---|
| Car physics in the game | cannon-es `RaycastVehicle`, `WheelInfo`, `Body`, `World` | MIT | As-is, vendored |
| Car physics for training | The same code, ported to batched PyTorch, including cannon's own ray-against-triangle test (closest hit, back faces skipped) | MIT | Line-by-line port; the only change is batching. No existing batched port was found; SceneFactory-3D (arXiv 2610.02874) describes the same design but its code is not public. NVIDIA Warp's `mesh_query_ray` (Apache-2.0) is the fallback for the ray test |
| Vehicle setup and tuning | pmndrs/racing-game `store.ts`; nadevrix/micro-volt-rc and icurtis1/raycast-vehicle (three.js + cannon `RaycastVehicle`, with ramps, jumps and respawn) | MIT | Setup and respawn code copied with notices; starting values, then tuned by hand; changes listed |
| Rendering | three.js and its examples | MIT | As-is |
| PPO | CleanRL `ppo_continuous_action_isaacgym.py` | MIT, NVIDIA BSD-3 header | Changed: two categorical heads and observation normalisation (Huang et al. 2022), bootstrapping on time limits (Rudin et al. 2021), checkpointing |
| Hyperparameters | CleanRL defaults; Rudin et al. 2021; Makoviychuk et al. 2021 | — | As-is to start; the pilot may change them on validation tracks |
| Baseline algorithm | PQN: CleanRL `pqn.py` (Gallici et al. 2025) | MIT | Adapted to the GPU environment; other settings from the paper |
| Reward | Centreline progress: Fuchs et al. 2021, Linesight, GT Sophy; gate progress in drone racing: Song et al. 2021 and 2023, Kaufmann et al. 2023 | — | As-is, no added shaping; time cost and crash penalty are ablations |
| Start states | Random starts along the track: Song et al. 2021 and 2023, Kaufmann et al. 2023 | — | Random centreline points with no scripted-driver states, in every phase except phase 4; starting from the scripted driver's run is a phase 4 hybrid |
| Imitation and hybrids (phase 4 only) | Behaviour cloning and DAgger (Ross et al. 2011); a reward for staying close to a classical driver's speed and steering (Evans et al. 2023); Fuchs et al. 2021 report that a behaviour-cloning warm start did not help | — | Provisional; a focused search before phase 4 fixes the final list |
| Observations | Upcoming track points in the car frame: Linesight, GT Sophy, Neinders 2023 | — | Adapted to the ribbon |
| Actions | Keyboard combinations: Linesight; PPO with discrete driving actions: AWS DeepRacer (Balaji et al. 2019) | — | Changed: two 3-way choices instead of 12 combined actions, to fit PPO's factorised heads. Seyde et al. 2021 show on-off control is enough on standard control benchmarks, but they did not test driving, so it is cited only as background |
| Track model and generator | Ribbon: Perantoni & Limebeer 2015; segment schema (straight or arc, start and end height, start and end banking): TORCS track manual | — | Schema reused, code ours. TORCS's own generator is GPL; juangallostra/procedural-tracks (MIT) makes closed 2D loops without the straights that jumps need, so it is not used |
| Generalisation protocol | Kirk et al. 2023 (held-out seeds, one test set per kind of generalisation, several training-set sizes, test score before gap); CoinRun and Procgen, Cobbe et al. 2019 and 2020 | — | Protocol as-is |
| Closest prior study | Green et al. 2026: PPO generalist against specialists on generated drone-racing tracks | — | Their findings are our stated expectations; our design is compared against theirs |
| Difficulty curriculum, only if needed | Track complexity adapted from the crash ratio: Song et al. 2021; robust level replay: Jiang et al. 2021 | — | Held in reserve as the preregistered fallback |
| Scripted driver | Pure pursuit: Coulter 1992; speed profile: TUM `trajectory_planning_helpers` | LGPL-3.0, imported as a library | As-is if it handles hills and jumps, otherwise reimplemented from Heilmeier et al. 2020 |
| Near-optimal reference | CEM: Pinneri et al. 2020; MPPI: Williams et al. 2016 | — | Algorithm from the paper |
| Primary metric | Success-weighted scores: SPL, Anderson et al. 2018; success × score in Green et al. 2026 | — | Changed: time instead of path length, and not capped at 1 because the reference is not the optimum |
| Statistics | `rliable` (Agarwal et al. 2021); Patterson et al. 2024; Colas et al. 2018 | Apache-2.0 | Library used directly |
| Time-limit handling | Gymnasium 1.4 documentation | MIT | As-is |
| Policy in the browser | `torch.onnx` and onnxruntime-web; Mondo-Robotics/open-beni's web worker, inference loop and export script (a browser racing game that runs PPO policies as ONNX) | BSD-3, MIT | open-beni's files copied with notice and adapted to our observation |
| Hugging Face model cards | The Stable-Baselines3 convention (`model-index`, task `reinforcement-learning`, metric with mean ± spread) | — | Convention followed |
| Dashboard and paper build | TensorBoard; pandoc with typst | Apache-2.0; tools only | As-is |
| Exploit reporting | Columns of the specification-gaming list (Krakovna et al. 2020); Baker et al. 2020 (hide-and-seek); Lehman et al. 2018 | — | Format followed, with three columns added |
| Research practice | Henderson et al. 2018; Agarwal et al. 2021; Patterson et al. 2024; Lones 2021; Nanda; Schulman | — | Checklists followed |

## Design

Sources named here come from the research reviews already done; each is re-read and linked in the literature-review report before it is cited anywhere.

**Where things run** — one working copy (this folder, seen from WSL under `/mnt/c`). Training and official evaluation run in WSL2 Ubuntu, where `torch.compile` can fuse the simulator's many small tensor operations (it is less well supported on native Windows). Windows runs the browser game, rendering, the parity checks against the browser physics and the test suite.
- The benchmark in the testing stage confirms this split; native Windows is the fallback if WSL2 turns out slower or unstable.
- The environment is fixed once and recorded in every `run.json` and in the simulator-freeze amendment, because results are bitwise reproducible only within one environment.
- In WSL the Linux virtual environment lives on the Linux filesystem (`UV_PROJECT_ENVIRONMENT`), and long runs sit in `tmux` so they survive a closed terminal. If file access through `/mnt/c` proves slow, training uses a clone in `~/projects` instead.

**Physics: reuse tested code** — the car is cannon-es's `RaycastVehicle` (MIT), a port of Bullet's `btRaycastVehicle` that shipped browser racing games already use. A survey found no permissively licensed TrackMania clone to reuse whole (Stunt Rally is GPL-3, PolyTrack is closed), so the car model is reused instead.
- **In the browser** the game runs a pinned, vendored copy of cannon-es directly. It is pure JavaScript, about 900 lines of car code that only casts rays and applies impulses to one rigid body.
- **For training** the same code is ported line by line to batched PyTorch (`RaycastVehicle.ts`, `WheelInfo.ts`, `Body.integrate`, the step order of `World.internalStep`, `Quaternion.integrate`, ray against triangle). cannon-es has no substeps and no gyroscopic term, so the port is a few dozen tensor lines per function.
- **Testing** is mostly comparison with the original: Node.js runs the vendored code on random states and inputs, and the port must produce the same next state. This replaces most hand-derived physics tests. A few closed-form checks stay as a sanity check.
- **Tuning** starts from the values in pmndrs/racing-game (MIT): mass 500, wheel radius 0.38, suspension stiffness 30, rest length 0.35, friction slip 1.5, engine force 1800, steering 0.3. I adjust them by driving the game by hand, then freeze them in one JSON file that both the game and the port read.
- **Known quirks, handled the same way on both sides:**
  - Straight-line braking has effectively unlimited grip in cannon-es and Bullet (fixed in Rapier 0.35). It is left as it is in the first physics version, mirrored exactly in the port, and listed as a candidate exploit; the patched physics version fixes it.
  - Chassis-to-ground contacts go through cannon-es's solver and cannot be ported exactly, so they are switched off; only the wheel rays touch the track.
  - Each wheel is a single ray and the suspension can bottom out, so the generator smooths ramp lips and limits landing height, with a test at the maximum.
  - An open cannon-es pull request (#215) reports the car launching into the air when driving slowly over a raised body. It is left in and listed as a candidate exploit.
- **Facts the port must mirror** (read from the cannon-es source): the car updates in the `preStep` event, after the solver and before integration; wheel rays return the closest hit and skip back faces; in cannon-es a triangle mesh only collides with spheres and planes, so the chassis box would not touch the track anyway.
- **Two implementations of one model** is a deliberate exception to "no duplicated code": the original stays untouched as the tested reference, and the parity tests fail if the port drifts.
- 100 Hz tick, five ticks per agent action, fixed-step loop in the game (no variable frame-time stepping).
- Rules (ours, on both sides): checkpoints in order; a run is out when the car is more than 3 m below the track, upside down on the ground, airborne for 5 s, or makes no progress for 2 s. No walls and no grass: the track is a ribbon in the sky with a 2 m low-grip shoulder each side.

Alternatives considered and not chosen:

| Alternative | Why not |
|---|---|
| Train on the browser engine itself (many Node.js or Rapier-Python worlds on the CPU) | One Rapier world step measures about 8–11 µs, so the target of 500,000 ticks/s would need five or more fully busy cores before messaging costs; the GPU port has more headroom |
| Rapier's car as the source | Apache-2.0 and actively changing; its solver uses substeps and a gyroscopic term, which makes an exact port much harder |
| Jolt's car | Runs inside its constraint solver; far more detailed than needed |
| Own physics exported to the browser, existing engine only as a test oracle | Loses the ready-made tuned game and needed an unproven export of a whole physics step |
| MuJoCo (MJX in WSL2, or MuJoCo Warp) | A different track per parallel car is undocumented (MJX) or costly (Warp); Warp's GPU runs are not deterministic; the browser would run a different engine; no car example exists |
| Brax, Isaac Lab | Superseded by MJX; needs 16 GB VRAM |

**Tracks** — a 3D ribbon (Perantoni & Limebeer 2015): centreline, tangent, banked lateral vector, half-width, gap mask and checkpoints, stored as padded tensors so every car in a batch can be on a different track.
- The ribbon is triangulated once, in Python, and the triangles are saved in the track file. The game loads them as a cannon-es triangle mesh and the port casts wheel rays against the same triangles near the car, so both sides hit identical geometry.
- Generator: seeded sequence of straights and arcs with elevation, banking, ramps and gap jumps (TORCS segment style), 8 m vertical clearance wherever the track crosses itself. A ribbon cannot represent loops or wall-rides; this limit is stated in the docs.
- Jumps are built so that "flat out, straight" always clears them, then each track is certified by a frozen scripted driver finishing it in the real simulator at three speed settings. Its action sequence and time are stored with the track as the reference time.
- Splits by disjoint seed ranges: 50,000 training tracks, 200 validation, 1,000 test. Every evaluation on test tracks is appended to an access log.
- Five hand-made campaign tracks for the game and for the specialist experiment.

**Agent** — PPO in a single file adapted from CleanRL's GPU script (`ppo_continuous_action_isaacgym.py`, licence headers kept), plus bootstrapping on time limits (Rudin et al. 2021), running observation normalisation and full checkpoint/resume every 10 minutes.
- Why PPO: it is what the closest prior work uses for minimum-time racing in fast parallel simulators (Song et al. 2021 and 2023, Kaufmann et al. 2023, Green et al. 2026). PQN (Gallici et al. 2025; CleanRL `pqn.py`, MIT) is trained as the one alternative-algorithm baseline, because it takes the same parallel rollouts and discrete actions.
- Actions every 50 ms: steer left/none/right × gas/coast/brake, exactly what a keyboard player has; the steering angle is rate-limited the same way in the game and in the port. AWS DeepRacer also trained PPO on discrete driving actions (Balaji et al. 2019).
- Observations: car-frame velocity and spin, up vector, wheel contacts, steering angle, previous action, offset and heading error to the centreline, and about 20 upcoming centreline points in the car's frame with width and gap flags (Linesight, GT Sophy, Neinders 2023).
- Reward: metres of centreline progress per step and nothing else, the common core of Fuchs et al. 2021, TMRL and the drone-racing work. A per-step time cost rewards dying early. A crash penalty is standard in drone racing, but fixed penalties caused stalling in Fuchs et al. 2021. Both become ablations.
- Training episodes mix standing starts with random starts along the centreline (heading along the track, random speed), so late sections are sampled (Song et al. 2021 and 2023, Kaufmann et al. 2023, Bharthulwar et al. 2025). No state from the scripted driver is used.
- The agent learns on its own in every phase except phase 4: it never sees the scripted driver's actions, states or times. The scripted driver only checks that a generated track can be finished and serves as the yardstick in the metric. Imitation and hybrids are confined to phase 4, after the single-seed experiments.
- Starting hyperparameters are the CleanRL/Rudin defaults (4,096–16,384 cars, 24–32 steps per rollout, γ 0.99, λ 0.95, clip 0.2); the pilot run fixes them using validation tracks only.

**Browser game** — static site with no build step: three.js low-poly chase camera, keyboard controls, timer, checkpoint splits, medals, ghosts of the AI and of your best run, "watch the AI drive live", campaign tracks and "random unseen track by seed".
- Physics in the browser is the vendored cannon-es car described above; the game's physics and rules live in DOM-free modules that the Node.js parity tests import too.
- "Watch the AI drive live" runs one ONNX graph containing observation building, normaliser and network (onnxruntime-web, WASM, single thread, which GitHub Pages requires), so JavaScript never rebuilds observations. Exporting observation terms this way has a precedent (mjswan).
- Ghosts are state trajectories recorded by the Python evaluator.
- Falling off respawns a human at the last checkpoint with the clock running; in official evaluation it is a did-not-finish. All reported numbers come from the Python evaluator on this GPU.

## Experiments (to be preregistered before any result exists)

**Primary metric — success-weighted speed.** On each held-out test track the agent scores (scripted driver's time ÷ agent's time) if it finishes within 3× the reference time, otherwise 0; the metric is the mean over the 1,000 test tracks. 1.0 means "finishes everything, as fast as the scripted driver". Confidence intervals come from a bootstrap over seeds and tracks (10,000 resamples), with Holm correction across the three hypotheses.
- It is adapted from SPL (Anderson et al. 2018, Eq. 1: mean of success × shortest path ÷ max(path taken, shortest path)); Green et al. 2026 use a success × score metric for racing.
- One deliberate change from SPL: the ratio is not capped at 1, because the reference here is a scripted driver, not the optimum, and H1 asks whether the agent beats it.

| Hypothesis | Test | Refuted if |
|---|---|---|
| H1: the generalist drives unseen tracks better than the scripted driver | score > 1.0 | 95 % interval lies below 1.0 |
| H2: track diversity is what generalises | score(50,000 tracks) − score(1 track) > 0.10, from the training-set-size ablation | interval lies below 0.10 |
| H3: the agent uses the look-ahead | score(generalist) − score(blind) > 0.10 | interval lies below 0.10 |

The three hypotheses are tested once, in the seed sensitivity phase, when all seeds exist. Single-seed results before that are reported as preliminary.

**Closest prior result.** Green et al. 2026 (a preprint) trained PPO on many generated drone-racing tracks: the generalist finished 69.8 % of 40 unseen tracks against 0.0 % for single-track policies, was 14.52 % slower than per-track specialists, and fine-tuning it on one track took 23.19 % of a specialist's training. H2 and the specialist comparison test whether the same pattern holds for a car with keyboard actions, hills and jumps, with a larger test set and more seeds. What is new here is the car setting, the playable browser game on the same physics, and the exploit reporting.

Because test tracks are filtered to ones the scripted driver finishes, it scores 1.0 by construction; rejection rates and a secondary evaluation on unfiltered tracks are reported.

**Physics exploits are results, not failures.** If an agent gains time through a quirk of the engine (the way TrackMania players use uber bugs and nose boosts, or OpenAI's hide-and-seek agents learned box surfing), the models and their scores are kept and the exploit is reported: what it is, which runs use it, how much time it is worth, and a replay in the game.
- Detection: the evaluator flags speed above 1.2 × top speed, energy gained while coasting and airborne distance beyond the ballistic range; I also watch the best and strangest replays of each block, on a fixed schedule (Lehman et al. 2018 advise visualising solutions regularly).
- Reporting: one row per exploit with the columns of the public specification-gaming list (Krakovna et al. 2020: intended goal, behaviour, and so on), plus the first training step it appears at, how many seeds find it, and the result after patching. Baker et al. 2020 report box surfing the same way, with figures and videos.
- The preregistered analysis stays on the first physics version, exploit or not. H1 is additionally reported on the patched version if an exploit is found.
- Every result records its physics version. The patch is applied identically to the vendored cannon-es copy and to the port, the parity tests are rerun, and the patched version gets its own tag (phase 3).

**Track sets** (generated tracks are split by seed range; no track is in two sets)

| Set | Tracks | Used for |
|---|---|---|
| Training pool | 50,000 generated | training; a fixed 1,000-track sample is also scored to measure the train–test gap |
| Validation | 200 generated | pilot decisions, tuning the scripted driver, live dashboard curves |
| Test | 1,000 generated | every reported score; read only by the evaluator, each access logged |
| Out-of-range test | 200 generated, each with one factor beyond the training range (sharper turns, steeper hills, longer jumps, longer tracks) | secondary: extrapolation, one test set per kind of generalisation as Kirk et al. 2023 recommend |
| Campaign | 5 hand-made | the game, the specialist experiment, and a hand-made out-of-distribution set |

**The standard recipe** (every trained run uses it unless its row below says otherwise)
- **Model:** PPO with separate actor and critic networks, each two hidden layers of 256 tanh units, about 0.2 million parameters in total. The input is the observation described above (about 120 numbers); the actor outputs the two 3-way choices (steer, pedal).
- **Training maps:** the 50,000-track pool, with 4,096 tracks on the GPU at a time and rotated. Budget: 1 billion agent steps, which is about 1.6 years of simulated driving and an estimated 2 hours.
- **Evaluation:** the checkpoint at the end of the step budget (no picking the best one), most-probable action, standing start, one run per track because the simulator is deterministic, time cap 3× the reference time. Scored on the 1,000 test tracks (primary metric), the 1,000-track training sample, the 200 out-of-range tracks and the 5 campaign tracks. Test performance is reported first and the train–test gap second, since a random policy also has a gap of zero (Kirk et al. 2023).
- **Secondary numbers for every run:** finish rate, time ratio on tracks both agents finish, finish rate by track feature (jumps, banking, hills), train–test gap.

**Runs, by phase** (hours assume about 150,000 agent-steps/s; the benchmark gate measures the real figure before anything is scheduled)

*Phase 1 — the first two models*

| Run | What is trained | Trained on | What its evaluation answers | Runs | Hours per run | Hours |
|---|---|---|---|---|---|---|
| Training-free baselines | Nothing. Random actions; an untrained network (10 random initialisations); the scripted driver (pure-pursuit steering + curvature speed profile, 2 parameters tuned on validation) | — | The floor; whether training is what matters; the reference time (score 1.0) | — | About 0.1 each | 0.3 |
| First specialist (model 1) | Standard recipe, seed 0, 200 million steps to start, extended for as long as its time on the track keeps improving | Campaign track 1 only | Does the whole pipeline learn one track, and how its time compares with the scripted driver. Reported separately because it is trained before the recipe is frozen | 1 | 0.4 | 0.4 |
| First generalist (model 2, the pilot) | Standard recipe, seed 0: about 4 h to start, extended for as long as its validation score keeps rising | Training pool | On validation only: where the score plateaus (fixes the step budget for all later runs) and whether the agent has found a physics exploit (a finding to report, not a failure). If it does not pass the learning gate, the fix loop below applies. Scored on test once, after the recipe is frozen, and reported separately | 1 | Up to 5 | 5 |

**Learning gate before the main experiments.** Phases 2–6 start only after both first models learn properly. This is judged on campaign track 1 and the validation tracks, never on the test tracks.
- **Specialist:** it finishes campaign track 1 on every evaluation run and is at least as fast as the scripted driver.
- **Generalist:** it finishes at least 80 % of the validation tracks, with a score of at least 0.8.
- These bars are my choice of "clearly working", set below the hypotheses on purpose. For scale, Green et al. 2026 report about 70 % success on unseen tracks for their generalist.

If a model misses its bar, the run queue stops and the fix loop runs until it passes, with no time limit inside the week:
1. Train longer. If the learning curve is still rising when the starting budget ends, more training time is the first remedy, before any change to the recipe; the run is resumable, so it simply continues.
2. If the curve is flat, check the basics: PPO on the flat one-bend track, reward and observation scales, the action mapping, overfitting a single track.
3. Watch replays to name the failure (stalling, falling at one feature, spinning).
4. Search how others fixed that failure before changing anything.
5. Try changes in a pre-listed order: PPO settings (learning rate, entropy bonus, rollout length, number of cars, discount, network size); then environment settings (share of random starts, no-progress timeout, look-ahead range, action rate); then a track-difficulty curriculum; then the reward.

Rules for the fix loop:
- Every attempt is logged with what was tried, why and what happened, failed ones included, and each change to the recipe is a dated amendment to the preregistration.
- The recipe is frozen only once both models pass. The later phases use fresh seeds, and the test tracks are first read after the freeze, so fixing cannot tune on them.
- The step budget frozen for the later runs is whatever the first models needed. If that is more than 1 billion steps for the generalist or 200 million for a specialist, the hours of every later run scale with it.
- Time spent on longer training and on fixing comes out of the later phases through the trim order.
- If the week ends without a properly learning model, that is reported as a negative result with the full log of attempts.

*Phase 2 — one seed of every experiment (all seed 1, frozen recipe); results are preliminary*

| Run | What is trained | Trained on | What its evaluation answers | Runs | Hours per run | Hours |
|---|---|---|---|---|---|---|
| CEM reference | Nothing: a planner that searches action sequences in the simulator | — | A near-optimal time on 25 test + 5 campaign tracks | — | About 0.12 per track | 3.7 |
| Generalist | Standard recipe | Training pool | Test score against the scripted driver (H1, preliminary); zero-shot times on the campaign tracks; the reference for every comparison below | 1 | 2 | 2 |
| Blind control | Standard recipe with the upcoming-track points removed from the input (offset and heading to the centreline stay) | Training pool | Generalist minus blind on the test tracks (H3, preliminary) | 1 | 2 | 2 |
| Baseline algorithm: PQN | PQN from CleanRL's `pqn.py` with the same network size, inputs, actions, reward and step budget | Training pool | Whether the result depends on PPO: test score against the PPO generalist | 1 | 2 | 2 |
| Specialists (secondary) | Standard recipe, 200 million steps | One campaign track each, 5 tracks | Time on its own track vs the generalist zero-shot, the scripted driver and CEM; also scored on the test tracks | 5 | 0.4 | 2 |
| Specialists fine-tuned from the generalist (exploratory, one seed) | The seed-1 generalist's weights, trained on for 200 million steps | One campaign track each, 5 tracks | Whether starting from the generalist reaches the from-scratch specialist's time sooner, as Green et al. 2026 found | 5 | 0.4 | 2 |
| Ablation: training-set size | Standard recipe, fewer training tracks | Only N tracks, drawn from the pool with the run's seed: N = 1, 10, 100, 1,000 | 50,000 tracks vs 1 (H2, preliminary); curve of test score and train–test gap against N; each run is also scored on its own N tracks | 4 | 2 | 8 |
| Ablations: inputs, actions and reward (exploratory) | Standard recipe with one change: (a) continuous steer and pedal outputs; (b) 5 upcoming points; (c) 40 upcoming points; (d) Linesight-style time cost added to the reward; (e) one action per 100 ms for the same simulated driving time (500 million steps); (f) a penalty when the run ends by falling or flipping, as in drone racing | Training pool | Change in test score against the generalist | 6 | 2 | 12 |

*Phase 3 — patched physics (only if an agent used an exploit or engine quirk in phases 1–2; seed 1)*

| Run | What is trained | Trained on | What its evaluation answers | Runs | Hours per run | Hours |
|---|---|---|---|---|---|---|
| Earlier models on patched physics | Nothing: every generalist and specialist from phases 1–2 is re-scored with the patch applied; models from later phases are re-scored as they appear | — | How much of each agent's score depended on the exploit | — | Minutes | 0.5 |
| Generalist, patched | Standard recipe on the patched physics | Training pool, with reference times re-measured on the patched physics | Test score and driving style without the exploit, against the unpatched generalist | 1 | 2 | 2 |
| Specialists, patched | Standard recipe, 200 million steps | One campaign track each, 5 tracks | Time lost on each track without the exploit | 5 | 0.4 | 2 |

If an exploit first shows up in a later phase, this phase is run then.

*Phase 4 — help from the scripted driver: imitation and hybrids (after the single-seed experiments in which the agent learns on its own; first physics version; seed 1, exploratory)*

| Run | What is trained | Trained on | What its evaluation answers | Runs | Hours per run | Hours |
|---|---|---|---|---|---|---|
| Imitation only: behaviour cloning | The standard network, trained by supervised learning to copy the scripted driver's actions from its stored runs; no reward | Training pool | How far pure copying gets on unseen tracks | 1 | 0.5 | 0.5 |
| Imitation only: DAgger | The same, but the scripted driver is also asked what it would do in the states the learner reaches | Training pool | Whether correcting the learner's own mistakes closes the gap to the scripted driver | 1 | 1 | 1 |
| Hybrid: imitation, then PPO | Behaviour-cloned weights as the starting point, then the standard recipe | Training pool | Whether a warm start reaches the generalist's score sooner or higher (it did not help in Fuchs et al. 2021) | 1 | 2 | 2 |
| Hybrid: reward hints | Standard recipe plus a reward for staying close to the scripted driver's speed and steering | Training pool | Whether hints speed learning up, and whether they hold the agent down at the scripted driver's level | 1 | 2 | 2 |
| Hybrid: scripted start states | Standard recipe, but mid-track starts are taken from the scripted driver's stored runs | Training pool | Whether realistic start states matter | 1 | 2 | 2 |

Each is scored on the same test sets and compared with the seed-1 generalist that learned on its own. The list is provisional: a focused literature search on imitation and demonstration-guided RL for racing comes first and fixes the final variants. More seeds are run only if hours remain.

*Phase 5 — hyperparameter sensitivity (seed 1)*

| Run | What is trained | Trained on | What its evaluation answers | Runs | Hours per run | Hours |
|---|---|---|---|---|---|---|
| Hyperparameter sensitivity | Standard recipe with one change: learning rate ÷ 3; discount 0.995; networks of 3 × 512 units (larger networks generalised better in Procgen); weight decay 1e-5 (it helped in Green et al. 2026) | Training pool | How far the test score moves when a setting changes; none of these replaces the frozen recipe | 4 | 2 | 8 |

*Phase 6 — seed sensitivity: earlier runs repeated with new seeds, nothing else changed*

| Run repeated | New seeds | Seeds in total afterwards | Runs | Hours per run | Hours |
|---|---|---|---|---|---|
| Generalist | 2–10 | 10 | 9 | 2 | 18 |
| Blind control | 2–5 | 5 | 4 | 2 | 8 |
| Ablation: training-set size N = 1 | 2–5 | 5 | 4 | 2 | 8 |
| Specialists, 5 tracks | 2–5 | 5 each | 20 | 0.4 | 8 |
| Ablation: training-set sizes N = 10, 100, 1,000 | 2–3 | 3 each | 6 | 2 | 12 |
| Ablations: inputs, actions and reward, 6 variants | 2–3 | 3 each | 12 | 2 | 24 |
| Hyperparameter settings, 4 | 2 | 2 each | 4 | 2 | 8 |
| Baseline algorithm: PQN | 2–3 | 3 | 2 | 2 | 4 |
| Generalist, patched (only if phase 3 ran) | 2–3 | 3 | 2 | 2 | 4 |

What the seed sensitivity phase tests:
- **Spread between seeds** for every configuration: minimum, maximum and standard deviation of the test score, and how much of the total variation comes from seeds versus tracks.
- **H1–H3, formally.** The confidence intervals need the seeds, so the three hypotheses are tested here and nowhere earlier. The first three rows (generalist, blind control, single-track ablation) are the confirmatory ones and run first.
- **Whether the single-seed picture holds:** for each earlier single-seed comparison, whether the sign and rough size of the difference survive across seeds.

| Phase | Trained runs | Hours |
|---|---|---|
| 1. First two models | 2 | 5.7 |
| 2. One seed of every experiment | 23 | 33.7 |
| 3. Patched physics, if an exploit is found | 6 | 4.5 |
| 4. Imitation and hybrids | 5 | 7.5 |
| 5. Hyperparameter sensitivity | 4 | 8 |
| 6. Seed sensitivity | 61, or 63 if phase 3 ran | 90, or 94 if phase 3 ran |
| **Total** | **95, or 103 with patched physics** | **≈ 145, or 153 with patched physics, of 168** |

Hours per run are steps ÷ 150,000 steps/s plus a few minutes of evaluation: 1 billion steps is about 1.9 h and 200 million about 0.4 h. Runs go one after another on the single GPU. All of these are re-computed from the measured throughput once the benchmark gate has run.

The recipe stays frozen after the pilot: the hyperparameter sweep is descriptive and cannot change what the seed sensitivity runs use.

If throughput is lower, runs are cut in this preregistered order, decided before phase 5 starts so the confirmatory seeds keep their hours: hyperparameter settings, the PQN baseline's extra seeds, the action-rate and look-ahead ablations, the training-set-size ablations at 10 and 1,000, specialist seeds. The ten generalist seeds, five blind seeds and five single-track seeds are never cut.

## Repo layout

```
pyproject.toml  uv.lock  .env.example  THIRD_PARTY_NOTICES.md  .github/workflows/pages.yml
configs/     sim.yaml  env.yaml  ppo.yaml  experiments/*.yaml
src/racingmania/
  sim/       state.py rigid_body.py raycast_vehicle.py ray_trimesh.py step.py rules.py   (port of cannon-es)
  tracks/    ribbon.py triangulate.py generator.py validate.py campaign.py store.py
  env/       obs.py reward.py vec_env.py
  rl/        ppo.py pqn.py networks.py normalizer.py checkpoint.py
  drivers/   pure_pursuit.py speed_profile.py cem.py random_policy.py
  eval/      evaluator.py metrics.py stats.py
  export/    onnx_export.py parity.py
  report/    tables.py figures.py render.py
  utils/     config.py logging.py paths.py seeding.py
scripts/     01_generate_tracks.py 02_train.py 03_evaluate.py 04_baselines.py 05_run_matrix.py
             06_export_web.py 07_make_report.py 08_publish.py bench_sim.py serve.py
tests/       unit/ parity/ (Node.js oracle + Python comparison) web/ (Playwright)
web/         index.html results.html
             js/core/ (DOM-free physics setup, fixed-tick loop, rules; shared with the parity tests)
             js/ (render, input, hud, ghost, policy)
             vendor/ (pinned cannon-es, unmodified in the first physics version; three.js; ORT wasm)
             assets/ (vehicle.json read by game and port, tracks, ghosts, models, data)
docs/        physics.md architecture.md usage.md
reports/     plan.md literature-review.md source-ledger.md preregistration.md expNN_*.md summary.md
paper/       paper.md.j2 refs.bib → paper.pdf (pandoc + typst)
results/     <experiment>/<run>/run.json + eval CSVs (small, tracked in git)
```

`uv run python scripts/07_make_report.py` is the one command that turns `results/` into every table, figure, report, the README results block, `paper.pdf` and the website's data. No number is typed by hand.

## Order of work

Each step is a feature branch with a self-merged pull request; commits carry no AI attribution.

The order is: literature review, then write everything and push it, then test, then run the training that takes long.

### Publishing rule (every model, in every phase)

Everything is pushed to GitHub and Hugging Face at three points for each model:

| When | What the push adds |
|---|---|
| Before its training starts | The code and config the run will use, with the run registered as "planned", so the commit recorded in the run is public before it starts |
| After its training finishes | The checkpoint and training curves; the run marked "trained, not yet evaluated" |
| After its evaluation | Its scores, replays and exported policy; the run marked "evaluated" |

Each push carries the whole project in its current state, regenerated by one command (`08_publish.py`): the detailed paper (PDF), the results files, the per-experiment reports and the summary, the detailed README with its results tables, the docs and the website (GitHub Pages redeploys).
- **On Hugging Face** the model's repo gets its checkpoint, exported policy, config, results and model card. The paper, reports and results are mirrored to a project repo in the same collection.
- **Unfinished runs are shown as such.** The paper, README and site carry a run-status table (planned, training, trained, evaluated) built from the results files; sections whose runs are not finished say so instead of showing numbers.
- **In a queue,** the push after one model's evaluation also registers the next model, so it serves as that model's "before training" push.
- **Mechanics:** the run queue makes these pushes itself, each as its own commit on a short-lived branch merged by pull request. Credentials and `.env` files are never included.
- **Cost:** an estimated 2–3 minutes per run, about 4 hours over all runs, on top of the run tables.

### Stage A — literature review, then write everything and push it (target: hours 0–5)

The literature review is finished and pushed before any code is written. Then all code, the README and the preregistration are written and pushed to GitHub before the testing stage and before any training. Tests are written next to each module; at this stage only quick checks are run (imports, a module's own fast tests), so nothing long holds up the push.

The hour targets are estimates of my own working speed, not measurements; I report progress against them. Stage A is roughly 9,000–10,000 lines including tests and docs. The slowest parts are re-checking the literature claims against their sources, reading the cannon-es source closely enough to port it exactly, and the detailed comments. To shorten it, the browser front end and the reporting and website pages are written in parallel by subagents against fixed file formats (track file, vehicle parameters, results schema); I write the simulator port, training and evaluation code myself and review everything before it is merged.

| Order | What is written |
|---|---|
| 1 | **Scaffold:** `uv init` with Python 3.13, PyTorch 2.15 `cu132` index, `.env`, git, public GitHub repo, and a Linux virtual environment in WSL2. **This plan in full** is written to `reports/plan.md` in the first commit: every section, table and decision, nothing summarised away. It is kept up to date afterwards, with each later change added as a dated amendment. |
| 2 | **Literature review**, completed and pushed before any code. The key papers are read in full (Fuchs et al. 2021, GT Sophy, Song et al. 2021 and 2023, Kaufmann et al. 2023, Green et al. 2026, Rudin et al. 2021, Huang et al. 2022, Gallici et al. 2025, Kirk et al. 2023, Cobbe et al. 2019 and 2020, Agarwal et al. 2021, Patterson et al. 2024) and the cannon-es source is read file by file. Every claim from the research reviews is re-checked against its source and logged in the source ledger. Any better existing solution it turns up replaces the plan's choice → `reports/literature-review.md`, `reports/source-ledger.md`. |
| 3 | **Preregistration** → `reports/preregistration.md`, fixing procedures (metric, splits, how the reference time and step budget are chosen, trim order). **README** with what the project is, how it works, and how to install, play, train and reproduce; its results block is filled by the report script later. |
| 4 | **Game:** vendored cannon-es (pinned, unmodified), ribbon and triangle mesh, five campaign tracks with hills, banking and jumps, three.js chase camera, keyboard controls, timer, checkpoints, restart and respawn, ghosts, medals, "watch the AI drive live" and random unseen tracks (both switch on once model and track files exist), local server and GitHub Pages workflow. |
| 5 | **Simulator:** the batched PyTorch port (rigid body, wheel rays against triangles, suspension, friction, rules) and the Node.js parity harness. |
| 6 | **Tracks:** generator with certification, track-set builder with disjoint seed ranges, test-access log. |
| 7 | **Model and training:** observations, environment, PPO with checkpoint and resume, the PQN baseline, TensorBoard dashboard, timestamped logging. Also the phase 4 options (behaviour cloning, DAgger, scripted-driver reward term, scripted-driver start states), switched off by default. |
| 8 | **Baselines:** random, untrained network, scripted driver, CEM planner. |
| 9 | **Experiments:** one evaluator for all drivers, metric and statistics, a config for every run in phases 1–6, the resumable run queue (`05_run_matrix.py`), the publish command for GitHub and Hugging Face (`08_publish.py`) that the queue calls at the three push points, policy export. |
| 10 | **Reporting:** the one-command report pipeline, report and paper templates, the results explorer page, docs. |

### Stage B — test and fix (target: hours 5–9)

Everything is run the way a user would run it, and each fix is pushed. This is the least predictable stage: parity debugging and getting PPO to learn are where time can go.

- The full test suite on Windows and in WSL2: unit tests, tick-by-tick parity against cannon-es, browser tests.
- Hand-driving the game to tune the handling, then freezing the vehicle parameters.
- Quirks of the original engine found while testing are written down as candidate exploits, not fixed.
- `bench_sim.py` on native Windows and in WSL2 (with and without `torch.compile`): at least 60,000 agent-steps/s, and the measured figure replaces the hour estimates.
- An end-to-end smoke run: two minutes of training on a flat track with one bend → evaluate → export → report → website.
- Generating and certifying the three track sets; split and leakage tests on the real files.
- A hand check of the metric code on a tiny case and a read-through for leakage.
- Tag `v0.1-core`, push, and add the dated simulator-freeze amendment to the preregistration.

### Stage C — training (from about hour 9)

| Step | Runs | Target |
|---|---|---|
| C1 | **Model 1 — specialist, seed 0, campaign track 1** (about 0.4 h, longer if it is still improving) with the random, untrained and scripted baselines. Evaluate and watch its replay for physics exploits. Pushed three times, per the publishing rule. | by hour 10 |
| C2 | **Model 2 — generalist, seed 0 (the pilot)** on the training pool, about 4 h, longer if it is still improving. Evaluate on the test tracks against the training-free baselines; add a short report and both models as ghosts in the game. Pushed three times, per the publishing rule. If its validation score is still rising, the same run resumes afterwards until it plateaus. | by hour 15 |
| Gate | **Learning gate.** Both first models must learn properly before anything else is trained. If one does not, the run queue stops and the fix loop runs until it does; then the recipe is frozen. | no time limit |
| C3 | **Run phase 2:** one seed of every experiment (generalist, blind control, PQN baseline, specialists from scratch and fine-tuned, ablations including training-set size) and the CEM reference times, through the resumable queue. Results are labelled preliminary. | |
| C4 | **Run phase 3, if an exploit or quirk was used:** write the exploit report with replays, patch both physics implementations, rerun parity, re-score the earlier models, then train the patched generalist and specialists. | |
| C5 | **Run phase 4:** a focused literature search on imitation and demonstration-guided RL for racing, then behaviour cloning, DAgger and the three hybrids, each compared with the generalist that learned on its own. | |
| C6 | **Run phase 5:** hyperparameter sensitivity. Report how far the test score moves. | |
| C7 | **Run phase 6:** seed sensitivity, confirmatory runs first (generalist, blind control, single-track ablation). Compute the spread between seeds, test H1–H3, check each single-seed result against its multi-seed version. | |
| C8 | **Final checks:** triple check of claims, citations and statistics; final paper (author Libor "Burny" Burian, dated, AI statement at the end), README, docs, site. | |

These targets put the first two models at about hour 15, three hours inside the 18-hour target. The target is soft: it is fine to pass it. Nothing is cut short to meet it, neither the generalist's training time nor tests, the preregistration or the game. Models that learn properly matter more than the clock. The aim is at least one interesting good result, and negative results alone are also an acceptable outcome, reported in full.

The laptop has to stay plugged in and awake for the training week; I will ask before changing any sleep setting.

## Verification

- **Physics, against the original (the main physics test):** for thousands of random states and inputs, one tick of the PyTorch port in float64 equals one tick of the vendored cannon-es code in Node.js within 1e-9; ten-second driven runs stay close; the scripted driver's finish tick on campaign track 1 is the same in both; rules (checkpoints, finish, out) fire on the same tick in both.
- **Physics, sanity checks kept:** free fall, rest compression, jump range = v² sin 2θ / g, parked on a 10° bank stays put, coasting never gains energy, a drop from the generator's maximum landing height is absorbed, float32 on the GPU stays close to float64, identical replays are bitwise equal on one device, results do not depend on batch size.
- **Tracks and environment:** seeds reproduce byte-identical tracks; splits are disjoint; the stored witness replays to the finish; the training sampler never serves a test track; progress reward sums to track length; terminate and truncate flags are correct.
- **Learning code:** advantage estimates match a NumPy reference; resuming from a checkpoint equals an uninterrupted run; PPO solves a straight track.
- **Export and browser:** the exported observation-and-policy graph picks the same action as PyTorch ≥ 99.9 % of the time; Playwright tests for no console errors, tick time budget, the AI finishing campaign track 1 within 0.05 s of the Python time, and the site working under the Pages sub-path. I also drive the game by hand in the built-in browser.
- **End to end:** a two-minute smoke run through train → evaluate → export → report, then a check that the Pages URL and the Hugging Face collection load.

## Main risks

| Risk | Early check | Response |
|---|---|---|
| An exploit goes unnoticed and is reported as ordinary skill | evaluator flags (speed, energy, airborne distance); watching the best and strangest replays of each block | none needed beyond reporting it: exploits are kept as findings, and phase 3 retrains on patched physics for comparison |
| Simulator slower than estimated | `bench_sim.py` in the testing stage, Windows vs WSL2 | train in WSL2 with `torch.compile`; then fewer tensor operations per tick; then the preregistered trim order |
| The PyTorch port drifts from the cannon-es original | tick-by-tick parity tests, run on every change | fix the port; the original is not edited in the first physics version, and later patches go into both |
| Hard landings bottom out the suspension, or wheels sink into ramp lips | drop test at the generator's maximum landing height; hand-driving the jump on track 1 | smoother ramp transitions and a lower landing-height limit in the generator; if needed, one bump-stop rule added identically on both sides |
| cannon-es is no longer maintained (last release 0.20.0) | — | it is vendored and pinned, so nothing can change underneath; Rapier's car is the documented second choice |
| Generalist never learns jumps | pilot finish rate by track feature | preregistered difficulty curriculum in training only (track complexity adapted from the crash ratio, as in Song et al. 2021); test distribution unchanged |
| Tuning on test tracks | split tests and the test-access log | all choices made on validation tracks |
| Week-long run interrupted | resume test | 10-minute checkpoints and the resumable queue |
| 18-hour target missed because writing, testing or fixing take longer | progress against the hour targets | accepted: the target is soft and nothing is skipped or cut short to meet it |
| The first models do not learn properly | the learning gate on campaign track 1 and the validation tracks | stop the run queue and run the fix loop until they pass; later phases shrink by the trim order; a model that never learns is reported as a negative result |

## Amendments

Changes made after the plan was approved, newest last. Changes to hypotheses, metric or analysis are also recorded in `reports/preregistration.md`.

| Date | Change | Why |
|---|---|---|
| 2026-11-14 | The report files are named `reports/literature-review.md`, `reports/preregistration.md` and `reports/source-ledger.md` (the approved plan said `00_literature_review.md`, `01_preregistration.md` and `source_ledger.md`). | The standing project instructions name these files. |
| 2026-11-14 | The standing instructions give 14 hours for the first training run with one seed; the plan's table uses the 18-hour figure agreed during planning. Both are soft targets: nothing is cut short to meet either, and the first models get more training time if they need it. | Recorded so the two figures are not confused. |
| 2026-11-14 | The GitHub repository is `BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game` (it was created as `simplified-trackmania-reinforcement-learning` and renamed). The project's own code is MIT-licensed. | Repository rename; a licence was needed for a public repository and MIT matches the reused code. |
| 2026-11-14 | No low-grip shoulder and no separate roll-over rule. The road has one grip level to its edge; a tipped car falls through the ribbon and is out by the "3 m below the track" rule. | Read from the cannon-es source: grip is per wheel, not per surface, and only the wheel rays touch the road. |
| 2026-11-14 | The car's top speed comes from an engine force that falls off linearly with speed, and coasting applies a small brake impulse; both are part of the action-to-wheel mapping in `docs/formats.md`, identical in the game and the port. | cannon-es has no rolling resistance when the brake is off and no drag beyond 1 % per second damping, so the car would otherwise accelerate without limit. |
| 2026-11-14 | Run matrix additions after the fact-check: a 10,000-track level in the training-set-size ablation and a PPO ablation with a single 9-way action head (each one seed in phase 2 and seeds 2–3 in phase 6, about 12 hours in total). Both are cut together with the PQN baseline's extra seeds if time is short. Confirmation of H1–H3 uses t-tests on per-seed scores; see the preregistration's amendments. | Cobbe et al. see the gap close near 10,000 levels; PQN needs a 9-way head; Colas et al. advise against bootstrap tests below about 50 samples. |
| 2026-11-14 | Starting PPO settings: entropy coefficient 0.01 and a KL-adaptive learning rate in [1e-5, 1e-2] (Rudin et al. 2021). | CleanRL's GPU file defaults to no entropy bonus. |
| 2026-11-14 | Training and official evaluation run in WSL2 with `torch.compile`: the eager simulator is limited by kernel-launch overhead (about 280 ms per agent step at any batch size, ~13,000–64,000 agent steps/s), compiled it reaches about 640,000 agent steps/s with 16,384 cars (first compilation about 200 s, then cached). The run queue itself runs on Windows (where `gh` and `hf` are logged in) and sends GPU work to WSL. | Measured in stage A (`scripts/bench_sim.py` writes the figures to `results/bench/`). |
| 2026-11-14 | The browser front end and the report pipeline were written by subagents against the fixed formats in `docs/formats.md`, as planned; everything else by the main session. | As planned in stage A. |
| 2026-11-14 | Out-of-range test set variants (one factor each, by seed): tight turns (radius 14–22 m instead of 22–120 m), steep hills (grade 6–10 % instead of 0–6 %), long jumps (gap 14–20 m instead of 6–14 m, jump probability 0.8), long tracks (1,400–2,400 m instead of 600–1,400 m). | Fixes the "one factor beyond the training range" definition before any test-set read. |
| 2026-11-14 | Time-cost ablation value: 0.06 per agent step (break-even at 12 m/s with the 0.1-per-metre progress reward); fall-penalty ablation: 5.0 (equal to 50 m of progress). | Linesight's ratio would make every step negative below 120 m/s; the ablation keeps its form at a scale where slow driving, not all driving, is penalised. |
| 2026-11-14 | Simulator freeze (physics `v1`, vehicle.json `37bc8a37f90ab12d`, tag `v0.1-core`) and scripted driver tuned (a_lat 6.0, gain 0.3); see the preregistration amendment. The car was not hand-tuned: its starting values drove sensibly (the browser agent and the user each finished campaign track 1, drops up to 15 m are absorbed, the scripted driver certifies 92–100 % of generated tracks). | Stage B gate before training. |
| 2026-11-14 | Exploit detector: the coasting-energy flag counts rotational kinetic energy (½ωᵀIω/m); the airborne-distance flag is dropped because the energy flag covers it. | The first specialist's validation run raised a false flag in a crash where spin turned back into linear speed (preregistration amendment). |
| 2026-11-14 | Evaluation reproducibility: every official evaluation resets the compiler (`torch.compiler.reset`) and compiles the simulator for exactly its batch, so a set's result does not depend on what was evaluated before it in the same process; two fresh evaluations of the generalist on the 1,000-track training sample agreed on every track. Before the fix, the training-sample evaluation ran after validation in the same process and reused a kernel recompiled for dynamic shapes: 741 of the 1,000 tracks ended on a different tick and the score moved from SWS 1.234 / 82.6 % finished (fresh) to 1.218 / 81.6 %. Float32 rounding differences of this size change individual outcomes on chaotic tracks, so this is also a measure of the evaluation's numerical noise floor. A track evaluated alone can still differ from the same track inside a batch (the policy's matrix products round differently at different batch sizes); official scores always evaluate a whole set together. All models and baselines evaluated before the fix were re-scored. | Found while replaying exploit-flagged tracks: one replayed exactly, another did not. |
| 2026-11-14 | Every run evaluated on the full set list (generalists, blind control, PQN, training-set-size and other ablations, hyperparameter and phase-4 runs) is also evaluated on the 200 unfiltered test seeds (`test_unfiltered`), and so are the pilot generalist and the training-free baselines. | Implements the preregistered secondary evaluation on unfiltered test seeds (preregistration, section 3), which the generated run matrix had left out. |
