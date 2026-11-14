---
title: "RacingMania: can one reinforcement-learning driver handle unseen tracks in a 3D time-attack racing game?"
author: 'Libor "Burny" Burian'
date: "2026-11-14"
papersize: a4
fontsize: 10pt
header-includes: |
  ```{=typst}
  #show table: set par(justify: false)
  #show table: set text(size: 7.5pt)
  #show figure: set text(size: 8.5pt)
  ```
margin:
  x: 2cm
  y: 2.2cm
---

# Abstract

Reinforcement-learning (RL) drivers in racing games are usually trained and tested on the same track, and the few published tests on unseen car tracks are negative or small. We built RacingMania, an original browser time-attack racer with hills, banked turns and jumps on the car physics of cannon-es, ported the same car physics line by line to batched PyTorch so that thousands of cars, each on its own procedurally generated track, train on one GPU, and preregistered three hypotheses: that a PPO policy trained on many generated tracks drives unseen tracks better than a scripted pure-pursuit driver (H1), that track diversity is what makes this possible (H2), and that the policy uses its view of the road ahead (H3). The primary metric is success-weighted speed (SWS), the mean over held-out tracks of finished × reference time / agent time.
The Generalist (PPO) (1 seed) finished 81.5 % of the 971 unseen test tracks (95 % CI 79.0 % to 83.8 %), with a success-weighted speed (SWS) of 1.214 [1.177, 1.250]; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.
On the tracks it finished, it needed 0.67 times the scripted driver's time (32.7 % less time on average).
Its finish rate was 8.1 percentage points more than the blind control (73.3 %, 1 seed); 1.6 percentage points more than the pqn (79.8 %, 1 seed); 81.5 percentage points more than the untrained network (0.0 %, 10 seeds); 81.5 percentage points more than the random actions (0.0 %, 3 seeds).
The preregistered hypotheses H1-H3 are not tested yet: they need test results from 10 generalist seeds and 5 seeds each of the blind control and the single-track ablation (1 generalist seeds so far).

# Introduction

TrackMania-style time-attack games ask for one thing: finish an unfamiliar track as fast as possible. RL agents have reached record level on fixed tracks ([Linesight](https://github.com/Linesight-RL/linesight); GT Sophy, [Wurman et al. 2022](https://doi.org/10.1038/s41586-021-04357-7); [Fuchs et al. 2021](https://arxiv.org/abs/2008.07971)), but evidence that one policy can drive *unseen* car tracks is thin: the Gran Turismo agent of Fuchs et al. got stuck in a hairpin on a new track, and the Learn-to-Race SAC baseline dropped from 31.1 % to 11.8 % track completion on an unseen track ([Herman et al. 2021](https://arxiv.org/abs/2103.11575)). The closest positive result is for drones: [Green et al. 2026](https://arxiv.org/abs/2606.27348) (a preprint) report 69.8 % success on unseen generated tracks for a PPO generalist against 0.0 % for single-track policies.

This paper asks four questions, fixed in the preregistration ([`reports/preregistration.md`](https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game/blob/main/reports/preregistration.md)) before any model was trained:

1. Can one policy, trained on many procedurally generated tracks, drive tracks it has never seen, and how does it compare with a classical scripted driver?
2. Is track diversity in training what makes that possible?
3. Does the policy use its view of the upcoming track?
4. How much time does a generalist lose against a specialist trained on one track?

**What is new** is a combination, limited to what our searches found ([`reports/literature-review.md`](https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game/blob/main/reports/literature-review.md), section 14): the same car code runs in a playable browser game and, as a tested batched port, in training; the generalist is tested on held-out generated 3D tracks with hills, banking and jumps, with several seeds against a scripted driver, a blind control and a training-set-size ablation; and exploits of the physics engine are reported as findings rather than removed.

**Status of this version.** 20 of 116 runs in the preregistered run matrix are evaluated. The confirmatory tests of H1-H3 need all preregistered seeds and are not complete; every number from fewer seeds or from the validation set is labelled preliminary.

# Background and related work

**RL drivers for racing games.** Linesight (IQN, TrackMania Nations Forever) observes 40 upcoming reference-line points in the car frame and acts with 12 keyboard combinations every 50 ms; TMRL ([repository](https://github.com/trackmania-rl/tmrl)) uses SAC with lidar over screenshots; Fuchs et al. 2021 and GT Sophy use SAC variants with centreline progress rewards and upcoming course points in Gran Turismo. All train and test on the same tracks. Our observation (upcoming centreline points in the car frame) and our progress reward follow this line of work; unlike these projects we use progress alone, so the extra terms they add (time cost, crash penalty) are ablations here.

**Minimum-time racing in fast simulators.** In drone racing, PPO with a gate-progress reward and random start states along the course is standard ([Song et al. 2021](https://arxiv.org/abs/2103.08624), [Song et al. 2023](https://arxiv.org/abs/2310.10943), [Kaufmann et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/)). Green et al. 2026 trained a PPO generalist on generated tracks, compared it with per-track specialists (the generalist was 14.52 % slower) and fine-tuned specialists from it; their findings are our stated expectations.

**Zero-shot generalisation.** [Kirk et al. 2023](https://arxiv.org/abs/2111.09794) recommend held-out seeds, one test set per kind of generalisation, several training-set sizes and reporting test performance before the train-test gap. CoinRun and Procgen ([Cobbe et al. 2019](https://arxiv.org/abs/1812.02341), [Cobbe et al. 2020](https://arxiv.org/abs/1912.01588)) show the gap closing with about 10,000 training levels; level-replay methods ([Jiang et al. 2021](https://arxiv.org/abs/2010.03934)) are our reserve curriculum.

**Algorithms.** PPO ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)) adapted from CleanRL's GPU script ([Huang et al. 2022](https://jmlr.org/papers/v23/21-1342.html), [implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)) with the massively parallel settings of [Rudin et al. 2021](https://arxiv.org/abs/2109.11978); PQN ([Gallici et al. 2025](https://arxiv.org/abs/2407.04811)) is the one alternative algorithm.

**Evaluation.** We follow [Henderson et al. 2018](https://arxiv.org/abs/1709.06560), [Colas et al. 2019](https://arxiv.org/abs/1904.06979) (Welch's t-test on per-seed scores), [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264) (interquartile mean, probability of improvement, bootstrap intervals) and [Patterson et al. 2024](https://arxiv.org/abs/2304.01315). SWS adapts SPL ([Anderson et al. 2018](https://arxiv.org/abs/1807.06757)) from path length to time.

**Car physics.** The car is the raycast vehicle of [cannon-es](https://github.com/pmndrs/cannon-es) (MIT), whose code follows Bullet's `btRaycastVehicle`: one rigid chassis and four suspension rays that apply spring, damper, side-friction and engine impulses. We found no permissively licensed TrackMania clone and no public batched GPU port of a raycast vehicle.

# Method

## Game and physics

The browser game (three.js and the unmodified, vendored cannon-es 0.20.0) and the training simulator share one car model and one track format ([`docs/formats.md`](https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game/blob/main/docs/formats.md)). Physics runs at a fixed tick of 0.01 s; the agent, like a keyboard player, acts once every 5 ticks. The training simulator is a line-by-line batched PyTorch port of `RaycastVehicle`, `WheelInfo`, `Body.integrate`, the step order of `World.internalStep` and cannon's ray-against-triangle test ([`docs/physics.md`](https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game/blob/main/docs/physics.md)). Its tests compare it with the original running in Node.js: one tick from random states must agree to 1e-9 in 64-bit floats, and driven runs must stay close (Appendix C). Known quirks of the original (unlimited straight-line braking grip, single-ray wheels, no rolling resistance when coasting) are kept and mirrored, and listed as candidate exploits.

A run ends *finished* when the car passes the last checkpoint, and *out* when the chassis falls more than 3 m below the track, no wheel touches the track for 5 s, or progress stalls for 2 s.

## Tracks

Tracks are ribbons in the sky: a centreline with banking, width, gaps for jumps and checkpoints ([Perantoni & Limebeer 2015](https://doi.org/10.1115/1.4028253)), generated from a seeded sequence of TORCS-style straights and arcs ([TORCS track manual](https://torcs.sourceforge.net/api/track_manual.html)) with height changes, banking and parabolic-kicker jumps (ranges in Appendix B). A generated track enters a set only if the frozen scripted driver finishes it at three speed settings; its finishing time is the track's reference time. Sets use disjoint seed ranges:

| Set | Seeds | Generated | Geometric rejects | Certification rejects | Kept | Mean length (m) | Mean reference time (s) |
|---|---|---|---|---|---|---|---|
| oor | 3,000,000 - 3,000,199 | 200 | 0 | 13 | 187 | 1335 | 66.0 |
| test | 2,000,000 - 2,000,999 | 1,000 | 0 | 29 | 971 | 1117 | 50.9 |
| test_unfiltered | 2,000,000 - 2,000,199 | 200 | 0 | 5 | 200 | 1102 | 50.6 |
| train | 0 - 49,999 | 50,000 | 0 | 1,051 | 48,949 | 1112 | 50.8 |
| val | 1,000,000 - 1,000,199 | 200 | 0 | 3 | 197 | 1130 | 51.5 |

Five hand-made campaign tracks serve the game, the specialists and a hand-made out-of-distribution test. Every read of the test and out-of-range sets is logged.

## Agent

The observation holds the car-frame velocity and spin, the up vector, wheel contacts, steering angle, previous action, offset and heading error to the centreline, and upcoming centreline points in the car frame with width and gap flags. The action is two 3-way choices (steer left/none/right; gas/coast/brake). The reward is metres of centreline progress per step and nothing else. Training episodes mix standing starts with random starts along the centreline. The agent never sees the scripted driver's actions, states or times outside phase 4. PPO uses separate actor and critic networks, a KL-adaptive learning rate, observation normalisation, bootstrapping on time limits and summed log-probabilities over the two heads (settings in Appendix A).

## Metric

For a track set $T$, with $T_{\mathrm{ref},i}$ the scripted driver's time on track $i$ and $T_i$ the agent's,

$$\mathrm{SWS} = \frac{1}{|T|} \sum_{i \in T} \text{finished}_i \cdot \frac{T_{\mathrm{ref},i}}{T_i},$$

where $\text{finished}_i = 1$ if the agent reaches the last checkpoint within three times the reference time without being out. Each agent is run once per track from a standing start with its most probable action (the simulator is deterministic), with the checkpoint at the end of its step budget. SWS = 1.0 means finishing every track as fast as the scripted driver, which scores 1.0 by construction on the filtered sets. Secondary numbers are the finish rate, the time ratio on finished tracks, the finish rate by track feature, and the train-test gap.

## Baselines, controls and ablations

Training-free: uniformly random actions and an untrained network (the floor) and the scripted driver (the reference). Controls: a blind generalist without the upcoming-track points (H3) and generalists trained on 1 to 10,000 tracks (H2). Alternative algorithm: PQN, with a PPO ablation using the same single 9-way action head. Exploratory ablations: continuous actions, 5 or 40 look-ahead points, a time cost, 10 Hz actions, a fall penalty. Specialists are trained per campaign track from scratch and fine-tuned from the generalist. Phase 4 adds imitation of the scripted driver and hybrids; phase 5 a descriptive hyperparameter sweep.

## Statistical analysis

All three hypotheses are tested once, on the test set and the first physics version, after the seed-sensitivity phase. Intervals are 95 % two-level percentile bootstrap intervals with 10,000 resamples: seeds and tracks are resampled with replacement, the same resampled tracks are used for both agents of a difference (paired on tracks), which is Owen's pigeonhole bootstrap for crossed designs ([Owen 2007](https://arxiv.org/abs/0712.1111)). Confirmation uses one-sided t-tests on per-seed scores (one-sample against 1.0 for H1; Welch, difference above 0.1, for H2 and H3), Holm-corrected across the three ([Holm 1979](https://www.jstor.org/stable/4615733)) at α = 0.05. A hypothesis is *refuted* if its interval lies entirely below its threshold, *confirmed* if its Holm-adjusted p-value is below α, and *inconclusive* otherwise. The interquartile mean and the probability of improvement ([Agarwal et al. 2021](https://arxiv.org/abs/2108.13264)) are reported as cross-checks, and seed sensitivity as the spread over seeds and the share of variance from seeds versus tracks (ANOVA estimators of a crossed random-effects model).

# Experiments

Runs follow the preregistered order: (1) the first two models (seed 0), which must pass a learning gate on validation tracks before the recipe is frozen; (2) one seed of every experiment; (3) patched physics, only if an exploit is used; (4) imitation and hybrids; (5) hyperparameter sensitivity; (6) seed sensitivity. Current status of every run in the matrix:

| Phase | not started | planned | training | trained | evaluated | total |
|---|---|---|---|---|---|---|
| baselines | 0 | 0 | 0 | 0 | 15 | 15 |
| 1-first-models | 0 | 0 | 0 | 0 | 2 | 2 |
| 2-single-seed | 21 | 0 | 0 | 1 | 3 | 25 |
| 4-imitation-hybrids | 5 | 0 | 0 | 0 | 0 | 5 |
| 5-hyperparameters | 4 | 0 | 0 | 0 | 0 | 4 |
| 6-seed-sensitivity | 65 | 0 | 0 | 0 | 0 | 65 |
| **all** | 95 | 0 | 0 | 1 | 20 | 116 |

# Results

## Learning gate (first models)

| Model | Bar (preregistration, section 7) | Result | Gate |
|---|---|---|---|
| Specialist (campaign-01) [pilot, seed 0] | finishes its track and is at least as fast as the scripted driver | finished: yes; time 19.71 s vs reference 30.17 s | **passed** |
| Generalist (PPO) [pilot, seed 0] | finishes >= 80 % of validation tracks with SWS >= 0.8 | finished 82.7 %, SWS 1.242 | **passed** |

## Main evaluation

| Configuration | Role | Seeds | Test SWS [95 % CI] | Test finished | OOR SWS [95 % CI] | OOR finished | Campaign SWS [95 % CI] | Campaign finished |
|---|---|---|---|---|---|---|---|---|
| CEM planner | Reference | 1 | — | — | — | — | 1.551 [1.409, 1.726] | 100.0 % |
| Scripted driver | Reference | 1 | 0.998 [0.995, 1.000] | 99.8 % | 1.000 [1.000, 1.000] | 100.0 % | 1.000 [1.000, 1.000] | 100.0 % |
| Random actions | Control | 3 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Untrained network | Control | 10 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Blind control | Control | 1 | 1.010 [0.971, 1.048] | 73.3 % | 0.605 [0.506, 0.703] | 43.9 % | 1.443 [1.321, 1.590] | 100.0 % |
| Generalist (PPO) [pilot, seed 0] | Main method | 1 | 1.199 [1.160, 1.237] | 80.2 % | 0.788 [0.678, 0.899] | 52.4 % | 1.596 [1.447, 1.799] | 100.0 % |
| Generalist (PPO) | Main method | 1 | 1.214 [1.177, 1.250] | 81.5 % | 0.875 [0.769, 0.982] | 57.8 % | 1.590 [1.451, 1.783] | 100.0 % |
| PQN | Baseline algorithm | 1 | 1.054 [1.020, 1.086] | 79.8 % | 0.725 [0.631, 0.820] | 55.1 % | 1.391 [1.270, 1.517] | 100.0 % |
| Specialist (campaign-01) [pilot, seed 0] | Specialist | 1 | — | — | — | — | 0.306 [0.000, 0.918] | 20.0 % |

Track counts: Test (unseen generated): 971 tracks; Validation: 197 tracks; Training sample: 1,000 tracks; Own training tracks: 1,000 tracks; Out-of-range test: 187 tracks; Campaign (hand-made): 5 tracks; Unfiltered test: 200 tracks.

![SWS and finish rate on the test set (971 tracks), mean over seeds with 95 % two-level bootstrap intervals (10,000 resamples of seeds and tracks); n = number of seeds. With one seed the interval reflects track sampling only.](../reports/figures/sws_by_method.svg){width=100%}

## Hypotheses

H1: the generalist's test SWS exceeds 1.0, the scripted driver's score. H2: SWS(trained on the whole pool) minus SWS(trained on 1 track) exceeds 0.1. H3: SWS(generalist) minus SWS(blind control) exceeds 0.1.

|  | Tested quantity | Seeds (have/required) | Estimate [95 % CI] | Test statistic | p (one-sided) | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| H1 | SWS > 1.0 | generalist 1/10 | 1.214 [1.177, 1.250] (preliminary) | — | — | — | **not yet tested** |
| H2 | SWS difference > 0.1 | generalist 1/10, size-1 0/5 | — | — | — | — | **not yet tested** |
| H3 | SWS difference > 0.1 | generalist 1/10, blind 1/5 | 0.204 [0.164, 0.245] (preliminary) | — | — | — | **not yet tested** |

Hypotheses marked "not yet tested" lack the preregistered seeds; any estimate shown for them is preliminary and is not a test.

Cross-checks (exploratory, Agarwal et al. 2021):

| Generalist vs | Set | Seeds (generalist / other) | IQM generalist | IQM other | P(generalist > other) | Time ratio where both finish | Tracks both finish |
|---|---|---|---|---|---|---|---|
| Blind control | test | 1 / 1 | 1.464 | 1.291 | 0.874 | 0.927 | 650 |
| PQN | test | 1 / 1 | 1.464 | 1.295 | 0.854 | 0.886 | 671 |
| Scripted driver | test | 1 / 1 | 1.464 | 1.000 | 0.815 | 0.673 | 790 |
| Random actions | test | 1 / 3 | 1.464 | 0.000 | 0.907 | — | 0 |
| Untrained network | test | 1 / 10 | 1.464 | 0.000 | 0.907 | — | 0 |

## Training-set size

| Training tracks | Configuration | Seeds | SWS [95 % CI] | Train - test gap [95 % CI] | Training score from |
|---|---|---|---|---|---|
| 48,949 | Generalist (PPO) | 1 | 1.214 [1.177, 1.250] | 0.036 [-0.015, 0.086] | train_own |

![Training-set-size ablation: score (top) and train-test gap (bottom) against the number of training tracks; the largest value is the generalist trained on the whole pool. Mean over seeds (n per point) with 95 % two-level bootstrap intervals.](../reports/figures/training_set_size.svg){width=70%}

## Specialists and the generalist on the campaign tracks

| Campaign track | Scripted time (s) | Agent | Kind | Finished | Mean time (s) | Per seed (s) |
|---|---|---|---|---|---|---|
| campaign-01 | 30.17 | CEM planner | reference | 1/1 | 20.33 | s0: 20.33 |
| campaign-01 | 30.17 | Scripted driver | reference | 1/1 | 30.16 | s0: 30.16 |
| campaign-01 | 30.17 | Generalist (PPO) [pilot, seed 0] | zero-shot | 1/1 | 20.02 | s0: 20.02 |
| campaign-01 | 30.17 | Generalist (PPO) | zero-shot | 1/1 | 20.29 | s1: 20.29 |
| campaign-01 | 30.17 | Specialist (campaign-01) [pilot, seed 0] | own track | 1/1 | 19.71 | s0: 19.71 |
| campaign-02 | 36.43 | CEM planner | reference | 1/1 | 25.25 | s0: 25.25 |
| campaign-02 | 36.43 | Scripted driver | reference | 1/1 | 36.43 | s0: 36.43 |
| campaign-02 | 36.43 | Generalist (PPO) [pilot, seed 0] | zero-shot | 1/1 | 24.45 | s0: 24.45 |
| campaign-02 | 36.43 | Generalist (PPO) | zero-shot | 1/1 | 24.60 | s1: 24.60 |
| campaign-03 | 48.50 | CEM planner | reference | 1/1 | 36.23 | s0: 36.23 |
| campaign-03 | 48.50 | Scripted driver | reference | 1/1 | 48.52 | s0: 48.52 |
| campaign-03 | 48.50 | Generalist (PPO) [pilot, seed 0] | zero-shot | 1/1 | 35.15 | s0: 35.15 |
| campaign-03 | 48.50 | Generalist (PPO) | zero-shot | 1/1 | 34.56 | s1: 34.56 |
| campaign-04 | 42.52 | CEM planner | reference | 1/1 | 22.38 | s0: 22.38 |
| campaign-04 | 42.52 | Scripted driver | reference | 1/1 | 42.52 | s0: 42.52 |
| campaign-04 | 42.52 | Generalist (PPO) [pilot, seed 0] | zero-shot | 1/1 | 21.27 | s0: 21.27 |
| campaign-04 | 42.52 | Generalist (PPO) | zero-shot | 1/1 | 21.60 | s1: 21.60 |
| campaign-05 | 56.07 | CEM planner | reference | 1/1 | 35.27 | s0: 35.27 |
| campaign-05 | 56.07 | Scripted driver | reference | 1/1 | 56.03 | s0: 56.03 |
| campaign-05 | 56.07 | Generalist (PPO) [pilot, seed 0] | zero-shot | 1/1 | 34.96 | s0: 34.96 |
| campaign-05 | 56.07 | Generalist (PPO) | zero-shot | 1/1 | 34.86 | s1: 34.86 |

![Finishing time on each campaign track: small marks are single seeds, large marks the mean over finishing seeds, x marks runs that did not finish (DNF); the black bar is the scripted driver's reference time. Specialists are scored on their own track; the generalist zero-shot.](../reports/figures/specialists_campaign.svg){width=100%}

## Finish rate by track feature

![Finish rate by track feature on the test set: jumps counted per track, banking and height range split into tertiles of that set (ranges in the labels). Mean over seeds and tracks in each bin with 95 % two-level bootstrap intervals. Secondary, descriptive analysis.](../reports/figures/finish_by_feature.svg){width=100%}

## Seed sensitivity

| Configuration | Seeds | Mean SWS | Min | Max | SD | Variance: seeds | Variance: tracks | Variance: seed x track | Tracks |
|---|---|---|---|---|---|---|---|---|---|
| Random actions | 3 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | 971 |
| Untrained network | 10 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | 971 |

_Not yet available: needs test results from two or more seeds of a configuration and of the generalist._

![Seed sensitivity: test SWS of every seed (dots) and the mean with its 95 % two-level bootstrap interval, for every configuration with at least two seeds.](../reports/figures/seed_spread.svg){width=80%}

## Physics exploits

_No exploit list yet (`results/exploits.json` does not exist)._

| Configuration | Set | Overspeed flags | Coasting energy-gain flags | Seeds | Tracks |
|---|---|---|---|---|---|
| Untrained network | OOR | 0 | 1 | 10 | 187 |
| Blind control | Own train | 0 | 2 | 1 | 1000 |
| Blind control | Train | 0 | 2 | 1 | 1000 |
| Blind control | Val | 0 | 1 | 1 | 197 |
| Generalist (PPO) [pilot, seed 0] | Train | 0 | 3 | 1 | 1000 |
| Generalist (PPO) | OOR | 0 | 3 | 1 | 187 |
| Generalist (PPO) | Test | 0 | 1 | 1 | 971 |
| PQN | Test | 0 | 1 | 1 | 971 |
| PQN | Own train | 0 | 1 | 1 | 1000 |
| PQN | Train | 0 | 1 | 1 | 1000 |

## Training curves of the generalist

![Training curves of Generalist (PPO); one line per seed (seeds 1). Return and finish rate are over training episodes (standing and random starts); validation SWS is the evaluator's score on the validation tracks.](../reports/figures/curves_generalist.svg){width=100%}

# Discussion

The results so far, in plain words:

- The Generalist (PPO) (1 seed) finished 81.5 % of the 971 unseen test tracks (95 % CI 79.0 % to 83.8 %), with a success-weighted speed (SWS) of 1.214 [1.177, 1.250]; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.
- On the tracks it finished, it needed 0.67 times the scripted driver's time (32.7 % less time on average).
- Its finish rate was 8.1 percentage points more than the blind control (73.3 %, 1 seed); 1.6 percentage points more than the pqn (79.8 %, 1 seed); 81.5 percentage points more than the untrained network (0.0 %, 10 seeds); 81.5 percentage points more than the random actions (0.0 %, 3 seeds).
- The preregistered hypotheses H1-H3 are not tested yet: they need test results from 10 generalist seeds and 5 seeds each of the blind control and the single-track ablation (1 generalist seeds so far).

A written interpretation will be added once the confirmatory runs are complete.

# Limitations

- **The reference is not optimal.** SWS compares with a scripted driver tuned on validation tracks, not with the fastest possible time; SWS above 1.0 means "faster than this driver", not near-optimal. A CEM planner gives near-optimal times on a few tracks only.
- **Filtered test sets.** Test tracks are kept only if the scripted driver finishes them, so the scripted driver scores 1.0 by construction and the test distribution excludes tracks it cannot drive; an unfiltered test set is reported as a secondary check.
- **One simulator.** All numbers come from the PyTorch port of one engine; quirks of cannon-es's raycast vehicle may be learned and exploited, and transfer to other physics is not tested.
- **Track model.** A ribbon cannot represent loops, wall-rides or open terrain; the generator's ranges define what "unseen" means (held-out seeds of the same distribution, plus one out-of-range set).
- **Seeds and tuning.** Five to ten seeds per configuration support the preregistered tests but not fine-grained comparisons; exploratory comparisons with one to three seeds are descriptive. The alternative algorithm gets the same tuning effort as PPO, which may still favour PPO, whose settings come from more prior work.
- **Prior-work comparison.** The expectations taken from Green et al. 2026 come from a preprint with a different vehicle and task.

# Conclusion

The game, the tested physics port, the track generator, the training and evaluation code and this report pipeline are complete; the confirmatory results are not yet available. This document is regenerated from the result files after every run.

# Appendix A: hyperparameters

PPO settings (logged in `results/generalist/seed1/run.json`):

| Setting | Value |
|---|---|
| `total_steps` | `1000000000` |
| `num_envs` | `16384` |
| `num_steps` | `32` |
| `lr` | `0.001` |
| `adaptive_lr` | `True` |
| `desired_kl` | `0.01` |
| `lr_min` | `1e-05` |
| `lr_max` | `0.01` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `num_minibatches` | `4` |
| `update_epochs` | `5` |
| `clip_coef` | `0.2` |
| `ent_coef` | `0.01` |
| `vf_coef` | `1.0` |
| `max_grad_norm` | `1.0` |
| `weight_decay` | `0.0` |
| `hidden` | `[256, 256]` |
| `action_mode` | `factored` |
| `eval_every_minutes` | `10.0` |
| `checkpoint_every_minutes` | `10.0` |
| `rotate_tracks_every` | `20` |
| `loaded_tracks` | `4096` |
| `seed` | `1` |
| `compile` | `True` |

Environment settings:

| Setting | Value |
|---|---|
| `random_start_prob` | `0.5` |
| `random_start_speed` | `[0.0, 30.0]` |
| `random_start_lateral` | `0.5` |
| `max_steps` | `1500` |
| `reward_scale` | `0.1` |
| `window_k` | `20` |
| `window_stride` | `3` |
| `blind` | `False` |
| `time_cost` | `0.0` |
| `fall_penalty` | `0.0` |
| `hint_weight` | `0.0` |
| `scripted_starts` | `False` |

# Appendix B: track generator ranges

| Range | Training distribution | Out-of-range variants |
|---|---|---|
| `length` | (600.0, 1400.0) | long_track: (1400.0, 2400.0) |
| `radius` | (22.0, 120.0) | tight_turns: (14.0, 22.0) |
| `angle` | (20.0, 150.0) | - |
| `straight` | (30.0, 140.0) | - |
| `grade` | (0.0, 0.06) | steep_hills: (0.06, 0.1) |
| `bank` | (0.0, 14.0) | - |
| `p_bank` | 0.6 | - |
| `p_jump` | 0.4 | long_jumps: 0.8 |
| `jump_min_straight` | 80.0 | - |
| `gap` | (6.0, 14.0) | long_jumps: (14.0, 20.0) |
| `kicker_h` | (0.8, 1.6) | - |
| `kicker_len` | (10.0, 16.0) | - |
| `drop` | (0.0, 2.0) | - |
| `landing_len` | (22.0, 30.0) | - |

Seed ranges: train 0-49,999; val 1,000,000-1,000,199; test 2,000,000-2,000,999; oor 3,000,000-3,000,199.

Scripted driver parameters (tuned on validation candidates, then frozen): `a_lat` = 6.0, `lookahead_gain` = 0.3, `lookahead_min` = 8.0, `lookahead_max` = 40.0, `a_brake` = 8.0, `jump_speed` = 25.0, `brake_margin` = 1.5, `speed_scale` = 1.0.

# Appendix C: physics parity

Measured agreement between the PyTorch port and cannon-es (from `results/parity.json`):

- `single_tick`: {'campaign-01': {'cases': 2000, 'max_abs_state_diff': 3.552713678800501e-15, 'median_abs_state_diff': 0.0, 'flag_mismatches': 0, 'contact_histogram': [658, 211, 254, 280, 597]}, 'campaign-03': {'cases': 2000, 'max_abs_state_diff': 7.105427357601002e-15, 'median_abs_state_diff': 0.0, 'flag_mismatches': 0, 'contact_histogram': [654, 208, 261, 285, 592]}, 'campaign-05': {'cases': 2000, 'max_abs_state_diff': 7.105427357601002e-15, 'median_abs_state_diff': 0.0, 'flag_mismatches': 0, 'contact_histogram': [657, 207, 255, 286, 595]}}
- `sequences`: {'campaign-01': {'ticks_js': 540, 'ticks_port': 540, 'max_abs_state_diff': 1.4210854715202004e-14, 'same_end': True}, 'campaign-03': {'ticks_js': 586, 'ticks_port': 586, 'max_abs_state_diff': 2.842170943040401e-14, 'same_end': True}, 'campaign-05': {'ticks_js': 545, 'ticks_port': 545, 'max_abs_state_diff': 3.552713678800501e-14, 'same_end': True}}

# Appendix D: reproducing this paper

`uv sync`, then `uv run python scripts/01_generate_tracks.py` (track sets), `scripts/04_baselines.py`, `scripts/05_run_matrix.py` (training, evaluation and publishing in the preregistered order) and `scripts/07_make_report.py` (every table, figure and this paper). Code: <https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game>.

# References

- Agarwal, R. et al. (2021). Deep reinforcement learning at the edge of the statistical precipice. NeurIPS. <https://arxiv.org/abs/2108.13264>
- Anderson, P. et al. (2018). On evaluation of embodied navigation agents. <https://arxiv.org/abs/1807.06757>
- Cobbe, K. et al. (2019). Quantifying generalization in reinforcement learning. ICML. <https://arxiv.org/abs/1812.02341>
- Cobbe, K. et al. (2020). Leveraging procedural generation to benchmark reinforcement learning. ICML. <https://arxiv.org/abs/1912.01588>
- Colas, C., Sigaud, O., Oudeyer, P.-Y. (2019). A hitchhiker's guide to statistical comparisons of reinforcement learning algorithms. <https://arxiv.org/abs/1904.06979>
- Fuchs, F. et al. (2021). Super-human performance in Gran Turismo Sport using deep reinforcement learning. IEEE RA-L. <https://arxiv.org/abs/2008.07971>
- Gallici, M. et al. (2025). Simplifying deep temporal difference learning. ICLR. <https://arxiv.org/abs/2407.04811>
- Green, Xing, Messikommer, Romero, Scaramuzza (2026). Bridging performance and generalization in reinforcement learning for agile flight. Preprint. <https://arxiv.org/abs/2606.27348>
- Henderson, P. et al. (2018). Deep reinforcement learning that matters. AAAI. <https://arxiv.org/abs/1709.06560>
- Herman, J. et al. (2021). Learn-to-Race: a multimodal control environment for autonomous racing. <https://arxiv.org/abs/2103.11575>
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2), 65-70. <https://www.jstor.org/stable/4615733>
- Huang, S. et al. (2022). CleanRL: high-quality single-file implementations of deep reinforcement learning algorithms. JMLR. <https://jmlr.org/papers/v23/21-1342.html>
- Jiang, M., Grefenstette, E., Rocktäschel, T. (2021). Prioritized level replay. ICML. <https://arxiv.org/abs/2010.03934>
- Kaufmann, E. et al. (2023). Champion-level drone racing using deep reinforcement learning. Nature. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/>
- Kirk, R. et al. (2023). A survey of zero-shot generalisation in deep reinforcement learning. JAIR. <https://arxiv.org/abs/2111.09794>
- Owen, A. B. (2007). The pigeonhole bootstrap. Annals of Applied Statistics 1(2), 386-411. <https://arxiv.org/abs/0712.1111>
- Patterson, A. et al. (2024). Empirical design in reinforcement learning. <https://arxiv.org/abs/2304.01315>
- Perantoni, G., Limebeer, D. J. N. (2015). Optimal control of a Formula One car on a three-dimensional track, part 1: track modeling and identification. <https://doi.org/10.1115/1.4028253>
- Rudin, N. et al. (2021). Learning to walk in minutes using massively parallel deep reinforcement learning. CoRL. <https://arxiv.org/abs/2109.11978>
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. <https://arxiv.org/abs/1707.06347>
- Song, Y. et al. (2021). Autonomous drone racing with deep reinforcement learning. IROS. <https://arxiv.org/abs/2103.08624>
- Song, Y. et al. (2023). Reaching the limit in autonomous racing: optimal control versus reinforcement learning. Science Robotics. <https://arxiv.org/abs/2310.10943>
- Wurman, P. R. et al. (2022). Outracing champion Gran Turismo drivers with deep reinforcement learning. Nature. <https://doi.org/10.1038/s41586-021-04357-7>
- cannon-es (2022). Version 0.20.0, MIT licence. <https://github.com/pmndrs/cannon-es>
- Linesight. TrackMania reinforcement-learning agent. <https://github.com/Linesight-RL/linesight>
- TORCS track manual. <https://torcs.sourceforge.net/api/track_manual.html>
- The complete annotated literature review with evidence labels: <https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game/blob/main/reports/literature-review.md>

# AI statement

This codebase and paper were written by Claude Code (Anthropic); human cleanup and rewriting are in progress; a human provided the exact specification that was executed.
