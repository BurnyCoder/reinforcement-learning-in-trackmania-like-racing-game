# Preregistration

**Project:** RacingMania — reinforcement-learning drivers for an original 3D time-attack racing game.
**Author:** Libor "Burny" Burian. **Written:** 2026-11-14, before any model was trained and before any result existed.
**Binding for:** hypotheses, primary metric, data splits, analysis and falsification criteria. Design background is in [`plan.md`](plan.md); sources are in [`literature-review.md`](literature-review.md).

Anything decided after this date is added under [Amendments](#10-amendments) with its date and reason. Analyses not listed here are reported as exploratory.

## 1. Questions

1. Can one policy, trained on many procedurally generated tracks, drive tracks it has never seen, and how does it compare with a classical scripted driver?
2. Is track diversity in training what makes that possible?
3. Does the policy use its view of the upcoming track?
4. Secondary: how much time does a generalist lose against a specialist trained on one track?

## 2. Setting

- **Game and simulator.** A car (cannon-es `RaycastVehicle`) on a ribbon of road with hills, banked turns and jumps. Training uses a batched PyTorch port of that code, tested tick by tick against the original (`docs/physics.md`). Physics tick 0.01 s; the agent acts every 5 ticks.
- **Actions.** Two 3-way choices per step: steer left / none / right, and gas / coast / brake.
- **Reward.** Metres of progress along the track's centreline per step. Nothing else.
- **Episode end.** *Finished:* the car reaches the final checkpoint. *Out:* more than 3 m below the track, airborne for 5 s, or no progress for 2 s. *Time limit:* a fixed cap; in evaluation it is 3× the track's reference time.
- **Learning on its own.** In every phase except phase 4 the agent never sees the scripted driver's actions, states or times.

## 3. Data: track sets

Generated tracks are identified by an integer seed. The sets use disjoint seed ranges.

| Set | Seeds | Size | Use |
|---|---|---|---|
| Training pool | 0 – 49,999 | 50,000 | Training only. A fixed sample of 1,000 (seeds 0–999) is also scored, to measure the train–test gap |
| Validation | 1,000,000 – 1,000,199 | 200 | Every decision made during development: the learning gate, the step budget, tuning the scripted driver, live dashboard curves |
| Test | 2,000,000 – 2,000,999 | 1,000 | Reported scores only |
| Out-of-range test | 3,000,000 – 3,000,199 | 200 | Secondary: each track has one generator factor beyond the training range (tighter turns, steeper hills, longer jumps or longer track) |
| Campaign | hand-made | 5 | The game, the specialist experiment, and a hand-made out-of-distribution set |

- A generated track enters a set only if a frozen scripted driver finishes it in the simulator at three speed settings. The number of rejected seeds is reported, and a secondary evaluation on the first 200 unfiltered test seeds is reported as well.
- The test and out-of-range sets are read only by the evaluator. Every read is appended to `results/test_access_log.jsonl`. They are first read after the recipe is frozen (section 7).
- The training sampler is tested never to serve a validation or test seed.

## 4. Primary metric: success-weighted speed

For an agent and a track set `T`, with `T_ref,i` the frozen scripted driver's finishing time on track `i` and `T_i` the agent's:

`SWS = (1/|T|) · Σ_i finished_i · T_ref,i / T_i`

- `finished_i` is 1 if the agent reaches the final checkpoint within `3 · T_ref,i` without being out, else 0.
- The agent is run once per track from a standing start with its most probable action at every step; the simulator is deterministic, so one run per track is enough.
- The checkpoint evaluated is the one at the end of the fixed step budget. The best checkpoint is never picked.
- 1.0 means "finishes every track as fast as the scripted driver". The scripted driver scores 1.0 by construction on filtered sets.
- The metric follows SPL (Anderson et al. 2018, Eq. 1) with time in place of path length. Unlike SPL the ratio is not capped at 1, because the reference is not the optimum.

## 5. Hypotheses and falsification

All three are tested once, after the seed sensitivity phase, on the **test** set and the **first physics version**.

| | Hypothesis | Quantity | Confirmed if | Refuted if |
|---|---|---|---|---|
| H1 | The generalist drives unseen tracks better than the scripted driver | `SWS(generalist)`, 10 seeds | the Holm-adjusted one-sided p-value for `SWS > 1.0` is below 0.05 | the 95 % interval lies entirely below 1.0 |
| H2 | Track diversity is what generalises | `SWS(trained on 50,000 tracks) − SWS(trained on 1 track)`, 10 and 5 seeds | the Holm-adjusted p-value for a difference above 0.10 is below 0.05 | the 95 % interval lies entirely below 0.10 |
| H3 | The agent uses the look-ahead | `SWS(generalist) − SWS(blind control)`, 10 and 5 seeds | as H2 | as H2 |

Otherwise the outcome is reported as inconclusive. 0.10 is the smallest difference treated as meaningful; it was chosen before any result.

**Expectations from prior work.** Green et al. 2026 (drone racing, PPO) report 69.8 % success on unseen tracks for a generalist against 0.0 % for single-track policies, and a generalist 14.52 % slower than specialists. We therefore expect H2 to hold, the generalist to finish well under 100 % of test tracks, and specialists to be faster on their own tracks. H1 is the least certain.

## 6. Analysis

- **Intervals.** Two-level percentile bootstrap, 10,000 resamples: seeds are resampled with replacement, then tracks. For a difference between two agents the same resampled tracks are used for both, so the comparison is paired on tracks. Intervals are 95 %.
- **p-values.** The share of bootstrap replicates on the wrong side of the threshold (1.0 for H1, 0.10 for H2 and H3), Holm-corrected across the three hypotheses.
- **Cross-checks, reported next to the main analysis:** Welch's t-test on per-seed scores (Colas et al. 2018); interquartile mean and probability of improvement from `rliable` (Agarwal et al. 2021).
- **Secondary numbers for every run:** finish rate; time ratio on tracks that both compared agents finish; finish rate by track feature (jumps, banking, hills); score on the training sample and the train–test gap; score on the out-of-range and campaign sets. Test performance is reported before the gap (Kirk et al. 2023).
- **Seed sensitivity.** For every configuration: minimum, maximum and standard deviation of the score over seeds, and the share of variance due to seeds versus tracks.
- **Every number** in the README, reports and paper is produced by the report script from result files.

## 7. Procedures fixed in advance

**Standard recipe.** PPO with separate actor and critic networks, two hidden layers of 256 units each; 1 billion agent steps for a generalist and 200 million for a specialist as starting budgets; training on the 50,000-track pool with 4,096 tracks loaded at a time.

**Learning gate.** No run beyond the first specialist (campaign track 1, seed 0) and the first generalist (seed 0) starts until both learn properly, judged on campaign track 1 and the validation set only:
- specialist: finishes its track in evaluation and is at least as fast as the scripted driver;
- generalist: finishes at least 80 % of validation tracks with a score of at least 0.8.

If a model misses its bar, the order of remedies is: more training time while its curve still rises; then checking the basics; then watching replays; then searching how others fixed the failure; then changes in this order: PPO settings, environment settings, a track-difficulty curriculum, the reward. Every attempt is logged, failed ones included, and every change to the recipe is an amendment here.

**Freeze.** Once both models pass, the recipe and step budgets are frozen and recorded as an amendment, together with the simulator version tag and the training environment (operating system, GPU, PyTorch version). All later runs use fresh seeds (1 and up); the two seed-0 models are reported separately. The test sets are first read after the freeze.

**Scripted driver.** Pure-pursuit steering with a curvature-based speed profile. Its two parameters are tuned on the validation set to minimise its own mean time while finishing, then frozen before the freeze above.

**Phases, in order.** (1) the first two models; (2) one seed of every experiment; (3) patched physics, only if an agent used an exploit or engine quirk; (4) imitation and hybrids; (5) hyperparameter sensitivity; (6) seed sensitivity, confirmatory runs first. The runs in each phase are listed in `plan.md`.

**Hyperparameter sweep.** It is descriptive. No setting from it replaces the frozen recipe.

**Trimming.** If training is slower than planned, runs are cut in this order, decided before phase 5 starts: hyperparameter settings, the PQN baseline's extra seeds, the action-rate and look-ahead ablations, the training-set-size ablations at 10 and 1,000 tracks, specialist seeds. The ten generalist seeds, five blind-control seeds and five single-track seeds are never cut. If they cannot all be run, the hypotheses are reported as not tested.

## 8. Physics exploits

If an agent gains time through a quirk of the engine, its models and scores are kept and the exploit is reported: what it is, which runs use it, when in training it first appears, how many seeds find it, how much time it is worth, and a replay.

- Detection: automatic flags (speed above 1.2 × top speed, energy gained while coasting; see the amendment of 2026-11-14 on the energy definition and the airborne-distance flag) and scheduled viewing of the best and strangest replays of each block.
- The confirmatory analysis stays on the first physics version, exploit or not. If an exploit is found, the patched physics is a separate tagged version; earlier models are re-scored on it, a generalist and specialists are retrained on it, and H1 is additionally reported on it.
- Known candidate exploits before any training are listed in `docs/physics.md`.

## 9. What is exploratory

The ablations other than training-set size, the PQN baseline, the fine-tuned specialists, phase 4 (imitation and hybrids), the hyperparameter sweep, the out-of-range and campaign evaluations, and everything about exploits. They are reported with intervals where seeds allow, and never as confirmation of a hypothesis.

## 10. Amendments

| Date | Change | Reason |
|---|---|---|
| 2026-11-14 (before any training) | **Confirmation test changed from a bootstrap p-value to a t-test on per-seed scores.** H1: one-sided one-sample t-test of the 10 per-seed scores against 1.0. H2 and H3: one-sided Welch's t-test that the difference between the two groups' per-seed scores exceeds 0.10. Holm correction across the three is unchanged. The two-level bootstrap interval is still reported and still decides refutation. | The fact-check of the literature review found that Colas et al. advise against bootstrap tests below about 50 samples and recommend Welch's t-test; we have 5–10 seeds per group. |
| 2026-11-14 (before any training) | **Training-set-size ablation gains a 10,000-track level** (one seed in phase 2, seeds 2–3 in phase 6). | Cobbe et al. see the generalisation gap close at about 10,000 levels; with only 1, 10, 100, 1,000 and 50,000 the sweep would skip that region. |
| 2026-11-14 (before any training) | **One more exploratory ablation: PPO with a single 9-way action head** instead of two 3-way heads. | PQN needs a single 9-way head, so without this ablation a PPO-versus-PQN difference could be due to the action parameterisation. |
| 2026-11-14 (before any training) | **Starting PPO settings made explicit:** entropy coefficient 0.01 and a learning rate adapted to a KL target of 0.01 within [1e-5, 1e-2], as in Rudin et al. 2021; log-probabilities and entropies are summed over the two action heads. | The fact-check showed CleanRL's GPU file defaults to no entropy bonus and continuous actions only. |
| 2026-11-14 (before any training) | **Rules simplified:** no low-grip shoulder (one grip level to the road's edge), and no separate roll-over rule. | Read from the cannon-es source: grip is a per-wheel parameter, not a surface property, and only the wheel rays touch the road, so a tipped car falls through and is out by the "3 m below" rule. |
| 2026-11-14 (before any training) | **Out-of-range variants fixed:** each out-of-range seed gets exactly one factor beyond the training range, by seed modulo 4: tight turns (arc radius 14–22 m; training 22–120 m), steep hills (grade 6–10 %; training 0–6 %), long jumps (gap 14–20 m; training 6–14 m), long tracks (1,400–2,400 m; training 600–1,400 m). Ablation values: time cost 0.06 per step, fall penalty 5.0. | Written before any track set is generated or any test set read. |
| 2026-11-14 (before any training) | **Simulator and scripted driver frozen.** Physics version `v1`: vendored cannon-es 0.20.0 unmodified, `web/assets/vehicle.json` at SHA-256 prefix `37bc8a37f90ab12d` (starting values from pmndrs/racing-game and cannon-es defaults; no hand-tuning changes were made after two people drove campaign track 1 to the finish), git tag `v0.1-core`. Scripted driver tuned on the 200 validation candidates by the procedure in section 7: a_lat = 6.0 m/s², look-ahead gain = 0.3 s (grid: a_lat 6/8/10/12, gain 0.3/0.45/0.6; table in `results/tracks/scripted_tuning.json`). The chosen a_lat is the lowest grid value, so the reference driver is conservative: faster settings finish fewer tracks at all three speed scales. Training and official evaluation run in WSL2 (Ubuntu, PyTorch 2.15.0+cu132, RTX 5070 Laptop GPU) with `torch.compile`, float32. | Freeze before the first RL run, as section 7 requires. |
| 2026-11-14 (before any training) | **Statistics library:** interquartile mean and probability of improvement follow `rliable`'s definitions; the library itself is used only if it imports in the project environment. | `rliable` 1.2.0 fails to import with pandas 3. |
| 2026-11-14 (after the first specialist's evaluation; no test set read) | **Exploit detector: energy now includes rotational kinetic energy; no airborne-distance flag.** The coasting-energy flag compares ½v² + g·y + ½ωᵀIω/m (per kg, ω in the chassis frame) before and after each agent step while the coast pedal is held; the threshold stays 2 J/kg. The planned airborne-distance flag is not implemented: an unpowered jump that flies further than a ballistic arc must gain energy in the air, which the energy flag already catches. Every flag is still confirmed by hand. | The first specialist's validation evaluation raised one energy flag (track val-1000079, +4.5 J/kg). A step-by-step replay in the same batch showed a crash: the chassis spun up and then turned 8.2 J/kg of spin back into linear speed, while its total energy including rotation fell by 3.7 J/kg; over the whole run the largest gain while coasting, counting rotation, was −0.63 J/kg. The flag was a false positive from leaving rotation out. |
| 2026-11-14 (after the first two models' evaluation on validation and training tracks; no test set read) | **Every official evaluation compiles the simulator afresh** for its own batch of tracks, so the score of a set does not depend on which sets were evaluated before it in the same process. Earlier evaluations were re-scored. | Without the reset, a later set reused a kernel recompiled for dynamic shapes; float32 rounding differences then changed the outcome on many tracks (details and numbers in `plan.md`, amendments). |
| 2026-11-14 (after the learning gate; no test set read) | **Recipe frozen unchanged.** Both first models passed the learning gate with the starting recipe (PPO settings as amended above, environment defaults in `src/racingmania/env/vec_env.py`), so no fix-loop change was made. Step budgets stay at the preregistered 200 million (specialists) and 1 billion (generalists): the specialist's lap time and the generalist's validation score stopped improving well before the end of their budgets (`results/specialist-campaign-01/seed0/val_metrics.csv`, `results/generalist/seed0/val_metrics.csv`). The final checkpoint is still the one evaluated. | Section 7: the recipe is frozen once both first models pass. The generalist's validation score moved by up to about ±3.5 percentage points between 10-minute checkpoints after it plateaued; this checkpoint-to-checkpoint variation is reported and is one reason the hypotheses are tested across seeds. |
| 2026-11-15 (training paused; no change to any hypothesis, metric, threshold or analysis) | **Training paused on 2026-11-14 with the confirmatory runs not done.** The preregistered tests of H1-H3 need 10 generalist seeds and 5 seeds each of the blind control and the single-track ablation; one seed of each exists. Until the seed-sensitivity phase runs, every reported hypothesis reading is labelled preliminary (one seed per group, track resampling only) and no hypothesis is declared confirmed or refuted. The postponed runs keep their preregistered settings, order and seeds. | The user postponed the rest of training to a later date. |
