_Results generated from `results/` by `scripts/07_make_report.py` on 2026-11-14; 28 of 116 planned runs evaluated. Full tables: [`reports/summary.md`](reports/summary.md)._

- The Generalist (PPO) (1 seed) finished 81.5 % of the 971 unseen test tracks (95 % CI 79.0 % to 83.8 %), with a success-weighted speed (SWS) of 1.214 [1.177, 1.250]; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.
- On the tracks it finished, it needed 0.67 times the scripted driver's time (32.7 % less time on average).
- Its finish rate was 8.1 percentage points more than the blind control (73.3 %, 1 seed); 1.6 percentage points more than the PQN (79.8 %, 1 seed); 81.5 percentage points more than the untrained network (0.0 %, 10 seeds); 81.5 percentage points more than the random actions (0.0 %, 3 seeds).
- The preregistered tests of H1-H3 are not done yet: they need test results from 10 generalist seeds and 5 seeds each of the blind control and the single-track ablation (so far: generalist 1, single-track ablation (N = 1) 0, blind control 1). Preliminary readings from the seeds available, not tests:
- H1: The generalist drives unseen tracks better than the scripted driver. SWS 1.214 [1.177, 1.250]; preliminary (1 seed, track resampling only): the interval lies above 1.0, consistent with H1.
- H2: Track diversity is what generalises (50,000 tracks vs 1 track). Not estimable yet: the single-track ablation (N = 1) has no test results yet.
- H3: The agent uses the look-ahead (generalist vs blind control). SWS difference, paired on the same tracks, 0.204 [0.164, 0.245]; preliminary (1 seed per group, track resampling only): the interval lies above 0.1, consistent with H3.
- Exploratory and not H2 itself: specialists trained on one hand-made track each finished 5.3 % to 45.2 % of the unseen test tracks (5 runs, seed 1), against 81.5 % for the generalist.
- With one seed per configuration these intervals reflect only which tracks were sampled; seed-to-seed and checkpoint-to-checkpoint variation are not in them yet, so they can be too narrow. Not done yet: one seed of every experiment (phase 2): 11 of 25 runs evaluated; imitation and hybrids (phase 4): 0 of 5 runs evaluated; hyperparameter sensitivity (phase 5): 0 of 4 runs evaluated; seed sensitivity (phase 6): 0 of 65 runs evaluated.
- On campaign track campaign-01 the generalist (zero-shot) was 2.8 % slower than the specialist trained on that track.
- On campaign track campaign-02 the generalist (zero-shot) was 5.5 % slower than the specialist trained on that track.
- On campaign track campaign-03 the generalist (zero-shot) was 3.3 % slower than the specialist trained on that track.
- On campaign track campaign-04 the generalist (zero-shot) was 7.6 % slower than the specialist trained on that track.
- On campaign track campaign-05 the generalist (zero-shot) was 6.3 % slower than the specialist trained on that track.

**Main evaluation table** (SWS = success-weighted speed, 1.0 = as fast as the scripted driver on every track; 95 % two-level bootstrap intervals over seeds and tracks):

| Configuration | Role | Seeds | Test SWS [95 % CI] | Test finished | OOR SWS [95 % CI] | OOR finished | Campaign SWS [95 % CI] | Campaign finished | Val SWS [95 % CI] | Val finished | Train SWS [95 % CI] | Train finished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEM planner | Reference | 1 | — | — | — | — | 1.551 [1.409, 1.726] | 100.0 % | — | — | — | — |
| Scripted driver | Reference | 1 | 0.998 [0.995, 1.000] | 99.8 % | 1.000 [1.000, 1.000] | 100.0 % | 1.000 [1.000, 1.000] | 100.0 % | 1.000 [1.000, 1.000] | 100.0 % | 0.999 [0.997, 1.000] | 99.9 % |
| Random actions | Control | 3 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Untrained network | Control | 10 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Blind control | Control | 1 | 1.010 [0.971, 1.048] | 73.3 % | 0.605 [0.506, 0.703] | 43.9 % | 1.443 [1.321, 1.590] | 100.0 % | 1.003 [0.913, 1.086] | 72.6 % | 1.008 [0.970, 1.046] | 73.2 % |
| Generalist (PPO) [pilot, seed 0] | Main method | 1 | 1.199 [1.160, 1.237] | 80.2 % | 0.788 [0.678, 0.899] | 52.4 % | 1.596 [1.447, 1.799] | 100.0 % | 1.242 [1.159, 1.319] | 82.7 % | 1.234 [1.198, 1.269] | 82.6 % |
| Generalist (PPO) | Main method | 1 | 1.214 [1.177, 1.250] | 81.5 % | 0.875 [0.769, 0.982] | 57.8 % | 1.590 [1.451, 1.783] | 100.0 % | 1.236 [1.153, 1.315] | 82.7 % | 1.250 [1.214, 1.283] | 83.9 % |
| PQN | Baseline algorithm | 1 | 1.054 [1.020, 1.086] | 79.8 % | 0.725 [0.631, 0.820] | 55.1 % | 1.391 [1.270, 1.517] | 100.0 % | 1.034 [0.956, 1.109] | 78.2 % | 1.052 [1.019, 1.085] | 79.7 % |
| Specialist (campaign-01) [pilot, seed 0] | Specialist | 1 | — | — | — | — | 0.306 [0.000, 0.918] | 20.0 % | 0.188 [0.123, 0.260] | 13.2 % | — | — |
| Specialist (campaign-01) | Specialist | 1 | 0.286 [0.250, 0.323] | 19.8 % | — | — | 0.577 [0.000, 1.188] | 40.0 % | — | — | — | — |
| Specialist (campaign-02) | Specialist | 1 | 0.312 [0.274, 0.351] | 21.0 % | — | — | 0.312 [0.000, 0.937] | 20.0 % | — | — | — | — |
| Specialist (campaign-03) | Specialist | 1 | 0.329 [0.291, 0.367] | 22.3 % | — | — | 0.290 [0.000, 0.870] | 20.0 % | — | — | — | — |
| Specialist (campaign-04) | Specialist | 1 | 0.076 [0.056, 0.096] | 5.3 % | — | — | 0.424 [0.000, 1.271] | 20.0 % | — | — | — | — |
| Specialist (campaign-05) | Specialist | 1 | 0.687 [0.640, 0.735] | 45.2 % | — | — | 0.648 [0.000, 1.332] | 40.0 % | — | — | — | — |
| Fine-tuned specialist (campaign-01) | Specialist | 1 | 0.478 [0.435, 0.523] | 31.8 % | — | — | 0.586 [0.000, 1.196] | 40.0 % | — | — | — | — |
| Fine-tuned specialist (campaign-02) | Specialist | 1 | 0.446 [0.403, 0.490] | 29.7 % | — | — | 0.892 [0.278, 1.507] | 60.0 % | — | — | — | — |
| Fine-tuned specialist (campaign-03) | Specialist | 1 | 0.636 [0.589, 0.684] | 42.2 % | — | — | 0.971 [0.292, 1.651] | 60.0 % | — | — | — | — |

**Preregistered hypotheses:**

|  | Hypothesis | Tested quantity | Seeds (have/required) | Estimate [95 % CI] | Test statistic | p (one-sided) | p (Holm) | Decision | Preliminary reading (not a test) |
|---|---|---|---|---|---|---|---|---|---|
| H1 | The generalist drives unseen tracks better than the scripted driver | SWS > 1.0 | generalist 1/10 | 1.214 [1.177, 1.250] (preliminary) | — | — | — | **not yet tested** | preliminary (1 seed, track resampling only): the interval lies above 1.0, consistent with H1 |
| H2 | Track diversity is what generalises (50,000 tracks vs 1 track) | SWS difference > 0.1 | generalist 1/10, size-1 0/5 | — | — | — | — | **not yet tested** | not estimable yet: the single-track ablation (N = 1) has no test results yet |
| H3 | The agent uses the look-ahead (generalist vs blind control) | SWS difference > 0.1 | generalist 1/10, blind 1/5 | 0.204 [0.164, 0.245] (preliminary) | — | — | — | **not yet tested** | preliminary (1 seed per group, track resampling only): the interval lies above 0.1, consistent with H3 |

![SWS and finish rate on the test set (971 tracks), mean over seeds with 95 % two-level bootstrap intervals (10,000 resamples of seeds and tracks); n = number of seeds. With one seed the interval reflects track sampling only.](reports/figures/sws_by_method.png)
