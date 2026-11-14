_Results generated from `results/` by `scripts/07_make_report.py` on 2026-11-14; 16 of 115 planned runs evaluated. Full tables: [`reports/summary.md`](reports/summary.md)._

- The Generalist (PPO) [pilot, seed 0] (1 seed) finished 82.7 % of the 197 validation tracks (the development set used for decisions, so these numbers are preliminary; test tracks are read only after the recipe is frozen) (95 % CI 77.2 % to 87.8 %), with a success-weighted speed (SWS) of 1.242 [1.159, 1.319]; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.
- On the tracks it finished, it needed 0.67 times the scripted driver's time (33.2 % faster on average).
- Its finish rate was 82.7 percentage points more than the untrained network (0.0 %, 10 seeds); 82.7 percentage points more than the random actions (0.0 %, 3 seeds).
- The preregistered hypotheses H1-H3 are not tested yet: they need test results from 10 generalist seeds and 5 seeds each of the blind control and the single-track ablation (0 generalist seeds so far).

**Main evaluation table** (SWS = success-weighted speed, 1.0 = as fast as the scripted driver on every track; 95 % two-level bootstrap intervals over seeds and tracks):

| Configuration | Role | Seeds | Campaign SWS [95 % CI] | Campaign finished | Val SWS [95 % CI] | Val finished | Train SWS [95 % CI] | Train finished |
|---|---|---|---|---|---|---|---|---|
| Scripted driver | Reference | 1 | 1.000 [1.000, 1.000] | 100.0 % | 1.000 [1.000, 1.000] | 100.0 % | 0.999 [0.997, 1.000] | 99.9 % |
| Random actions | Control | 3 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Untrained network | Control | 10 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Generalist (PPO) [pilot, seed 0] | Main method | 1 | 1.596 [1.447, 1.799] | 100.0 % | 1.242 [1.159, 1.319] | 82.7 % | 1.234 [1.198, 1.269] | 82.6 % |
| Specialist (campaign-01) [pilot, seed 0] | Specialist | 1 | 0.306 [0.000, 0.918] | 20.0 % | 0.188 [0.123, 0.260] | 13.2 % | — | — |

**Preregistered hypotheses:**

|  | Hypothesis | Tested quantity | Seeds (have/required) | Estimate [95 % CI] | Test statistic | p (one-sided) | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|---|
| H1 | The generalist drives unseen tracks better than the scripted driver | SWS > 1.0 | generalist 0/10 | — | — | — | — | **not yet tested** |
| H2 | Track diversity is what generalises (50,000 tracks vs 1 track) | SWS difference > 0.1 | generalist 0/10, size-1 0/5 | — | — | — | — | **not yet tested** |
| H3 | The agent uses the look-ahead (generalist vs blind control) | SWS difference > 0.1 | generalist 0/10, blind 0/5 | — | — | — | — | **not yet tested** |

![SWS and finish rate on the validation set (development set; preliminary) (197 tracks), mean over seeds with 95 % two-level bootstrap intervals (10,000 resamples of seeds and tracks); n = number of seeds. With one seed the interval reflects track sampling only.](reports/figures/sws_by_method.png)
