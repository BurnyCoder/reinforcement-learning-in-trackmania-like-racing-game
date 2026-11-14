_Results generated from `results/` by `scripts/07_make_report.py` on 2026-11-14; 18 of 116 planned runs evaluated. Full tables: [`reports/summary.md`](reports/summary.md)._

- The Generalist (PPO) (1 seed) finished 81.5 % of the 971 unseen test tracks (95 % CI 79.0 % to 83.8 %), with a success-weighted speed (SWS) of 1.214 [1.177, 1.250]; an SWS of 1.0 would mean finishing every track as fast as the scripted driver.
- On the tracks it finished, it needed 0.67 times the scripted driver's time (32.7 % less time on average).
- Its finish rate was 81.5 percentage points more than the untrained network (0.0 %, 10 seeds); 81.5 percentage points more than the random actions (0.0 %, 3 seeds).
- The preregistered hypotheses H1-H3 are not tested yet: they need test results from 10 generalist seeds and 5 seeds each of the blind control and the single-track ablation (1 generalist seeds so far).

**Main evaluation table** (SWS = success-weighted speed, 1.0 = as fast as the scripted driver on every track; 95 % two-level bootstrap intervals over seeds and tracks):

| Configuration | Role | Seeds | Test SWS [95 % CI] | Test finished | OOR SWS [95 % CI] | OOR finished | Campaign SWS [95 % CI] | Campaign finished | Val SWS [95 % CI] | Val finished | Train SWS [95 % CI] | Train finished |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEM planner | Reference | 1 | — | — | — | — | 1.551 [1.409, 1.726] | 100.0 % | — | — | — | — |
| Scripted driver | Reference | 1 | 0.998 [0.995, 1.000] | 99.8 % | 1.000 [1.000, 1.000] | 100.0 % | 1.000 [1.000, 1.000] | 100.0 % | 1.000 [1.000, 1.000] | 100.0 % | 0.999 [0.997, 1.000] | 99.9 % |
| Random actions | Control | 3 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Untrained network | Control | 10 | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % | 0.000 [0.000, 0.000] | 0.0 % |
| Generalist (PPO) [pilot, seed 0] | Main method | 1 | 1.199 [1.160, 1.237] | 80.2 % | 0.788 [0.678, 0.899] | 52.4 % | 1.596 [1.447, 1.799] | 100.0 % | 1.242 [1.159, 1.319] | 82.7 % | 1.234 [1.198, 1.269] | 82.6 % |
| Generalist (PPO) | Main method | 1 | 1.214 [1.177, 1.250] | 81.5 % | 0.875 [0.769, 0.982] | 57.8 % | 1.590 [1.451, 1.783] | 100.0 % | 1.236 [1.153, 1.315] | 82.7 % | 1.250 [1.214, 1.283] | 83.9 % |
| Specialist (campaign-01) [pilot, seed 0] | Specialist | 1 | — | — | — | — | 0.306 [0.000, 0.918] | 20.0 % | 0.188 [0.123, 0.260] | 13.2 % | — | — |

**Preregistered hypotheses:**

|  | Hypothesis | Tested quantity | Seeds (have/required) | Estimate [95 % CI] | Test statistic | p (one-sided) | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|---|
| H1 | The generalist drives unseen tracks better than the scripted driver | SWS > 1.0 | generalist 1/10 | 1.214 [1.177, 1.250] (preliminary) | — | — | — | **not yet tested** |
| H2 | Track diversity is what generalises (50,000 tracks vs 1 track) | SWS difference > 0.1 | generalist 1/10, size-1 0/5 | — | — | — | — | **not yet tested** |
| H3 | The agent uses the look-ahead (generalist vs blind control) | SWS difference > 0.1 | generalist 1/10, blind 0/5 | — | — | — | — | **not yet tested** |

![SWS and finish rate on the test set (971 tracks), mean over seeds with 95 % two-level bootstrap intervals (10,000 resamples of seeds and tracks); n = number of seeds. With one seed the interval reflects track sampling only.](reports/figures/sws_by_method.png)
