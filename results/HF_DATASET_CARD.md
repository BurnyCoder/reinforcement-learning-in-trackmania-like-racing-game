---
license: mit
pretty_name: RacingMania reinforcement-learning results
language:
- en
task_categories:
- reinforcement-learning
size_categories:
- 10K<n<100K
tags:
- reinforcement-learning
- racingmania
- racing
- procedural-generation
- generalization
- ppo
- pqn
---


# RacingMania results

Every result file of the RacingMania project: per-run `run.json` (status, configuration, hardware, physics version), per-track evaluation CSVs (`eval_<set>.csv`: track, finished, ticks, time, reference ticks, score, end reason, progress, maximum speed, exploit flags), training and validation curves, track-set summaries, the test-access log, and the generated reports and paper. Column definitions: `docs/formats.md` in the repository.

Snapshot of 2026-11-14: 21 registered runs (20 evaluated), 49,312 per-track evaluation rows.

## Headline (Test (unseen generated))

| Configuration | Seeds | SWS [95 % CI] | Finished |
|---|---|---|---|
| Scripted driver | 1 | 0.998 [0.995, 1.000] | 99.8 % |
| Random actions | 3 | 0.000 [0.000, 0.000] | 0.0 % |
| Untrained network | 10 | 0.000 [0.000, 0.000] | 0.0 % |
| Blind control | 1 | 1.010 [0.971, 1.048] | 73.3 % |
| Generalist (PPO) [pilot, seed 0] | 1 | 1.199 [1.160, 1.237] | 80.2 % |
| Generalist (PPO) | 1 | 1.214 [1.177, 1.250] | 81.5 % |
| PQN | 1 | 1.054 [1.020, 1.086] | 79.8 % |

## Layout

- `results/<experiment>[-<track>]/seed<k>/`: one trained run
- `results/baselines/<name>/seed<k>/`: training-free baselines
- `results/tracks/`: track-set summaries and scripted-driver tuning
- `reports/`, `paper/`: generated reports, figures and the paper

## Source

Code: <https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game>. Author: Libor "Burny" Burian. Licence: MIT.
