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

Snapshot of 2026-11-14: 31 registered runs (30 evaluated), 59,072 per-track evaluation rows.

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
| Specialist (campaign-01) | 1 | 0.286 [0.250, 0.323] | 19.8 % |
| Specialist (campaign-02) | 1 | 0.312 [0.274, 0.351] | 21.0 % |
| Specialist (campaign-03) | 1 | 0.329 [0.291, 0.367] | 22.3 % |
| Specialist (campaign-04) | 1 | 0.076 [0.056, 0.096] | 5.3 % |
| Specialist (campaign-05) | 1 | 0.687 [0.640, 0.735] | 45.2 % |
| Fine-tuned specialist (campaign-01) | 1 | 0.478 [0.435, 0.523] | 31.8 % |
| Fine-tuned specialist (campaign-02) | 1 | 0.446 [0.403, 0.490] | 29.7 % |
| Fine-tuned specialist (campaign-03) | 1 | 0.636 [0.589, 0.684] | 42.2 % |
| Fine-tuned specialist (campaign-04) | 1 | 0.265 [0.229, 0.301] | 17.9 % |
| Fine-tuned specialist (campaign-05) | 1 | 0.842 [0.794, 0.890] | 54.9 % |

## Layout

- `results/<experiment>[-<track>]/seed<k>/`: one trained run
- `results/baselines/<name>/seed<k>/`: training-free baselines
- `results/tracks/`: track-set summaries and scripted-driver tuning
- `reports/`, `paper/`: generated reports, figures and the paper

## Source

Code: <https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game>. Author: Libor "Burny" Burian. Licence: MIT.
