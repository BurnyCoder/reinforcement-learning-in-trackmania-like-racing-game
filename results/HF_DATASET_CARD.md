---
license: mit
pretty_name: RacingMania reinforcement-learning results
language:
- en
task_categories:
- reinforcement-learning
size_categories:
- n<1K
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

Snapshot of 2026-11-14: 2 registered runs (1 evaluated), 202 per-track evaluation rows.

## Headline (Validation)

| Configuration | Seeds | SWS [95 % CI] | Finished |
|---|---|---|---|
| Specialist (campaign-01) [pilot, seed 0] | 1 | 0.188 [0.123, 0.260] | 13.2 % |

## Layout

- `results/<experiment>[-<track>]/seed<k>/`: one trained run
- `results/baselines/<name>/seed<k>/`: training-free baselines
- `results/tracks/`: track-set summaries and scripted-driver tuning
- `reports/`, `paper/`: generated reports, figures and the paper

## Source

Code: <https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game>. Author: Libor "Burny" Burian. Licence: MIT.
