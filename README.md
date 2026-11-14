# RacingMania

An original 3D time-attack racing game for the browser, and reinforcement-learning drivers trained to play it.

**Status (2026-11-14): planning and literature review done; code is being written. No model has been trained and no result exists yet.** This README is replaced by the detailed version, with install and usage instructions, when the code lands.

## What this project is

- **A game.** One car, checkpoints, hills, banked turns and jumps on a ribbon of road in the sky; the fastest time wins. It runs in the browser with [three.js](https://threejs.org/) for graphics and the car physics of [cannon-es](https://github.com/pmndrs/cannon-es) (`RaycastVehicle`, MIT). It is inspired by the time-attack genre of TrackMania but copies none of its tracks, art or code.
- **A simulator for learning.** The same car physics is ported line by line to batched PyTorch, so thousands of cars, each on a different generated track, can be simulated on one GPU. The port is tested tick by tick against the original code.
- **An experiment.** A PPO agent with keyboard-style actions is trained on tens of thousands of generated tracks and tested on 1,000 tracks it has never seen, against a scripted driver, a random policy and a blind control, and compared with specialists trained on single tracks.

## Where to read more

| Document | Content |
|---|---|
| [`reports/plan.md`](reports/plan.md) | The full approved plan: design, experiments, run matrix, order of work, risks, amendments |
| [`reports/literature-review.md`](reports/literature-review.md) | Prior work on racing RL, generalisation, algorithms, car physics and tooling, with sources |

The preregistration (hypotheses, primary metric, analysis plan, falsification criteria) is published in `reports/preregistration.md` before any result exists.

## Licence and third-party code

Third-party code that is vendored or ported is listed with its licence in `THIRD_PARTY_NOTICES.md` as it is added.
