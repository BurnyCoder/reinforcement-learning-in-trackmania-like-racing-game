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

Snapshot of 2026-11-15: 31 registered runs (31 evaluated), 61,633 per-track evaluation rows.

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
| Trained on 1 track | 1 | 0.539 [0.495, 0.583] | 36.5 % |
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

## Videos

The README's clips: official evaluation runs replayed in the browser game, chosen by fixed rules (best, typical and failure cases, then the controls). Each model's own clips are on its model card.

### Best specialist lap

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/specialist-campaign-01-seed0_campaign-01.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** The specialist has driven only this one track during training, so it knows every corner. Watch it accelerate hard from the start, hold high speed on the straights, keep its speed through the bends by staying close to the edge, and take the jump without slowing down. It is the fastest of all drivers evaluated so far on this track, but what it learned is how to drive this one track: on tracks with a different layout it mostly falls off (see the results).
- **Model:** `specialist-campaign-01/seed0`: PPO, trained by reinforcement learning for 200 million agent steps on campaign-01 only (seed 0, 0.3 h on one GPU).
- **Map:** First Flight (`campaign-01`), a hand-made campaign track, 628 m long with 1 jump and up to 6° of banking; the track this model was trained on.
- **Evaluation:** official evaluation in `results/specialist-campaign-01/seed0/eval_campaign.csv` (standing start, most probable action, final checkpoint): finished in 19.71 s; the scripted driver needs 30.17 s, so this is 1.53× its speed. The clip replays this run.
- **Why this clip:** chosen by a fixed rule as its only finished campaign track in its official evaluation.

### Best generalist lap on a hand-made track it never saw in training: it generalized

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/generalist-seed0_campaign-04.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** This is generalization: driving well on a track the model never saw in training. The generalist learned on tens of thousands of generated tracks and never on these hand-made ones, which were designed separately. It reads the road ahead, picks a speed for each bend, and laps this track in about half the scripted driver's time. On the specialist's own track it is only slightly slower than the specialist (see the results). Compare clip 7 among the negative results: the specialist on this same track.
- **Model:** `generalist/seed0`: PPO, trained by reinforcement learning for 1 billion agent steps on 48,949 generated training tracks (seed 0, 1.3 h on one GPU).
- **Map:** Technical (`campaign-04`), a hand-made campaign track, 666 m long with no jumps and up to 5° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/generalist/seed0/eval_campaign.csv` (standing start, most probable action, final checkpoint): finished in 21.27 s; the scripted driver needs 42.52 s, so this is 2.00× its speed. The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the highest-scoring finished track of the 5 campaign tracks it finished in its official evaluation.

### Best generalist lap on a held-out generated track it never saw in training: it generalized

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/generalist-seed0_test-2000747.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** Generalization again, now on a generated track from the held-out test seeds: the same kind of track as in training, but never this one, driven from the standing start to the finish. Watch it slow down for each bend, hug the inside edge through it and accelerate out again, using only what it sees of the road ahead. Compare it with the next clip, the scripted driver on the same track.
- **Model:** `generalist/seed0`: PPO, trained by reinforcement learning for 1 billion agent steps on 48,949 generated training tracks (seed 0, 1.3 h on one GPU).
- **Map:** Unseen test track 2000747 (`test-2000747`), a held-out generated test track, 1,461 m long with no jumps and up to 11° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/generalist/seed0/eval_test.csv` (standing start, most probable action, final checkpoint): finished in 43.35 s; the scripted driver needs 79.77 s, so this is 1.84× its speed. The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the highest-scoring finished track of the 779 test tracks it finished in its official evaluation.

### Scripted driver on the same held-out track, for comparison

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/scripted-driver_test-2000747.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** The same track as the clip before, driven by the hand-written scripted driver. It drives the track cleanly, but it is cautious: in the bends it slows to well under half the generalist's speed, and it never reaches the speeds the generalist holds on the straights, so it needs almost twice as long (speeds measured from both recorded runs). Its time on each track is the yardstick of the score: 1.0 means as fast as this driver.
- **Model:** `baselines/scripted/seed0`: the scripted driver, a hand-written controller (pure-pursuit steering and a curvature-based speed profile, see [The scripted driver](#the-scripted-driver)) with its 2 parameters tuned on validation tracks before any training (a_lat = 6.0 m/s², look-ahead gain = 0.3 s); nothing is learned, and its time on each track is the reference that scores 1.0.
- **Map:** Unseen test track 2000747 (`test-2000747`), a held-out generated test track, 1,461 m long with no jumps and up to 11° of banking; the scripted driver is not trained; it drove this track once before, to certify it and set its reference time.
- **Evaluation:** official evaluation in `results/baselines/scripted/seed0/eval_test.csv` (standing start, most probable action, final checkpoint): finished in 79.75 s (the track's reference time, from the scripted driver's separate certification run, is 79.77 s). The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the same track as the generalist/seed0 clip: the highest-scoring finished track of the 779 test tracks `generalist/seed0` finished in its official evaluation.

### Typical generalist lap on a held-out generated track it never saw in training

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/generalist-seed0_test-2000386.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** Not the best lap but a typical one: the median of the generalist's finished test tracks that have a jump. It shows what most of its successful laps on unseen tracks look like: a clean run with a jump, clearly faster than the scripted driver but less spectacular than the best lap.
- **Model:** `generalist/seed0`: PPO, trained by reinforcement learning for 1 billion agent steps on 48,949 generated training tracks (seed 0, 1.3 h on one GPU).
- **Map:** Unseen test track 2000386 (`test-2000386`), a held-out generated test track, 1,097 m long with 1 jump and up to 10° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/generalist/seed0/eval_test.csv` (standing start, most probable action, final checkpoint): finished in 33.01 s; the scripted driver needs 49.36 s, so this is 1.50× its speed. The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the median-scoring finished track among those with at least one jump (397 test tracks with a jump that it finished) in its official evaluation.

### Seed 1's specialist finishes a held-out track it never trained on: it generalized

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/specialist-campaign-01-seed1_test-2000510.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** This is generalization too, by a specialist: driving well on a track it never saw in training. It is a second specialist, trained the same way on campaign track 1 only but with seed 1. It finishes about one unseen test track in five, and this is a typical one of them (median score). Its finishes lean towards easier tracks like this one: shorter, less banked and with fewer jumps than average. Where its driving carries over it is fast: on the tracks both finish it is as fast as the generalist, slightly faster on most of them, and on this track the two times are almost equal (see the evaluation line). But the generalist also finishes almost all of the tracks the specialist finishes, plus many harder ones the specialist falls off: about four unseen test tracks in five. That is the difference between learning one track and learning many.
- **Model:** `specialist-campaign-01/seed1`: PPO, trained by reinforcement learning for 200 million agent steps on campaign-01 only (seed 1, 0.2 h on one GPU).
- **Map:** Unseen test track 2000510 (`test-2000510`), a held-out generated test track, 934 m long with no jumps and up to 3° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/specialist-campaign-01/seed1/eval_test.csv` (standing start, most probable action, final checkpoint): finished in 27.63 s; the scripted driver needs 40.29 s, so this is 1.46× its speed; on the same track `generalist/seed0` finished in 27.61 s. The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the median-scoring finished track of the 192 test tracks it finished in its official evaluation.

### The specialist does not generalize: clip 1's model on clip 2's track, which it never trained on

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/specialist-campaign-01-seed0_campaign-04.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** The specialist from clip 1, on the track of clip 2, which it never trained on. For the first seconds it accelerates exactly like the generalist, but then it brakes to a crawl, drifts towards the edge of a bend and drops off the road, while the generalist on the same track keeps accelerating (clip 2; speeds compared from both recorded runs). Trained on one track, it learned to drive that track, and that driving does not carry over to a different layout, which is why the README's results show it finishing so few unseen tracks.
- **Model:** `specialist-campaign-01/seed0`: PPO, trained by reinforcement learning for 200 million agent steps on campaign-01 only (seed 0, 0.3 h on one GPU).
- **Map:** Technical (`campaign-04`), a hand-made campaign track, 666 m long with no jumps and up to 5° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/specialist-campaign-01/seed0/eval_campaign.csv` (standing start, most probable action, final checkpoint): did not finish: fell after 7.44 s and 73 m (the scripted driver finishes in 42.52 s). The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the same track as the generalist/seed0 clip: the highest-scoring finished track of the 5 campaign tracks `generalist/seed0` finished in its official evaluation.

### Generalization failure on a held-out track: the generalist falls off halfway round

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/generalist-seed0_test-2000430.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** Generalization does not always work. About one unseen track in five ends like this: the generalist enters a bend too fast, cannot hold the line and leaves the road on the outside. Most of its falls come right after tight corners rather than at jumps (failure analysis in `results/generalist/seed*/failures_*.json`): it trades safety for speed.
- **Model:** `generalist/seed0`: PPO, trained by reinforcement learning for 1 billion agent steps on 48,949 generated training tracks (seed 0, 1.3 h on one GPU).
- **Map:** Unseen test track 2000430 (`test-2000430`), a held-out generated test track, 1,559 m long with no jumps and up to 14° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/generalist/seed0/eval_test.csv` (standing start, most probable action, final checkpoint): did not finish: fell after 24.43 s and 776 m (the scripted driver finishes in 66.30 s). The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the fall whose position is closest to the middle of its track among its 190 falls on test tracks in its official evaluation.

### Generalization failure beyond the training range: a corner tighter than any training track

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/generalist-seed0_oor-3000123.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** This track was generated with tighter corners than any track the generalist trained on. It approaches the corner at a speed that would suit the wider corners it knows and flies off the outside. Learning from many tracks lets it drive new combinations of familiar features, but not features outside what it has seen: on tracks like this it almost always falls, while longer jumps than in training, for example, are handled well (see the results).
- **Model:** `generalist/seed0`: PPO, trained by reinforcement learning for 1 billion agent steps on 48,949 generated training tracks (seed 0, 1.3 h on one GPU).
- **Map:** Out-of-range track 3000123 (`oor-3000123`), a held-out out-of-range test track with one factor beyond the training range, arc radius 14–22 m (training: 22–120 m), 855 m long with 2 jumps and up to 9° of banking; this model never saw it during training.
- **Evaluation:** official evaluation in `results/generalist/seed0/eval_oor.csv` (standing start, most probable action, final checkpoint): did not finish: fell after 17.76 s and 449 m (the scripted driver finishes in 52.16 s). The clip replays this run.
- **Why this clip:** chosen by a fixed rule as the fall whose position is closest to the middle of its track among its 43 falls on tight-turn out-of-range tracks in its official evaluation.

### Untrained network (control) on the specialist's track

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/untrained-seed0_campaign-01.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** Like every evaluated driver, the network takes its most probable action at each step instead of sampling one. With random initial weights its output layer starts close to zero (standard deviation 0.01, the CleanRL convention), so it picks almost the same keys every step, here steering right while coasting, and never presses gas. Sampled instead of taken greedily, the same network would press keys almost uniformly at random, which is how PPO starts exploring.
- **Model:** `baselines/untrained/seed0`: the same network as the trained agents (2 × 256 units) with random initial weights (seed 0) and no training; a control.
- **Map:** First Flight (`campaign-01`), a hand-made campaign track, 628 m long with 1 jump and up to 6° of banking; this network was never trained on any track.
- **Evaluation:** official evaluation in `results/baselines/untrained/seed0/eval_campaign.csv` (standing start, most probable action, final checkpoint): did not finish: stalled after 2.00 s and 6 m (the scripted driver finishes in 30.17 s). The clip replays this run.

### Untrained network with sampled actions: almost uniformly random keys, as PPO starts exploring

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/untrained-seed0-sampled_campaign-01.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** The same network and weights as the clip before. The official evaluation takes the most probable action; here the keys of every step are drawn from the network's own probabilities, which are close to one in three for each steering and each pedal choice, so they are almost uniformly random, as at the start of PPO training. The car jitters between gas, coast and brake and drifts a few dozen metres before it is out for lack of progress. This is a demonstration run, not part of the scored evaluation.
- **Model:** `demos/untrained-sampled/seed0`: the untrained control network (random initial weights, seed 0) with each action sampled from its probabilities (sampling seed 1000) instead of the most probable one.
- **Map:** First Flight (`campaign-01`), a hand-made campaign track, 628 m long with 1 jump and up to 6° of banking; this network was never trained on any track.
- **Evaluation:** demonstration run through the same evaluator, recorded in `results/demos/untrained-sampled/seed0/eval_campaign.csv` (standing start, sampled actions; not scored): did not finish: stalled after 28.99 s and 52 m (the scripted driver finishes in 30.17 s). The clip replays this run.
- **Over many runs:** in 200 runs each on this track, random actions are out after 9.4 s on average (median 6.1 s, 10–90 % range 2.0–21.0 s) and the untrained network with sampled actions after 9.8 s (median 6.5 s, 2.0–23.1 s), with no detectable difference between them (Mann–Whitney p = 0.50 for the time, 0.48 for the progress). The gap between the two random clips is therefore consistent with luck, and this run is a lucky one, above the 90th percentile of its runs (`results/demos/random-vs-sampled/campaign-01.json`, `scripts/14_random_vs_sampled.py`).

### Random actions (control) on the specialist's track

<video src="https://huggingface.co/datasets/BurnyCoder/racingmania-results/resolve/main/videos/random-seed0_campaign-01.mp4" controls muted loop playsinline width="100%"></video>

- **What you see:** This control presses a uniformly random key combination every agent step (0.05 s). Gas, coast and brake, which reverses at a standstill, each come up a third of the time and cancel out, so the car makes almost no progress and is out under the "no 0.5 m of progress in 2 s" rule.
- **Model:** `baselines/random/seed0`: random actions, a uniformly random key combination every agent step from a seeded generator (seed 0); no network and no training; a control.
- **Map:** First Flight (`campaign-01`), a hand-made campaign track, 628 m long with 1 jump and up to 6° of banking; random actions ignore the track entirely.
- **Evaluation:** official evaluation in `results/baselines/random/seed0/eval_campaign.csv` (standing start, most probable action, final checkpoint): did not finish: stalled after 2.00 s and 6 m (the scripted driver finishes in 30.17 s). The clip replays this run.
- **Over many runs:** in 200 runs each on this track, random actions are out after 9.4 s on average (median 6.1 s, 10–90 % range 2.0–21.0 s) and the untrained network with sampled actions after 9.8 s (median 6.5 s, 2.0–23.1 s), with no detectable difference between them (Mann–Whitney p = 0.50 for the time, 0.48 for the progress). The gap between the two random clips is therefore consistent with luck, and this run is an unlucky one: 27% of its runs end at 2.0 s like this one (`results/demos/random-vs-sampled/campaign-01.json`, `scripts/14_random_vs_sampled.py`).

## Source

Code: <https://github.com/BurnyCoder/reinforcement-learning-in-trackmania-like-racing-game>. Author: Libor "Burny" Burian. Licence: MIT.
