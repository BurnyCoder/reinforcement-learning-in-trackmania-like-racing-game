### Learnings while running phase 2 (one seed of every experiment)

**Run queue crash on a report change (2026-11-14, 21:39).**
- **What happened:** the run queue (`scripts/05_run_matrix.py`) crashed when it published the trained `specialist-finetune-campaign-04/seed1` run. The error was jinja2 `UndefinedError: 'preliminary_hypotheses' is undefined`. Training itself had finished (200 million steps), and no result was lost.
- **Why:** the queue is one Python process that runs for days, and it built the report in that same process. A pull request merged minutes earlier had added preliminary hypothesis readings: a new paper template variable plus the `render.py` code that fills it. The template on disk was new, but `render.py` was still the version imported when the queue started.
- **A second, latent problem:** the publish built the report first and pulled `main` only afterwards. Any report change merged into `main` while the queue ran would therefore have left regenerated files in the way of the pull and made the publish's pull request fail.
- **Fix:** `racingmania.matrix.publish_all` now first fast-forwards `main`, while the generated files are still clean, and then runs `scripts/08_publish.py` in a child process that imports the code just pulled (test: `tests/unit/test_matrix.py`).
- **Recovery:** the missed "trained" publish was run by hand (pull request #75) and the queue was restarted. The queue resumes from `run.json` statuses, so it went straight on to evaluating the run.

**Run queue stop on a Hugging Face rate limit (2026-11-14, 23:33).**
- **What happened:** the "trained" publish of `size-1/seed1` failed with `429 Too Many Requests`: "You have exceeded the rate limit for collection creation (100 per day)". The GitHub part of the publish had succeeded. The queue stopped; the trained run was not lost.
- **Why:** every publish looked up the collection with `HfApi.create_collection(..., exists_ok=True)`, and Hugging Face counts each such call as a creation even when the collection exists. With three publishes per run plus model uploads, a long day of runs passed 100 calls.
- **Fix:** the collection is found with the read-only `HfApi.list_collections(owner=...)` (or `HF_COLLECTION_SLUG`) and created only when missing, once per process. The queue also no longer stops when a publish fails: it logs the error and goes on, since the next publish carries the whole state (tests: `tests/unit/test_matrix.py`).
- **What came next:** training was paused at this point at the user's request (see below), so the queue was not restarted. The trained size-1 run was evaluated by hand with the queue's own commands.

**Training paused (2026-11-14, about 23:50).**
- **Decision:** the user postponed the rest of training to a later date and asked to wrap up with the results so far.
- **State at the pause:** phase 1 done; 14 of 25 phase-2 runs evaluated (generalist, blind control, PQN, five specialists, five specialists fine-tuned from the generalist, and the single-track ablation `size-1/seed1`, evaluated by hand after the queue stopped); phases 4, 5 and 6 not started; phase 3 has no runs because no agent used a physics exploit.
- **How it is recorded:** a `paused` block in `configs/run_matrix.yaml`. While it is present the queue does not start (`racingmania.matrix.run_queue`), and the report lists runs without results as `postponed` (README results block, paper status, run-status diagram, website status board).
- **Resuming:** delete the block and run `uv run python scripts/05_run_matrix.py --phases 2-single-seed 4-imitation-hybrids 5-hyperparameters 6-seed-sensitivity`. Phase 4 first needs its focused literature search and a preregistration amendment fixing the imitation and hybrid variants. With the measured throughput (about 1.5 hours per 1-billion-step run including evaluation and publishing, about 25 minutes per 200-million-step specialist), the postponed runs need about 104 GPU hours.


**Fact-check of the documentation (2026-11-15, training still paused).**
- **What was checked:** every claim in the README, AGENTS.md, `docs/`, the paper template, these notes, the Hugging Face cards, the video captions, code comments and the citations, against the code, the configs, the result files and the cited sources. The generated numbers all matched the result files; the errors were in hand-written text, caption templates and comments. The corrections are listed in the amendment of 2026-11-15 in `reports/plan.md`.
- **One metadata fix in a result file:** `results/pqn/seed1/run.json` recorded `ppo.action_mode: "factored"`, a field PQN inherited from the PPO settings and never used: the PQN network (`rl/pqn.py`, `QAgent`) always has one 9-way head. The field now says `"joint"`, and `PQNConfig` defaults to it, so future PQN runs record it correctly. Nothing about the trained model or its scores changed.
- **Warm starts are now recorded:** `run.json` gets `init_from` and `bc_init` for future runs. The five fine-tuned specialists, trained before this change, started from `generalist/seed1` as their experiment file says (`configs/experiments/specialist-finetune.yaml`).
