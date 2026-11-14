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
