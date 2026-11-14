### Learnings while running phase 2 (one seed of every experiment)

**Run queue crash on a report change (2026-11-14, 21:39).**
- **What happened:** the run queue (`scripts/05_run_matrix.py`) crashed when it published the trained `specialist-finetune-campaign-04/seed1` run. The error was jinja2 `UndefinedError: 'preliminary_hypotheses' is undefined`. Training itself had finished (200 million steps), and no result was lost.
- **Why:** the queue is one Python process that runs for days, and it built the report in that same process. A pull request merged minutes earlier had added preliminary hypothesis readings: a new paper template variable plus the `render.py` code that fills it. The template on disk was new, but `render.py` was still the version imported when the queue started.
- **A second, latent problem:** the publish built the report first and pulled `main` only afterwards. Any report change merged into `main` while the queue ran would therefore have left regenerated files in the way of the pull and made the publish's pull request fail.
- **Fix:** `racingmania.matrix.publish_all` now first fast-forwards `main`, while the generated files are still clean, and then runs `scripts/08_publish.py` in a child process that imports the code just pulled (test: `tests/unit/test_matrix.py`).
- **Recovery:** the missed "trained" publish was run by hand (pull request #75) and the queue was restarted. The queue resumes from `run.json` statuses, so it went straight on to evaluating the run.
