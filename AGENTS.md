# AGENTS.md

Notes for coding agents (and people) working on this repository. The README explains the project; this file says how to work on it without breaking the experiment.

## State (2026-11-15)

- **Training is paused** since 2026-11-14 at the owner's request; the remaining runs are postponed to a later date. `configs/run_matrix.yaml` has a `paused` block: while it is present `scripts/05_run_matrix.py` refuses to start, and the report lists runs without results as `postponed`.
- Done: phase 1 (first two models) and 14 of the 25 phase-2 runs (generalist, blind control, PQN, five specialists, five specialists fine-tuned from the generalist, single-track ablation). Phase 3 has no runs (no physics exploit was used). Phases 4, 5 and 6 have not started.
- The preregistered tests of H1-H3 run only after phase 6; until then every hypothesis reading is preliminary (`reports/preregistration.md`, amendments).
- History and incidents: `reports/plan.md` (amendments), `reports/notes/first-models.md`, `reports/notes/phase-2.md`.

## Layout

| Path | What |
|---|---|
| `src/racingmania/sim/` | Batched PyTorch port of cannon-es `RaycastVehicle` (the training simulator) |
| `src/racingmania/tracks/`, `env/`, `rl/`, `drivers/`, `eval/` | Track generator and pools, vectorised environment and observation, PPO and PQN, scripted/CEM/random drivers, evaluator |
| `src/racingmania/report/` | One-command report: `collect` → `analysis` → `figures` + `diagrams` → `render` (README block, reports, paper) → `webdata`, `cards` (Hugging Face) |
| `src/racingmania/matrix.py`, `publish.py` | Resumable run queue and the GitHub/Hugging Face publish step |
| `scripts/NN_*.py` | Entry points in pipeline order (generate tracks, train, evaluate, baselines, queue, export to web, report, publish, analyses, videos) |
| `web/` | The browser game (three.js + vendored, unmodified cannon-es) and the results explorer (`results.html`) |
| `configs/` | `run_matrix.yaml` (preregistered run order) and `experiments/*.yaml` |
| `results/` | Small result files tracked in git (`run.json`, per-track CSVs, replays); checkpoints live under `RACINGMANIA_DATA_DIR` |
| `reports/`, `paper/` | Generated reports, figures, diagrams and the paper (`paper.md.j2` is the template) |
| `docs/` | `formats.md` (track, model, replay and result file formats), `physics.md`, `videos/` (README clips) and `videos/cards/` (Hugging Face card clips) |

## Commands

```bash
uv run pytest tests/unit -q
```

```bash
uv run pytest tests/web -q
```

```bash
uv run python scripts/07_make_report.py
```

- Unit tests take about a minute; browser tests (Playwright) about a minute. `tests/parity` compares the PyTorch port with cannon-es in Node.js.
- Training and official evaluation run in WSL2 (`torch.compile`); the queue sends GPU work there itself (`matrix.run_gpu`). The WSL environment lives at `$HOME/venvs/racingmania` (`UV_PROJECT_ENVIRONMENT`).
- Videos: serve the game (`uv run python -m http.server 8123 --directory web`), then `scripts/11_record_videos.py` (README clips) or with `--clips docs/videos/cards/clips.yaml --out docs/videos/cards --no-readme --no-gif` (card clips).

## Rules of this project

- **No hand-typed result numbers** in the README results block, reports or paper: they come from `scripts/07_make_report.py`, which reads `results/`. Hand-written prose may describe results qualitatively.
- **Test tracks are protected.** Only evaluation-type code reads them (the evaluator, the CEM baseline, `scripts/10_diagnose_flag.py` and `scripts/12_export_unseen_track.py`), and every read is appended to `results/test_access_log.jsonl` (`eval.evaluator._log_access`). Never tune anything on the test set.
- **The preregistered recipe is frozen.** Any change to settings, rules or analysis is a dated amendment in `reports/preregistration.md` and `reports/plan.md`.
- **Two physics implementations on purpose.** `web/vendor/cannon-es` is the unmodified reference; `src/racingmania/sim` must match it tick by tick (parity tests). Patches, if ever needed, go into both and get a new physics version.
- **Git:** work on a branch, open a pull request, merge it yourself; never commit credentials or `.env`; no AI attribution in commits or PRs.

## Pitfalls already hit

- **Merging while the queue runs.** The queue is one long-lived Python process. Since 2026-11-14 its publish step pulls `main` first and builds the report in a fresh process, so report changes merged mid-queue are picked up. Changes to `matrix.py` itself need a queue restart. Do not merge in the window between a run's export and its publish.
- **Hugging Face limits.** `create_collection` counts against a limit of 100 collection creations per day even with `exists_ok=True`; `publish.py` looks the collection up with `list_collections` instead. A failed publish is logged and the queue continues.
- **Restarting the queue.** Kill the Windows process tree (`taskkill /T /F`), then check WSL for leftover `02_train.py` jobs before relaunching (they survive). Launch the queue with PowerShell `Start-Process` (a `&` job inside `wsl -e bash -lc` dies silently).
- **Windows console** is cp1252: scripts reconfigure stdout to UTF-8 (`scripts/_common.py`); set `PYTHONIOENCODING=utf-8` for ad-hoc runs.
- **Determinism.** Official evaluation is deterministic per batch of tracks, not per track (GPU rounding depends on batch size); reproduce a single track inside its full evaluation batch.
- **Browser caches.** The local `http.server` sends no cache headers, so a browser can mix new and old ES modules after an edit; reload with the cache bypassed. GitHub Pages caches for about 10 minutes.
- **`web/assets/index.json` conflicts** between branches: resolve as the union of tracks, models and ghosts.
