# Source ledger

Every factual claim in [`literature-review.md`](literature-review.md) was checked against its primary source on 2026-11-14 by three independent passes. Each part lists, per claim: the review section, the claim as written, the source URL, where in the source it is (section, table, equation, or file and function), a short exact supporting quote, and a verdict (`confirmed`, `corrected`, `not found in source`, `contradicted`, `source unreachable`).

| Part | Review sections | Claims | File |
|---|---|---|---|
| A | 2–4: RL agents for TrackMania and Gran Turismo, minimum-time racing in parallel simulators, design implications | 99 | [`ledger/part-a-racing-rl.md`](ledger/part-a-racing-rl.md) |
| B | 5–7 and 13: generalisation, algorithms for parallel environments, evaluation and statistics, research practice (includes a research-practice checklist) | 79 | [`ledger/part-b-methods.md`](ledger/part-b-methods.md) |
| C | 8–12: car physics and engines, tracks, baselines, exploits, browser and tooling (includes step-by-step notes for porting cannon-es) | 108 | [`ledger/part-c-physics-tooling.md`](ledger/part-c-physics-tooling.md) |

The corrections found by these passes are applied in the review; its [section 15](literature-review.md#15-fact-check-log) summarises them and lists what remains unverified.

New sources cited later in code, reports or the paper are added to this ledger in the same commit, with the same columns.
