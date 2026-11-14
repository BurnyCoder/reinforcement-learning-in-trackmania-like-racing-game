"""Phase 1: tune the scripted driver, then generate and certify every track set (preregistration, section 3).

Usage: uv run python scripts/01_generate_tracks.py [--splits val test oor train test_unfiltered] [--limit N] [--skip-tune]
"""
import argparse


def main():
    import _common  # noqa: F401  (imported here, not at module level: worker processes re-import this file)
    from racingmania.tracks import build_sets

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["val", "test", "oor", "test_unfiltered", "train"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N seeds of each set (smoke tests)")
    ap.add_argument("--skip-tune", action="store_true", help="reuse results/tracks/scripted_tuning.json")
    ap.add_argument("--processes", type=int, default=8)
    a = ap.parse_args()

    params = build_sets.load_tuned() if a.skip_tune else build_sets.tune(processes=a.processes)
    build_sets.build_campaign(params)
    for split in a.splits:
        build_sets.build_split(split, params, processes=a.processes, limit=a.limit, keep_failed=split == "test_unfiltered")


# Guard required by multiprocessing's "spawn" start method: worker processes import this module and
# must not run the pipeline again (https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods).
if __name__ == "__main__":
    main()
