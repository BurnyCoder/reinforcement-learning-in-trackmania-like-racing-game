"""Track pools: many tracks packed into padded arrays, saved as one file per set.

Global context: training samples from tens of thousands of generated tracks, too many to keep as
JSON files. A pool stores the arrays the simulator needs (centreline, left vectors, floor, gaps,
checkpoints, start, reference time) for every certified track of one set (train, val, test, oor),
padded to a common length, in float32 (training precision). ``TrackBatch.from_pool`` turns any
subset of a pool into GPU tensors; float64 parity tests use the JSON track files instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..sim.track_tensors import TrackBatch
from ..utils.paths import data_dir

POOL_KEYS = ("center", "left", "floor", "gap", "n", "checkpoints", "n_cp", "start_pos", "start_quat", "start_index",
             "half_width", "ref_ticks", "seeds")


def pool_path(split: str) -> Path:
    return data_dir() / "tracks" / f"{split}.pt"


def pack(tracks: list[dict], ref_ticks: list[int] | None = None) -> dict:
    """Pack track dictionaries into padded numpy arrays."""
    T = len(tracks)
    M = max(len(t["center"]) for t in tracks)
    C = max(len(t["checkpoints"]) for t in tracks)
    out = {
        "center": np.zeros((T, M, 3), np.float32), "left": np.zeros((T, M, 3), np.float32),
        "floor": np.full((T, M), -1e9, np.float32), "gap": np.ones((T, M - 1), bool), "n": np.zeros(T, np.int64),
        "checkpoints": np.full((T, C), -1, np.int64), "n_cp": np.zeros(T, np.int64),
        "start_pos": np.zeros((T, 3), np.float32), "start_quat": np.zeros((T, 4), np.float32),
        "start_index": np.zeros(T, np.int64), "half_width": np.zeros(T, np.float32),
        "ref_ticks": np.zeros(T, np.int64), "seeds": np.full(T, -1, np.int64),
    }
    for i, t in enumerate(tracks):
        m = len(t["center"])
        c = np.asarray(t["center"]); l = np.asarray(t["left"])
        out["center"][i, :m] = c; out["center"][i, m:] = c[-1]
        out["left"][i, :m] = l; out["left"][i, m:] = l[-1]
        out["floor"][i, :m] = t["floor"]
        out["gap"][i, : m - 1] = np.asarray(t["gap"], bool)
        out["n"][i] = m
        cps = t["checkpoints"]; out["checkpoints"][i, : len(cps)] = cps; out["n_cp"][i] = len(cps)
        out["start_pos"][i] = t["start"]["position"]; out["start_quat"][i] = t["start"]["quaternion"]
        out["start_index"][i] = t["start"]["index"]; out["half_width"][i] = t["half_width"]
        out["seeds"][i] = t["seed"] if t.get("seed") is not None else -1
        if ref_ticks is not None:
            out["ref_ticks"][i] = ref_ticks[i]
        elif t.get("reference"):
            out["ref_ticks"][i] = t["reference"]["ticks"]
    out["names"] = [t["name"] for t in tracks]
    return out


def save_pool(pool: dict, path: Path, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**{k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in pool.items()},
                "meta": meta or {}}, path)


def load_pool(path: Path) -> dict:
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def subset(pool: dict, ids) -> dict:
    """Pool restricted to track indices ``ids`` (padding kept)."""
    ids = np.asarray(ids)
    return {k: (v[ids] if isinstance(v, np.ndarray) else ([v[i] for i in ids] if k == "names" else v)) for k, v in pool.items()}


def batch_from_pool(pool: dict, ids=None, device="cpu", dtype=torch.float32) -> TrackBatch:
    """Build a TrackBatch (with triangles, normals, arc lengths) from a pool or a subset of it."""
    p = pool if ids is None else subset(pool, ids)
    return TrackBatch.from_arrays(p, device=device, dtype=dtype)
