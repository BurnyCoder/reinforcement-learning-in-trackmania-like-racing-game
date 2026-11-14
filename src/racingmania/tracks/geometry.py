"""Track geometry shared by the game files and the simulator, in plain NumPy (no PyTorch).

Global context: these functions define the triangles the game's cannon-es mesh and the port's
wheel rays both use (docs/formats.md). They live in a torch-free module so that the track
generator's worker processes stay small (each PyTorch import costs ~600 MB of RAM).
"""

from __future__ import annotations

import numpy as np


def f32(x: np.ndarray) -> np.ndarray:
    """Round to the nearest float32 and return as float64 (how cannon-es stores mesh data)."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def ribbon_edges(center: np.ndarray, left: np.ndarray, half_width: float) -> np.ndarray:
    """Left and right edge vertices ``[M, 2, 3]``: ``center ± half_width * left``, rounded to float32.

    The same function writes the ``vertices`` of track files (interleaved left, right per sample),
    so the game's mesh and the port's triangles are bit-identical.
    """
    lft = center + half_width * left
    rgt = center - half_width * left
    return f32(np.stack((lft, rgt), axis=1))


def triangle_normals(edges: np.ndarray) -> np.ndarray:
    """Face normals ``[M-1, 2, 3]`` as Trimesh.updateNormals computes and stores them.

    Triangle 0 of segment k is (L_k, R_k, L_k+1), triangle 1 is (R_k, R_k+1, L_k+1). cannon-es calls
    ``computeNormal(vb, va, vc)``, i.e. ``normalize((c - a) × (a - b))``, then stores it as float32.
    """
    lk, rk = edges[:-1, 0], edges[:-1, 1]
    lk1, rk1 = edges[1:, 0], edges[1:, 1]
    out = []
    for a, b, c in ((lk, rk, lk1), (rk, rk1, lk1)):
        ab = a - b
        cb = c - a
        n = np.stack((cb[:, 1] * ab[:, 2] - cb[:, 2] * ab[:, 1],
                      cb[:, 2] * ab[:, 0] - cb[:, 0] * ab[:, 2],
                      cb[:, 0] * ab[:, 1] - cb[:, 1] * ab[:, 0]), axis=-1)
        length = np.sqrt(n[:, 0] * n[:, 0] + n[:, 1] * n[:, 1] + n[:, 2] * n[:, 2])
        inv = np.where(length > 0, 1.0 / np.where(length > 0, length, 1.0), 0.0)
        out.append(n * inv[:, None])
    return f32(np.stack(out, axis=1))


def cumulative_length(center: np.ndarray) -> np.ndarray:
    """Arc length at each sample, summed sequentially as rules.js does (np.add.accumulate is sequential)."""
    d = center[1:] - center[:-1]
    seg = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
    return np.concatenate(([0.0], np.add.accumulate(seg)))
