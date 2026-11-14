"""Tracks packed into padded tensors so that every car in a batch can be on a different track.

Global context: a track is a ribbon (docs/formats.md, "Track file"): centreline samples, a banked
left vector per sample, a half width, a gap mask and checkpoints. The game turns the ribbon into a
cannon-es triangle mesh; the port derives the same triangles here, rounded to 32-bit floats as
cannon-es stores them, and gathers the few segments around each car for its wheel rays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# Segments around a car's nearest sample that its wheel rays are tested against: [index-3, index+4].
# With 2 m samples this covers ±6 m around the chassis, more than the 1.35 m wheel offset plus
# 0.5 m of travel per tick at 50 m/s.
WINDOW_OFFSETS = tuple(range(-3, 5))


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


@dataclass
class TrackBatch:
    """Padded track tensors ``[T, ...]`` on one device in one dtype."""

    center: Tensor  # [T, M, 3] centreline samples
    left: Tensor  # [T, M, 3] unit vector to the left edge (includes banking)
    floor: Tensor  # [T, M] height below which a car is out
    cum: Tensor  # [T, M] arc length (m)
    n: Tensor  # [T] number of real samples (long)
    seg_valid: Tensor  # [T, M-1] segment has road (not a gap, not padding)
    edges: Tensor  # [T, M, 2, 3] left/right edge vertices (float32 values)
    normals: Tensor  # [T, M-1, 2, 3] triangle face normals (float32 values)
    cp_s: Tensor  # [T, C] arc length of each checkpoint; padding is +inf
    n_cp: Tensor  # [T] number of checkpoints (long)
    start_pos: Tensor  # [T, 3]
    start_quat: Tensor  # [T, 4]
    start_index: Tensor  # [T] (long)
    ref_ticks: Tensor  # [T] scripted driver's finishing ticks (0 if unknown)
    half_width: Tensor  # [T]
    names: list

    @property
    def num_tracks(self) -> int:
        return self.center.shape[0]

    @classmethod
    def from_tracks(cls, tracks: list[dict], device="cpu", dtype=torch.float64) -> "TrackBatch":
        """Pack track dictionaries (the track-file format) into padded tensors."""
        T = len(tracks)
        M = max(len(t["center"]) for t in tracks)
        C = max(len(t["checkpoints"]) for t in tracks)
        center = np.zeros((T, M, 3)); left = np.zeros((T, M, 3)); floor = np.full((T, M), -1e9)
        cum = np.zeros((T, M)); n = np.zeros(T, dtype=np.int64); seg_valid = np.zeros((T, M - 1), dtype=bool)
        edges = np.zeros((T, M, 2, 3)); normals = np.zeros((T, M - 1, 2, 3))
        cp_s = np.full((T, C), np.inf); n_cp = np.zeros(T, dtype=np.int64)
        start_pos = np.zeros((T, 3)); start_quat = np.zeros((T, 4)); start_index = np.zeros(T, dtype=np.int64)
        ref_ticks = np.zeros(T); half_width = np.zeros(T)
        for i, t in enumerate(tracks):
            c = np.asarray(t["center"], dtype=np.float64); m = len(c)
            l = np.asarray(t["left"], dtype=np.float64)
            center[i, :m] = c; left[i, :m] = l; n[i] = m
            floor[i, :m] = np.asarray(t["floor"], dtype=np.float64)
            cl = cumulative_length(c); cum[i, :m] = cl
            # Pad positions repeat the last sample so that clamped window lookups stay finite.
            center[i, m:] = c[-1]; left[i, m:] = l[-1]; cum[i, m:] = cl[-1]
            e = ribbon_edges(c, l, float(t["half_width"])); edges[i, :m] = e; edges[i, m:] = e[-1]
            normals[i, : m - 1] = triangle_normals(e)
            seg_valid[i, : m - 1] = ~np.asarray(t["gap"], dtype=bool)
            cps = list(t["checkpoints"]); cp_s[i, : len(cps)] = cl[cps]; n_cp[i] = len(cps)
            start_pos[i] = t["start"]["position"]; start_quat[i] = t["start"]["quaternion"]
            start_index[i] = t["start"]["index"]
            ref_ticks[i] = (t.get("reference") or {}).get("ticks", 0)
            half_width[i] = t["half_width"]
        as_t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        as_l = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
        return cls(center=as_t(center), left=as_t(left), floor=as_t(floor), cum=as_t(cum), n=as_l(n),
                   seg_valid=torch.as_tensor(seg_valid, device=device), edges=as_t(edges), normals=as_t(normals),
                   cp_s=as_t(cp_s), n_cp=as_l(n_cp), start_pos=as_t(start_pos), start_quat=as_t(start_quat),
                   start_index=as_l(start_index), ref_ticks=as_t(ref_ticks), half_width=as_t(half_width),
                   names=[t.get("name", str(t.get("seed"))) for t in tracks])

    @classmethod
    def from_arrays(cls, p: dict, device="cpu", dtype=torch.float32) -> "TrackBatch":
        """Vectorised construction from packed pool arrays (``tracks.store.pack``).

        Same geometry as ``from_tracks``: edges ``center ± half_width·left`` rounded to float32,
        normals ``normalize((c - a) × (a - b))`` per triangle, arc length summed sequentially.
        """
        center = p["center"].astype(np.float64); left = p["left"].astype(np.float64)
        T, M = center.shape[:2]
        hw = p["half_width"].astype(np.float64)[:, None, None]
        edges = f32(np.stack((center + hw * left, center - hw * left), axis=2))  # [T, M, 2, 3]
        lk, rk, lk1, rk1 = edges[:, :-1, 0], edges[:, :-1, 1], edges[:, 1:, 0], edges[:, 1:, 1]
        normals = []
        for a, b, c in ((lk, rk, lk1), (rk, rk1, lk1)):
            ab, cb = a - b, c - a
            nrm = np.stack((cb[..., 1] * ab[..., 2] - cb[..., 2] * ab[..., 1],
                            cb[..., 2] * ab[..., 0] - cb[..., 0] * ab[..., 2],
                            cb[..., 0] * ab[..., 1] - cb[..., 1] * ab[..., 0]), axis=-1)
            ln = np.sqrt((nrm * nrm).sum(-1, keepdims=True))
            normals.append(nrm / np.where(ln > 0, ln, 1.0))
        normals = f32(np.stack(normals, axis=2))  # [T, M-1, 2, 3]
        d = center[:, 1:] - center[:, :-1]
        seg = np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])
        cum = np.concatenate((np.zeros((T, 1)), np.add.accumulate(seg, axis=1)), axis=1)
        n = p["n"].astype(np.int64)
        real_seg = np.arange(M - 1)[None, :] < (n - 1)[:, None]
        seg_valid = ~p["gap"].astype(bool) & real_seg
        cps = p["checkpoints"]; cp_s = np.full(cps.shape, np.inf)
        ok = cps >= 0
        cp_s[ok] = np.take_along_axis(cum, np.where(ok, cps, 0), axis=1)[ok]
        as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        as_l = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        return cls(center=as_t(center), left=as_t(left), floor=as_t(p["floor"]), cum=as_t(cum), n=as_l(n),
                   seg_valid=torch.as_tensor(seg_valid, device=device), edges=as_t(edges), normals=as_t(normals),
                   cp_s=as_t(cp_s), n_cp=as_l(p["n_cp"]), start_pos=as_t(p["start_pos"]), start_quat=as_t(p["start_quat"]),
                   start_index=as_l(p["start_index"]), ref_ticks=as_t(p["ref_ticks"]), half_width=as_t(p["half_width"]),
                   names=list(p["names"]))

    def window_triangles(self, track_id: Tensor, index: Tensor):
        """Triangles of the segments around each car's nearest sample.

        ``track_id``, ``index``: ``[N]`` long. Returns ``a, b, c, normal`` ``[N, K, 3]`` and ``valid``
        ``[N, K]`` with ``K = 2 * len(WINDOW_OFFSETS)``, in segment order, triangle 0 before 1.
        """
        off = torch.tensor(WINDOW_OFFSETS, device=index.device)
        k = index.unsqueeze(1) + off  # [N, W] candidate segment indices
        n_seg = self.n[track_id] - 1  # segments per car's track
        in_range = (k >= 0) & (k < n_seg.unsqueeze(1))
        kc = k.clamp(0, self.seg_valid.shape[1] - 1)
        tid = track_id.unsqueeze(1).expand_as(kc)
        valid = in_range & self.seg_valid[tid, kc]
        lk = self.edges[tid, kc, 0]; rk = self.edges[tid, kc, 1]  # [N, W, 3]
        lk1 = self.edges[tid, kc + 1, 0]; rk1 = self.edges[tid, kc + 1, 1]
        nrm = self.normals[tid, kc]  # [N, W, 2, 3]
        a = torch.stack((lk, rk), dim=2).flatten(1, 2)  # [N, 2W, 3]: triangle 0 a=L_k, triangle 1 a=R_k
        b = torch.stack((rk, rk1), dim=2).flatten(1, 2)
        c = torch.stack((lk1, lk1), dim=2).flatten(1, 2)
        normal = nrm.flatten(1, 2)
        valid2 = valid.unsqueeze(2).expand(-1, -1, 2).flatten(1, 2)
        return a, b, c, normal, valid2

    def subset(self, ids) -> "TrackBatch":
        """A TrackBatch with only the tracks ``ids`` (keeps padding size)."""
        ids = torch.as_tensor(ids, device=self.center.device, dtype=torch.long)
        fields = {f: getattr(self, f)[ids] for f in ("center", "left", "floor", "cum", "n", "seg_valid", "edges",
                                                       "normals", "cp_s", "n_cp", "start_pos", "start_quat",
                                                       "start_index", "ref_ticks", "half_width")}
        return TrackBatch(**fields, names=[self.names[i] for i in ids.tolist()])

    def to(self, device=None, dtype=None) -> "TrackBatch":
        """Move to another device and/or floating dtype (integer and bool fields keep their type)."""
        out = {}
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if isinstance(v, Tensor):
                out[f] = v.to(device=device, dtype=dtype if v.is_floating_point() and dtype else v.dtype)
            else:
                out[f] = v
        return TrackBatch(**out)
