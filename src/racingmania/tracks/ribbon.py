"""Build a 3D ribbon track (centreline, banked left vectors, gaps, checkpoints) from a segment plan.

Global context: both the hand-made campaign tracks (``campaign.py``) and the procedural generator
(``generator.py``) describe a track as a list of segments in the style of the TORCS track format
(https://torcs.sourceforge.net/api/track_manual.html): straights and arcs, each with a height
change, arcs with banking, and optional jumps on straights. This module turns such a plan into the
track-file dictionary of docs/formats.md, which the game (cannon-es triangle mesh) and the training
simulator (``sim/track_tensors.py``) both read. The ribbon parameterisation (a spine curve with a
lateral coordinate) follows Perantoni & Limebeer 2015 (doi:10.1115/1.4028253).
"""

from __future__ import annotations

import math

import numpy as np

from ..sim.track_tensors import ribbon_edges

SPACING = 2.0  # metres between centreline samples (horizontal arc length)
FINE = 0.25  # metres between samples of the fine construction path
HALF_WIDTH = 6.0  # road half width (m)
FLOOR_DROP = 3.0  # a car this far below the local track height is out
CHECKPOINT_EVERY = 150.0  # metres between checkpoints
START_INDEX = 3  # the car starts above this centreline sample
RUNOUT = 30.0  # road after the finish line (m)


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """3u² - 2u³: C1 blend from 0 to 1 with zero slope at both ends."""
    return u * u * (3.0 - 2.0 * u)


def quat_from_frame(forward: np.ndarray, up: np.ndarray) -> list[float]:
    """Unit quaternion (x, y, z, w) whose rotation maps body +X to ``forward`` and +Y to ``up``.

    Body +Z (right) becomes forward × up. Standard rotation-matrix-to-quaternion conversion
    (Shepperd's method, branch on the largest diagonal term).
    """
    f = forward / np.linalg.norm(forward)
    u = up - f * np.dot(up, f)
    u = u / np.linalg.norm(u)
    r = np.cross(f, u)
    m = np.stack((f, u, r), axis=1)  # columns are the body axes in world coordinates
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def _fine_path(plan: list[dict]):
    """Integrate the plan on a fine grid. Returns per-fine-sample arrays: x, z (plan position),
    h (height), bank (signed rad, positive raises the left edge), gap (bool), seg (segment id)."""
    # Start with the origin sample so that the path begins exactly at (0, 0, 0).
    xs, zs, hs, banks, gaps, segs = [0.0], [0.0], [0.0], [0.0], [False], [0]
    x = z = heading = 0.0  # heading 0 = +X; positive turns toward -Z (left)
    h0 = 0.0
    for k, seg in enumerate(plan):
        if seg["kind"] == "straight":
            length = float(seg["length"])
            curvature = 0.0
        else:
            length = float(seg["radius"]) * math.radians(float(seg["angle"]))
            curvature = float(seg["dir"]) / float(seg["radius"])  # +1 left, -1 right
        n = max(1, int(round(length / FINE)))
        u = (np.arange(n) + 1) / n  # end of each fine step within the segment
        dh = float(seg.get("dh", 0.0))
        h = h0 + dh * _smoothstep(u)
        gap = np.zeros(n, dtype=bool)
        jump = seg.get("jump")
        if jump:
            s_local = u * length
            a = float(jump["at"])  # kicker start (m from segment start), a multiple of SPACING
            lk, hk, lg = float(jump["kicker_len"]), float(jump["kicker_h"]), float(jump["gap"])
            ll, hl = float(jump["landing_len"]), float(jump["kicker_h"]) - float(jump["drop"])
            kick = (s_local > a) & (s_local <= a + lk)
            uk = (s_local - a) / lk
            h = h + np.where(kick, hk * uk * uk, 0.0)  # parabolic kicker: take-off slope 2 hk / lk
            in_gap = (s_local > a + lk) & (s_local < a + lk + lg)
            gap |= in_gap
            land = (s_local >= a + lk + lg) & (s_local <= a + lk + lg + ll)
            ul = (s_local - (a + lk + lg)) / ll
            h = h + np.where(land, hl * (1 - ul) * (1 - ul), 0.0)  # landing descends to the road
            # Inside the gap the centreline is the straight line from the lip to the landing start.
            g = (s_local - (a + lk)) / lg
            h_lip = h0 + dh * _smoothstep((a + lk) / length) + hk
            h_land = h0 + dh * _smoothstep((a + lk + lg) / length) + hl
            h = np.where(in_gap, h_lip + (h_land - h_lip) * g, h)
        bank_target = math.radians(float(seg.get("bank", 0.0))) * (-float(seg.get("dir", 0)))  # inner edge lower
        ramp = np.minimum(1.0, np.minimum(u, 1 - u + 1.0 / n) * length / max(1e-9, min(15.0, 0.3 * length)))
        bank = bank_target * _smoothstep(np.clip(ramp, 0, 1))
        for i in range(n):
            heading += curvature * FINE * (length / (n * FINE))
            step = length / n
            x += math.cos(heading) * step
            z -= math.sin(heading) * step
            xs.append(x); zs.append(z); hs.append(h[i]); banks.append(bank[i]); gaps.append(gap[i]); segs.append(k)
        h0 += dh
    return (np.array(xs), np.array(zs), np.array(hs), np.array(banks), np.array(gaps), np.array(segs))


def build_track(plan: list[dict], name: str, seed=None, half_width: float = HALF_WIDTH,
                spawn_height: float = 0.95, generator_info: dict | None = None) -> dict:
    """Turn a segment plan into a track-file dictionary (docs/formats.md), without the reference run.

    The plan's last segment should be a straight of at least ``RUNOUT`` + 20 m: the finish line is
    placed ``RUNOUT`` metres before the end of the road.
    """
    x, z, h, bank, gap, _ = _fine_path(plan)
    s2d = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))))  # horizontal arc length
    m = int(s2d[-1] // SPACING) + 1
    st = np.arange(m) * SPACING
    cx = np.interp(st, s2d, x); cz = np.interp(st, s2d, z); cy = np.interp(st, s2d, h)
    cb = np.interp(st, s2d, bank)
    center = np.stack((cx, cy, cz), axis=1)
    # Gap per segment: the segment between samples i and i+1 has no road if its midpoint lies in a gap.
    mid = (st[:-1] + st[1:]) / 2
    seg_gap = np.interp(mid, s2d, gap.astype(float)) > 0.5
    # Tangent by central differences (one-sided at the ends).
    t = np.gradient(center, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    left_h = np.cross(np.array([0.0, 1.0, 0.0]), t)  # horizontal left: Y × t
    left_h /= np.linalg.norm(left_h, axis=1, keepdims=True)
    up_r = np.cross(t, left_h)  # road "up" before banking
    left = np.cos(cb)[:, None] * left_h + np.sin(cb)[:, None] * up_r  # rotate about t by the bank angle
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    # Floor: 3 m below the road; inside a gap, 3 m below the lower of the lip and the landing.
    floor = cy - FLOOR_DROP
    i = 0
    while i < m - 1:
        if seg_gap[i]:
            j = i
            while j < m - 1 and seg_gap[j]:
                j += 1  # segments i..j-1 are gaps: samples i (lip) .. j (landing start)
            low = min(cy[i], cy[j]) - FLOOR_DROP
            floor[i + 1:j] = low
            i = j
        else:
            i += 1
    # Checkpoints every CHECKPOINT_EVERY metres on road samples, and the finish RUNOUT before the end.
    finish = int(round((st[-1] - RUNOUT) / SPACING))
    cps = []
    s_next = CHECKPOINT_EVERY
    for idx in range(START_INDEX + 1, finish):
        on_road = not seg_gap[min(idx, m - 2)] and not seg_gap[max(idx - 1, 0)]
        if st[idx] >= s_next and on_road:
            cps.append(idx)
            s_next = st[idx] + CHECKPOINT_EVERY
    if cps and finish - cps[-1] < 20:
        cps.pop()  # avoid a checkpoint right before the finish
    cps.append(finish)

    edges = ribbon_edges(center, left, half_width)  # float32-rounded, same as the port
    vertices = edges.reshape(-1, 3)  # interleaved: 2i = left edge, 2i+1 = right edge
    tris, tri_seg = [], []
    for k in range(m - 1):
        if seg_gap[k]:
            continue
        tris.append([2 * k, 2 * k + 1, 2 * k + 2]); tri_seg.append(k)
        tris.append([2 * k + 1, 2 * k + 3, 2 * k + 2]); tri_seg.append(k)
    up = np.cross(t[START_INDEX], left[START_INDEX])
    start_pos = center[START_INDEX] + up / np.linalg.norm(up) * spawn_height
    return {
        "format": 1,
        "name": name,
        "seed": seed,
        "spacing": SPACING,
        "half_width": half_width,
        "center": center.tolist(),
        "left": left.tolist(),
        "gap": seg_gap.astype(int).tolist(),
        "floor": floor.tolist(),
        "checkpoints": cps,
        "vertices": vertices.tolist(),
        "triangles": tris,
        "triangle_segment": tri_seg,
        "start": {"position": start_pos.tolist(), "quaternion": quat_from_frame(t[START_INDEX], up), "index": START_INDEX},
        "reference": None,
        "length": float(np.sum(np.linalg.norm(np.diff(center, axis=0), axis=1))),
        "generator": generator_info or {},
        "plan": plan,
    }


def crossing_ok(track: dict, min_vertical: float = 8.0, min_arc: float = 30.0) -> bool:
    """True if no two parts of the track that are far apart along it come within the road width
    horizontally while being less than ``min_vertical`` metres apart vertically."""
    c = np.asarray(track["center"])
    hw = float(track["half_width"])
    sp = float(track["spacing"])
    xz = c[:, [0, 2]]
    d2 = ((xz[:, None, :] - xz[None, :, :]) ** 2).sum(-1)
    idx = np.arange(len(c))
    far_along = np.abs(idx[:, None] - idx[None, :]) * sp > min_arc
    close = d2 < (2 * hw + 6.0) ** 2
    dy = np.abs(c[:, None, 1] - c[None, :, 1])
    return not np.any(far_along & close & (dy < min_vertical))
