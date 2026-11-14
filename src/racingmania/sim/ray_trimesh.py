"""Batched ray casts against candidate triangles, reproducing cannon-es ``World.rayTest``.

Global context: each car's four suspension rays are cast against the triangles of the track
segments near the car (``track_tensors.TrackBatch.window_triangles``). cannon-es casts against the
whole mesh through an octree, but its octree query is conservative (``Octree.aabbQuery``), so the
closest hit is the same as long as every triangle the ray could reach is in the window.

The arithmetic follows ``Ray._intersectTrimesh``, ``Ray.pointInTriangle`` and
``Ray.reportIntersection`` (mode ``CLOSEST``, ``skipBackfaces: true``) for a mesh body at the origin
with identity rotation and unit scale, where cannon-es's local/world transforms are exact.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .math3d import dot, normalize


def _point_in_triangle(p: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Ray.pointInTriangle: barycentric test without division (strict on the far edge)."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = dot(v0, v0)
    dot01 = dot(v0, v1)
    dot02 = dot(v0, v2)
    dot11 = dot(v1, v1)
    dot12 = dot(v1, v2)
    u = dot11 * dot02 - dot01 * dot12
    v = dot00 * dot12 - dot01 * dot02
    return (u >= 0) & (v >= 0) & (u + v < dot00 * dot11 - dot01 * dot01)


def raycast_closest(origin: Tensor, target: Tensor, tri_a: Tensor, tri_b: Tensor, tri_c: Tensor,
                    tri_n: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Closest front-face hit of segment ``origin → target`` among K candidate triangles per ray.

    Shapes: ``origin``, ``target`` ``[R, 3]``; ``tri_*`` ``[R, K, 3]`` (vertices a, b, c in mesh order
    and the stored face normal); ``valid`` ``[R, K]`` marks real triangles (False for padding/gaps).
    Returns ``hit [R]`` (bool), ``distance [R]``, ``point [R, 3]``, ``normal [R, 3]``; rows without a
    hit have distance ``inf`` and zero point/normal.
    """
    direction = normalize(target - origin)  # Ray.updateDirection: normalize(to - from)
    from_to_sq = dot(target - origin, target - origin)  # localFrom.distanceSquared(localTo)
    o = origin.unsqueeze(1)  # [R, 1, 3], broadcast over candidates
    d = direction.unsqueeze(1)
    vector = tri_a - o  # a - localFrom
    dn = dot(d, tri_n)  # localDirection · normal (no parallel check in cannon-es)
    scalar = dot(tri_n, vector) / dn  # distance along the ray to the triangle's plane
    point = d * scalar.unsqueeze(-1) + o  # localDirection*scalar + localFrom
    diff = point - o
    sq_dist = dot(diff, diff)  # intersectPoint.distanceSquared(localFrom)
    inside = _point_in_triangle(point, tri_b, tri_a, tri_c) | _point_in_triangle(point, tri_a, tri_b, tri_c)
    ok = valid & (scalar >= 0) & inside & (sq_dist <= from_to_sq.unsqueeze(1))  # NaN comparisons are False, as in JS
    ok = ok & ~(dot(tri_n, d) > 0)  # reportIntersection: skip back faces (normal · direction > 0)
    distance = torch.sqrt(dot(point - o, point - o))  # from.distanceTo(hitPointWorld)
    distance = torch.where(ok, distance, torch.full_like(distance, float("inf")))
    best = torch.argmin(distance, dim=1)  # CLOSEST: smallest distance wins (ties: first candidate)
    idx = best.view(-1, 1)
    best_dist = distance.gather(1, idx).squeeze(1)
    hit = torch.isfinite(best_dist)
    best_point = point.gather(1, idx.unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)
    best_normal = tri_n.gather(1, idx.unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)
    zero = torch.zeros_like(best_point)
    best_point = torch.where(hit.unsqueeze(-1), best_point, zero)
    best_normal = torch.where(hit.unsqueeze(-1), best_normal, zero)
    return hit, best_dist, best_point, best_normal
