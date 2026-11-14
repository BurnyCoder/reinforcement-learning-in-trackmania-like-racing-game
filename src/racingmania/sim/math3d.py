"""Batched 3D vector and quaternion helpers that reproduce cannon-es arithmetic exactly.

Global context: the training simulator is a line-by-line port of the cannon-es car physics
(docs/physics.md). For the parity tests to agree with cannon-es to ~1e-9 in float64, every
helper here performs the same floating-point operations in the same order as the cannon-es
function it names (cannon-es 0.20.0, ``web/vendor/cannon-es/cannon-es.js``). Shapes: vectors are
``[..., 3]`` tensors, quaternions ``[..., 4]`` in ``(x, y, z, w)`` order, matrices ``[..., 3, 3]``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Vec3.dot: ``a.x*b.x + a.y*b.y + a.z*b.z`` (summed left to right)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def cross(a: Tensor, b: Tensor) -> Tensor:
    """Vec3.cross ``a × b``: ``(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)``."""
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return torch.stack((x, y, z), dim=-1)


def length_squared(a: Tensor) -> Tensor:
    """Vec3.lengthSquared."""
    return dot(a, a)


def normalize(a: Tensor) -> Tensor:
    """Vec3.normalize: multiply by ``1/n`` (not divide by n); a zero vector becomes zero."""
    n = torch.sqrt(a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1] + a[..., 2] * a[..., 2])
    inv = torch.where(n > 0.0, 1.0 / torch.where(n > 0.0, n, torch.ones_like(n)), torch.zeros_like(n))
    return a * inv.unsqueeze(-1)


def quat_mult(a: Tensor, b: Tensor) -> Tensor:
    """Quaternion.mult ``a ⊗ b`` with cannon-es's term order."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = ax * bw + aw * bx + ay * bz - az * by
    y = ay * bw + aw * by + az * bx - ax * bz
    z = az * bw + aw * bz + ax * by - ay * bx
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((x, y, z, w), dim=-1)


def quat_conj(q: Tensor) -> Tensor:
    """Quaternion.conjugate: ``(-x, -y, -z, w)``."""
    return torch.stack((-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]), dim=-1)


def quat_normalize(q: Tensor) -> Tensor:
    """Quaternion.normalize: multiply every component by ``1/|q|``."""
    x, y, z, w = q.unbind(-1)
    length = torch.sqrt(x * x + y * y + z * z + w * w)
    inv = torch.where(length == 0, torch.zeros_like(length), 1.0 / torch.where(length == 0, torch.ones_like(length), length))
    return q * inv.unsqueeze(-1)


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Quaternion.vmult: rotate vector ``v`` by unit quaternion ``q`` (computes q * v * q⁻¹)."""
    qx, qy, qz, qw = q.unbind(-1)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return torch.stack((rx, ry, rz), dim=-1)


def quat_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """Quaternion.setFromAxisAngle: ``(axis*sin(a/2), cos(a/2))``."""
    s = torch.sin(angle * 0.5)
    return torch.stack((axis[..., 0] * s, axis[..., 1] * s, axis[..., 2] * s, torch.cos(angle * 0.5)), dim=-1)


def quat_integrate(q: Tensor, w: Tensor, dt: float) -> Tensor:
    """Quaternion.integrate with angular factor 1: ``q += dt/2 * (w ⊗ q)`` component by component."""
    ax, ay, az = w[..., 0], w[..., 1], w[..., 2]
    bx, by, bz, bw = q.unbind(-1)
    half_dt = dt * 0.5
    x = bx + half_dt * (ax * bw + ay * bz - az * by)
    y = by + half_dt * (ay * bw + az * bx - ax * bz)
    z = bz + half_dt * (az * bw + ax * by - ay * bx)
    ww = bw + half_dt * (-ax * bx - ay * by - az * bz)
    return torch.stack((x, y, z, ww), dim=-1)


def quat_to_matrix(q: Tensor) -> Tensor:
    """Mat3.setRotationFromQuaternion: rotation matrix ``R`` (row-major elements e[3*row + col])."""
    x, y, z, w = q.unbind(-1)
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    row0 = torch.stack((1 - (yy + zz), xy - wz, xz + wy), dim=-1)
    row1 = torch.stack((xy + wz, 1 - (xx + zz), yz - wx), dim=-1)
    row2 = torch.stack((xz - wy, yz + wx, 1 - (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def mat_vec(m: Tensor, v: Tensor) -> Tensor:
    """Mat3.vmult: ``(e0*x + e1*y + e2*z, e3*x + e4*y + e5*z, e6*x + e7*y + e8*z)``."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r0 = m[..., 0, 0] * x + m[..., 0, 1] * y + m[..., 0, 2] * z
    r1 = m[..., 1, 0] * x + m[..., 1, 1] * y + m[..., 1, 2] * z
    r2 = m[..., 2, 0] * x + m[..., 2, 1] * y + m[..., 2, 2] * z
    return torch.stack((r0, r1, r2), dim=-1)


def inv_inertia_world(q: Tensor, inv_inertia: Tensor) -> Tensor:
    """Body.updateInertiaWorld: ``R · diag(invInertia) · Rᵀ`` with cannon-es's Mat3 operation order.

    ``Mat3.scale(I)`` multiplies column j by ``I[j]``; ``Mat3.mmult`` sums ``a_i1*b_1j + a_i2*b_2j + a_i3*b_3j``
    left to right. ``inv_inertia`` is ``[3]`` (local principal inverse inertia, same for all cars).
    """
    r = quat_to_matrix(q)
    rs = r * inv_inertia  # column j scaled by inv_inertia[j]: (R·diag(I))[i,j] = R[i,j]*I[j]
    rt = r.transpose(-1, -2)  # Rᵀ
    # (R·diag(I))·Rᵀ, each entry summed in cannon-es order: a1*b1 + a2*b2 + a3*b3.
    out = rs[..., :, 0:1] * rt[..., 0:1, :] + rs[..., :, 1:2] * rt[..., 1:2, :] + rs[..., :, 2:3] * rt[..., 2:3, :]
    return out
