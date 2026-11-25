"""Small 3D math for retargeting.

Conventions used everywhere in this package:

- Vectors are ``(x, y, z)`` tuples.
- Matrices are row-major 3x3 tuples. They multiply **column** vectors:
  ``v' = M @ v``.
- Quaternions are ``(w, x, y, z)``. Multiplication applies the right-hand
  factor first: ``R(quat_mul(a, b)) = R(a) @ R(b)``.
- Positive rotations are right-handed.
- BVH Euler channels follow Blender's BVH importer. The importer builds
  ``Euler((x, y, z), channel_order[::-1])``, so a file that lists
  ``Zrotation Yrotation Xrotation`` is the matrix ``Rz @ Ry @ Rx``.
  The last channel hits the bone first. Putting the first channel on the
  inside instead leaves a CMU walk's arms stuck behind the torso.

Roblox is Y-up and characters face **-Z**. A 180 degree yaw takes a BVH
skeleton that faces **+Z** onto that frame.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]

QUAT_IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)
VEC_X: Vec3 = (1.0, 0.0, 0.0)
VEC_Y: Vec3 = (0.0, 1.0, 0.0)
VEC_Z: Vec3 = (0.0, 0.0, 1.0)

_AXES = {
    "X": VEC_X,
    "Y": VEC_Y,
    "Z": VEC_Z,
}


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def v_add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_len(a: Sequence[float]) -> float:
    return math.sqrt(v_dot(a, a))


def v_normalize(a: Sequence[float], fallback: Vec3 = VEC_Y) -> Vec3:
    length = v_len(a)
    if length < 1e-12:
        return fallback
    return (a[0] / length, a[1] / length, a[2] / length)


def v_lerp(a: Sequence[float], b: Sequence[float], t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def v_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return v_len(v_sub(a, b))


def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mat_identity() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def mat_vec(m: Mat3, v: Sequence[float]) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat_transpose(m: Mat3) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def mat_from_columns(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> Mat3:
    return (
        (x[0], y[0], z[0]),
        (x[1], y[1], z[1]),
        (x[2], y[2], z[2]),
    )


def rot_x(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rot_y(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rot_z(angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def quat_normalize(q: Sequence[float]) -> Quat:
    length = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if length < 1e-12:
        return QUAT_IDENTITY
    return (q[0] / length, q[1] / length, q[2] / length, q[3] / length)


def quat_inv(q: Sequence[float]) -> Quat:
    """Inverse of a (near) unit quaternion."""
    n = quat_normalize(q)
    return (n[0], -n[1], -n[2], -n[3])


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Quat:
    """Hamilton product. ``R(a*b) = R(a) @ R(b)`` — ``b`` is applied first."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return quat_normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> Quat:
    axis_n = v_normalize(axis)
    half = angle * 0.5
    s = math.sin(half)
    return (math.cos(half), axis_n[0] * s, axis_n[1] * s, axis_n[2] * s)


def quat_to_mat(q: Sequence[float]) -> Mat3:
    w, x, y, z = quat_normalize(q)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def mat_to_quat(m: Mat3) -> Quat:
    """Shepperd's method. Stable when the matrix is a pure rotation."""
    r00, r01, r02 = m[0]
    r10, r11, r12 = m[1]
    r20, r21, r22 = m[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s
    return quat_normalize((w, x, y, z))


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> Vec3:
    return mat_vec(quat_to_mat(q), v)


def quat_angle(q: Sequence[float]) -> float:
    """Shortest rotation angle of ``q``, in radians."""
    w = abs(quat_normalize(q)[0])
    return 2.0 * math.acos(clamp(w, 0.0, 1.0))


def quat_between_angle(a: Sequence[float], b: Sequence[float]) -> float:
    return quat_angle(quat_mul(quat_inv(a), b))


def quat_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_slerp(a: Sequence[float], b: Sequence[float], t: float) -> Quat:
    """Shortest-path slerp. ``t`` is not clamped, so callers can extrapolate."""
    a_n = quat_normalize(a)
    b_n = quat_normalize(b)
    dot = quat_dot(a_n, b_n)
    if dot < 0.0:
        b_n = (-b_n[0], -b_n[1], -b_n[2], -b_n[3])
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(
            (
                a_n[0] + (b_n[0] - a_n[0]) * t,
                a_n[1] + (b_n[1] - a_n[1]) * t,
                a_n[2] + (b_n[2] - a_n[2]) * t,
                a_n[3] + (b_n[3] - a_n[3]) * t,
            )
        )
    dot = clamp(dot, -1.0, 1.0)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    return quat_normalize(
        (
            a_n[0] * w1 + b_n[0] * w2,
            a_n[1] * w1 + b_n[1] * w2,
            a_n[2] * w1 + b_n[2] * w2,
            a_n[3] * w1 + b_n[3] * w2,
        )
    )


def euler_to_quat(order: str, degrees: Sequence[float]) -> Quat:
    """BVH Euler rotation, matching Blender's importer.

    ``order`` is the channel sequence as written in the file.
    ``degrees`` lines up with ``order`` (not with XYZ storage).

    Blender does ``Euler((x, y, z), channel_order[::-1])``. For ``ZYX`` that
    is Euler order ``XYZ``: ``Rz @ Ry @ Rx``. The last listed channel is
    applied to the bone first, and the first listed channel is outermost.
    """
    if len(order) != len(degrees):
        raise ValueError(f"Euler order {order!r} does not match {len(degrees)} angles")
    q: Quat = QUAT_IDENTITY
    for axis_name, deg in zip(order, degrees):
        key = axis_name.upper()
        if key not in _AXES:
            raise ValueError(f"Unknown Euler axis {axis_name!r}")
        step = quat_from_axis_angle(_AXES[key], math.radians(deg))
        # Newer channel goes on the right, so it is applied to the vector first.
        q = quat_mul(q, step)
    return q


def euler_matrix(order: str, degrees: Sequence[float]) -> Mat3:
    return quat_to_mat(euler_to_quat(order, degrees))


def basis_quat(primary: Sequence[float], hint: Sequence[float], fallback: Sequence[float]) -> Quat:
    """Rotation whose local +Y is ``primary`` and whose local +Z leans toward ``hint``.

    ``fallback`` is used when ``hint`` is parallel to ``primary`` (feet, which
    point along the character forward axis). The result is right-handed.
    """
    y = v_normalize(primary)
    hint_v = hint
    if abs(v_dot(y, v_normalize(hint_v))) > 0.95:
        hint_v = fallback
    z_raw = v_sub(hint_v, v_mul(y, v_dot(hint_v, y)))
    if v_len(z_raw) < 1e-8:
        z_raw = v_sub(fallback, v_mul(y, v_dot(fallback, y)))
    z = v_normalize(z_raw, VEC_Z)
    x = v_normalize(v_cross(y, z), VEC_X)
    # Re-orthogonalize Z so the basis stays right-handed after normalization.
    z = v_normalize(v_cross(x, y), VEC_Z)
    return mat_to_quat(mat_from_columns(x, y, z))


def yaw_about_y(q: Sequence[float], forward: Vec3 = (0.0, 0.0, -1.0)) -> Tuple[Quat, Quat]:
    """Split ``q`` into a yaw about +Y and the leftover pitch/roll.

    Returns ``(q_yaw, q_rest)`` with ``q = q_yaw * q_rest``. ``forward`` is the
    vector that should be treated as the character's facing direction before
    the rotation (Roblox forward is -Z).
    """
    faced = quat_rotate(q, forward)
    # Ry(+θ) sends -Z to (-sin θ, 0, -cos θ). Recover θ from that XZ projection.
    yaw = math.atan2(-faced[0], -faced[2])
    q_yaw = quat_from_axis_angle(VEC_Y, yaw)
    q_rest = quat_mul(quat_inv(q_yaw), q)
    return q_yaw, q_rest


def forward_axis_quaternion(axis: str) -> Quat:
    """Yaw that maps a BVH forward axis onto Roblox forward ``(0, 0, -1)``.

    ``axis`` is the direction the **source character faces** in BVH coordinates:
    ``z``, ``-z``, ``x``, or ``-x``. Y stays up.
    """
    key = axis.lower().replace("plus", "").replace("+", "")
    # Ry(θ) @ forward_bvh = (0, 0, -1).
    #   +Z --180°--> -Z,  +X --+90°--> -Z,  -X --−90°--> -Z,  -Z already there.
    angles = {
        "z": math.pi,
        "-z": 0.0,
        "x": math.pi / 2.0,
        "-x": -math.pi / 2.0,
    }
    if key not in angles:
        raise ValueError(f"forward axis must be z, -z, x, or -x (got {axis!r})")
    return quat_from_axis_angle(VEC_Y, angles[key])


def quat_between(a: Sequence[float], b: Sequence[float]) -> Quat:
    """Shortest rotation that takes vector ``a`` onto vector ``b``."""
    a_n = v_normalize(a)
    b_n = v_normalize(b)
    d = v_dot(a_n, b_n)
    if d > 0.999999:
        return QUAT_IDENTITY
    if d < -0.999999:
        axis = v_cross(a_n, VEC_X)
        if v_len(axis) < 1e-6:
            axis = v_cross(a_n, VEC_Y)
        return quat_from_axis_angle(axis, math.pi)
    axis = v_cross(a_n, b_n)
    return quat_normalize((1.0 + d, axis[0], axis[1], axis[2]))


def conjugate(q: Sequence[float], by: Sequence[float]) -> Quat:
    """``by * q * by^-1``. Changes the basis of a rotation."""
    return quat_mul(by, quat_mul(q, quat_inv(by)))


def almost(a: float, b: float, eps: float = 1e-5) -> bool:
    return abs(a - b) <= eps


def almost_vec(a: Sequence[float], b: Sequence[float], eps: float = 1e-5) -> bool:
    return all(abs(a[i] - b[i]) <= eps for i in range(min(len(a), len(b))))


def median(values: Iterable[float]) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return 0.5 * (data[mid - 1] + data[mid])


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * clamp(pct, 0.0, 100.0) / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (k - lo)
