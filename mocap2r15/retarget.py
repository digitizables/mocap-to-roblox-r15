"""Collapse a BVH pose onto R15 Motor6D transforms.

For each R15 part the local BVH rotations of the mapped chain are composed
the way forward kinematics does (``L = R_parent @ R_child @ ...``). That rotation is moved into
Roblox axes by conjugation with the forward-axis yaw ``C``, then a rest
correction lines the bind pose up with the R15 stance::

    T = D_parent^-1 * (C * L * C^-1) * D_part

``D_part`` is the rotation that takes the R15 rest bone direction onto the
axis-converted BVH bone direction, with a shared "character forward" twist
hint so both arms do not roll randomly. At the BVH bind pose (``L = I``) a
T-pose arm (bone along +X) becomes the R15 arms-down rest plus the shoulder
correction, and a skeleton whose offsets already hang the arms down stays
near identity. Animated frames ride on top of that correction.

Fingers and toes are not parts. Their offsets are only used as the direction
of the hand or the foot. Missing joints leave that R15 pose at identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from mocap2r15.math3d import (
    QUAT_IDENTITY,
    Quat,
    Vec3,
    basis_quat,
    conjugate,
    quat_between_angle,
    quat_inv,
    quat_mul,
    quat_rotate,
)
from mocap2r15.presets import Mapping
from mocap2r15.r15 import PART_ORDER, POSE_PARENT, target_hint


@dataclass
class R15Frame:
    time: float
    local: Dict[str, Quat]
    root_pos: Vec3


def compose_chain(quats: Iterable[Quat]) -> Quat:
    """Compose a parent-to-child chain the way forward kinematics does.

    The first quaternion is the outermost parent. A vector in the last joint's
    local frame sees the last quaternion first and the parent last:
    ``R_parent @ R_child @ v``.
    """
    composed: Quat = QUAT_IDENTITY
    for quat in quats:
        composed = quat_mul(composed, quat)
    return composed


def rest_corrections(mapping: Mapping, conv: Quat) -> Dict[str, Quat]:
    """``D_part`` quaternions, including identity for the HumanoidRootPart."""
    corrections = {"HumanoidRootPart": QUAT_IDENTITY}
    # BVH characters are described as facing +Z before ``conv`` is applied.
    # ``conv`` itself is what turns that axis onto Roblox -Z, so the hint is
    # converted with the same yaw as the bone direction.
    src_hint = quat_rotate(conv, (0.0, 0.0, 1.0))
    src_fallback = quat_rotate(conv, (0.0, 1.0, 0.0))
    tgt_fallback = (0.0, 1.0, 0.0)
    for part, spec in mapping.parts.items():
        src_rbx = quat_rotate(conv, spec.src_dir)
        q_src = basis_quat(src_rbx, src_hint, src_fallback)
        q_tgt = basis_quat(spec.tgt_dir, target_hint(part), tgt_fallback)
        corrections[part] = quat_mul(q_src, quat_inv(q_tgt))
    return corrections


def part_transform(l_src: Quat, conv: Quat, d_part: Quat, d_parent: Quat) -> Quat:
    """One Motor6D transform. See the module docstring for the formula."""
    l_rbx = conjugate(l_src, conv)
    return quat_mul(quat_inv(d_parent), quat_mul(l_rbx, d_part))


def retarget_pose(rotations: Dict[str, Quat], mapping: Mapping, conv: Quat, corrections: Dict[str, Quat]) -> Dict[str, Quat]:
    local = {part: QUAT_IDENTITY for part in PART_ORDER}
    for part, spec in mapping.parts.items():
        if not spec.joints:
            continue
        chain = [rotations.get(name, QUAT_IDENTITY) for name in spec.joints]
        parent = POSE_PARENT[part]
        local[part] = part_transform(
            compose_chain(chain),
            conv,
            corrections[part],
            corrections[parent if parent is not None else "HumanoidRootPart"],
        )
    return local


def subtract_rest_pose(frames: List[R15Frame], index: int) -> None:
    """Make ``frames[index]`` the R15 rest and express the clip as deltas.

    Rotations become ``inv(T_rest) * T``. Root translation is left alone:
    the caller already subtracted the origin frame's position.
    """
    if not frames:
        return
    index = max(0, min(index, len(frames) - 1))
    # Copy first. The rest pose is one of the frames, and the loop writes
    # those quaternions in place.
    rest = dict(frames[index].local)
    for frame in frames:
        for part, quat in list(frame.local.items()):
            frame.local[part] = quat_mul(quat_inv(rest[part]), quat)


def pose_angle_warnings(frames: Sequence[R15Frame], limit_degrees: float = 150.0) -> List[str]:
    """Warn when a joint spends the clip a long way from the R15 rest.

    A median over ``limit_degrees`` usually means the forward axis is wrong
    or the file's zero rotation is already the standing pose and
    ``--rest-frame 0`` is appropriate.
    """
    if not frames:
        return []
    warnings = []
    watched = (
        "LeftUpperArm",
        "RightUpperArm",
        "LeftUpperLeg",
        "RightUpperLeg",
        "UpperTorso",
        "Head",
    )
    for part in watched:
        angles = [math.degrees(quat_between_angle(QUAT_IDENTITY, frame.local[part])) for frame in frames]
        angles.sort()
        median = angles[len(angles) // 2]
        if median > limit_degrees:
            warnings.append(
                f"{part} stays about {median:.0f}° away from the R15 rest. "
                "Check --forward-axis. If the BVH offsets are already a standing "
                "pose, try --rest-frame 0."
            )
    return warnings
