"""Plant the feet during detected ground contact.

Contact is a run of frames where the ankle is near the low end of its height
range and is not moving horizontally very fast. Inside a run the ankle is
pulled toward the median contact position (with a short blend at each edge)
and a two-bone IK rebuilds the thigh and shin. The original twist around
each bone is kept, and the foot keeps its world orientation so the sole does
not roll while the leg reaches.

This is a cleanup, not a full mocap solve. Corrections that would need more
than about a third of the leg length are clamped. Turn it on with
``--foot-lock``; the default path leaves the source rotations alone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from mocap2r15.bvh import BVH
from mocap2r15.clip import SourceFrame
from mocap2r15.math3d import (
    QUAT_IDENTITY,
    Quat,
    Vec3,
    median,
    percentile,
    quat_between,
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_rotate,
    smoothstep,
    v_add,
    v_dist,
    v_len,
    v_lerp,
    v_mul,
    v_normalize,
    v_sub,
)


def leg_triples(chains: Dict[str, Sequence[str]]) -> List[Tuple[str, str, str]]:
    """``(thigh, shin, foot)`` joint names from resolved R15 chains."""
    triples = []
    for side in ("Left", "Right"):
        upper = list(chains.get(f"{side}UpperLeg", []))
        lower = list(chains.get(f"{side}LowerLeg", []))
        foot = list(chains.get(f"{side}Foot", []))
        if upper and lower and foot:
            triples.append((upper[-1], lower[-1], foot[-1]))
    return triples


def apply_foot_lock(
    bvh: BVH,
    frames: Sequence[SourceFrame],
    triples: Sequence[Tuple[str, str, str]],
    blend_frames: int = 2,
) -> List[str]:
    """Rewrite thigh/shin/foot rotations in ``frames``. Returns warnings."""
    warnings: List[str] = []
    if len(frames) < 2 or not triples:
        if not triples:
            warnings.append("foot lock found no mapped legs; skipped")
        return warnings
    for thigh, shin, foot in triples:
        _lock_leg(bvh, frames, thigh, shin, foot, blend_frames, warnings)
    return warnings


def two_bone_ik(hip: Vec3, target: Vec3, length_a: float, length_b: float, pole: Vec3) -> Tuple[Vec3, Vec3]:
    """Return ``(knee, ankle)`` reaching toward ``target`` with the knee near ``pole``.

    The ankle is pulled in along the hip-to-target line when the target is
    outside the reachable range, so the bone lengths stay exact.
    """
    direction = v_sub(target, hip)
    dist = v_len(direction)
    if dist < 1e-8:
        direction = (0.0, -1.0, 0.0)
        dist = 1.0
    else:
        direction = v_mul(direction, 1.0 / dist)
    reach = max(length_a + length_b - 1e-4, 1e-4)
    minimum = abs(length_a - length_b) + 1e-4
    clamped = min(max(dist, minimum), reach)
    target_on_ray = v_add(hip, v_mul(direction, clamped))
    cos_a = (length_a * length_a + clamped * clamped - length_b * length_b) / (2.0 * length_a * clamped)
    cos_a = max(-1.0, min(1.0, cos_a))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    pole_v = v_sub(pole, hip)
    pole_flat = v_sub(pole_v, v_mul(direction, _dot(pole_v, direction)))
    if v_len(pole_flat) < 1e-6:
        # Straight source pose: bend the knee toward BVH forward (+Z), which
        # is "back of the calf swings back" once the thigh points down.
        pole_flat = v_sub((0.0, 0.0, 1.0), v_mul(direction, direction[2]))
        if v_len(pole_flat) < 1e-6:
            pole_flat = v_sub((1.0, 0.0, 0.0), v_mul(direction, direction[0]))
    pole_flat = v_normalize(pole_flat)
    knee = v_add(hip, v_add(v_mul(direction, length_a * cos_a), v_mul(pole_flat, length_a * sin_a)))
    return knee, target_on_ray


def _lock_leg(
    bvh: BVH,
    frames: Sequence[SourceFrame],
    thigh: str,
    shin: str,
    foot: str,
    blend_frames: int,
    warnings: List[str],
) -> None:
    length_a, length_b = _bone_lengths(bvh, thigh, shin, foot)
    if length_a < 1e-5 or length_b < 1e-5:
        warnings.append(f"foot lock skipped {foot}: zero bone length")
        return
    samples = [_fk(bvh, frame) for frame in frames]
    ankles = [sample[0][foot] for sample in samples]
    ground = percentile([ankle[1] for ankle in ankles], 15.0)
    leg = length_a + length_b
    height_limit = ground + 0.12 * leg
    dt = max(frames[1].time - frames[0].time, 1e-4)
    speed_limit = 0.65 * leg  # studs-or-file-units per second; scale cancels

    contact = []
    for index, ankle in enumerate(ankles):
        if index == 0:
            speed = 0.0
        else:
            step = v_sub(ankle, ankles[index - 1])
            speed = math.sqrt(step[0] * step[0] + step[2] * step[2]) / dt
        contact.append(ankle[1] <= height_limit and speed <= speed_limit)

    runs = _runs(contact)
    minimum = max(2, int(round(0.04 / dt)))
    runs = [run for run in runs if run[1] - run[0] >= minimum]
    if not runs:
        warnings.append(f"foot lock found no contact on {foot}")
        return

    weights = [0.0] * len(frames)
    targets: List[Vec3] = [(0.0, 0.0, 0.0)] * len(frames)
    blend = max(0, blend_frames)
    for start, stop in runs:
        window = ankles[start:stop]
        plant = (
            median(p[0] for p in window),
            median(p[1] for p in window),
            median(p[2] for p in window),
        )
        for index in range(start, stop):
            edge = min(index - start, stop - 1 - index)
            if blend > 0 and edge < blend:
                weights[index] = smoothstep((edge + 1) / float(blend + 1))
            else:
                weights[index] = 1.0
            targets[index] = plant

    bind_thigh = _bind_dir(bvh, thigh, shin)
    bind_shin = _bind_dir(bvh, shin, foot)
    for index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        positions, orientations = samples[index]
        hip = positions[thigh]
        original_ankle = positions[foot]
        pinned = v_lerp(original_ankle, targets[index], weight)
        if v_dist(pinned, original_ankle) > 0.35 * leg:
            pinned = v_lerp(original_ankle, pinned, (0.35 * leg) / v_dist(pinned, original_ankle))
        knee_guess = positions[shin]
        knee, ankle = two_bone_ik(hip, pinned, length_a, length_b, knee_guess)
        parent_name = bvh.joints[thigh].parent
        parent_rot = orientations[parent_name] if parent_name else QUAT_IDENTITY
        desired_thigh = quat_rotate(quat_inv(parent_rot), v_sub(knee, hip))
        frames[index].rotations[thigh] = _aim(bind_thigh, desired_thigh, frames[index].rotations[thigh])
        hip_world = quat_mul(parent_rot, frames[index].rotations[thigh])
        desired_shin = quat_rotate(quat_inv(hip_world), v_sub(ankle, knee))
        frames[index].rotations[shin] = _aim(bind_shin, desired_shin, frames[index].rotations[shin])
        # Keep the sole's world orientation from the source frame.
        foot_world = orientations[foot]
        shin_world = quat_mul(hip_world, frames[index].rotations[shin])
        frames[index].rotations[foot] = quat_mul(quat_inv(shin_world), foot_world)


def _bone_lengths(bvh: BVH, thigh: str, shin: str, foot: str) -> Tuple[float, float]:
    bind = bvh.bind_positions()
    return v_dist(bind[thigh], bind[shin]), v_dist(bind[shin], bind[foot])


def _bind_dir(bvh: BVH, parent: str, child: str) -> Vec3:
    bind = bvh.bind_positions()
    return v_normalize(v_sub(bind[child], bind[parent]), (0.0, -1.0, 0.0))


def _aim(bind_dir: Vec3, desired_dir: Vec3, original: Quat) -> Quat:
    if v_len(desired_dir) < 1e-8:
        return original
    swing = quat_between(bind_dir, desired_dir)
    twist = _twist(original, bind_dir)
    return quat_mul(swing, twist)


def _twist(quat: Quat, axis: Vec3) -> Quat:
    axis_n = v_normalize(axis)
    w, x, y, z = quat_normalize(quat)
    proj = x * axis_n[0] + y * axis_n[1] + z * axis_n[2]
    return quat_normalize((w, axis_n[0] * proj, axis_n[1] * proj, axis_n[2] * proj))


def _fk(bvh: BVH, frame: SourceFrame) -> Tuple[Dict[str, Vec3], Dict[str, Quat]]:
    positions: Dict[str, Vec3] = {}
    orientations: Dict[str, Quat] = {}

    def walk(name: str, parent_pos: Vec3, parent_rot: Quat) -> None:
        joint = bvh.joints[name]
        local_pos = frame.root_pos if name == bvh.root else joint.offset
        local_rot = frame.rotations.get(name, QUAT_IDENTITY)
        world_pos = v_add(parent_pos, quat_rotate(parent_rot, local_pos))
        world_rot = quat_mul(parent_rot, local_rot)
        positions[name] = world_pos
        orientations[name] = world_rot
        for child in joint.children:
            walk(child, world_pos, world_rot)

    walk(bvh.root, (0.0, 0.0, 0.0), QUAT_IDENTITY)
    return positions, orientations


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
