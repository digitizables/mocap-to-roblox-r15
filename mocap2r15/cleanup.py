"""Resample, thin, and seam-blend clips.

Keyframe reduction is Ramer-Douglas-Peucker on the whole R15 pose: a sample
stays if any joint's rotation, or the root translation, cannot be recovered
by slerp/lerp of its neighbors within the given epsilon. Endpoints always
stay. Static holds collapse to two keys.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from mocap2r15.clip import SourceFrame
from mocap2r15.math3d import (
    QUAT_IDENTITY,
    Quat,
    quat_between_angle,
    quat_inv,
    quat_mirror_x,
    quat_mul,
    quat_slerp,
    quat_strip_twist,
    smoothstep,
    v_dist,
    v_lerp,
    yaw_about_y,
)
from mocap2r15.retarget import R15Frame

# Left limb, then the right limb that should copy it. Both sides of a default
# R15 have the same Motor6D axis directions, so one X-mirror fits every pair.
LIMB_MIRROR = (
    ("LeftUpperArm", "RightUpperArm"),
    ("LeftLowerArm", "RightLowerArm"),
    ("LeftHand", "RightHand"),
    ("LeftUpperLeg", "RightUpperLeg"),
    ("LeftLowerLeg", "RightLowerLeg"),
    ("LeftFoot", "RightFoot"),
)


def trim_frames(frames: Sequence[SourceFrame], start: float | None, end: float | None) -> List[SourceFrame]:
    kept = []
    for frame in frames:
        if start is not None and frame.time < start - 1e-8:
            continue
        if end is not None and frame.time > end + 1e-8:
            continue
        kept.append(frame)
    if not kept:
        return []
    origin = kept[0].time
    return [
        SourceFrame(time=frame.time - origin, rotations=dict(frame.rotations), root_pos=frame.root_pos)
        for frame in kept
    ]


def resample_frames(frames: Sequence[SourceFrame], fps: float) -> List[SourceFrame]:
    if fps <= 0 or len(frames) < 2:
        return [SourceFrame(f.time, dict(f.rotations), f.root_pos) for f in frames]
    duration = frames[-1].time - frames[0].time
    if duration <= 1e-8:
        return [SourceFrame(0.0, dict(frames[0].rotations), frames[0].root_pos)]
    steps = max(1, int(round(duration * fps)))
    # ``steps`` intervals cover the clip, so the last sample lands on the end
    # and a 1 second 30 fps clip stays 31 samples (0, 1/30, ..., 1).
    times = [frames[0].time + i * (duration / steps) for i in range(steps + 1)]
    names = list(frames[0].rotations.keys())
    output: List[SourceFrame] = []
    cursor = 0
    for time in times:
        while cursor + 1 < len(frames) and frames[cursor + 1].time < time - 1e-8:
            cursor += 1
        a = frames[cursor]
        b = frames[min(cursor + 1, len(frames) - 1)]
        span = b.time - a.time
        u = 0.0 if span <= 1e-8 else (time - a.time) / span
        u = 0.0 if u < 0 else 1.0 if u > 1 else u
        rotations: Dict[str, Quat] = {}
        for name in names:
            rotations[name] = quat_slerp(a.rotations[name], b.rotations.get(name, a.rotations[name]), u)
        output.append(SourceFrame(time - frames[0].time, rotations, v_lerp(a.root_pos, b.root_pos, u)))
    return output


def upright_torso(frames: Sequence[R15Frame]) -> None:
    """Stand the torso up without throwing away the limb swing.

    ``LowerTorso`` keeps the yaw about +Y, so the character still faces the
    same way, and loses pitch and roll. ``UpperTorso`` becomes identity.
    The head keeps the direction it was looking and loses roll around that
    look. Upper arms and upper legs are re-expressed on the upright chest
    and pelvis so their world directions stay put. Forearms, hands, shins,
    and feet hang off those and do not need their own fix.
    """
    for frame in frames:
        local = frame.local
        hrp = local.get("HumanoidRootPart", QUAT_IDENTITY)
        lower = local["LowerTorso"]
        upper = local["UpperTorso"]
        world_lower = quat_mul(hrp, lower)
        world_upper = quat_mul(world_lower, upper)
        world_head = quat_mul(world_upper, local["Head"])
        yaw, _rest = yaw_about_y(world_lower)
        local["LowerTorso"] = quat_mul(quat_inv(hrp), yaw)
        local["UpperTorso"] = QUAT_IDENTITY
        local["Head"] = quat_mul(quat_inv(yaw), quat_strip_twist(world_head))
        for part in ("LeftUpperArm", "RightUpperArm"):
            world_arm = quat_mul(world_upper, local[part])
            local[part] = quat_mul(quat_inv(yaw), world_arm)
        for part in ("LeftUpperLeg", "RightUpperLeg"):
            world_leg = quat_mul(world_lower, local[part])
            local[part] = quat_mul(quat_inv(yaw), world_leg)


def _sample_local(frames: Sequence[R15Frame], part: str, time: float, period: float) -> Quat:
    """Slerp ``part`` at ``time``. Times outside one cycle wrap.

    A time that already sits on the final key is kept. Wrapping that key
    onto the first pose would ignore it whenever the clip is not closed.
    """
    if period <= 1e-8 or len(frames) == 1:
        return frames[0].local[part]
    start = frames[0].time
    end = start + period
    if time < start - 1e-8 or time > end + 1e-8:
        time = start + (time - start) % period
    if time >= end - 1e-8 and abs(frames[-1].time - end) <= 1e-5:
        return frames[-1].local[part]
    if time >= end - 1e-8:
        time = start
    if time > frames[-1].time + 1e-8:
        span = end - frames[-1].time
        u = 0.0 if span <= 1e-8 else (time - frames[-1].time) / span
        return quat_slerp(frames[-1].local[part], frames[0].local[part], u)
    hi = 1
    while hi < len(frames) - 1 and frames[hi].time < time - 1e-10:
        hi += 1
    lo = hi - 1
    span = frames[hi].time - frames[lo].time
    u = 0.0 if span <= 1e-8 else (time - frames[lo].time) / span
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    return quat_slerp(frames[lo].local[part], frames[hi].local[part], u)


def mirror_left_to_right(frames: Sequence[R15Frame], phase: float = 0.5) -> None:
    """Copy each left limb onto its right partner, reflected across X.

    ``phase`` is a fraction of the clip. ``0.5`` reads the left limb half a
    cycle later, so the two sides stay out of phase and a walk still walks.
    ``0`` copies the same frame, which steps both feet together. Call this
    before ``drop_matching_end``: the last key's time is then one full cycle.
    The left side is not modified.
    """
    if not frames:
        return
    period = frames[-1].time - frames[0].time if len(frames) > 1 else 0.0
    delay = phase * period
    copied: List[dict] = []
    for frame in frames:
        pose = {}
        when = frame.time + delay
        for left, right in LIMB_MIRROR:
            if left not in frame.local:
                continue
            pose[right] = quat_mirror_x(_sample_local(frames, left, when, period))
        copied.append(pose)
    for frame, pose in zip(frames, copied):
        frame.local.update(pose)


def loop_blend(frames: List[R15Frame], count: int, blend_translation: bool) -> None:
    """Pull the tail of the clip toward the first pose so a loop does not pop.

    ``count`` is the number of trailing frames that participate. The last of
    them matches the first pose. Translation is blended only for in-place
    clips; root-motion XZ is left so the trajectory is not rewound.
    """
    if count <= 0 or len(frames) < 3:
        return
    count = min(count, len(frames) - 1)
    first = frames[0]
    for step in range(1, count + 1):
        index = len(frames) - count + step - 1
        t = smoothstep(step / float(count))
        frame = frames[index]
        for part, quat in list(frame.local.items()):
            frame.local[part] = quat_slerp(quat, first.local[part], t)
        if blend_translation:
            frame.root_pos = v_lerp(frame.root_pos, first.root_pos, t)
        else:
            frame.root_pos = (
                frame.root_pos[0],
                frame.root_pos[1] + (first.root_pos[1] - frame.root_pos[1]) * t,
                frame.root_pos[2],
            )


def drop_matching_end(frames: List[R15Frame], angle_degrees: float = 0.75, position: float = 0.02) -> List[R15Frame]:
    """Drop a final key that repeats the first. Looping that pair hitches."""
    if len(frames) < 3:
        return frames
    first, last = frames[0], frames[-1]
    for part in first.local:
        if math.degrees(quat_between_angle(first.local[part], last.local[part])) > angle_degrees:
            return frames
    if v_dist(first.root_pos, last.root_pos) > position:
        return frames
    return frames[:-1]


def reduce_frames(frames: Sequence[R15Frame], degrees: float, position_epsilon: float) -> List[R15Frame]:
    if degrees <= 0 or len(frames) <= 2:
        return list(frames)
    rot_eps = math.radians(degrees)
    pos_eps = max(position_epsilon, 1e-6)

    def deviation(index: int, left: int, right: int) -> tuple:
        t0 = frames[left].time
        t1 = frames[right].time
        span = t1 - t0
        u = 0.0 if span <= 1e-8 else (frames[index].time - t0) / span
        ang = 0.0
        for part, quat in frames[index].local.items():
            ref = quat_slerp(frames[left].local[part], frames[right].local[part], u)
            ang = max(ang, quat_between_angle(quat, ref))
        pref = v_lerp(frames[left].root_pos, frames[right].root_pos, u)
        return ang, v_dist(frames[index].root_pos, pref)

    keep = {0, len(frames) - 1}

    def mark(left: int, right: int) -> None:
        if right <= left + 1:
            return
        worst = left + 1
        worst_score = -1.0
        worst_ang = 0.0
        worst_pos = 0.0
        for index in range(left + 1, right):
            ang, pos = deviation(index, left, right)
            score = ang / rot_eps + pos / pos_eps
            if score > worst_score:
                worst = index
                worst_score = score
                worst_ang = ang
                worst_pos = pos
        if worst_ang > rot_eps or worst_pos > pos_eps:
            keep.add(worst)
            mark(left, worst)
            mark(worst, right)

    mark(0, len(frames) - 1)
    return [frames[index] for index in sorted(keep)]
