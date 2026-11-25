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
    Quat,
    Vec3,
    quat_between_angle,
    quat_slerp,
    smoothstep,
    v_dist,
    v_lerp,
)
from mocap2r15.retarget import R15Frame


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
