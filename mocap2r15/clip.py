"""Per-frame source poses, independent of the raw BVH channel layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from mocap2r15.bvh import BVH
from mocap2r15.math3d import Quat, Vec3


@dataclass
class SourceFrame:
    time: float
    rotations: Dict[str, Quat]
    root_pos: Vec3


def frames_from_bvh(bvh: BVH) -> List[SourceFrame]:
    frames: List[SourceFrame] = []
    for index, raw in enumerate(bvh.frames):
        rotations = {name: bvh.rotation_quat(name, raw) for name in bvh.order}
        frames.append(
            SourceFrame(
                time=index * bvh.frame_time,
                rotations=rotations,
                root_pos=bvh.translation(bvh.root, raw),
            )
        )
    return frames
