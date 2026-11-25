"""A one-second CMU-style walk, generated with no outside data.

The skeleton uses CMU joint names and faces +Z in a T-pose bind. Frame
rotations hang the arms and bend the knees, so the clip is a standing walk
rather than an unposed T-pose. Root X sits at 40 so a converter that forgets
to subtract the first frame will throw the character sideways. Root Z travels
one meter. The last frame repeats the first pose; its translation does not,
so an in-place loop can drop it and a root-motion export keeps the stride.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

FRAME_COUNT = 31
CYCLES = 30
FRAME_TIME = 1.0 / 30.0
TRAVEL = 1.0  # meters of +Z travel from the first frame to the last

Offset = Tuple[float, float, float]


def _joint(name: str, offset: Offset, children: Sequence[dict] = (), end: Offset = None) -> dict:
    return {"name": name, "offset": offset, "children": list(children), "end": end}


def _arm(side: str, sign: float) -> dict:
    return _joint(
        f"{side}Shoulder",
        (sign * 0.06, 0.08, 0.0),
        [
            _joint(
                f"{side}Arm",
                (sign * 0.12, 0.0, 0.0),
                [
                    _joint(
                        f"{side}ForeArm",
                        (sign * 0.28, 0.0, 0.0),
                        [_joint(f"{side}Hand", (sign * 0.25, 0.0, 0.0), end=(sign * 0.08, 0.0, 0.0))],
                    )
                ],
            )
        ],
    )


def _leg(side: str, hip_name: str, sign: float) -> dict:
    up = "LeftUpLeg" if side == "Left" else "RightUpLeg"
    return _joint(
        hip_name,
        (sign * 0.10, -0.05, 0.0),
        [
            _joint(
                up,
                (0.0, -0.02, 0.0),
                [
                    _joint(
                        f"{side}Leg",
                        (0.0, -0.45, 0.0),
                        [
                            _joint(
                                f"{side}Foot",
                                (0.0, -0.44, 0.0),
                                [_joint(f"{side}ToeBase", (0.0, -0.05, 0.16), end=(0.0, 0.0, 0.06))],
                            )
                        ],
                    )
                ],
            )
        ],
    )


def skeleton() -> dict:
    """Bind pose in meters. Height is about 1.75, arms out, toes toward +Z."""
    chest = [
        _joint(
            "Neck",
            (0.0, 0.12, 0.0),
            [_joint("Neck1", (0.0, 0.04, 0.0), [_joint("Head", (0.0, 0.07, 0.0), end=(0.0, 0.16, 0.0))])],
        ),
        _arm("Left", 1.0),
        _arm("Right", -1.0),
    ]
    return _joint(
        "Hips",
        (0.0, 0.0, 0.0),
        [
            _joint(
                "LowerBack",
                (0.0, 0.10, 0.0),
                [_joint("Spine", (0.0, 0.12, 0.0), [_joint("Spine1", (0.0, 0.13, 0.0), chest)])],
            ),
            _leg("Left", "LHipJoint", 1.0),
            _leg("Right", "RHipJoint", -1.0),
        ],
    )


def frame_channels(index: int) -> Dict[str, Sequence[float]]:
    """One frame. Root values are position then ZYX degrees. Joints are ZYX degrees."""
    phase = 2.0 * math.pi * (index / CYCLES)
    swing = math.sin(phase)
    bob = math.sin(2.0 * phase)
    # Thighs point down, so the X channel swings them: negative X is +Z.
    # Arms point along X, so the Y channel swings them. Positive Y sends
    # the left arm toward -Z and the right arm toward +Z.
    return {
        "Hips": (40.0, 1.02 + 0.025 * bob, TRAVEL * (index / CYCLES), 3.0 * swing, 6.0 * swing, 2.0 * bob),
        "LowerBack": (1.5 * swing, 0.0, 1.0 * bob),
        "Spine": (4.0 * swing, 3.0 * swing, 2.0 * bob),
        "LeftUpLeg": (0.0, 0.0, -28.0 * swing),
        "LeftLeg": (0.0, 0.0, 22.0 + 18.0 * max(0.0, -swing)),
        "LeftFoot": (0.0, 0.0, 8.0 * swing),
        "RightUpLeg": (0.0, 0.0, 28.0 * swing),
        "RightLeg": (0.0, 0.0, 22.0 + 18.0 * max(0.0, swing)),
        "RightFoot": (0.0, 0.0, -8.0 * swing),
        "LeftArm": (-90.0, 20.0 * swing, 0.0),
        "RightArm": (90.0, 20.0 * swing, 0.0),
    }


def build_bvh_text() -> str:
    """The full BVH, including the one-second walk."""
    lines = ["HIERARCHY"]
    order: List[Tuple[str, int]] = []

    def emit(node: dict, indent: int, is_root: bool) -> None:
        pad = "  " * indent
        lines.append(f"{pad}{'ROOT' if is_root else 'JOINT'} {node['name']}")
        lines.append(f"{pad}{{")
        ox, oy, oz = node["offset"]
        lines.append(f"{pad}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        if is_root:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
            order.append((node["name"], 6))
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Yrotation Xrotation")
            order.append((node["name"], 3))
        for child in node["children"]:
            emit(child, indent + 1, False)
        if node["end"] is not None:
            ex, ey, ez = node["end"]
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET {ex:.6f} {ey:.6f} {ez:.6f}")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")

    emit(skeleton(), 0, True)
    lines.append("MOTION")
    lines.append(f"Frames: {FRAME_COUNT}")
    lines.append(f"Frame Time: {FRAME_TIME:.10f}")
    for index in range(FRAME_COUNT):
        channels = frame_channels(index)
        values: List[float] = []
        for name, count in order:
            given = channels.get(name)
            if given is None:
                values.extend([0.0] * count)
            else:
                values.extend(given)
        lines.append(" ".join(f"{value:.6f}" for value in values))
    return "\n".join(lines) + "\n"


def write_bvh(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_bvh_text())
