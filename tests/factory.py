"""Build small BVH files for tests. Channel order is ZYX, root has translation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

Offset = Tuple[float, float, float]
Frame = Dict[str, Sequence[float]]


def joint(
    name: str,
    offset: Offset,
    children: Sequence[dict] = (),
    end: Optional[Offset] = None,
) -> dict:
    return {"name": name, "offset": offset, "children": list(children), "end": end}


def render_bvh(root: dict, frames: Sequence[Frame], frame_time: float = 0.04) -> str:
    """Serialize ``root`` and one dict of channel values per frame.

    Root values are ``(x, y, z, zrot, yrot, xrot)``. Other joints are
    ``(zrot, yrot, xrot)``. Omitted joints are zero.
    """
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
        if node.get("end") is not None:
            ex, ey, ez = node["end"]
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET {ex:.6f} {ey:.6f} {ez:.6f}")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")

    emit(root, 0, True)
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {frame_time}")
    for frame in frames:
        values: List[float] = []
        for name, count in order:
            given = frame.get(name)
            if given is None:
                values.extend([0.0] * count)
            elif len(given) != count:
                raise ValueError(f"{name} expects {count} channels, got {len(given)}")
            else:
                values.extend(given)
        lines.append(" ".join(f"{value:.6f}" for value in values))
    return "\n".join(lines) + "\n"


def cmu_skeleton() -> dict:
    """CMU-like names. The clavicle offset is longer than the neck on purpose."""

    def arm(side: str, sign: float) -> dict:
        return joint(
            f"{side}Shoulder",
            (sign * 5.0, 0.0, 0.0),
            [
                joint(
                    f"{side}Arm",
                    (sign * 1.0, 0.0, 0.0),
                    [
                        joint(
                            f"{side}ForeArm",
                            (sign * 4.0, 0.0, 0.0),
                            [joint(f"{side}Hand", (sign * 3.0, 0.0, 0.0), end=(sign * 1.0, 0.0, 0.0))],
                        )
                    ],
                )
            ],
        )

    def leg(side: str, joint_prefix: str, sign: float) -> dict:
        up = "LeftUpLeg" if side == "Left" else "RightUpLeg"
        return joint(
            f"{joint_prefix}HipJoint",
            (sign * 1.2, -0.2, 0.0),
            [
                joint(
                    up,
                    (0.0, -0.2, 0.0),
                    [
                        joint(
                            f"{side}Leg",
                            (0.0, -6.0, 0.0),
                            [
                                joint(
                                    f"{side}Foot",
                                    (0.0, -6.0, 0.0),
                                    [joint(f"{side}ToeBase", (0.0, 0.0, 2.0), end=(0.0, 0.0, 1.0))],
                                )
                            ],
                        )
                    ],
                )
            ],
        )

    spine1_children = [
        joint(
            "Neck",
            (0.0, 0.0, 0.0),
            [joint("Neck1", (0.0, 3.0, 0.0), [joint("Head", (0.0, 2.0, 0.0), end=(0.0, 2.0, 0.0))])],
        ),
        arm("Left", 1.0),
        arm("Right", -1.0),
    ]
    return joint(
        "Hips",
        (0.0, 0.0, 0.0),
        [
            joint(
                "LowerBack",
                (0.0, 2.0, 0.0),
                [joint("Spine", (0.0, 2.0, 0.0), [joint("Spine1", (0.0, 2.0, 0.0), spine1_children)])],
            ),
            leg("Left", "L", 1.0),
            leg("Right", "R", -1.0),
        ],
    )
