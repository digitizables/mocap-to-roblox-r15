"""Write a Roblox KeyframeSequence as an ``.rbxmx`` model file.

The hierarchy matches an R15 rig: one Pose per part, nested the way
Motor6Ds (and the newer AnimationConstraints) are parented. Studio applies
the pose named after the part to the joint that drives that part. Insert the
file with Explorer → Insert from File, then move the KeyframeSequence under
the rig's ``AnimSaves`` model and open it in the Animation Editor.

CFrame rotation is the standard Roblox 3x3 (rows R00..R22) acting on column
vectors. Right is +X, up is +Y, and back is +Z because characters face -Z.
"""

from __future__ import annotations

from typing import List, Sequence

from mocap2r15.math3d import Quat, Vec3, quat_to_mat
from mocap2r15.r15 import ANIMATION_PRIORITY, PART_ORDER, children_of
from mocap2r15.retarget import R15Frame


def write_rbxmx(
    frames: Sequence[R15Frame],
    path: str,
    name: str,
    loop: bool,
    priority: str,
    root_bone: str = "humanoid",
) -> None:
    if priority not in ANIMATION_PRIORITY:
        known = ", ".join(ANIMATION_PRIORITY)
        raise ValueError(f"Unknown animation priority {priority!r}. Use one of: {known}")
    if root_bone not in ("humanoid", "lower-torso"):
        raise ValueError("root_bone must be 'humanoid' or 'lower-torso'")
    text = render_rbxmx(frames, name=name, loop=loop, priority=priority, root_bone=root_bone)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def render_rbxmx(
    frames: Sequence[R15Frame],
    name: str,
    loop: bool,
    priority: str,
    root_bone: str = "humanoid",
) -> str:
    if priority not in ANIMATION_PRIORITY:
        known = ", ".join(ANIMATION_PRIORITY)
        raise ValueError(f"Unknown animation priority {priority!r}. Use one of: {known}")
    if root_bone not in ("humanoid", "lower-torso"):
        raise ValueError("root_bone must be 'humanoid' or 'lower-torso'")
    priority_id = ANIMATION_PRIORITY[priority]
    lines: List[str] = ['<roblox version="4">']
    refs = _Refs()
    lines.append(f'\t<Item class="KeyframeSequence" referent="{refs.next()}">')
    lines.append("\t\t<Properties>")
    lines.append(f"\t\t\t<string name=\"Name\">{_xml(name)}</string>")
    lines.append(f"\t\t\t<bool name=\"Loop\">{'true' if loop else 'false'}</bool>")
    lines.append(f"\t\t\t<token name=\"Priority\">{priority_id}</token>")
    lines.append("\t\t</Properties>")
    for index, frame in enumerate(frames):
        _write_keyframe(lines, refs, frame, index, root_bone)
    lines.append("\t</Item>")
    lines.append("</roblox>")
    lines.append("")
    return "\n".join(lines)


class _Refs:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> str:
        ref = f"RBX{self._n}"
        self._n += 1
        return ref


def _write_keyframe(lines: List[str], refs: _Refs, frame: R15Frame, index: int, root_bone: str) -> None:
    lines.append(f'\t\t<Item class="Keyframe" referent="{refs.next()}">')
    lines.append("\t\t\t<Properties>")
    lines.append(f"\t\t\t\t<string name=\"Name\">KF{index}</string>")
    lines.append(f"\t\t\t\t<float name=\"Time\">{_num(frame.time)}</float>")
    lines.append("\t\t\t</Properties>")
    _write_pose(lines, refs, frame, "HumanoidRootPart", 3, root_bone)
    lines.append("\t\t</Item>")


def _write_pose(
    lines: List[str],
    refs: _Refs,
    frame: R15Frame,
    part: str,
    depth: int,
    root_bone: str,
) -> None:
    pad = "\t" * depth
    position = _position_for(part, frame.root_pos, root_bone)
    quat = frame.local.get(part, (1.0, 0.0, 0.0, 0.0))
    lines.append(f'{pad}<Item class="Pose" referent="{refs.next()}">')
    lines.append(f"{pad}\t<Properties>")
    lines.append(f'{pad}\t\t<string name="Name">{part}</string>')
    _write_cframe(lines, pad + "\t\t", position, quat)
    lines.append(f'{pad}\t\t<float name="Weight">1</float>')
    # EasingStyle Linear = 0, EasingDirection In = 0. Mocap is linearly sampled.
    lines.append(f'{pad}\t\t<token name="EasingStyle">0</token>')
    lines.append(f'{pad}\t\t<token name="EasingDirection">0</token>')
    lines.append(f"{pad}\t</Properties>")
    for child in children_of(part):
        if child in PART_ORDER:
            _write_pose(lines, refs, frame, child, depth + 1, root_bone)
    lines.append(f"{pad}</Item>")


def _position_for(part: str, root_pos: Vec3, root_bone: str) -> Vec3:
    if root_bone == "lower-torso" and part == "LowerTorso":
        return root_pos
    if root_bone == "humanoid" and part == "HumanoidRootPart":
        return root_pos
    return (0.0, 0.0, 0.0)


def _write_cframe(lines: List[str], pad: str, position: Vec3, quat: Quat) -> None:
    matrix = quat_to_mat(quat)
    lines.append(f'{pad}<CoordinateFrame name="CFrame">')
    lines.append(f"{pad}\t<X>{_num(position[0])}</X>")
    lines.append(f"{pad}\t<Y>{_num(position[1])}</Y>")
    lines.append(f"{pad}\t<Z>{_num(position[2])}</Z>")
    for row in range(3):
        for col in range(3):
            lines.append(f"{pad}\t<R{row}{col}>{_num(matrix[row][col])}</R{row}{col}>")
    lines.append(f"{pad}</CoordinateFrame>")


def _num(value: float) -> str:
    if abs(value) < 5e-8:
        value = 0.0
    text = f"{value:.7f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
