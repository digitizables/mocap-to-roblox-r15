"""BVH reader.

The parser accepts the layouts exporters actually write: braces on their own
line or on the joint line, ``Frames:`` with or without a space, Windows and
Unix newlines, and a per-joint channel order. End sites are kept because the
retargeter uses them as bone directions when a joint has no child.

Motion rows may wrap. All numeric tokens after ``Frame Time`` are read and
split by the channel count, so a short or long file fails with an explicit
count instead of a silent shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from mocap2r15.math3d import (
    QUAT_IDENTITY,
    Quat,
    Vec3,
    euler_to_quat,
    quat_mul,
    quat_rotate,
    v_add,
    v_len,
)


class BVHError(ValueError):
    """The file is not a usable BVH clip."""


@dataclass
class Joint:
    name: str
    parent: Optional[str]
    offset: Vec3
    channels: List[str]
    channel_start: int
    children: List[str] = field(default_factory=list)
    end_offset: Optional[Vec3] = None
    depth: int = 0

    @property
    def rotation_order(self) -> str:
        letters = []
        for channel in self.channels:
            lower = channel.lower()
            if lower.endswith("rotation") and lower[0] in "xyz":
                letters.append(lower[0].upper())
        return "".join(letters)

    @property
    def has_position(self) -> bool:
        return any(ch.lower().endswith("position") for ch in self.channels)


@dataclass
class BVH:
    """One clip. Joint names are the original strings from the file."""

    joints: Dict[str, Joint]
    order: List[str]
    frames: List[List[float]]
    frame_time: float
    root: str
    path: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        if self.frame_time <= 0:
            return 0.0
        return 1.0 / self.frame_time

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return self.frame_time * (len(self.frames) - 1)

    def find(self, *candidates: str) -> Optional[str]:
        """Resolve a joint by exact name, then by a normalized alias.

        Normalization drops a ``prefix:`` (Mixamo), underscores, and case.
        Matching is exact on that form, so ``Spine`` does not hit ``Spine1``.
        """
        if not hasattr(self, "_index"):
            self._build_index()
        index: Dict[str, str] = self._index  # type: ignore[attr-defined]
        for candidate in candidates:
            if candidate in self.joints:
                return candidate
            key = normalize_name(candidate)
            if key in index:
                return index[key]
        return None

    def _build_index(self) -> None:
        index: Dict[str, str] = {}
        for name in self.order:
            key = normalize_name(name)
            # First joint wins. A later duplicate alias is unusual and would
            # hide a real bone, so leave the earlier one in place.
            index.setdefault(key, name)
        self._index = index  # type: ignore[attr-defined]

    def channel_value(self, joint_name: str, frame: Sequence[float], channel: str) -> Optional[float]:
        joint = self.joints[joint_name]
        want = channel.lower()
        for offset, name in enumerate(joint.channels):
            if name.lower() == want:
                idx = joint.channel_start + offset
                if idx >= len(frame):
                    raise BVHError(
                        f"Frame is missing channel {channel} for {joint_name} "
                        f"(index {idx}, frame length {len(frame)})"
                    )
                return frame[idx]
        return None

    def translation(self, joint_name: str, frame: Sequence[float]) -> Vec3:
        """Local translation of a joint.

        Root position channels are absolute and replace the offset (CMU and
        most locomotion BVHs store the trajectory there, with a zero or
        duplicate offset). Other joints add position channels on top of the
        offset, which is what the rare per-joint translation channels mean.
        """
        joint = self.joints[joint_name]
        px = self.channel_value(joint_name, frame, "Xposition")
        py = self.channel_value(joint_name, frame, "Yposition")
        pz = self.channel_value(joint_name, frame, "Zposition")
        if joint_name == self.root and joint.has_position:
            return (
                0.0 if px is None else px,
                0.0 if py is None else py,
                0.0 if pz is None else pz,
            )
        return (
            joint.offset[0] + (0.0 if px is None else px),
            joint.offset[1] + (0.0 if py is None else py),
            joint.offset[2] + (0.0 if pz is None else pz),
        )

    def rotation_quat(self, joint_name: str, frame: Sequence[float]) -> Quat:
        joint = self.joints[joint_name]
        order = joint.rotation_order
        if not order:
            return QUAT_IDENTITY
        degrees = []
        for letter in order:
            value = self.channel_value(joint_name, frame, f"{letter}rotation")
            degrees.append(0.0 if value is None else value)
        return euler_to_quat(order, degrees)

    def is_ancestor(self, ancestor: str, joint: str) -> bool:
        current: Optional[str] = joint
        while current is not None:
            if current == ancestor:
                return True
            current = self.joints[current].parent
        return False

    def bind_positions(self) -> Dict[str, Vec3]:
        """Joint positions with every rotation at zero and the root at the origin.

        Offsets add in the parent frame. This is the pose the CHANNELS rotate
        away from, which is the right rest pose for a standard BVH even when
        frame 0 is already a standing walk pose.
        """
        positions: Dict[str, Vec3] = {}

        def walk(name: str, parent_pos: Vec3) -> None:
            joint = self.joints[name]
            pos = v_add(parent_pos, joint.offset)
            positions[name] = pos
            for child in joint.children:
                walk(child, pos)

        walk(self.root, (0.0, 0.0, 0.0))
        return positions

    def skeleton_height(self) -> float:
        """Vertical extent of the bind pose, from the lowest point to the highest.

        End sites are included so the top of the head and the tip of the toe
        count. The number is in file units. Callers turn it into studs.
        """
        positions = self.bind_positions()
        ys = [p[1] for p in positions.values()]
        for joint in self.joints.values():
            if joint.end_offset is not None:
                tip = v_add(positions[joint.name], joint.end_offset)
                ys.append(tip[1])
        if not ys:
            return 0.0
        return max(ys) - min(ys)

    def forward_hint(self) -> Optional[str]:
        """Guess which horizontal axis the character faces from toe offsets.

        Toes and feet in a bind T-pose or A-pose point forward. The dominant
        horizontal component across those offsets is the facing axis. Returns
        ``z``, ``-z``, ``x``, ``-x``, or None when the skeleton has no signal.
        """
        sx = 0.0
        sz = 0.0
        hits = 0
        for joint in self.joints.values():
            key = normalize_name(joint.name)
            interesting = key.endswith("toe") or key.endswith("toes") or key.endswith("toebase") or key.endswith("foot")
            offset = joint.end_offset if joint.end_offset is not None else None
            # Prefer the offset of a toe child, which is the foot pointing forward.
            if interesting and offset is None:
                continue
            if not interesting:
                # A foot whose child is a toe: use that child's offset.
                continue
            if offset is None:
                continue
            if abs(offset[0]) + abs(offset[2]) < 1e-6:
                continue
            sx += offset[0]
            sz += offset[2]
            hits += 1
        if hits == 0:
            # Fall back to any end site that is mostly horizontal.
            for joint in self.joints.values():
                if joint.end_offset is None:
                    continue
                off = joint.end_offset
                if abs(off[2]) > abs(off[1]) and abs(off[2]) > abs(off[0]):
                    sz += off[2]
                    hits += 1
                elif abs(off[0]) > abs(off[1]) and abs(off[0]) > abs(off[2]):
                    sx += off[0]
                    hits += 1
        if hits == 0 or (abs(sx) < 1e-8 and abs(sz) < 1e-8):
            return None
        if abs(sz) >= abs(sx):
            return "z" if sz >= 0 else "-z"
        return "x" if sx >= 0 else "-x"


def normalize_name(name: str) -> str:
    base = name.split(":")[-1].split("|")[-1]
    return "".join(ch for ch in base.lower() if ch.isalnum())


def fk_frame(
    bvh: BVH,
    frame: Sequence[float],
    rotations: Optional[Dict[str, Quat]] = None,
) -> Tuple[Dict[str, Vec3], Dict[str, Quat]]:
    """World position and world rotation of every joint for one frame.

    ``rotations``, when given, replaces the Euler channels. The root
    translation still comes from the frame. World rotation of a joint includes
    its own local rotation, so a child offset is expressed in the parent's
    world rotation (not the child's).
    """
    positions: Dict[str, Vec3] = {}
    orientations: Dict[str, Quat] = {}

    def walk(name: str, parent_pos: Vec3, parent_rot: Quat) -> None:
        joint = bvh.joints[name]
        local_rot = rotations[name] if rotations and name in rotations else bvh.rotation_quat(name, frame)
        local_pos = bvh.translation(name, frame)
        world_pos = v_add(parent_pos, quat_rotate(parent_rot, local_pos))
        world_rot = quat_mul(parent_rot, local_rot)
        positions[name] = world_pos
        orientations[name] = world_rot
        for child in joint.children:
            walk(child, world_pos, world_rot)

    walk(bvh.root, (0.0, 0.0, 0.0), QUAT_IDENTITY)
    return positions, orientations


def parse_bvh(text: str, path: str = "") -> BVH:
    tokens = _tokenize(text)
    if not tokens or tokens[0].lower() != "hierarchy":
        raise BVHError(f"{_where(path)} does not start with HIERARCHY")

    joints: Dict[str, Joint] = {}
    order: List[str] = []
    stack: List[Optional[str]] = [None]
    index = 1
    channel_cursor = 0
    root_name: Optional[str] = None

    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()
        if lower in ("root", "joint"):
            if index + 1 >= len(tokens):
                raise BVHError(f"{_where(path)} ends in the middle of a joint name")
            raw_name = tokens[index + 1]
            name = _unique_name(raw_name, joints)
            index += 2
            if index < len(tokens) and tokens[index] == "{":
                index += 1
            offset, index = _expect_offset(tokens, index, path, name)
            channels, index = _expect_channels(tokens, index, path, name)
            parent = stack[-1]
            joint = Joint(
                name=name,
                parent=parent,
                offset=offset,
                channels=channels,
                channel_start=channel_cursor,
                depth=0 if parent is None else joints[parent].depth + 1,
            )
            channel_cursor += len(channels)
            if parent is None:
                root_name = name
            else:
                joints[parent].children.append(name)
            joints[name] = joint
            order.append(name)
            stack.append(name)
            continue
        if lower == "end" and index + 1 < len(tokens) and tokens[index + 1].lower() == "site":
            index += 2
            if index < len(tokens) and tokens[index] == "{":
                index += 1
            # Some files name the site. Skip a non-numeric token before OFFSET.
            if index < len(tokens) and tokens[index].lower() not in ("offset", "{"):
                if tokens[index] != "{":
                    # A name. If the next token is '{', consume it too.
                    index += 1
                    if index < len(tokens) and tokens[index] == "{":
                        index += 1
            offset, index = _expect_offset(tokens, index, path, "End Site")
            owner = stack[-1]
            if owner is None or owner not in joints:
                raise BVHError(f"{_where(path)} has an End Site outside a joint")
            joints[owner].end_offset = offset
            # The site's own closing brace must not pop the joint that owns it.
            # The joint was pushed after its CHANNELS, and the site's '{' was
            # consumed here rather than pushed, so swallow exactly one '}'.
            if index < len(tokens) and tokens[index] == "}":
                index += 1
            continue
        if token == "{":
            index += 1
            continue
        if token == "}":
            if len(stack) <= 1:
                raise BVHError(f"{_where(path)} has an extra closing brace")
            stack.pop()
            index += 1
            continue
        if lower == "motion":
            index += 1
            break
        raise BVHError(f"{_where(path)} has unexpected token {token!r} in the hierarchy")

    if root_name is None or not joints:
        raise BVHError(f"{_where(path)} has no ROOT joint")
    if len(stack) != 1:
        raise BVHError(f"{_where(path)} has unclosed braces in the hierarchy")

    frame_count, frame_time, index = _expect_motion_header(tokens, index, path)
    values: List[float] = []
    while index < len(tokens):
        try:
            values.append(float(tokens[index]))
        except ValueError as exc:
            raise BVHError(f"{_where(path)} has a non-numeric motion token {tokens[index]!r}") from exc
        index += 1

    if channel_cursor <= 0:
        raise BVHError(f"{_where(path)} has no motion channels")
    if len(values) % channel_cursor != 0:
        raise BVHError(
            f"{_where(path)} motion data has {len(values)} numbers, which is not a "
            f"multiple of {channel_cursor} channels"
        )
    got_frames = len(values) // channel_cursor
    if frame_count is not None and got_frames != frame_count:
        raise BVHError(
            f"{_where(path)} declares {frame_count} frames but has {got_frames}"
        )
    if got_frames == 0:
        raise BVHError(f"{_where(path)} has no motion frames")

    frames = [values[i * channel_cursor : (i + 1) * channel_cursor] for i in range(got_frames)]
    return BVH(
        joints=joints,
        order=order,
        frames=frames,
        frame_time=frame_time,
        root=root_name,
        path=path,
    )


def load_bvh(path: str) -> BVH:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return parse_bvh(text, path=path)


def _tokenize(text: str) -> List[str]:
    cleaned = text.replace("{", " { ").replace("}", " } ")
    cleaned = cleaned.replace(",", " ")
    raw = cleaned.split()
    tokens: List[str] = []
    for token in raw:
        # "Frames:" and "Time:" arrive either split or glued.
        if token.endswith(":") and token != ":":
            tokens.append(token[:-1])
            tokens.append(":")
        else:
            tokens.append(token)
    return tokens


def _where(path: str) -> str:
    return path or "BVH"


def _unique_name(name: str, joints: Dict[str, Joint]) -> str:
    if name not in joints:
        return name
    index = 2
    while f"{name}.{index}" in joints:
        index += 1
    return f"{name}.{index}"


def _expect_offset(tokens: List[str], index: int, path: str, owner: str) -> Tuple[Vec3, int]:
    if index >= len(tokens) or tokens[index].lower() != "offset":
        got = tokens[index] if index < len(tokens) else "EOF"
        raise BVHError(f"{_where(path)} joint {owner} is missing OFFSET (found {got})")
    if index + 3 >= len(tokens):
        raise BVHError(f"{_where(path)} joint {owner} has a truncated OFFSET")
    try:
        offset = (float(tokens[index + 1]), float(tokens[index + 2]), float(tokens[index + 3]))
    except ValueError as exc:
        raise BVHError(f"{_where(path)} joint {owner} has a non-numeric OFFSET") from exc
    return offset, index + 4


def _expect_channels(tokens: List[str], index: int, path: str, owner: str) -> Tuple[List[str], int]:
    if index >= len(tokens) or tokens[index].lower() != "channels":
        # End sites have no channels. Joints must have them.
        got = tokens[index] if index < len(tokens) else "EOF"
        raise BVHError(f"{_where(path)} joint {owner} is missing CHANNELS (found {got})")
    if index + 1 >= len(tokens):
        raise BVHError(f"{_where(path)} joint {owner} has a truncated CHANNELS line")
    try:
        count = int(tokens[index + 1])
    except ValueError as exc:
        raise BVHError(f"{_where(path)} joint {owner} has a bad channel count") from exc
    start = index + 2
    end = start + count
    if end > len(tokens):
        raise BVHError(f"{_where(path)} joint {owner} declares {count} channels but the line ends")
    channels = tokens[start:end]
    for channel in channels:
        lower = channel.lower()
        if not (lower.endswith("position") or lower.endswith("rotation")):
            raise BVHError(f"{_where(path)} joint {owner} has unknown channel {channel}")
    return channels, end


def _expect_motion_header(tokens: List[str], index: int, path: str) -> Tuple[Optional[int], float, int]:
    frame_count: Optional[int] = None
    frame_time: Optional[float] = None
    while index < len(tokens):
        lower = tokens[index].lower()
        if lower == "frames":
            index += 1
            if index < len(tokens) and tokens[index] == ":":
                index += 1
            if index >= len(tokens):
                raise BVHError(f"{_where(path)} is missing the frame count")
            frame_count = int(float(tokens[index]))
            index += 1
            continue
        if lower == "frame":
            index += 1
            if index < len(tokens) and tokens[index].lower() == "time":
                index += 1
            if index < len(tokens) and tokens[index] == ":":
                index += 1
            if index >= len(tokens):
                raise BVHError(f"{_where(path)} is missing the frame time")
            frame_time = float(tokens[index])
            index += 1
            break
        # A number here means the header was skipped and motion has started.
        try:
            float(tokens[index])
        except ValueError:
            raise BVHError(f"{_where(path)} has unexpected motion header token {tokens[index]!r}")
        else:
            break
    if frame_time is None or frame_time < 0:
        raise BVHError(f"{_where(path)} is missing a positive Frame Time")
    return frame_count, frame_time, index


def bone_length(offset: Vec3) -> float:
    return v_len(offset)
