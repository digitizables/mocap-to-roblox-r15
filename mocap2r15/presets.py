"""Skeleton presets: which BVH joints drive which R15 part.

Adding a preset
---------------
Copy an entry in ``PRESETS``. ``parts`` maps an R15 part to BVH joint names
in parent-to-child order. Names that are not in the file are skipped, so a
clavicle (``LeftShoulder``) can be listed in front of ``LeftArm`` and files
without a clavicle still work. ``ends`` names extra joints that can supply
the bone direction (a toe for the foot, a finger for the hand). The measured
direction prefers the next R15 part (the neck for the chest, the forearm for
the upper arm) and skips a zero-length connector such as CMU's Neck. It does
not pick the longest child, because a clavicle is often longer than the neck.

``auto`` picks the preset with the most joints actually present. ``generic``
uses the alias table and is the fallback when no named preset matches well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from mocap2r15.bvh import BVH, normalize_name
from mocap2r15.math3d import Vec3, v_add, v_len, v_normalize, v_sub
from mocap2r15.r15 import DEFAULT_SRC_DIR, PART_ORDER, TARGET_DIR

# The bone that continues this part. Arms are not a continuation of the spine,
# so the spine aims at the neck even when the clavicle offset is longer.
_SUCCESSOR = {
    "LowerTorso": "UpperTorso",
    "UpperTorso": "Head",
    "LeftUpperArm": "LeftLowerArm",
    "LeftLowerArm": "LeftHand",
    "RightUpperArm": "RightLowerArm",
    "RightLowerArm": "RightHand",
    "LeftUpperLeg": "LeftLowerLeg",
    "LeftLowerLeg": "LeftFoot",
    "RightUpperLeg": "RightLowerLeg",
    "RightLowerLeg": "RightFoot",
}


# R15 part -> BVH names, parent first. Absent names are skipped.
_CMU = {
    "LowerTorso": ["Hips"],
    "UpperTorso": ["LowerBack", "Spine", "Spine1"],
    "Head": ["Neck", "Neck1", "Head"],
    "LeftUpperArm": ["LeftShoulder", "LeftArm"],
    "LeftLowerArm": ["LeftForeArm"],
    "LeftHand": ["LeftHand"],
    "RightUpperArm": ["RightShoulder", "RightArm"],
    "RightLowerArm": ["RightForeArm"],
    "RightHand": ["RightHand"],
    "LeftUpperLeg": ["LHipJoint", "LeftUpLeg"],
    "LeftLowerLeg": ["LeftLeg"],
    "LeftFoot": ["LeftFoot"],
    "RightUpperLeg": ["RHipJoint", "RightUpLeg"],
    "RightLowerLeg": ["RightLeg"],
    "RightFoot": ["RightFoot"],
}

_MIXAMO = {
    "LowerTorso": ["Hips"],
    "UpperTorso": ["Spine", "Spine1", "Spine2"],
    "Head": ["Neck", "Head"],
    "LeftUpperArm": ["LeftShoulder", "LeftArm"],
    "LeftLowerArm": ["LeftForeArm"],
    "LeftHand": ["LeftHand"],
    "RightUpperArm": ["RightShoulder", "RightArm"],
    "RightLowerArm": ["RightForeArm"],
    "RightHand": ["RightHand"],
    "LeftUpperLeg": ["LeftUpLeg"],
    "LeftLowerLeg": ["LeftLeg"],
    "LeftFoot": ["LeftFoot"],
    "RightUpperLeg": ["RightUpLeg"],
    "RightLowerLeg": ["RightLeg"],
    "RightFoot": ["RightFoot"],
}

# Ubisoft LaFAN1 / Bandai Namco style. Same limb names as Mixamo, toe joint
# is often "LeftToe" rather than "LeftToeBase" (ends list accepts both).
_LAFAN1 = {
    "LowerTorso": ["Hips"],
    "UpperTorso": ["Spine", "Spine1", "Spine2"],
    "Head": ["Neck", "Head"],
    "LeftUpperArm": ["LeftShoulder", "LeftArm"],
    "LeftLowerArm": ["LeftForeArm"],
    "LeftHand": ["LeftHand"],
    "RightUpperArm": ["RightShoulder", "RightArm"],
    "RightLowerArm": ["RightForeArm"],
    "RightHand": ["RightHand"],
    "LeftUpperLeg": ["LeftUpLeg"],
    "LeftLowerLeg": ["LeftLeg"],
    "LeftFoot": ["LeftFoot"],
    "RightUpperLeg": ["RightUpLeg"],
    "RightLowerLeg": ["RightLeg"],
    "RightFoot": ["RightFoot"],
}

# SOMA / SMPL BVH exports (Kimodo and similar). "LeftLeg" is the thigh here,
# unlike CMU where "LeftLeg" is the shin. Keep this preset separate.
_SOMA = {
    "LowerTorso": ["Root", "Hips"],
    "UpperTorso": ["Spine1", "Spine2", "Chest"],
    "Head": ["Neck1", "Neck2", "Head"],
    "LeftUpperArm": ["LeftShoulder", "LeftArm"],
    "LeftLowerArm": ["LeftForeArm"],
    "LeftHand": ["LeftHand"],
    "RightUpperArm": ["RightShoulder", "RightArm"],
    "RightLowerArm": ["RightForeArm"],
    "RightHand": ["RightHand"],
    "LeftUpperLeg": ["LeftLeg"],
    "LeftLowerLeg": ["LeftShin"],
    "LeftFoot": ["LeftFoot"],
    "RightUpperLeg": ["RightLeg"],
    "RightLowerLeg": ["RightShin"],
    "RightFoot": ["RightFoot"],
}

_ENDS = {
    "LowerTorso": ["Spine", "LowerBack", "Spine1"],
    "UpperTorso": ["Neck", "Neck1", "Chest"],
    "Head": ["Head"],
    "LeftUpperArm": ["LeftForeArm"],
    "LeftLowerArm": ["LeftHand"],
    "LeftHand": ["LeftFingerBase", "LeftHandIndex1"],
    "RightUpperArm": ["RightForeArm"],
    "RightLowerArm": ["RightHand"],
    "RightHand": ["RightFingerBase", "RightHandIndex1"],
    "LeftUpperLeg": ["LeftLeg", "LeftShin"],
    "LeftLowerLeg": ["LeftFoot"],
    "LeftFoot": ["LeftToeBase", "LeftToe", "LeftToeBase"],
    "RightUpperLeg": ["RightLeg", "RightShin"],
    "RightLowerLeg": ["RightFoot"],
    "RightFoot": ["RightToeBase", "RightToe"],
}

# Alias stems for the generic matcher. Compared to normalize_name(joint),
# which strips prefixes and underscores. Do not substring-match: "spine"
# must not consume "spine1".
_GENERIC = {
    "LowerTorso": ["hips", "pelvis", "hip"],
    "UpperTorso": [
        "lowerback", "spine", "spine1", "spine2", "spine3",
        "chest", "upperback", "thorax", "abdomen",
    ],
    "Head": ["neck", "neck1", "neck2", "head"],
    "LeftUpperArm": ["leftshoulder", "leftclavicle", "lclavicle", "leftarm", "leftupperarm", "lhumerus", "lefthumerus"],
    "LeftLowerArm": ["leftforearm", "leftlowerarm", "lradius", "leftradius", "leftelbow"],
    "LeftHand": ["lefthand", "lwrist", "leftwrist"],
    "RightUpperArm": ["rightshoulder", "rightclavicle", "rclavicle", "rightarm", "rightupperarm", "rhumerus", "righthumerus"],
    "RightLowerArm": ["rightforearm", "rightlowerarm", "rradius", "rightradius", "rightelbow"],
    "RightHand": ["righthand", "rwrist", "rightwrist"],
    "LeftUpperLeg": ["leftupleg", "leftupperleg", "leftthigh", "lfemur", "leftfemur", "lthigh"],
    "LeftLowerLeg": ["leftleg", "leftlowerleg", "leftshin", "ltibia", "lefttibia", "leftknee"],
    "LeftFoot": ["leftfoot", "leftankle", "lfoot"],
    "RightUpperLeg": ["rightupleg", "rightupperleg", "rightthigh", "rfemur", "rightfemur", "rthigh"],
    "RightLowerLeg": ["rightleg", "rightlowerleg", "rightshin", "rtibia", "righttibia", "rightknee"],
    "RightFoot": ["rightfoot", "rightankle", "rfoot"],
}


PRESETS = {
    "cmu": {
        "description": "CMU Graphics Lab / cgspeed MotionBuilder BVH (Hips, LeftUpLeg, LeftLeg, LHipJoint).",
        "parts": _CMU,
    },
    "mixamo": {
        "description": "Mixamo (mixamorig:Hips, Spine, Spine1, Spine2, LeftUpLeg, LeftLeg).",
        "parts": _MIXAMO,
    },
    "lafan1": {
        "description": "LaFAN1 / Bandai-style (Hips, Spine2, LeftUpLeg, LeftToe).",
        "parts": _LAFAN1,
    },
    "soma": {
        "description": "SOMA / SMPL BVH (Root, Hips, Chest, LeftLeg thigh, LeftShin).",
        "parts": _SOMA,
    },
}


@dataclass
class PartSpec:
    part: str
    joints: List[str]
    src_dir: Vec3
    tgt_dir: Vec3
    # True when src_dir came from a real offset rather than the T-pose fallback.
    direction_from_skeleton: bool = True


@dataclass
class Mapping:
    preset: str
    parts: Dict[str, PartSpec]
    warnings: List[str] = field(default_factory=list)
    found: int = 0
    total: int = 0

    @property
    def ratio(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.found / self.total


def preset_names() -> List[str]:
    return ["auto", "generic", *PRESETS.keys()]


def score_preset(bvh: BVH, preset: str) -> Tuple[int, int]:
    """Return ``(names found, names listed)`` for a named preset."""
    parts = PRESETS[preset]["parts"]
    found = 0
    total = 0
    for names in parts.values():
        for name in names:
            total += 1
            if bvh.find(name):
                found += 1
    return found, total


def detect_preset(bvh: BVH) -> Tuple[str, Dict[str, float]]:
    """Pick the named preset that shares the most joints with the file."""
    ratios: Dict[str, float] = {}
    best_name = "generic"
    best_key = (-1, -1.0)
    for name in PRESETS:
        found, total = score_preset(bvh, name)
        ratio = found / total if total else 0.0
        ratios[name] = ratio
        # More confirmed joints wins. Ratio breaks ties between a long list
        # and a short one that happens to be a subset.
        key = (found, ratio)
        if key > best_key:
            best_key = key
            best_name = name
    if best_key[1] < 0.4:
        return "generic", ratios
    return best_name, ratios


def resolve_mapping(bvh: BVH, preset: str = "auto") -> Mapping:
    """Bind preset names to joints in this file and measure bone directions."""
    warnings: List[str] = []
    chosen = preset
    if preset == "auto":
        chosen, ratios = detect_preset(bvh)
        ranked = ", ".join(f"{name} {ratio:.0%}" for name, ratio in sorted(ratios.items(), key=lambda kv: -kv[1]))
        warnings.append(f"auto-selected preset {chosen} ({ranked})")
        if chosen == "generic":
            warnings.append("no named preset matched well; using generic aliases")
    elif preset != "generic" and preset not in PRESETS:
        known = ", ".join(preset_names())
        raise ValueError(f"Unknown preset {preset!r}. Known presets: {known}")

    if chosen == "generic":
        listed = _GENERIC
        found_count, total = _generic_score(bvh)
    else:
        listed = PRESETS[chosen]["parts"]
        found_count, total = score_preset(bvh, chosen)

    raw: Dict[str, List[str]] = {}
    for part, names in listed.items():
        raw[part] = _resolve_names(bvh, names)

    # A joint drives one part. Distal parts claim it first so a neck listed on
    # both the chest and the head stays on the head.
    claimed: set = set()
    for part in reversed(PART_ORDER):
        if part not in raw:
            continue
        kept = []
        for joint in raw[part]:
            if joint in claimed:
                warnings.append(f"{joint} was listed for more than one part; keeping it on the distal part")
                continue
            kept.append(joint)
            claimed.add(joint)
        raw[part] = kept

    # Wrapper roots (SOMA "Root", some Mixamo exports) must contribute their
    # rotation to the hips or the character will not turn.
    hips = raw.get("LowerTorso", [])
    if bvh.root not in hips and bvh.joints[bvh.root].rotation_order:
        if not hips or bvh.is_ancestor(bvh.root, hips[0]):
            hips = [bvh.root] + hips
            raw["LowerTorso"] = hips

    parts: Dict[str, PartSpec] = {}
    for part in PART_ORDER:
        if part == "HumanoidRootPart":
            continue
        joints = raw.get(part, [])
        if not joints:
            warnings.append(f"no BVH joint for {part}; that pose stays at the R15 rest")
            parts[part] = PartSpec(
                part=part,
                joints=[],
                src_dir=DEFAULT_SRC_DIR[part],
                tgt_dir=TARGET_DIR[part],
                direction_from_skeleton=False,
            )
            continue
        # Placeholder; directions are filled once every chain exists so the
        # spine can aim at the head chain instead of at a shoulder.
        parts[part] = PartSpec(
            part=part,
            joints=joints,
            src_dir=DEFAULT_SRC_DIR[part],
            tgt_dir=TARGET_DIR[part],
            direction_from_skeleton=False,
        )

    for part, spec in parts.items():
        if not spec.joints:
            continue
        successor = parts.get(_SUCCESSOR.get(part, ""))
        successor_joints = successor.joints if successor else []
        direction, from_skeleton = _chain_direction(
            bvh, spec.joints, _ENDS.get(part, []), successor_joints
        )
        if not from_skeleton:
            warnings.append(
                f"{part} has no usable bone offset; assuming a default {DEFAULT_SRC_DIR[part]} direction"
            )
            direction = DEFAULT_SRC_DIR[part]
        spec.src_dir = direction
        spec.direction_from_skeleton = from_skeleton

    if not parts["LowerTorso"].joints:
        warnings.append("hips joint was not found")
    return Mapping(preset=chosen, parts=parts, warnings=warnings, found=found_count, total=total)


def compose_names(mapping: Mapping, part: str) -> List[str]:
    spec = mapping.parts.get(part)
    if spec is None:
        return []
    return list(spec.joints)


def _resolve_names(bvh: BVH, names: Sequence[str]) -> List[str]:
    found: List[str] = []
    for name in names:
        hit = bvh.find(name)
        if hit and hit not in found:
            found.append(hit)
    found.sort(key=lambda joint: (bvh.joints[joint].depth, bvh.order.index(joint)))
    if not found:
        return []
    top = found[0]
    # Drop a name that resolved onto a different branch (a loose alias).
    return [joint for joint in found if bvh.is_ancestor(top, joint)]


def _generic_score(bvh: BVH) -> Tuple[int, int]:
    found = 0
    total = 0
    for names in _GENERIC.values():
        total += 1
        if _resolve_names(bvh, names):
            found += 1
    return found, total


def _chain_direction(
    bvh: BVH,
    chain: Sequence[str],
    end_hints: Sequence[str],
    successor_joints: Sequence[str],
) -> Tuple[Vec3, bool]:
    """Unit bone direction in the bind pose, in the chain parent's frame.

    Bind rotations are zero, so offsets already live in that frame. The first
    point with a real length wins, in this order: the next R15 part's joints
    (so the spine aims at the neck, not down an arm), explicit end hints
    (toes, fingers), then this joint's End Site. Zero-length connectors such
    as CMU's Neck are skipped.
    """
    bind = bvh.bind_positions()
    origin = chain[-1]
    start = bind[origin]
    points: List[Vec3] = []

    for joint_name in successor_joints:
        if joint_name != origin and bvh.is_ancestor(origin, joint_name):
            points.append(bind[joint_name])
    for hint in end_hints:
        found = bvh.find(hint)
        if found and found != origin and bvh.is_ancestor(origin, found):
            points.append(bind[found])
    joint = bvh.joints[origin]
    if joint.end_offset is not None:
        points.append(v_add(start, joint.end_offset))
    # A successor that starts on a zero offset (Neck on top of Spine1) still
    # needs the next segment. Walk only that chain, not sibling limbs.
    if successor_joints:
        cursor = successor_joints[0]
        guard = 0
        while guard < 6 and cursor in bvh.joints and v_len(v_sub(bind[cursor], start)) < 1e-4:
            kids = bvh.joints[cursor].children
            if bvh.joints[cursor].end_offset is not None and v_len(bvh.joints[cursor].end_offset) > 1e-4:
                points.append(v_add(bind[cursor], bvh.joints[cursor].end_offset))
                break
            if not kids:
                break
            cursor = kids[0]
            points.append(bind[cursor])
            guard += 1

    for point in points:
        delta = v_sub(point, start)
        if v_len(delta) > 1e-4:
            return v_normalize(delta), True
    return (0.0, 1.0, 0.0), False
