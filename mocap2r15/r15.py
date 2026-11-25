"""Roblox R15 pose tree.

A KeyframeSequence nests Pose instances by **part name**. Studio applies the
pose named ``LowerTorso`` to the Motor6D named ``Root``, the pose named
``UpperTorso`` to ``Waist``, and so on. The part hierarchy below is what the
importer expects. Identity CFrames are the Rig Builder rest: torso and head
up, arms and legs down, feet pointing along the character forward axis (-Z).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from mocap2r15.math3d import Vec3

# part -> parent part. Insertion order is parent-before-child.
POSE_PARENT: Dict[str, Optional[str]] = {
    "HumanoidRootPart": None,
    "LowerTorso": "HumanoidRootPart",
    "UpperTorso": "LowerTorso",
    "Head": "UpperTorso",
    "LeftUpperArm": "UpperTorso",
    "LeftLowerArm": "LeftUpperArm",
    "LeftHand": "LeftLowerArm",
    "RightUpperArm": "UpperTorso",
    "RightLowerArm": "RightUpperArm",
    "RightHand": "RightLowerArm",
    "LeftUpperLeg": "LowerTorso",
    "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg",
    "RightUpperLeg": "LowerTorso",
    "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg",
}

PART_ORDER: List[str] = list(POSE_PARENT.keys())

# Motor6D that actually moves when the part's Pose is played. Informational:
# the rbxmx names poses after parts, and Studio resolves these motors.
MOTOR6D_OF_PART = {
    "LowerTorso": "Root",
    "UpperTorso": "Waist",
    "Head": "Neck",
    "LeftUpperArm": "LeftShoulder",
    "LeftLowerArm": "LeftElbow",
    "LeftHand": "LeftWrist",
    "RightUpperArm": "RightShoulder",
    "RightLowerArm": "RightElbow",
    "RightHand": "RightWrist",
    "LeftUpperLeg": "LeftHip",
    "LeftLowerLeg": "LeftKnee",
    "LeftFoot": "LeftAnkle",
    "RightUpperLeg": "RightHip",
    "RightLowerLeg": "RightKnee",
    "RightFoot": "RightAnkle",
}

_UP: Vec3 = (0.0, 1.0, 0.0)
_DOWN: Vec3 = (0.0, -1.0, 0.0)
_FWD: Vec3 = (0.0, 0.0, -1.0)  # Roblox characters face -Z.

# Rest bone direction and the hint passed to the twist frame.
# Hint is character-forward, except feet, whose bone *is* forward, so the
# hint falls back to world up inside basis_quat.
TARGET_DIR: Dict[str, Vec3] = {
    "LowerTorso": _UP,
    "UpperTorso": _UP,
    "Head": _UP,
    "LeftUpperArm": _DOWN,
    "LeftLowerArm": _DOWN,
    "LeftHand": _DOWN,
    "RightUpperArm": _DOWN,
    "RightLowerArm": _DOWN,
    "RightHand": _DOWN,
    "LeftUpperLeg": _DOWN,
    "LeftLowerLeg": _DOWN,
    "LeftFoot": _FWD,
    "RightUpperLeg": _DOWN,
    "RightLowerLeg": _DOWN,
    "RightFoot": _FWD,
}

# Used only when a BVH bone has a zero-length offset and no descendant to aim at.
DEFAULT_SRC_DIR: Dict[str, Vec3] = {
    "LowerTorso": (0.0, 1.0, 0.0),
    "UpperTorso": (0.0, 1.0, 0.0),
    "Head": (0.0, 1.0, 0.0),
    "LeftUpperArm": (1.0, 0.0, 0.0),
    "LeftLowerArm": (1.0, 0.0, 0.0),
    "LeftHand": (1.0, 0.0, 0.0),
    "RightUpperArm": (-1.0, 0.0, 0.0),
    "RightLowerArm": (-1.0, 0.0, 0.0),
    "RightHand": (-1.0, 0.0, 0.0),
    "LeftUpperLeg": (0.0, -1.0, 0.0),
    "LeftLowerLeg": (0.0, -1.0, 0.0),
    "LeftFoot": (0.0, 0.0, 1.0),
    "RightUpperLeg": (0.0, -1.0, 0.0),
    "RightLowerLeg": (0.0, -1.0, 0.0),
    "RightFoot": (0.0, 0.0, 1.0),
}

ANIMATION_PRIORITY = {
    "Idle": 0,
    "Movement": 1,
    "Action": 2,
    "Action2": 3,
    "Action3": 4,
    "Action4": 5,
    "Core": 1000,
}


def children_of(part: str) -> List[str]:
    return [name for name, parent in POSE_PARENT.items() if parent == part]


def target_hint(part: str) -> Vec3:
    """Secondary axis for the R15 rest frame of ``part`` (Roblox space)."""
    direction = TARGET_DIR[part]
    # Feet already point forward, so the twist reference is up.
    if abs(direction[2]) > 0.9:
        return _UP
    return _FWD
