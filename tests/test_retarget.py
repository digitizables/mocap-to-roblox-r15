"""Direction invariants for the BVH → R15 rest correction."""

import math
import unittest

from mocap2r15.bvh import parse_bvh
from mocap2r15.math3d import (
    QUAT_IDENTITY,
    almost_vec,
    euler_to_quat,
    forward_axis_quaternion,
    quat_between_angle,
    quat_rotate,
)
from mocap2r15.presets import resolve_mapping
from mocap2r15.r15 import TARGET_DIR
from mocap2r15.retarget import (
    R15Frame,
    compose_chain,
    retarget_pose,
    rest_corrections,
    subtract_rest_pose,
)
from tests.factory import cmu_skeleton, joint, render_bvh


def _clip():
    bvh = parse_bvh(render_bvh(cmu_skeleton(), [{}]))
    mapping = resolve_mapping(bvh, "cmu")
    conv = forward_axis_quaternion("z")
    corrections = rest_corrections(mapping, conv)
    return bvh, mapping, conv, corrections


def _rotations(bvh, **degrees):
    """``degrees`` maps a joint to ``(z, y, x)`` in degrees, Z applied first."""
    rotations = {name: QUAT_IDENTITY for name in bvh.order}
    for name, angles in degrees.items():
        rotations[name] = euler_to_quat("ZYX", angles)
    return rotations


def _bone(local, part):
    """Where the R15 rest bone points after this Motor6D transform."""
    return quat_rotate(local[part], TARGET_DIR[part])


class RetargetDirectionTests(unittest.TestCase):
    def test_t_pose_bind_points_arms_out_and_legs_down(self):
        bvh, mapping, conv, corrections = _clip()
        local = retarget_pose(_rotations(bvh), mapping, conv, corrections)
        self.assertTrue(almost_vec(_bone(local, "LeftUpperArm"), (-1.0, 0.0, 0.0), 0.05))
        self.assertTrue(almost_vec(_bone(local, "RightUpperArm"), (1.0, 0.0, 0.0), 0.05))
        self.assertTrue(almost_vec(_bone(local, "LeftUpperLeg"), (0.0, -1.0, 0.0), 0.05))
        self.assertTrue(almost_vec(_bone(local, "LeftLowerLeg"), (0.0, -1.0, 0.0), 0.05))
        self.assertTrue(almost_vec(_bone(local, "LeftFoot"), (0.0, 0.0, -1.0), 0.05))
        self.assertTrue(almost_vec(_bone(local, "UpperTorso"), (0.0, 1.0, 0.0), 0.05))
        # Legs and spine already match the R15 rest. Shoulders do not: a
        # T-pose arm is a large correction away from arms-down.
        for part in ("LeftUpperLeg", "UpperTorso", "LeftFoot", "Head"):
            angle = math.degrees(quat_between_angle(QUAT_IDENTITY, local[part]))
            self.assertLess(angle, 15.0, part)
        shoulder = math.degrees(quat_between_angle(QUAT_IDENTITY, local["LeftUpperArm"]))
        self.assertGreater(shoulder, 60.0)
        self.assertLess(shoulder, 120.0)

    def test_arms_down_rotation_hangs_the_r15_arm(self):
        # Rz(-90) takes the left T-pose arm (+X) to -Y.
        bvh, mapping, conv, corrections = _clip()
        local = retarget_pose(_rotations(bvh, LeftArm=( -90.0, 0.0, 0.0)), mapping, conv, corrections)
        direction = _bone(local, "LeftUpperArm")
        self.assertLess(direction[1], -0.9)
        self.assertLess(abs(direction[0]), 0.15)
        self.assertLess(abs(direction[2]), 0.15)

    def test_forward_arm_swing_faces_roblox_negative_z(self):
        # Arms already down, then Rx(-20) swings the hand toward BVH +Z.
        bvh, mapping, conv, corrections = _clip()
        local = retarget_pose(_rotations(bvh, LeftArm=(-90.0, 0.0, -20.0)), mapping, conv, corrections)
        direction = _bone(local, "LeftUpperArm")
        self.assertLess(direction[1], -0.8)
        self.assertLess(direction[2], -0.25)
        self.assertLess(abs(direction[0]), 0.15)

    def test_elbow_flexion_on_a_hanging_arm_bends_backward(self):
        # Shoulder Rz(-90) hangs the T-pose arm. Forearm Ry(+30) then bends
        # the hand toward BVH -Z, which is behind the character in Roblox.
        bvh, mapping, conv, corrections = _clip()
        local = retarget_pose(
            _rotations(bvh, LeftArm=(-90.0, 0.0, 0.0), LeftForeArm=(0.0, 30.0, 0.0)),
            mapping,
            conv,
            corrections,
        )
        direction = quat_rotate(local["LeftUpperArm"], _bone(local, "LeftLowerArm"))
        self.assertLess(direction[1], -0.8)
        self.assertGreater(direction[2], 0.3)
        self.assertLess(abs(direction[0]), 0.2)

    def test_knee_flexion_bends_toward_roblox_back(self):
        # Rx(+30) on a shin that hangs down tilts it toward BVH -Z.
        # After the +Z → -Z yaw, that bend is Roblox +Z (behind the character).
        bvh, mapping, conv, corrections = _clip()
        local = retarget_pose(_rotations(bvh, LeftLeg=(0.0, 0.0, 30.0)), mapping, conv, corrections)
        direction = _bone(local, "LeftLowerLeg")
        self.assertLess(direction[1], -0.8)
        self.assertGreater(direction[2], 0.4)
        self.assertLess(abs(direction[0]), 0.1)

    def test_spine_chain_matches_forward_kinematics(self):
        # The chest bone lives in Spine's frame, so Spine's Rz(90) hits +Y
        # before LowerBack's Rx(90). That sends +Y to -X. The facing yaw then
        # flips X, and the R15 torso (rest +Y) points along +X.
        bvh, mapping, conv, corrections = _clip()
        local = retarget_pose(
            _rotations(bvh, LowerBack=(0.0, 0.0, 90.0), Spine=(90.0, 0.0, 0.0)),
            mapping,
            conv,
            corrections,
        )
        direction = _bone(local, "UpperTorso")
        self.assertTrue(almost_vec(direction, (1.0, 0.0, 0.0), 0.08))

    def test_compose_chain_applies_the_child_in_its_own_frame(self):
        parent = euler_to_quat("X", (90.0,))
        child = euler_to_quat("Z", (90.0,))
        composed = compose_chain([parent, child])
        # Rz(90) first: +Y -> -X. Rx(90) leaves -X alone.
        self.assertTrue(almost_vec(quat_rotate(composed, (0.0, 1.0, 0.0)), (-1.0, 0.0, 0.0), 1e-5))

    def test_missing_head_stays_at_identity(self):
        text = render_bvh(
            joint(
                "Hips",
                (0, 0, 0),
                [
                    joint(
                        "LowerBack",
                        (0, 1, 0),
                        [joint("Spine", (0, 1, 0), [joint("Spine1", (0, 1, 0), end=(0, 1, 0))])],
                    )
                ],
            ),
            [{}],
        )
        bvh = parse_bvh(text)
        mapping = resolve_mapping(bvh, "cmu")
        conv = forward_axis_quaternion("z")
        local = retarget_pose(
            {name: QUAT_IDENTITY for name in bvh.order},
            mapping,
            conv,
            rest_corrections(mapping, conv),
        )
        self.assertEqual(local["Head"], QUAT_IDENTITY)
        self.assertTrue(any("Head" in warning for warning in mapping.warnings))

    def test_rest_frame_makes_the_chosen_pose_identity(self):
        bvh, mapping, conv, corrections = _clip()
        first = retarget_pose(_rotations(bvh, LeftLeg=(0.0, 0.0, 30.0)), mapping, conv, corrections)
        second = retarget_pose(_rotations(bvh, LeftLeg=(0.0, 0.0, 50.0)), mapping, conv, corrections)
        bent = math.degrees(quat_between_angle(QUAT_IDENTITY, first["LeftLowerLeg"]))
        self.assertGreater(bent, 20.0)
        frames = [
            R15Frame(0.0, first, (0.0, 0.0, 0.0)),
            R15Frame(0.1, second, (0.0, 0.0, 0.0)),
        ]
        subtract_rest_pose(frames, 0)
        self.assertLess(math.degrees(quat_between_angle(QUAT_IDENTITY, frames[0].local["LeftLowerLeg"])), 1e-4)
        delta = math.degrees(quat_between_angle(QUAT_IDENTITY, frames[1].local["LeftLowerLeg"]))
        self.assertAlmostEqual(delta, 20.0, delta=1.5)
