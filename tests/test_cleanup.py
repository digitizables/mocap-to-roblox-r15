"""Trim, resample, loop blend, and keyframe reduction."""

import math
import unittest

from mocap2r15.cleanup import (
    drop_matching_end,
    loop_blend,
    mirror_left_to_right,
    reduce_frames,
    resample_frames,
    trim_frames,
    upright_torso,
)
from mocap2r15.clip import SourceFrame
from mocap2r15.math3d import (
    QUAT_IDENTITY,
    VEC_X,
    VEC_Y,
    VEC_Z,
    quat_between_angle,
    quat_from_axis_angle,
    quat_mirror_x,
    quat_mul,
    quat_rotate,
    quat_strip_twist,
)
from mocap2r15.r15 import PART_ORDER
from mocap2r15.retarget import R15Frame


def _source(times, angle_of):
    frames = []
    for time in times:
        rotations = {name: QUAT_IDENTITY for name in ("Hips", "LeftUpLeg")}
        rotations["LeftUpLeg"] = quat_from_axis_angle(VEC_Y, math.radians(angle_of(time)))
        frames.append(SourceFrame(time, rotations, (time, 0.0, 0.0)))
    return frames


def _r15(time, angle, pos=(0.0, 0.0, 0.0)):
    local = {part: QUAT_IDENTITY for part in PART_ORDER}
    local["LeftUpperLeg"] = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(angle))
    return R15Frame(time, local, pos)


class CleanupTests(unittest.TestCase):
    def test_trim_rebases_time(self):
        frames = _source([0.0, 0.5, 1.0, 1.5], lambda time: 0.0)
        kept = trim_frames(frames, 0.5, 1.0)
        self.assertEqual([frame.time for frame in kept], [0.0, 0.5])
        self.assertEqual(trim_frames(frames, 3.0, 4.0), [])

    def test_resample_includes_both_endpoints(self):
        frames = _source([0.0, 1.0], lambda time: 90.0 * time)
        out = resample_frames(frames, 30.0)
        self.assertEqual(len(out), 31)
        self.assertAlmostEqual(out[0].time, 0.0)
        self.assertAlmostEqual(out[-1].time, 1.0)
        mid = out[15]
        angle = math.degrees(quat_between_angle(QUAT_IDENTITY, mid.rotations["LeftUpLeg"]))
        self.assertAlmostEqual(angle, 45.0, delta=1.0)
        self.assertAlmostEqual(mid.root_pos[0], 0.5, places=5)

    def test_loop_blend_matches_the_first_pose_at_the_seam(self):
        frames = [
            _r15(0.0, 0.0, (0.0, 0.0, 0.0)),
            _r15(0.5, 40.0, (1.0, 0.2, 3.0)),
            _r15(1.0, 80.0, (2.0, 0.4, 6.0)),
        ]
        loop_blend(frames, 2, blend_translation=False)
        last = frames[-1]
        self.assertLess(math.degrees(quat_between_angle(frames[0].local["LeftUpperLeg"], last.local["LeftUpperLeg"])), 1e-4)
        # Root XZ stays. Only the bob (Y) is pulled back.
        self.assertAlmostEqual(last.root_pos[0], 2.0, places=5)
        self.assertAlmostEqual(last.root_pos[2], 6.0, places=5)
        self.assertAlmostEqual(last.root_pos[1], 0.0, places=5)

    def test_drop_matching_end_keeps_a_real_stride(self):
        same = [_r15(0.0, 0.0), _r15(0.5, 10.0), _r15(1.0, 0.0)]
        self.assertEqual(len(drop_matching_end(same)), 2)
        moved = [_r15(0.0, 0.0, (0.0, 0.0, 0.0)), _r15(0.5, 10.0), _r15(1.0, 0.0, (0.0, 0.0, 1.0))]
        self.assertEqual(len(drop_matching_end(moved)), 3)

    def test_upright_torso_is_vertical_and_keeps_the_limb_direction(self):
        yaw = quat_from_axis_angle(VEC_Y, math.radians(20))
        roll = quat_from_axis_angle(VEC_Z, math.radians(18))
        pitch = quat_from_axis_angle(VEC_X, math.radians(12))
        lower = quat_mul(yaw, quat_mul(roll, pitch))
        upper = quat_from_axis_angle(VEC_Z, math.radians(15))
        arm = quat_from_axis_angle(VEC_X, math.radians(40))
        leg = quat_from_axis_angle(VEC_X, math.radians(-30))
        head = quat_mul(quat_from_axis_angle(VEC_X, math.radians(-10)), quat_from_axis_angle(VEC_Z, math.radians(25)))
        local = {part: QUAT_IDENTITY for part in PART_ORDER}
        local["LowerTorso"] = lower
        local["UpperTorso"] = upper
        local["Head"] = head
        local["LeftUpperArm"] = arm
        local["LeftUpperLeg"] = leg
        frame = R15Frame(0.0, local, (0.0, 0.0, 0.0))
        world_upper = quat_mul(lower, upper)
        world_arm = quat_rotate(quat_mul(world_upper, arm), (0.0, -1.0, 0.0))
        world_leg = quat_rotate(quat_mul(lower, leg), (0.0, -1.0, 0.0))
        world_head = quat_mul(world_upper, head)
        upright_torso([frame])
        up = quat_rotate(frame.local["LowerTorso"], (0.0, 1.0, 0.0))
        self.assertAlmostEqual(up[0], 0.0, places=5)
        self.assertAlmostEqual(up[1], 1.0, places=5)
        self.assertAlmostEqual(up[2], 0.0, places=5)
        # Pitch and roll change the heading, so the kept yaw is the facing of
        # the original chest, flattened onto the ground.
        original = quat_rotate(lower, (0.0, 0.0, -1.0))
        horizontal = math.hypot(original[0], original[2])
        faced = quat_rotate(frame.local["LowerTorso"], (0.0, 0.0, -1.0))
        self.assertAlmostEqual(faced[1], 0.0, places=5)
        self.assertAlmostEqual(faced[0], original[0] / horizontal, places=5)
        self.assertAlmostEqual(faced[2], original[2] / horizontal, places=5)
        self.assertLess(math.degrees(quat_between_angle(QUAT_IDENTITY, frame.local["UpperTorso"])), 1e-4)
        new_arm = quat_rotate(quat_mul(frame.local["LowerTorso"], frame.local["LeftUpperArm"]), (0.0, -1.0, 0.0))
        new_leg = quat_rotate(quat_mul(frame.local["LowerTorso"], frame.local["LeftUpperLeg"]), (0.0, -1.0, 0.0))
        for got, want in ((new_arm, world_arm), (new_leg, world_leg)):
            self.assertAlmostEqual(got[0], want[0], places=5)
            self.assertAlmostEqual(got[1], want[1], places=5)
            self.assertAlmostEqual(got[2], want[2], places=5)
        new_head = quat_mul(frame.local["LowerTorso"], frame.local["Head"])
        self.assertLess(math.degrees(quat_between_angle(new_head, quat_strip_twist(world_head))), 0.05)

    def test_upright_torso_leaves_yaw_on_the_humanoid_root(self):
        yaw = quat_from_axis_angle(VEC_Y, math.radians(35))
        local = {part: QUAT_IDENTITY for part in PART_ORDER}
        local["HumanoidRootPart"] = yaw
        local["LowerTorso"] = quat_from_axis_angle(VEC_Z, math.radians(14))
        frame = R15Frame(0.0, local, (0.0, 0.0, 0.0))
        upright_torso([frame])
        self.assertLess(math.degrees(quat_between_angle(yaw, frame.local["HumanoidRootPart"])), 1e-4)
        self.assertLess(math.degrees(quat_between_angle(QUAT_IDENTITY, frame.local["LowerTorso"])), 0.05)

    def test_mirror_half_cycle_copies_the_left_limb(self):
        # Z rotation swings a down bone out to the side, so the X mirror is visible.
        # The last key is the cycle point and does not repeat the first pose.
        # Sampling half a cycle from the middle must use that key, not frame 0.
        angles = [10.0, 25.0, 40.0, -15.0, 55.0]
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        frames = []
        for time, angle in zip(times, angles):
            local = {part: QUAT_IDENTITY for part in PART_ORDER}
            local["LeftUpperArm"] = quat_from_axis_angle(VEC_Z, math.radians(angle))
            local["LeftLowerArm"] = quat_from_axis_angle(VEC_X, math.radians(angle * 0.5))
            local["RightUpperArm"] = quat_from_axis_angle(VEC_Y, math.radians(80))
            frames.append(R15Frame(time, local, (0.0, 0.0, 0.0)))
        mirror_left_to_right(frames, phase=0.5)
        # t = 0 reads the left arm at t = 0.5. The left side itself stays.
        self.assertLess(
            math.degrees(quat_between_angle(frames[0].local["LeftUpperArm"], quat_from_axis_angle(VEC_Z, math.radians(10)))),
            1e-4,
        )
        self.assertLess(
            math.degrees(
                quat_between_angle(
                    frames[0].local["RightUpperArm"],
                    quat_mirror_x(quat_from_axis_angle(VEC_Z, math.radians(40))),
                )
            ),
            0.05,
        )
        self.assertLess(
            math.degrees(
                quat_between_angle(
                    frames[2].local["RightLowerArm"],
                    quat_mirror_x(quat_from_axis_angle(VEC_X, math.radians(27.5))),
                )
            ),
            0.05,
        )
        swung = quat_rotate(frames[0].local["RightUpperArm"], (0.0, -1.0, 0.0))
        left = quat_rotate(quat_from_axis_angle(VEC_Z, math.radians(40)), (0.0, -1.0, 0.0))
        self.assertAlmostEqual(swung[0], -left[0], places=5)
        self.assertAlmostEqual(swung[1], left[1], places=5)
        self.assertAlmostEqual(swung[2], left[2], places=5)

    def test_mirror_phase_zero_copies_the_same_frame(self):
        frames = []
        for time, angle in ((0.0, 15.0), (0.5, 50.0), (1.0, 70.0)):
            local = {part: QUAT_IDENTITY for part in PART_ORDER}
            local["LeftUpperLeg"] = quat_from_axis_angle(VEC_Z, math.radians(angle))
            frames.append(R15Frame(time, local, (0.0, 0.0, 0.0)))
        mirror_left_to_right(frames, phase=0.0)
        for frame in frames:
            copied = quat_mirror_x(frame.local["LeftUpperLeg"])
            self.assertLess(math.degrees(quat_between_angle(frame.local["RightUpperLeg"], copied)), 0.05)

    def test_reduction_keeps_endpoints_and_a_position_spike(self):
        flat = [_r15(i * 0.1, 0.0) for i in range(6)]
        thin = reduce_frames(flat, 5.0, 0.05)
        self.assertEqual([frame.time for frame in thin], [0.0, 0.5])
        spiked = [_r15(i * 0.1, 0.0, (0.0, 0.0, 0.0)) for i in range(5)]
        spiked[2] = _r15(0.2, 0.0, (0.0, 1.0, 0.0))
        kept = reduce_frames(spiked, 5.0, 0.05)
        self.assertIn(0.2, [frame.time for frame in kept])
        self.assertEqual(kept[0].time, 0.0)
        self.assertEqual(kept[-1].time, 0.4)
