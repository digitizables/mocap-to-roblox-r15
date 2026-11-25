"""Scale, in-place root motion, rest frame, and the synthetic walk."""

import math
import unittest

from mocap2r15.bvh import parse_bvh
from mocap2r15.math3d import QUAT_IDENTITY, quat_between_angle, quat_rotate
from mocap2r15.pipeline import ConvertOptions, convert_bvh, guess_unit
from mocap2r15.r15 import TARGET_DIR
from mocap2r15.sample_walk import TRAVEL, build_bvh_text
from tests.factory import cmu_skeleton, joint, render_bvh


def _walk(frames):
    return parse_bvh(render_bvh(cmu_skeleton(), frames, frame_time=0.05))


def _bone(frame, part):
    return quat_rotate(frame.local[part], TARGET_DIR[part])


class PipelineTests(unittest.TestCase):
    def test_in_place_removes_a_huge_offset_and_keeps_the_bob(self):
        bvh = _walk(
            [
                {"Hips": (1.0e6, 10.0, -4.0e5, 0.0, 0.0, 0.0)},
                {"Hips": (1.0e6 + 100.0, 10.5, -4.0e5 + 50.0, 0.0, 0.0, 0.0)},
            ]
        )
        result = convert_bvh(bvh, ConvertOptions())
        self.assertEqual(result.preset, "cmu")
        self.assertEqual(result.forward_axis, "z")
        for frame in result.frames:
            self.assertAlmostEqual(frame.root_pos[0], 0.0, places=4)
            self.assertAlmostEqual(frame.root_pos[2], 0.0, places=4)
        self.assertAlmostEqual(result.frames[0].root_pos[1], 0.0, places=5)
        self.assertAlmostEqual(result.frames[1].root_pos[1], 0.5 * result.scale, places=4)

    def test_root_motion_turns_plus_z_into_roblox_forward(self):
        bvh = _walk(
            [
                {"Hips": (0.0, 10.0, 0.0, 0.0, 0.0, 0.0)},
                {"Hips": (100.0, 10.5, 50.0, 0.0, 0.0, 0.0)},
            ]
        )
        result = convert_bvh(bvh, ConvertOptions(in_place=False, unit="raw"))
        root = result.frames[1].root_pos
        self.assertAlmostEqual(result.scale, 1.0, places=6)
        self.assertAlmostEqual(root[0], -100.0, places=4)
        self.assertAlmostEqual(root[1], 0.5, places=4)
        self.assertAlmostEqual(root[2], -50.0, places=4)

    def test_auto_scale_fits_height_to_five_studs(self):
        root = joint(
            "Hips",
            (0, 0, 0),
            [
                joint("Head", (0, 180, 0), end=(0, 20, 0)),
                joint("LeftFoot", (0, -20, 0), [joint("LeftToeBase", (0, 0, 10), end=(0, 0, 5))]),
            ],
        )
        bvh = parse_bvh(render_bvh(root, [{}], frame_time=0.1))
        result = convert_bvh(bvh, ConvertOptions(preset="generic"))
        self.assertAlmostEqual(result.skeleton_height, 220.0, places=4)
        self.assertAlmostEqual(result.scale, 5.0 / 220.0, places=6)

    def test_metric_unit_uses_studs_per_meter(self):
        bvh = _walk([{"Hips": (0, 0, 0, 0, 0, 0)}, {"Hips": (0, 0, 1, 0, 0, 0)}])
        result = convert_bvh(bvh, ConvertOptions(in_place=False, unit="m", studs_per_meter=3.571))
        self.assertAlmostEqual(result.scale, 3.571, places=5)
        self.assertAlmostEqual(result.frames[1].root_pos[2], -3.571, places=4)

    def test_studs_per_unit_overrides_auto(self):
        bvh = _walk([{"Hips": (0, 0, 0, 0, 0, 0)}, {"Hips": (0, 0, 2, 0, 0, 0)}])
        result = convert_bvh(bvh, ConvertOptions(in_place=False, studs_per_unit=0.25))
        self.assertAlmostEqual(result.scale, 0.25, places=6)
        self.assertAlmostEqual(result.frames[1].root_pos[2], -0.5, places=5)

    def test_rest_frame_zero_keeps_only_the_extra_knee_bend(self):
        bvh = _walk(
            [
                {"LeftLeg": (0.0, 0.0, 30.0)},
                {"LeftLeg": (0.0, 0.0, 50.0)},
            ]
        )
        plain = convert_bvh(bvh, ConvertOptions())
        self.assertGreater(math.degrees(quat_between_angle(QUAT_IDENTITY, plain.frames[0].local["LeftLowerLeg"])), 20.0)
        rested = convert_bvh(bvh, ConvertOptions(rest_frame=0))
        self.assertLess(math.degrees(quat_between_angle(QUAT_IDENTITY, rested.frames[0].local["LeftLowerLeg"])), 1e-4)
        delta = math.degrees(quat_between_angle(QUAT_IDENTITY, rested.frames[1].local["LeftLowerLeg"]))
        self.assertAlmostEqual(delta, 20.0, delta=1.5)

    def test_upright_torso_and_same_frame_mirror(self):
        rows = []
        for _ in range(5):
            rows.append(
                {
                    "LowerBack": (20.0, 0.0, 8.0),
                    "LeftArm": (-90.0, 35.0, 0.0),
                    "RightArm": (90.0, -5.0, 0.0),
                }
            )
        bvh = _walk(rows)
        plain = convert_bvh(bvh, ConvertOptions())
        fixed = convert_bvh(bvh, ConvertOptions(upright_torso=True, mirror_left=True, mirror_phase=0.0))
        self.assertGreater(math.degrees(quat_between_angle(QUAT_IDENTITY, plain.frames[0].local["UpperTorso"])), 2.0)
        plain_left = _bone(plain.frames[0], "LeftUpperArm")
        plain_right = _bone(plain.frames[0], "RightUpperArm")
        self.assertGreater(abs(plain_right[0] + plain_left[0]) + abs(plain_right[2] - plain_left[2]), 0.05)
        for frame in fixed.frames:
            up = quat_rotate(frame.local["LowerTorso"], (0.0, 1.0, 0.0))
            self.assertAlmostEqual(up[0], 0.0, places=4)
            self.assertAlmostEqual(up[1], 1.0, places=4)
            self.assertLess(math.degrees(quat_between_angle(QUAT_IDENTITY, frame.local["UpperTorso"])), 0.05)
            left = _bone(frame, "LeftUpperArm")
            right = _bone(frame, "RightUpperArm")
            self.assertAlmostEqual(right[0], -left[0], places=4)
            self.assertAlmostEqual(right[1], left[1], places=4)
            self.assertAlmostEqual(right[2], left[2], places=4)

    def test_yaw_moves_onto_the_humanoid_root(self):
        bvh = _walk([{"Hips": (0.0, 0.0, 0.0, 0.0, 90.0, 0.0)}])
        result = convert_bvh(bvh, ConvertOptions(yaw_on_root=True))
        yaw = math.degrees(quat_between_angle(QUAT_IDENTITY, result.frames[0].local["HumanoidRootPart"]))
        leftover = math.degrees(quat_between_angle(QUAT_IDENTITY, result.frames[0].local["LowerTorso"]))
        self.assertAlmostEqual(yaw, 90.0, delta=1.0)
        self.assertLess(leftover, 5.0)

    def test_missing_head_still_converts(self):
        root = joint(
            "Hips",
            (0, 0, 0),
            [joint("LowerBack", (0, 1, 0), [joint("Spine", (0, 1, 0), [joint("Spine1", (0, 1, 0), end=(0, 1, 0))])])],
        )
        bvh = parse_bvh(render_bvh(root, [{}]))
        result = convert_bvh(bvh, ConvertOptions(preset="cmu"))
        self.assertEqual(result.frames[0].local["Head"], QUAT_IDENTITY)
        self.assertTrue(any("Head" in line for line in result.warnings))

    def test_resample_count_and_trim(self):
        rows = [{"Hips": (0, 0, index, 0, 0, 0)} for index in range(11)]
        bvh = parse_bvh(render_bvh(cmu_skeleton(), rows, frame_time=0.1))
        result = convert_bvh(bvh, ConvertOptions(fps=10, start=0.2, end=0.8))
        # 0.2..0.8 is 0.6 s. 10 fps → 6 intervals → 7 samples.
        self.assertEqual(result.output_frames, 7)
        self.assertAlmostEqual(result.frames[0].time, 0.0)
        self.assertAlmostEqual(result.frames[-1].time, 0.6, places=5)

    def test_synthetic_walk_is_an_in_place_cycle_with_a_real_stride(self):
        bvh = parse_bvh(build_bvh_text())
        self.assertGreater(bvh.skeleton_height(), 1.6)
        self.assertLess(bvh.skeleton_height(), 1.9)
        self.assertEqual(bvh.forward_hint(), "z")
        parked = convert_bvh(bvh, ConvertOptions(loop=True))
        self.assertEqual(parked.preset, "cmu")
        self.assertEqual(parked.output_frames, 30)  # the repeated last pose was dropped
        for frame in parked.frames:
            self.assertAlmostEqual(frame.root_pos[0], 0.0, places=4)
            self.assertAlmostEqual(frame.root_pos[2], 0.0, places=4)
        self.assertLess(_bone(parked.frames[0], "LeftUpperArm")[1], -0.85)
        self.assertGreater(_bone(parked.frames[0], "LeftLowerLeg")[2], 0.25)
        forward = min(range(len(parked.frames)), key=lambda index: _bone(parked.frames[index], "LeftUpperLeg")[2])
        self.assertLess(_bone(parked.frames[forward], "LeftUpperLeg")[2], -0.3)
        traveled = convert_bvh(bvh, ConvertOptions(in_place=False))
        end = traveled.frames[-1].root_pos
        self.assertAlmostEqual(end[0], 0.0, places=3)
        expected = -TRAVEL * traveled.scale
        self.assertAlmostEqual(end[2], expected, delta=0.05)
        self.assertLess(end[2], -2.0)
        self.assertFalse(any("away from the R15 rest" in line for line in parked.warnings))

    def test_unit_guess_recognizes_cmu_and_meters(self):
        self.assertIn("CMU", guess_unit(24.6))
        self.assertEqual(guess_unit(1.75), "meters")
