"""Trim, resample, loop blend, and keyframe reduction."""

import math
import unittest

from mocap2r15.cleanup import drop_matching_end, loop_blend, reduce_frames, resample_frames, trim_frames
from mocap2r15.clip import SourceFrame
from mocap2r15.math3d import QUAT_IDENTITY, VEC_Y, quat_between_angle, quat_from_axis_angle
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
