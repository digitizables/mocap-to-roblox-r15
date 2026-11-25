"""Two-bone IK and a planted foot on a bent leg."""

import unittest

from mocap2r15.bvh import parse_bvh
from mocap2r15.clip import frames_from_bvh
from mocap2r15.footlock import _fk, apply_foot_lock, two_bone_ik
from mocap2r15.math3d import almost_vec, v_dist
from tests.factory import joint, render_bvh


class TwoBoneTests(unittest.TestCase):
    def test_lengths_stay_exact_inside_reach(self):
        hip = (0.0, 0.0, 0.0)
        target = (0.0, -0.8, 0.2)
        knee, ankle = two_bone_ik(hip, target, 0.5, 0.5, (0.0, -0.2, 1.0))
        self.assertAlmostEqual(v_dist(hip, knee), 0.5, places=5)
        self.assertAlmostEqual(v_dist(knee, ankle), 0.5, places=5)
        self.assertTrue(almost_vec(ankle, target, 1e-5))
        self.assertGreater(knee[2], 0.05)

    def test_unreachable_target_is_pulled_onto_the_ray(self):
        knee, ankle = two_bone_ik((0.0, 0.0, 0.0), (0.0, -3.0, 0.0), 0.5, 0.5, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(v_dist((0.0, 0.0, 0.0), knee), 0.5, places=4)
        self.assertAlmostEqual(v_dist(knee, ankle), 0.5, places=4)
        self.assertAlmostEqual(v_dist((0.0, 0.0, 0.0), ankle), 1.0, delta=1e-3)
        self.assertLess(ankle[1], -0.9)


class FootLockTests(unittest.TestCase):
    def test_bent_leg_contact_stops_sliding(self):
        root = joint(
            "Hips",
            (0.0, 0.0, 0.0),
            [
                joint(
                    "LeftUpLeg",
                    (0.0, 0.0, 0.0),
                    [joint("LeftLeg", (0.0, -0.5, 0.0), [joint("LeftFoot", (0.0, -0.5, 0.0), end=(0.0, -0.05, 0.12))])],
                )
            ],
        )
        frames_spec = []
        for index in range(12):
            frames_spec.append({"Hips": (index * 0.0015, 1.0, 0.0, 0.0, 0.0, 0.0), "LeftLeg": (0.0, 0.0, 35.0)})
        bvh = parse_bvh(render_bvh(root, frames_spec, frame_time=1.0 / 30.0))
        frames = frames_from_bvh(bvh)
        before = [sample[0]["LeftFoot"][0] for sample in (_fk(bvh, frame) for frame in frames)]
        warnings = apply_foot_lock(bvh, frames, [("LeftUpLeg", "LeftLeg", "LeftFoot")], blend_frames=2)
        self.assertFalse(any("no contact" in line for line in warnings), warnings)
        after = []
        for frame in frames:
            positions, _ = _fk(bvh, frame)
            after.append(positions["LeftFoot"][0])
            self.assertAlmostEqual(v_dist(positions["LeftUpLeg"], positions["LeftLeg"]), 0.5, delta=0.01)
            self.assertAlmostEqual(v_dist(positions["LeftLeg"], positions["LeftFoot"]), 0.5, delta=0.01)
        before_span = max(before) - min(before)
        # Fully weighted frames sit in the middle of the contact run.
        planted = after[4:8]
        planted_span = max(planted) - min(planted)
        self.assertGreater(before_span, 0.01)
        self.assertLess(planted_span, before_span * 0.25)
