"""BVH parser: hierarchy, channel order, wrapped frames, and forward kinematics."""

import math
import unittest

from mocap2r15.bvh import BVHError, fk_frame, parse_bvh
from mocap2r15.math3d import almost_vec, quat_rotate


MINI = """
HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Chest
    {
        OFFSET 0 10 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0 5 0
        }
    }
    JOINT LeftUpLeg
    {
        OFFSET 2 -1 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {
            OFFSET 0 -8 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {
                OFFSET 0 -8 0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftToeBase
                {
                    OFFSET 0 -1 3
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site { OFFSET 0 0 1 }
                }
            }
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.04
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 2 3 0 0 0 0 90 0 10 0 0 0 0 0 0 0 0 0 0 0
"""


class ParserTests(unittest.TestCase):
    def test_counts_channels_and_end_site(self):
        bvh = parse_bvh(MINI, path="mini.bvh")
        self.assertEqual(bvh.root, "Hips")
        self.assertEqual(len(bvh), 2)
        self.assertAlmostEqual(bvh.frame_time, 0.04)
        self.assertAlmostEqual(bvh.fps, 25.0)
        self.assertEqual(bvh.joints["Chest"].rotation_order, "ZXY")
        self.assertTrue(almost_vec(bvh.joints["Chest"].end_offset, (0, 5, 0)))
        self.assertEqual(bvh.joints["LeftUpLeg"].parent, "Hips")
        self.assertIn("LeftLeg", bvh.joints["LeftUpLeg"].children)

    def test_root_position_channels_replace_offset(self):
        bvh = parse_bvh(MINI)
        pos = bvh.translation("Hips", bvh.frames[1])
        self.assertTrue(almost_vec(pos, (1, 2, 3)))

    def test_child_rotation_uses_its_channel_order(self):
        bvh = parse_bvh(MINI)
        # Chest channels are Z, X, Y and the second frame stores 0, 90, 0,
        # so this is a +90° rotation about X. +Y goes to +Z? 
        # Rx(90) @ (0, 1, 0) = (0, 0, 1).
        quat = bvh.rotation_quat("Chest", bvh.frames[1])
        self.assertTrue(almost_vec(quat_rotate(quat, (0, 1, 0)), (0, 0, 1), 1e-6))

    def test_fk_places_chest_above_hips(self):
        bvh = parse_bvh(MINI)
        positions, _ = fk_frame(bvh, bvh.frames[0])
        self.assertTrue(almost_vec(positions["Hips"], (0, 0, 0)))
        self.assertTrue(almost_vec(positions["Chest"], (0, 10, 0)))
        self.assertTrue(almost_vec(positions["LeftLeg"], (2, -1 - 8, 0)))

    def test_bind_height_includes_end_site(self):
        bvh = parse_bvh(MINI)
        # Lowest toe tip is around y = -1-8-8-1 = -18, plus end site y 0.
        # Highest is chest 10 + end 5 = 15. Height = 15 - (-18) = 33.
        self.assertAlmostEqual(bvh.skeleton_height(), 33.0, places=5)

    def test_toe_offset_guesses_forward_plus_z(self):
        bvh = parse_bvh(MINI)
        self.assertEqual(bvh.forward_hint(), "z")

    def test_same_line_braces_and_wrapped_motion(self):
        text = """HIERARCHY
ROOT Hips {
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Spine { OFFSET 0 1 0 CHANNELS 3 Zrotation Yrotation Xrotation
End Site { OFFSET 0 1 0 }
}
}
MOTION
Frames: 1
Frame Time: .033333
0 1 2
0 0 0
10 0 0
"""
        bvh = parse_bvh(text)
        self.assertEqual(bvh.joints["Hips"].rotation_order, "ZYX")
        self.assertEqual(bvh.joints["Spine"].rotation_order, "ZYX")
        self.assertEqual(len(bvh), 1)
        self.assertTrue(almost_vec(bvh.translation("Hips", bvh.frames[0]), (0, 1, 2)))
        quat = bvh.rotation_quat("Spine", bvh.frames[0])
        # Z=10°, Y=0, X=0. Rz(10) sends +X toward +Y.
        turned = quat_rotate(quat, (1, 0, 0))
        self.assertAlmostEqual(turned[1], math.sin(math.radians(10)), places=5)

    def test_frame_count_mismatch_is_an_error(self):
        text = MINI.replace("Frames: 2", "Frames: 3")
        with self.assertRaises(BVHError):
            parse_bvh(text)

    def test_missing_hierarchy_is_an_error(self):
        with self.assertRaises(BVHError):
            parse_bvh("MOTION\nFrames: 0\nFrame Time: 0.1\n")

    def test_duplicate_joint_names_are_made_unique(self):
        text = """HIERARCHY
ROOT Hips {
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Arm { OFFSET 1 0 0 CHANNELS 3 Zrotation Xrotation Yrotation End Site { OFFSET 1 0 0 } }
JOINT Arm { OFFSET -1 0 0 CHANNELS 3 Zrotation Xrotation Yrotation End Site { OFFSET -1 0 0 } }
}
MOTION
Frames: 1
Frame Time: 0.1
0 0 0 0 0 0 0 0 0 0 0 0
"""
        bvh = parse_bvh(text)
        self.assertIn("Arm", bvh.joints)
        self.assertIn("Arm.2", bvh.joints)

    def test_find_ignores_mixamo_prefix_and_case(self):
        text = """HIERARCHY
ROOT mixamorig:Hips {
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT mixamorig:LeftUpLeg { OFFSET 1 -1 0 CHANNELS 3 Zrotation Xrotation Yrotation
End Site { OFFSET 0 -1 0 } }
}
MOTION
Frames: 1
Frame Time: 0.1
0 0 0 0 0 0 0 0 0
"""
        bvh = parse_bvh(text)
        self.assertEqual(bvh.find("Hips"), "mixamorig:Hips")
        self.assertEqual(bvh.find("leftupleg"), "mixamorig:LeftUpLeg")
        self.assertIsNone(bvh.find("Spine"))


if __name__ == "__main__":
    unittest.main()
