"""Joint maps: named presets, generic aliases, and bone direction."""

import unittest

from mocap2r15.bvh import parse_bvh
from mocap2r15.math3d import almost_vec
from mocap2r15.presets import detect_preset, resolve_mapping
from tests.factory import cmu_skeleton, joint, render_bvh


def _parse(root, frames=None):
    return parse_bvh(render_bvh(root, frames or [{}]))


class PresetDetectionTests(unittest.TestCase):
    def test_cmu_names_select_cmu_and_aim_the_spine_up(self):
        bvh = _parse(cmu_skeleton())
        chosen, ratios = detect_preset(bvh)
        self.assertEqual(chosen, "cmu")
        self.assertGreater(ratios["cmu"], ratios["mixamo"])
        mapping = resolve_mapping(bvh, "cmu")
        spine = mapping.parts["UpperTorso"].src_dir
        # The clavicle offset is longer than the neck. The torso must still
        # aim at the head, not down the arm.
        self.assertGreater(spine[1], 0.9)
        self.assertLess(abs(spine[0]), 0.1)
        arm = mapping.parts["LeftUpperArm"].src_dir
        self.assertGreater(arm[0], 0.9)
        foot = mapping.parts["LeftFoot"].src_dir
        self.assertGreater(foot[2], 0.9)
        thigh = mapping.parts["LeftUpperLeg"].src_dir
        self.assertLess(thigh[1], -0.9)

    def test_zero_length_neck_is_skipped(self):
        bvh = _parse(cmu_skeleton())
        mapping = resolve_mapping(bvh, "cmu")
        # Neck sits on Spine1. The head chain still resolves, and the torso
        # direction comes from Neck1 rather than the zero offset.
        self.assertIn("Neck", mapping.parts["Head"].joints)
        self.assertGreater(mapping.parts["UpperTorso"].src_dir[1], 0.9)
        self.assertTrue(mapping.parts["Head"].direction_from_skeleton)

    def test_mixamo_prefix_selects_mixamo(self):
        root = joint(
            "mixamorig:Hips",
            (0, 0, 0),
            [
                joint(
                    "mixamorig:Spine",
                    (0, 1, 0),
                    [
                        joint(
                            "mixamorig:Spine1",
                            (0, 1, 0),
                            [
                                joint(
                                    "mixamorig:Spine2",
                                    (0, 1, 0),
                                    [
                                        joint("mixamorig:Neck", (0, 1, 0), [joint("mixamorig:Head", (0, 1, 0), end=(0, 1, 0))]),
                                        joint(
                                            "mixamorig:LeftArm",
                                            (1, 0, 0),
                                            [joint("mixamorig:LeftForeArm", (2, 0, 0), [joint("mixamorig:LeftHand", (2, 0, 0), end=(1, 0, 0))])],
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                joint(
                    "mixamorig:LeftUpLeg",
                    (1, -1, 0),
                    [joint("mixamorig:LeftLeg", (0, -3, 0), [joint("mixamorig:LeftFoot", (0, -3, 0), [joint("mixamorig:LeftToe", (0, -0.2, 1), end=(0, 0, 1))])])],
                ),
            ],
        )
        mapping = resolve_mapping(_parse(root), "auto")
        self.assertEqual(mapping.preset, "mixamo")
        self.assertIn("mixamorig:LeftUpLeg", mapping.parts["LeftUpperLeg"].joints)
        self.assertIn("mixamorig:LeftLeg", mapping.parts["LeftLowerLeg"].joints)

    def test_soma_leftleg_is_the_thigh(self):
        root = joint(
            "Root",
            (0, 0, 0),
            [
                joint(
                    "Hips",
                    (0, 1, 0),
                    [
                        joint(
                            "Spine1",
                            (0, 1, 0),
                            [
                                joint(
                                    "Spine2",
                                    (0, 1, 0),
                                    [
                                        joint(
                                            "Chest",
                                            (0, 1, 0),
                                            [
                                                joint("Neck1", (0, 1, 0), [joint("Neck2", (0, 0.4, 0), [joint("Head", (0, 0.5, 0), end=(0, 0.4, 0))])]),
                                                joint("LeftShoulder", (0.4, 0, 0), [joint("LeftArm", (0.2, 0, 0), [joint("LeftForeArm", (2, 0, 0), [joint("LeftHand", (2, 0, 0), end=(0.5, 0, 0))])])]),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                        joint("LeftLeg", (0.5, -0.5, 0), [joint("LeftShin", (0, -3, 0), [joint("LeftFoot", (0, -3, 0), end=(0, -0.2, 1))])]),
                        joint("RightLeg", (-0.5, -0.5, 0), [joint("RightShin", (0, -3, 0), [joint("RightFoot", (0, -3, 0), end=(0, -0.2, 1))])]),
                    ],
                )
            ],
        )
        bvh = _parse(root)
        mapping = resolve_mapping(bvh, "auto")
        self.assertEqual(mapping.preset, "soma")
        self.assertEqual(mapping.parts["LeftUpperLeg"].joints[-1], "LeftLeg")
        self.assertEqual(mapping.parts["LeftLowerLeg"].joints[-1], "LeftShin")
        self.assertEqual(mapping.parts["LowerTorso"].joints[0], "Root")
        # The generic table treats "leftleg" as a shin. Forcing it must not
        # be how a SOMA file is read.
        generic = resolve_mapping(bvh, "generic")
        self.assertIn("LeftLeg", generic.parts["LeftLowerLeg"].joints)
        self.assertNotIn("LeftLeg", generic.parts["LeftUpperLeg"].joints)

    def test_generic_aliases_do_not_substring_match(self):
        root = joint(
            "Pelvis",
            (0, 0, 0),
            [
                joint("SpineExtra", (0, 2, 0), [joint("Head", (0, 2, 0), end=(0, 1, 0))]),
                joint("LeftThigh", (1, -1, 0), [joint("LeftShin", (0, -3, 0), [joint("LeftFoot", (0, -3, 0), end=(0, 0, 1))])]),
            ],
        )
        mapping = resolve_mapping(_parse(root), "generic")
        self.assertEqual(mapping.preset, "generic")
        self.assertEqual(mapping.parts["LowerTorso"].joints, ["Pelvis"])
        self.assertEqual(mapping.parts["UpperTorso"].joints, [])
        self.assertEqual(mapping.parts["LeftUpperLeg"].joints, ["LeftThigh"])
        self.assertEqual(mapping.parts["LeftLowerLeg"].joints, ["LeftShin"])
        text = " ".join(mapping.warnings)
        self.assertIn("UpperTorso", text)

    def test_low_confidence_falls_back_to_generic(self):
        root = joint("Hips", (0, 0, 0), [joint("Head", (0, 5, 0), end=(0, 1, 0))])
        mapping = resolve_mapping(_parse(root), "auto")
        self.assertEqual(mapping.preset, "generic")
        self.assertTrue(any("generic" in warning for warning in mapping.warnings))

    def test_missing_joint_warns_and_wrapper_root_is_kept(self):
        root = joint(
            "Root",
            (0, 0, 0),
            [
                joint(
                    "Hips",
                    (0, 1, 0),
                    [
                        joint("LowerBack", (0, 1, 0), [joint("Spine", (0, 1, 0), [joint("Spine1", (0, 1, 0), end=(0, 1, 0))])]),
                        joint("LeftUpLeg", (1, -1, 0), [joint("LeftLeg", (0, -3, 0), [joint("LeftFoot", (0, -3, 0), end=(0, 0, 1))])]),
                    ],
                )
            ],
        )
        mapping = resolve_mapping(_parse(root), "cmu")
        self.assertEqual(mapping.parts["LowerTorso"].joints[0], "Root")
        self.assertIn("Hips", mapping.parts["LowerTorso"].joints)
        self.assertEqual(mapping.parts["Head"].joints, [])
        self.assertTrue(any("Head" in warning for warning in mapping.warnings))

    def test_unknown_preset_raises(self):
        bvh = _parse(joint("Hips", (0, 0, 0), end=(0, 1, 0)))
        with self.assertRaises(ValueError):
            resolve_mapping(bvh, "not-a-skeleton")

    def test_directions_are_unit_length(self):
        mapping = resolve_mapping(_parse(cmu_skeleton()), "cmu")
        for spec in mapping.parts.values():
            length = sum(component * component for component in spec.src_dir) ** 0.5
            self.assertAlmostEqual(length, 1.0, places=5)
            self.assertTrue(almost_vec(spec.tgt_dir, spec.tgt_dir))
