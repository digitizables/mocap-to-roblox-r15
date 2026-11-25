"""KeyframeSequence XML matches what Studio's Insert from File expects."""

import unittest
import xml.etree.ElementTree as ET

from mocap2r15.math3d import QUAT_IDENTITY, quat_from_axis_angle
from mocap2r15.r15 import PART_ORDER
from mocap2r15.rbxmx import render_rbxmx, write_rbxmx
from mocap2r15.retarget import R15Frame


def _frames():
    local = {part: QUAT_IDENTITY for part in PART_ORDER}
    local["LeftUpperLeg"] = quat_from_axis_angle((1.0, 0.0, 0.0), 0.4)
    return [
        R15Frame(0.0, dict(local), (1.5, 2.25, -3.5)),
        R15Frame(0.5, {part: QUAT_IDENTITY for part in PART_ORDER}, (0.0, 0.1, 0.0)),
    ]


def _prop(item, name):
    props = item.find("Properties")
    for child in list(props):
        if child.attrib.get("name") == name:
            return child
    return None


class RbxmxTests(unittest.TestCase):
    def test_document_has_fifteen_nested_poses(self):
        text = render_rbxmx(_frames(), name='Walk & "turn"', loop=True, priority="Movement")
        self.assertIn("&amp;", text)
        self.assertIn("&quot;", text)
        root = ET.fromstring(text)
        self.assertEqual(root.tag, "roblox")
        self.assertEqual(root.attrib["version"], "4")
        sequence = root.find("Item")
        self.assertEqual(sequence.attrib["class"], "KeyframeSequence")
        self.assertEqual(_prop(sequence, "Loop").text, "true")
        self.assertEqual(_prop(sequence, "Priority").text, "1")
        self.assertEqual(_prop(sequence, "Name").text, 'Walk & "turn"')
        keyframes = [item for item in sequence if item.attrib.get("class") == "Keyframe"]
        self.assertEqual(len(keyframes), 2)
        times = [float(_prop(frame, "Time").text) for frame in keyframes]
        self.assertEqual(times, [0.0, 0.5])
        for frame in keyframes:
            poses = [item for item in frame.iter("Item") if item.attrib.get("class") == "Pose"]
            names = [_prop(pose, "Name").text for pose in poses]
            self.assertEqual(names, PART_ORDER)
            root_pose = poses[0]
            self.assertEqual(_prop(root_pose, "Name").text, "HumanoidRootPart")
            children = [item for item in list(root_pose) if item.attrib.get("class") == "Pose"]
            self.assertEqual(_prop(children[0], "Name").text, "LowerTorso")
        hrp = [item for item in keyframes[0].iter("Item") if item.attrib.get("class") == "Pose"][0]
        cframe = _prop(hrp, "CFrame")
        self.assertEqual(cframe.find("X").text, "1.5")
        self.assertEqual(cframe.find("Y").text, "2.25")
        self.assertEqual(cframe.find("Z").text, "-3.5")
        self.assertEqual(cframe.find("R00").text, "1")
        self.assertEqual(cframe.find("R11").text, "1")
        self.assertEqual(cframe.find("R22").text, "1")

    def test_lower_torso_can_carry_translation(self):
        text = render_rbxmx(_frames(), name="Walk", loop=False, priority="Action", root_bone="lower-torso")
        root = ET.fromstring(text)
        sequence = root.find("Item")
        self.assertEqual(_prop(sequence, "Priority").text, "2")
        self.assertEqual(_prop(sequence, "Loop").text, "false")
        hrp = [item for item in sequence.iter("Item") if item.attrib.get("class") == "Pose"][0]
        self.assertEqual(_prop(hrp, "CFrame").find("X").text, "0")
        lower = [item for item in list(hrp) if item.attrib.get("class") == "Pose"][0]
        self.assertEqual(_prop(lower, "CFrame").find("X").text, "1.5")

    def test_unknown_priority_is_rejected(self):
        with self.assertRaises(ValueError):
            render_rbxmx(_frames(), name="Walk", loop=False, priority="Sometimes")

    def test_write_roundtrip_parses(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "walk.rbxmx"
            write_rbxmx(_frames(), str(path), name="Walk", loop=True, priority="Core")
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertNotIn("\r", text)
            root = ET.fromstring(text)
            self.assertEqual(_prop(root.find("Item"), "Priority").text, "1000")
