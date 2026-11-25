"""The clips checked into examples/ stay in sync with the generator."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from mocap2r15.r15 import PART_ORDER
from mocap2r15.sample_walk import build_bvh_text

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _load(name):
    text = (EXAMPLES / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    return ET.fromstring(text)


def _prop(item, name):
    for child in item.find("Properties"):
        if child.attrib.get("name") == name:
            return child.text
    return None


class ExampleTests(unittest.TestCase):
    def test_synthetic_bvh_matches_the_generator(self):
        text = (EXAMPLES / "synthetic_walk.bvh").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(text, build_bvh_text())

    def test_rbxmx_examples_are_r15_clips(self):
        synthetic = _load("synthetic_walk.rbxmx")
        cmu = _load("cmu_07_01_walk.rbxmx")
        for document, frames, name in (
            (synthetic, 30, "synthetic_walk"),
            (cmu, 33, "CMU 07_01 walk"),
        ):
            self.assertEqual(document.tag, "roblox")
            sequence = document.find("Item")
            self.assertEqual(sequence.attrib["class"], "KeyframeSequence")
            self.assertEqual(_prop(sequence, "Name"), name)
            self.assertEqual(_prop(sequence, "Loop"), "true")
            self.assertEqual(_prop(sequence, "Priority"), "1")
            keyframes = [item for item in sequence if item.attrib.get("class") == "Keyframe"]
            self.assertEqual(len(keyframes), frames)
            poses = [item for item in keyframes[0].iter("Item") if item.attrib.get("class") == "Pose"]
            self.assertEqual([_prop(pose, "Name") for pose in poses], PART_ORDER)
