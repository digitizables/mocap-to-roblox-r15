"""CLI exit codes, implied convert, batch, and inspect."""

import json
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from mocap2r15.cli import main
from tests.factory import cmu_skeleton, render_bvh


def _bvh(path: Path, rows=None) -> None:
    path.write_text(render_bvh(cmu_skeleton(), rows or [{}, {}]), encoding="utf-8", newline="\n")


class CliTests(unittest.TestCase):
    def test_implied_convert_writes_rbxmx(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            src = folder / "walk.bvh"
            _bvh(src)
            code = main([str(src), "--fps", "10", "--loop"])
            self.assertEqual(code, 0)
            text = (folder / "walk.rbxmx").read_text(encoding="utf-8")
            root = ET.fromstring(text)
            self.assertEqual(root.tag, "roblox")
            sequence = root.find("Item")
            props = {child.attrib["name"]: child.text for child in sequence.find("Properties")}
            self.assertEqual(props["Loop"], "true")
            self.assertEqual(props["Name"], "walk")

    def test_bad_arguments_exit_2_and_bad_bvh_exits_1(self):
        self.assertEqual(main(["--not-a-flag"]), 2)
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "broken.bvh"
            src.write_text("this is not bvh\n", encoding="utf-8")
            self.assertEqual(main([str(src)]), 1)

    def test_batch_continues_after_one_failure(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trials"
            folder.mkdir()
            _bvh(folder / "good.bvh")
            (folder / "bad.bvh").write_text("nope\n", encoding="utf-8")
            (folder / "notes.txt").write_text("ignore me\n", encoding="utf-8")
            out = Path(tmp) / "out"
            code = main(["batch", str(folder), "-o", str(out), "--meta"])
            self.assertEqual(code, 1)
            self.assertTrue((out / "good.rbxmx").is_file())
            self.assertFalse((out / "bad.rbxmx").exists())
            meta = json.loads((out / "good.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["preset"], "cmu")
            self.assertGreater(meta["output_frames"], 0)

    def test_inspect_and_foreign_files(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "walk.bvh"
            _bvh(src)
            code = main(["inspect", str(src)])
            self.assertEqual(code, 0)
            fbx = Path(tmp) / "clip.fbx"
            fbx.write_bytes(b"Kaydara FBX Binary  \x00")
            self.assertEqual(main(["convert", str(fbx)]), 1)
            self.assertEqual(main(["inspect", str(fbx)]), 0)

    def test_module_help(self):
        completed = subprocess.run(
            [sys.executable, "-m", "mocap2r15", "--help"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("convert", completed.stdout)
        self.assertIn("batch", completed.stdout)
