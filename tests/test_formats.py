"""Foreign-format notes, including a header-only look at AMASS npz files."""

import struct
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from mocap2r15.formats import classify, foreign_report, npy_header, read_array_headers


def _npy(shape, descr="<f4"):
    header = str({"descr": descr, "fortran_order": False, "shape": shape, })
    raw = header.encode("latin1")
    pad = (16 - ((10 + len(raw) + 1) % 16)) % 16
    raw = raw + b" " * pad + b"\n"
    body = b"\x00" * (4 * (1 if not shape else __import__("math").prod(shape)))
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(raw)) + raw + body


class FormatTests(unittest.TestCase):
    def test_suffixes(self):
        self.assertEqual(classify("walk.BVH"), "bvh")
        self.assertEqual(classify("trial.c3d"), "c3d")
        self.assertEqual(classify("clip.fbx"), "fbx")
        self.assertEqual(classify("amass.npz"), "npz")

    def test_fbx_and_c3d_explain_the_blender_step(self):
        fbx = foreign_report("clips/walk.fbx")
        self.assertIn("blender/export_bvh.py", fbx)
        self.assertIn("walk.bvh", fbx)
        c3d = foreign_report("trial.c3d")
        self.assertIn("C3D", c3d)
        self.assertIn("BVH", c3d)

    def test_npz_header_reports_pose_shape_without_loading_samples(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "walk.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("poses.npy", _npy((4, 72)))
                archive.writestr("trans.npy", _npy((4, 3)))
            arrays = read_array_headers(str(path))
            self.assertEqual(arrays["poses"][0], (4, 72))
            self.assertEqual(arrays["trans"][0], (4, 3))
            report = foreign_report(str(path))
            self.assertIn("SMPL", report)
            self.assertIn("24 packed", report)
            shape, descr = npy_header(_npy((2, 22, 3), "<f8"))
            self.assertEqual(shape, (2, 22, 3))
            self.assertEqual(descr, "<f8")
