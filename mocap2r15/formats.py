"""Tell the user what to do with files this tool does not retarget directly.

BVH is the format the converter reads. C3D and FBX become BVH in Blender
(see ``blender/export_bvh.py``). AMASS ``.npz`` files store SMPL pose
parameters, not a skeleton, so they need a body model before they can be a BVH.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path
from typing import Dict, Tuple


def classify(path: str) -> str:
    """``bvh``, ``c3d``, ``fbx``, ``npz``, ``npy``, or ``unknown``."""
    suffix = Path(path).suffix.lower()
    return {
        ".bvh": "bvh",
        ".c3d": "c3d",
        ".fbx": "fbx",
        ".npz": "npz",
        ".npy": "npy",
    }.get(suffix, "unknown")


def foreign_report(path: str) -> str:
    """Human-readable explanation for a file that is not ready to retarget."""
    kind = classify(path)
    name = Path(path).name
    if kind == "bvh":
        return f"{name} is BVH. Run convert on it directly."
    if kind == "fbx":
        return (
            f"{name} is FBX. mocap2r15 reads BVH. Export a BVH from Blender:\n"
            f"  blender --background --python blender/export_bvh.py -- "
            f"--input \"{path}\" --output \"{Path(path).with_suffix('.bvh')}\"\n"
            "Then convert the BVH. The script keeps the armature's frame range."
        )
    if kind == "c3d":
        return (
            f"{name} is C3D (marker trajectories, not a skeleton animation). "
            "mocap2r15 does not read C3D. Convert it to BVH with a tool that "
            "solves a skeleton — the Blender C3D add-on plus blender/export_bvh.py, "
            "or the BVH export from the CMU database — and then convert that BVH."
        )
    if kind in ("npz", "npy"):
        return _array_report(path)
    return (
        f"{name} is not a format mocap2r15 reads. Export BVH (hierarchy plus "
        "Euler channels) and convert that. FBX can go through blender/export_bvh.py."
    )


def _array_report(path: str) -> str:
    try:
        arrays = read_array_headers(path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return (
            f"{Path(path).name} looks like a NumPy archive but could not be read ({exc}). "
            "AMASS files need a SMPL body before they become a skeleton. "
            "Export BVH from a tool that has the body model, then convert the BVH."
        )
    lines = [
        f"{Path(path).name} is a NumPy archive, not a skeleton.",
        "AMASS / SMPL stores axis-angle poses. Turning those into joint positions",
        "needs the body model (shaped betas and the kinematic tree). This tool",
        "does not ship SMPL. Export a BVH from Blender (or any SMPL viewer) and",
        "convert that. Arrays found:",
    ]
    if not arrays:
        lines.append("  (none)")
    for name, (shape, descr) in arrays.items():
        lines.append(f"  {name}: shape {shape} {descr}".rstrip())
    poses = arrays.get("poses")
    trans = arrays.get("trans")
    if poses is not None:
        lines.append(_pose_hint(poses[0]))
    if trans is not None:
        lines.append(f"trans has shape {trans[0]} (root translation per frame, once you have a skeleton).")
    return "\n".join(lines)


def _pose_hint(shape: Tuple[int, ...]) -> str:
    if len(shape) == 3 and shape[-1] == 3:
        return f"poses looks like {shape[0]} frames of {shape[1]} axis-angle joints."
    if len(shape) == 2 and shape[1] % 3 == 0:
        return f"poses looks like {shape[0]} frames of {shape[1] // 3} packed axis-angle joints."
    return f"poses has shape {shape}, which is not the usual AMASS (frames, joints*3) layout."


def read_array_headers(path: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Shape and dtype of each array in an ``.npy`` or ``.npz`` file.

    Only the header is read. The samples themselves are not loaded.
    """
    file_path = Path(path)
    if file_path.suffix.lower() == ".npy":
        shape, descr = npy_header(file_path.read_bytes())
        return {file_path.stem: (shape, descr)}
    found: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    with zipfile.ZipFile(file_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            shape, descr = npy_header(archive.read(name))
            key = name[:-4]
            if key.endswith("/"):
                continue
            found[key.split("/")[-1]] = (shape, descr)
    return found


def npy_header(data: bytes) -> Tuple[Tuple[int, ...], str]:
    """Parse a ``.npy`` preamble. Returns ``(shape, descr)``."""
    if len(data) < 10 or data[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy array")
    major = data[6]
    if major == 1:
        header_len = int.from_bytes(data[8:10], "little")
        start = 10
    elif major == 2:
        header_len = int.from_bytes(data[8:12], "little")
        start = 12
    else:
        raise ValueError(f"unsupported npy version {major}")
    header = data[start:start + header_len].decode("latin1").strip()
    meta = ast.literal_eval(header)
    shape = meta.get("shape", ())
    if isinstance(shape, int):
        shape = (shape,)
    return tuple(int(dim) for dim in shape), str(meta.get("descr", ""))
