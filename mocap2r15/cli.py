"""Command line: convert, batch, inspect, sample.

``python -m mocap2r15 walk.bvh`` is the same as ``convert``. Exit status is
0 on success, 1 when a clip cannot be converted, and 2 when the arguments
themselves are wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mocap2r15 import __version__
from mocap2r15.errors import MocapError
from mocap2r15.formats import classify, foreign_report
from mocap2r15.pipeline import ConvertOptions, ConvertResult, convert_path, inspect_text
from mocap2r15.presets import PRESETS, preset_names
from mocap2r15.rbxmx import write_rbxmx
from mocap2r15.sample_walk import write_bvh


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args_list = list(argv)
    if not args_list:
        args_list = ["--help"]
    commands = {"convert", "batch", "inspect", "sample", "presets", "-h", "--help"}
    if args_list[0] not in commands and not args_list[0].startswith("-"):
        args_list = ["convert", *args_list]
    parser = build_parser()
    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        code = exc.code
        if code in (0, None):
            return 0
        return int(code)
    try:
        return int(args.handler(args))
    except MocapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mocap2r15",
        description="Retarget a locomotion BVH onto a Roblox R15 KeyframeSequence.",
    )
    parser.add_argument("--version", action="version", version=f"mocap2r15 {__version__}")
    sub = parser.add_subparsers(dest="command")

    convert = sub.add_parser("convert", help="Convert one or more BVH files to .rbxmx")
    convert.add_argument("inputs", nargs="+", help="BVH files")
    _add_output(convert)
    _add_convert_flags(convert)
    convert.set_defaults(handler=_cmd_convert)

    batch = sub.add_parser("batch", help="Convert every BVH in a folder")
    batch.add_argument("folder", help="Directory of .bvh trials")
    _add_output(batch)
    batch.add_argument("--recursive", action="store_true", help="Include subfolders")
    _add_convert_flags(batch)
    batch.set_defaults(handler=_cmd_batch)

    inspect = sub.add_parser("inspect", help="Print skeleton, fps, preset, and scale hints")
    inspect.add_argument("inputs", nargs="+")
    inspect.set_defaults(handler=_cmd_inspect)

    sample = sub.add_parser("sample", help="Write a synthetic one-second walk BVH")
    sample.add_argument("-o", "--output", default="synthetic_walk.bvh", help="Where to write the BVH")
    sample.add_argument("--rbxmx", default=None, help="Also write an in-place R15 animation to this path")
    sample.set_defaults(handler=_cmd_sample)

    presets = sub.add_parser("presets", help="List built-in skeleton presets")
    presets.set_defaults(handler=_cmd_presets)
    return parser


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .rbxmx file, or a directory when converting more than one clip. "
        "The default is the input name with a .rbxmx suffix.",
    )
    parser.add_argument("--meta", action="store_true", help="Write a .meta.json sidecar next to each animation")


def _add_convert_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default="auto", choices=preset_names(), help="Skeleton map (default: auto)")
    parser.add_argument("--forward-axis", default="auto", help="Source facing axis: auto, z, -z, x, -x")
    parser.add_argument("--unit", default="auto", help="File units: auto, m, cm, mm, in, cmu, raw")
    parser.add_argument("--target-height", type=float, default=5.0, help="Studs to fit the bind pose to when --unit auto (default 5)")
    parser.add_argument("--studs-per-meter", type=float, default=3.571, help="Used with m/cm/mm/in/cmu (default 3.571, about 28 cm per stud)")
    parser.add_argument("--studs-per-unit", type=float, default=None, help="Override scale: studs per file unit")
    parser.add_argument("--root-motion", action="store_true", help="Keep the traveled path. The default is an in-place cycle")
    parser.add_argument("--root-bone", default="humanoid", choices=("humanoid", "lower-torso"), help="Which pose carries root translation")
    parser.add_argument("--yaw-on-root", action="store_true", help="Put hip yaw on HumanoidRootPart and leave pitch/roll on LowerTorso")
    parser.add_argument(
        "--upright-torso",
        action="store_true",
        help="Remove sideways lean and chest pitch. LowerTorso keeps its yaw, the head keeps its nod and turn, and the limbs keep their swing.",
    )
    parser.add_argument(
        "--mirror-left",
        action="store_true",
        help="Copy the left limbs onto the right, reflected across X, half a cycle later so a walk stays out of phase.",
    )
    parser.add_argument(
        "--mirror-phase",
        type=float,
        default=0.5,
        help="Fraction of the clip to delay the copied side (default 0.5). 0 copies the same frame and steps both feet together.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Resample to this frame rate. The default keeps the source rate")
    parser.add_argument("--reduce-degrees", type=float, default=0.0, help="Drop keys a slerp can rebuild within this many degrees (0 keeps every frame)")
    parser.add_argument("--reduce-position", type=float, default=0.05, help="Root-position error allowed while thinning, in studs")
    parser.add_argument("--loop", action="store_true", help="Mark the clip looping and drop a last frame that repeats the first")
    parser.add_argument("--loop-blend", type=int, default=0, help="Frames at the end to blend back toward the first pose")
    parser.add_argument("--foot-lock", action="store_true", help="Plant feet during detected ground contact")
    parser.add_argument("--foot-blend", type=int, default=2, help="Contact frames to ease in and out of a plant")
    parser.add_argument("--start", type=float, default=None, help="Trim start, in seconds")
    parser.add_argument("--end", type=float, default=None, help="Trim end, in seconds")
    parser.add_argument("--priority", default="Movement", help="KeyframeSequence priority (default Movement)")
    parser.add_argument("--name", default=None, help="KeyframeSequence name. The default is the file stem")
    parser.add_argument(
        "--rest-frame",
        type=int,
        default=None,
        help="Frame index (after trim and resample) that should become the R15 rest pose. "
        "Do not point this at a T-pose or the arms will be baked outstretched.",
    )


def _options_from(args: argparse.Namespace, name: Optional[str]) -> ConvertOptions:
    return ConvertOptions(
        preset=args.preset,
        forward_axis=args.forward_axis,
        unit=args.unit,
        target_height=args.target_height,
        studs_per_meter=args.studs_per_meter,
        studs_per_unit=args.studs_per_unit,
        in_place=not args.root_motion,
        root_bone=args.root_bone,
        yaw_on_root=args.yaw_on_root,
        upright_torso=args.upright_torso,
        mirror_left=args.mirror_left,
        mirror_phase=args.mirror_phase,
        fps=args.fps,
        reduce_degrees=args.reduce_degrees,
        reduce_position=args.reduce_position,
        loop=args.loop,
        loop_blend=args.loop_blend,
        foot_lock=args.foot_lock,
        foot_blend=args.foot_blend,
        start=args.start,
        end=args.end,
        priority=args.priority,
        name=name,
        rest_frame=args.rest_frame,
    )


def _cmd_convert(args: argparse.Namespace) -> int:
    pairs = _destinations(args.inputs, args.output)
    failures = 0
    for src, dest in pairs:
        if not _convert_one(src, dest, args):
            failures += 1
    if failures and len(pairs) > 1:
        print(f"{failures} of {len(pairs)} clips failed", file=sys.stderr)
    return 1 if failures else 0


def _cmd_batch(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        raise MocapError(f"{folder} is not a directory")
    files = _bvh_in(folder, args.recursive)
    if not files:
        raise MocapError(f"No .bvh files in {folder}")
    output = args.output
    if output is None:
        output = str(folder / "r15")
    args.inputs = [str(path) for path in files]
    args.output = output
    return _cmd_convert(args)


def _cmd_inspect(args: argparse.Namespace) -> int:
    failures = 0
    for raw in args.inputs:
        kind = classify(raw)
        if kind != "bvh":
            print(foreign_report(raw))
            print("")
            if kind == "unknown" and not Path(raw).exists():
                failures += 1
            continue
        try:
            print(inspect_text(raw), end="")
        except MocapError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
        print("")
    return 1 if failures else 0


def _cmd_sample(args: argparse.Namespace) -> int:
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_bvh(str(destination))
    print(f"wrote {destination}")
    if args.rbxmx:
        rbx = Path(args.rbxmx)
        rbx.parent.mkdir(parents=True, exist_ok=True)
        # Reuse convert with the default in-place options.
        namespace = argparse.Namespace(
            preset="auto",
            forward_axis="auto",
            unit="auto",
            target_height=5.0,
            studs_per_meter=3.571,
            studs_per_unit=None,
            root_motion=False,
            root_bone="humanoid",
            yaw_on_root=False,
            upright_torso=False,
            mirror_left=False,
            mirror_phase=0.5,
            fps=None,
            reduce_degrees=0.0,
            reduce_position=0.05,
            loop=True,
            loop_blend=4,
            foot_lock=False,
            foot_blend=2,
            start=None,
            end=None,
            priority="Movement",
            rest_frame=None,
            meta=False,
            name=destination.stem,
        )
        if not _convert_one(destination, rbx, namespace):
            return 1
    return 0


def _cmd_presets(_args: argparse.Namespace) -> int:
    print("auto     pick the named preset with the most joints present")
    print("generic  alias table (pelvis, leftthigh, leftshin, ...)")
    for name, spec in PRESETS.items():
        print(f"{name:<8} {spec['description']}")
    return 0


def _convert_one(src: Path, dest: Path, args: argparse.Namespace) -> bool:
    kind = classify(str(src))
    if kind != "bvh":
        print(foreign_report(str(src)), file=sys.stderr)
        return False
    name = args.name or src.stem
    try:
        result = convert_path(str(src), _options_from(args, name))
    except MocapError as exc:
        print(f"error: {src}: {exc}", file=sys.stderr)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_rbxmx(
        result.frames,
        str(dest),
        name=result.name,
        loop=args.loop,
        priority=args.priority,
        root_bone=args.root_bone,
    )
    _report(src, dest, result)
    if getattr(args, "meta", False):
        _write_meta(src, dest, result, args)
    return True


def _report(src: Path, dest: Path, result: ConvertResult) -> None:
    print(
        f"wrote {dest}  {result.output_frames} frames  {result.duration:.3f}s  "
        f"{result.preset}  facing {result.forward_axis}  "
        f"scale {result.scale:.4g} ({result.unit_note})"
    )
    for line in result.warnings:
        if line.startswith("auto-selected"):
            print(line)
        else:
            print(f"warning: {line}", file=sys.stderr)


def _write_meta(src: Path, dest: Path, result: ConvertResult, args: argparse.Namespace) -> None:
    payload = {
        "source": str(src),
        "output": str(dest),
        "preset": result.preset,
        "forward_axis": result.forward_axis,
        "scale": result.scale,
        "unit_note": result.unit_note,
        "skeleton_height": result.skeleton_height,
        "fps": result.fps,
        "duration": result.duration,
        "source_frames": result.source_frames,
        "output_frames": result.output_frames,
        "in_place": not args.root_motion,
        "loop": args.loop,
        "warnings": result.warnings,
    }
    meta_path = dest.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _destinations(inputs: Sequence[str], output: Optional[str]) -> List[Tuple[Path, Path]]:
    sources = [Path(item) for item in inputs]
    if output is None:
        return [(src, src.with_suffix(".rbxmx")) for src in sources]
    destination = Path(output)
    if destination.suffix.lower() == ".rbxmx":
        if len(sources) != 1:
            raise MocapError("An .rbxmx output path only works with one input. Pass a directory for a batch.")
        return [(sources[0], destination)]
    return [(src, destination / f"{src.stem}.rbxmx") for src in sources]


def _bvh_in(folder: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    found: Dict[str, Path] = {}
    for path in folder.glob(pattern):
        if path.is_file() and path.suffix.lower() == ".bvh":
            found[str(path.resolve())] = path
    return [found[key] for key in sorted(found)]
