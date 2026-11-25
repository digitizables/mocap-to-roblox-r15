"""Convert one BVH clip into R15 frames.

Order of operations:

1. Trim, then plant the feet on the source clip (IK needs the original channels).
2. Resample.
3. Retarget rotations and turn root translation into studs. The origin is
   frame 0, or ``--rest-frame`` when that is set, so a hip height of 90 cm
   does not lift the Roblox avatar.
4. Optionally subtract that frame's pose so it becomes the R15 rest.
5. Optionally move yaw onto the HumanoidRootPart.
6. Optionally stand the torso upright, then blend the loop seam.
7. Optionally copy the left limbs onto the right, half a cycle later.
8. Drop a duplicated last key, then thin keyframes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from mocap2r15.bvh import BVH, BVHError, load_bvh
from mocap2r15.cleanup import (
    drop_matching_end,
    loop_blend,
    mirror_left_to_right,
    reduce_frames,
    resample_frames,
    trim_frames,
    upright_torso,
)
from mocap2r15.clip import frames_from_bvh
from mocap2r15.errors import MocapError
from mocap2r15.footlock import apply_foot_lock, leg_triples
from mocap2r15.math3d import QUAT_IDENTITY, forward_axis_quaternion, quat_rotate, yaw_about_y
from mocap2r15.presets import detect_preset, resolve_mapping
from mocap2r15.r15 import ANIMATION_PRIORITY, PART_ORDER
from mocap2r15.retarget import pose_angle_warnings, rest_corrections, retarget_pose, subtract_rest_pose, R15Frame

# CMU / cgspeed ASF units: the BVH numbers are (1/0.45) inches.
CMU_METERS_PER_UNIT = (1.0 / 0.45) * 0.0254

_UNIT_METERS = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "cmu": CMU_METERS_PER_UNIT,
}


@dataclass
class ConvertOptions:
    preset: str = "auto"
    forward_axis: str = "auto"
    unit: str = "auto"
    target_height: float = 5.0
    studs_per_meter: float = 3.571
    studs_per_unit: Optional[float] = None
    in_place: bool = True
    root_bone: str = "humanoid"
    yaw_on_root: bool = False
    upright_torso: bool = False
    mirror_left: bool = False
    mirror_phase: float = 0.5
    fps: Optional[float] = None
    reduce_degrees: float = 0.0
    reduce_position: float = 0.05
    loop: bool = False
    loop_blend: int = 0
    foot_lock: bool = False
    foot_blend: int = 2
    start: Optional[float] = None
    end: Optional[float] = None
    priority: str = "Movement"
    name: Optional[str] = None
    rest_frame: Optional[int] = None


@dataclass
class ConvertResult:
    frames: List[R15Frame]
    warnings: List[str] = field(default_factory=list)
    preset: str = ""
    fps: float = 0.0
    duration: float = 0.0
    scale: float = 1.0
    skeleton_height: float = 0.0
    forward_axis: str = "z"
    source_frames: int = 0
    output_frames: int = 0
    name: str = "Animation"
    unit_note: str = ""


def studs_per_file_unit(height: float, options: ConvertOptions) -> tuple:
    """Return ``(studs per file unit, short explanation)``."""
    if options.studs_per_unit is not None:
        return float(options.studs_per_unit), "explicit studs per file unit"
    unit = (options.unit or "auto").lower()
    if unit == "raw":
        return 1.0, "raw file units"
    if unit == "auto":
        if height < 1e-4:
            return 1.0, "skeleton height was ~0, so the scale was left at 1"
        return options.target_height / height, f"fit the bind pose to {options.target_height:g} studs"
    if unit not in _UNIT_METERS:
        known = "auto, m, cm, mm, in, cmu, raw"
        raise MocapError(f"Unknown unit {options.unit!r}. Use one of: {known}")
    scale = _UNIT_METERS[unit] * options.studs_per_meter
    return scale, f"{unit} converted at {options.studs_per_meter:g} studs per meter"


def convert_bvh(bvh: BVH, options: Optional[ConvertOptions] = None) -> ConvertResult:
    """Retarget ``bvh`` and return R15 frames. Does not write a file."""
    options = options or ConvertOptions()
    _validate(options)
    if not bvh.frames:
        raise MocapError("BVH has no motion frames")

    warnings: List[str] = []
    try:
        mapping = resolve_mapping(bvh, options.preset)
    except ValueError as exc:
        raise MocapError(str(exc)) from exc
    warnings.extend(mapping.warnings)

    axis = options.forward_axis
    if axis == "auto":
        guessed = bvh.forward_hint()
        if guessed is None:
            warnings.append("could not guess a forward axis from the feet; assuming +Z")
            axis = "z"
        else:
            axis = guessed
    try:
        conv = forward_axis_quaternion(axis)
    except ValueError as exc:
        raise MocapError(str(exc)) from exc

    height = bvh.skeleton_height()
    scale, unit_note = studs_per_file_unit(height, options)
    if scale <= 0:
        raise MocapError(f"Scale must be positive (got {scale})")
    if "left at 1" in unit_note:
        warnings.append(unit_note)

    source = frames_from_bvh(bvh)
    source_count = len(source)
    source = trim_frames(source, options.start, options.end)
    if not source:
        raise MocapError("The trim range removed every frame")

    if options.foot_lock:
        triples = leg_triples({part: spec.joints for part, spec in mapping.parts.items()})
        warnings.extend(apply_foot_lock(bvh, source, triples, options.foot_blend))

    if options.fps is not None and options.fps > 0:
        source = resample_frames(source, options.fps)

    corrections = rest_corrections(mapping, conv)
    origin_index = 0
    if options.rest_frame is not None:
        origin_index = max(0, min(int(options.rest_frame), len(source) - 1))
        if origin_index != options.rest_frame:
            warnings.append(f"rest frame {options.rest_frame} was clamped to {origin_index}")
    origin = source[origin_index].root_pos

    frames: List[R15Frame] = []
    for sample in source:
        local = retarget_pose(sample.rotations, mapping, conv, corrections)
        for part in PART_ORDER:
            local.setdefault(part, QUAT_IDENTITY)
        delta = (
            sample.root_pos[0] - origin[0],
            sample.root_pos[1] - origin[1],
            sample.root_pos[2] - origin[2],
        )
        moved = quat_rotate(conv, delta)
        root = (moved[0] * scale, moved[1] * scale, moved[2] * scale)
        if options.in_place:
            root = (0.0, root[1], 0.0)
        frames.append(R15Frame(time=sample.time, local=local, root_pos=root))

    if options.rest_frame is not None:
        subtract_rest_pose(frames, origin_index)

    if options.yaw_on_root:
        for frame in frames:
            yaw, rest = yaw_about_y(frame.local["LowerTorso"])
            frame.local["HumanoidRootPart"] = yaw
            frame.local["LowerTorso"] = rest

    if options.upright_torso:
        upright_torso(frames)

    if options.loop and options.loop_blend > 0:
        loop_blend(frames, options.loop_blend, blend_translation=options.in_place)
    if options.mirror_left:
        # The endpoint key is still present, so its time is one full cycle.
        # Dropping it first would make a half-cycle sample one frame short.
        mirror_left_to_right(frames, options.mirror_phase)
    if options.loop:
        frames = drop_matching_end(frames)
    if options.reduce_degrees > 0:
        frames = reduce_frames(frames, options.reduce_degrees, options.reduce_position)

    warnings.extend(pose_angle_warnings(frames))

    if len(frames) > 1:
        duration = frames[-1].time - frames[0].time
        fps = options.fps if options.fps else (1.0 / bvh.frame_time if bvh.frame_time else 0.0)
    else:
        duration = 0.0
        fps = options.fps or (1.0 / bvh.frame_time if bvh.frame_time else 0.0)

    return ConvertResult(
        frames=frames,
        warnings=warnings,
        preset=mapping.preset,
        fps=fps,
        duration=duration,
        scale=scale,
        skeleton_height=height,
        forward_axis=axis,
        source_frames=source_count,
        output_frames=len(frames),
        name=options.name or "Animation",
        unit_note=unit_note,
    )


def convert_path(path: str, options: Optional[ConvertOptions] = None) -> ConvertResult:
    """Load a BVH from disk and convert it."""
    try:
        bvh = load_bvh(path)
    except OSError as exc:
        raise MocapError(f"Could not read {path}: {exc}") from exc
    except BVHError as exc:
        raise MocapError(str(exc)) from exc
    return convert_bvh(bvh, options)


def guess_unit(height: float) -> str:
    """A one-line hint for ``inspect``. Auto-fit does not depend on this."""
    if 1.2 <= height <= 2.6:
        return "meters"
    if 120.0 <= height <= 260.0:
        return "centimeters"
    if 1200.0 <= height <= 2600.0:
        return "millimeters"
    if 15.0 <= height <= 45.0:
        return "CMU/ASF-style (about 0.056 m per unit)"
    if 4.0 <= height <= 8.0:
        return "already near a Roblox character's height in studs"
    return "unrecognized; auto-fit to 5 studs is the usual choice"


def inspect_text(path: str) -> str:
    """Multi-line report for the ``inspect`` command."""
    try:
        bvh = load_bvh(path)
    except OSError as exc:
        raise MocapError(f"Could not read {path}: {exc}") from exc
    except BVHError as exc:
        raise MocapError(str(exc)) from exc
    chosen, ratios = detect_preset(bvh)
    mapping = resolve_mapping(bvh, "auto")
    height = bvh.skeleton_height()
    missing = [spec.part for spec in mapping.parts.values() if not spec.joints]
    root = bvh.joints[bvh.root]
    lines = [
        f"file: {path}",
        f"root: {bvh.root}",
        f"joints: {len(bvh.order)}",
        f"frames: {len(bvh.frames)}",
        f"frame time: {bvh.frame_time:.6g} s",
        f"fps: {bvh.fps:.4g}",
        f"duration: {bvh.duration:.4g} s",
        f"channels on root: {root.rotation_order or '(no rotation)'} {'with translation' if root.has_position else ''}".rstrip(),
        f"skeleton height: {height:.4g} file units",
        f"unit guess: {guess_unit(height)}",
        f"forward axis: {bvh.forward_hint() or 'unknown (assume z)'}",
        f"preset: {chosen}",
        "preset scores: " + ", ".join(f"{name} {ratio:.0%}" for name, ratio in sorted(ratios.items(), key=lambda kv: -kv[1])),
        "missing parts: " + (", ".join(missing) if missing else "none"),
    ]
    for warning in mapping.warnings:
        if warning.startswith("auto-selected"):
            continue
        lines.append(f"warning: {warning}")
    lines.append("joints:")
    for name in bvh.order:
        joint = bvh.joints[name]
        parent = joint.parent or "-"
        lines.append(f"  {name}  parent={parent}  channels={joint.rotation_order or '-'}")
    return "\n".join(lines) + "\n"


def _validate(options: ConvertOptions) -> None:
    if options.priority not in ANIMATION_PRIORITY:
        known = ", ".join(ANIMATION_PRIORITY)
        raise MocapError(f"Unknown animation priority {options.priority!r}. Use one of: {known}")
    if options.root_bone not in ("humanoid", "lower-torso"):
        raise MocapError("root bone must be 'humanoid' or 'lower-torso'")
    if options.fps is not None and options.fps < 0:
        raise MocapError("fps must be positive")
    if options.target_height <= 0:
        raise MocapError("target height must be positive")
