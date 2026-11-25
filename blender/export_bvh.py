"""Export an armature from FBX (or glTF) to BVH for mocap2r15.

Blender is not required to run the converter. Use this when the clip you
have is FBX and you want a BVH the Python tool can retarget:

    blender --background --python blender/export_bvh.py -- --input walk.fbx --output walk.bvh

C3D is not imported here. Blender needs a C3D add-on before marker data
becomes an armature, and this script only exports an armature that is
already in the file.
"""

from __future__ import annotations

import sys


def _arguments(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    import argparse

    parser = argparse.ArgumentParser(description="Export the first armature to BVH")
    parser.add_argument("--input", required=True, help="FBX, glTF, or BVH file")
    parser.add_argument("--output", required=True, help="Destination .bvh")
    parser.add_argument("--scale", type=float, default=1.0, help="global_scale passed to the BVH exporter")
    return parser.parse_args(argv)


def _import_clip(bpy, path):
    lower = path.lower()
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
        return
    if lower.endswith(".glb") or lower.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
        return
    if lower.endswith(".bvh"):
        bpy.ops.import_anim.bvh(filepath=path)
        return
    if lower.endswith(".c3d"):
        raise SystemExit(
            "Blender does not import C3D in this script. Load a C3D add-on, "
            "solve or import an armature, then export BVH."
        )
    raise SystemExit(f"Unsupported input for the Blender exporter: {path}")


def _frame_range(arm, scene):
    action = None
    data = arm.animation_data
    if data is not None:
        action = getattr(data, "action", None)
    if action is not None:
        try:
            start, end = action.frame_range
            return int(start), int(end)
        except Exception:
            pass
    return int(scene.frame_start), int(scene.frame_end)


def main() -> None:
    import bpy

    args = _arguments(sys.argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import_clip(bpy, args.input)
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise SystemExit(f"No armature in {args.input}")
    arm = armatures[0]
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    start, end = _frame_range(arm, bpy.context.scene)
    try:
        bpy.ops.export_anim.bvh(
            filepath=args.output,
            frame_start=start,
            frame_end=end,
            global_scale=args.scale,
            root_transform_only=False,
        )
    except TypeError:
        # Older Blender builds expose a shorter operator. The frame range
        # still comes from the scene, which was just set from the action.
        bpy.context.scene.frame_start = start
        bpy.context.scene.frame_end = end
        bpy.ops.export_anim.bvh(filepath=args.output, global_scale=args.scale)
    print(f"exported {args.output} frames {start}-{end} from armature {arm.name}")


if __name__ == "__main__":
    main()
