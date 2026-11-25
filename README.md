# mocap-to-roblox-r15

Convert a publicly available motion-capture walk into a Roblox **R15**
`KeyframeSequence` (`.rbxmx`). The tool is pure Python 3.9+ and uses only
the standard library.

It reads **BVH**, maps the skeleton onto the R15 rig, scales the stride
into studs, and writes an animation you can insert in Roblox Studio. C3D,
FBX, and AMASS `.npz` are recognized and pointed at a conversion step. They
are not retargeted in process.

## Setup

```powershell
git clone https://github.com/digitizables/mocap-to-roblox-r15.git
cd mocap-to-roblox-r15
python -m unittest discover -s tests -v
python -m mocap2r15 --help
```

No install step is required. From the repo root, `python -m mocap2r15` is the
command. If you want a `mocap2r15` executable on `PATH`:

```powershell
pip install -e .
```

There are no third-party dependencies. Blender is optional and is only used
to turn FBX into BVH. See [Other formats](#other-formats).

## Quick start

A one-second synthetic walk ships in `examples/`. Regenerate it with:

```powershell
python -m mocap2r15 sample -o examples/synthetic_walk.bvh --rbxmx examples/synthetic_walk.rbxmx
```

Convert any BVH. The default result is an **in-place** cycle (the character
walks on the spot, which is what a Humanoid locomotion clip wants):

```powershell
python -m mocap2r15 walk.bvh -o walk.rbxmx
python -m mocap2r15 walk.bvh --fps 30 --loop --loop-blend 4 --foot-lock
python -m mocap2r15 inspect walk.bvh
python -m mocap2r15 batch trials -o r15 --fps 30
python -m mocap2r15 presets
```

Passing a path as the first argument implies `convert`.

Exit status is `0` when the clip is written, `1` when a file cannot be
converted, and `2` when the arguments are wrong. `batch` keeps going after a
bad file and returns `1` if any trial failed.

## Import into Roblox Studio

1. In Explorer, right-click and choose **Insert from File**. Pick the `.rbxmx`.
2. Move the `KeyframeSequence` under the R15 rig's `AnimSaves` model.
3. Open the Animation Editor on that rig and select the clip.
4. Publish it to get an animation asset id.

Poses are named after **parts** (`UpperTorso`, `LeftUpperArm`, …). Studio
applies each pose to the motor that drives that part (`Waist`,
`LeftShoulder`, …). The same names work for Motor6D rigs and for
AnimationConstraint rigs.

The first frame's translation is the origin. A BVH hip height of 90 cm does
not lift the avatar off its Humanoid. Playback priority defaults to
**Movement**. Pass `--loop` to set the Loop flag.

Some Humanoid playback paths ignore translation on `HumanoidRootPart`. If
the character does not bob or travel, export again with
`--root-bone lower-torso`.

## Commands

| Command | What it does |
| --- | --- |
| `convert` (or a bare path) | One or more BVH files. `-o clip.rbxmx` for one file, or `-o folder` for several. |
| `batch FOLDER` | Every `.bvh` in the folder. Default output is `FOLDER/r15/`. `--recursive` includes subfolders. Non-BVH files are skipped. |
| `inspect FILE` | Joints, fps, height, forward axis, preset scores, missing parts. |
| `sample` | Write a synthetic CMU-style walk. `--rbxmx PATH` also writes the in-place R15 clip. |
| `presets` | List the built-in skeleton maps. |

`--meta` writes a `.meta.json` sidecar (preset, scale, warnings). Those files
are gitignored.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--preset` | `auto` | `auto`, `cmu`, `mixamo`, `lafan1`, `soma`, or `generic` |
| `--forward-axis` | `auto` | Source facing: `z`, `-z`, `x`, `-x`. `auto` reads the toe offsets. |
| `--unit` | `auto` | `auto`, `m`, `cm`, `mm`, `in`, `cmu`, `raw` |
| `--target-height` | `5` | Studs the bind pose is fit to when the unit is `auto` |
| `--studs-per-meter` | `3.571` | Used with real-world units. About 28 cm per stud. |
| `--studs-per-unit` | | Overrides the scale outright |
| `--root-motion` | off | Keep the traveled path. Omit it for an in-place cycle. |
| `--root-bone` | `humanoid` | `humanoid` or `lower-torso` |
| `--yaw-on-root` | off | Hip yaw goes on `HumanoidRootPart`. Pitch and roll stay on `LowerTorso`. |
| `--fps` | source | Resample. A 1 second clip at 30 fps stays 31 samples, including both ends. |
| `--reduce-degrees` | `0` | Drop keys a slerp can rebuild within this angle. `0` keeps every frame. |
| `--reduce-position` | `0.05` | Root-position error allowed while thinning, in studs |
| `--loop` | off | Set Loop, and drop a final key that repeats the first |
| `--loop-blend N` | `0` | Ease the last N frames back to the first pose |
| `--foot-lock` | off | Plant the feet during detected ground contact |
| `--foot-blend` | `2` | Contact frames to ease in and out of a plant |
| `--start` / `--end` | | Trim, in seconds |
| `--priority` | `Movement` | `Idle`, `Movement`, `Action`, `Action2`, `Action3`, `Action4`, `Core` |
| `--name` | file stem | `KeyframeSequence` name |
| `--rest-frame N` | | Make this frame (after trim and resample) the R15 rest pose |

`--rest-frame` is for a clip whose **standing** pose should be the avatar's
idle, so later frames become deltas from that stance. Do not point it at a
T-pose. That would bake the arms outstretched.

## Joint mapping

The file stores one pose for `HumanoidRootPart` plus the 15 body parts
(16 poses in total). The importer stores a CFrame per part. Studio
plays that CFrame as the matching motor's `Transform`. Identity is the Rig
Builder rest: torso and head up, arms and legs down, feet pointing along
character forward (**-Z**).

| R15 part | Motor6D | CMU joints (parent first) |
| --- | --- | --- |
| HumanoidRootPart | translation, optional yaw | |
| LowerTorso | Root | Hips |
| UpperTorso | Waist | LowerBack, Spine, Spine1 |
| Head | Neck | Neck, Neck1, Head |
| LeftUpperArm | LeftShoulder | LeftShoulder, LeftArm |
| LeftLowerArm | LeftElbow | LeftForeArm |
| LeftHand | LeftWrist | LeftHand |
| LeftUpperLeg | LeftHip | LHipJoint, LeftUpLeg |
| LeftLowerLeg | LeftKnee | LeftLeg |
| LeftFoot | LeftAnkle | LeftFoot |

The right side mirrors the left (`RightUpLeg`, `RightLeg`, `RightFoot`,
`RHipJoint`). Fingers and toes are not parts. A toe offset is only used to
aim the foot. A finger offset aims the hand.

Several BVH joints collapse onto one part. Their local rotations are
composed in hierarchy order, the same way forward kinematics does:
`R_parent @ R_child`. A joint claimed by two parts stays on the distal one,
so a neck listed on both the chest and the head drives the head.

Bone direction is measured on the **bind** pose (every Euler zero), not on
frame 0. A walk's first frame is usually already standing, with non-zero
channels. The direction follows the next R15 part: the chest aims at the
neck, not down a clavicle, even when the shoulder offset is longer. A
zero-length connector such as CMU's `Neck` is skipped.

### Presets

`auto` picks the named preset with the most joints actually present. Below
40% overlap it falls back to `generic` and says so.

| Preset | Skeleton |
| --- | --- |
| `cmu` | CMU Graphics Lab / cgspeed MotionBuilder BVH. `LeftLeg` is the **shin**. |
| `mixamo` | `mixamorig:Hips`, `Spine`, `Spine1`, `Spine2`, `LeftUpLeg`, `LeftLeg` |
| `lafan1` | LaFAN1 / Bandai-style names. Same limbs as Mixamo. Toe may be `LeftToe`. |
| `soma` | SOMA / SMPL BVH (`Root`, `Hips`, `Chest`, `LeftShin`). Here `LeftLeg` is the **thigh**. |
| `generic` | Exact aliases: `pelvis`, `leftthigh`, `leftshin`, `leftfoot`, … |

Names are matched exactly after a light normalize: the `mixamorig:` prefix
is dropped, case and underscores are ignored. `Spine` does not match
`Spine1`. Generic aliases do not substring-match either. A wrapper `Root`
above the hips is kept on `LowerTorso` so its rotation is not thrown away.

A missing joint warns and that R15 pose stays at the rest. The conversion
still succeeds.

### How a pose becomes an R15 CFrame

For each part, `L` is the composed BVH rotation and `C` is a yaw about Y
that turns the source forward axis onto Roblox `(0, 0, -1)`. `D_part` lines
the bind bone up with the R15 rest bone (T-pose arms out become arms down,
with a shared forward twist so the elbows do not roll at random):

```text
T = D_parent^-1 * (C * L * C^-1) * D_part
```

At the bind pose a T-pose shoulder is a large correction. A frame that
already hangs the arm (`Rz(-90)` on a `+X` upper arm) lands near identity.
Knee flexion is a positive X rotation on a shin that hangs down: in BVH that
bends the calf toward `-Z`, and after the facing yaw it bends toward Roblox
`+Z` (behind the character).

`T` is written as a Roblox `CoordinateFrame` (position, then the rotation
matrix `R00`…`R22`). Easing is linear.

### Adding a preset

Edit `PRESETS` in `mocap2r15/presets.py`. `parts` maps an R15 part to BVH
joint names in parent-to-child order. Names that are absent from a file are
skipped, so listing a clavicle in front of the upper arm is safe. Copy
`_CMU` as a starting point. `auto` will pick the new preset up as soon as it
shares more joints with the file than the others do.

## Coordinates and scale

Both skeletons are Y-up and right-handed. CMU and most locomotion BVHs face
**+Z**. Roblox characters face **-Z**. The converter yaws about Y to close
that gap (`+Z` → 180°, `+X` → +90°, `-X` → −90°, `-Z` → none). Pass
`--forward-axis` when the toe guess is wrong. A joint that stays around
180° from the R15 rest prints a warning; that usually means the facing axis
is flipped.

Root translation in a `KeyframeSequence` is an **offset from the character**,
not a world position. Channels on the BVH root replace the root `OFFSET`
(CMU hips are stored that way). The clip subtracts frame 0, or
`--rest-frame` when you set it, then scales and yaws the delta. A constant
hip position of a million units does not move the avatar. In-place export
then zeroes X and Z and keeps the vertical bob. `--root-motion` keeps the
path: source `+Z` travel becomes Roblox `-Z`.

Default scale fits the bind-pose height (lowest end site to highest) to
**5 studs**, a standard R15. Override it when the file is not a full body:

| `--unit` | Scale |
| --- | --- |
| `auto` | `target_height / skeleton_height` |
| `m`, `cm`, `mm`, `in` | Real metres times `--studs-per-meter` (default 3.571) |
| `cmu` | cgspeed units. One unit is `(1/0.45) * 2.54 cm` ≈ 0.0564 m, then studs. |
| `raw` | 1 stud per file unit |

On CMU subject 07 trial 01 the bind height is 24.6 file units. Auto-fit
(0.203 studs per unit) and `--unit cmu` (about 0.202) land on nearly the
same stride. Auto-fit is the right default when you just want the walk to
look natural on an R15.

## Cleanup

Operations run in this order: trim, foot plant, resample, retarget, rest
subtraction, yaw split, loop blend, drop a duplicated last key, then thin.

**Foot plant** (`--foot-lock`) is off by default. It looks for runs where
the ankle is near the bottom of its height range and is not sliding fast,
pins the ankle toward the median of that run, and rebuilds the thigh and
shin with a two-bone IK. Bone lengths stay exact. A correction larger than
about a third of the leg is clamped. The foot keeps its world orientation.
This is a cleanup, not a new solve. Use it when a foot skates. Leave it off
when the source contact is already clean.

**Loop blend** eases rotation, and the vertical bob, back to the first pose.
With `--root-motion` the XZ path is not rewound. `--loop` also drops a final
key within 0.75° and 0.02 studs of the first, because playing that pair
hitches.

**Reduction** is Ramer–Douglas–Peucker on the whole pose. A sample stays if
any joint, or the root translation, cannot be rebuilt by interpolating its
neighbors. Endpoints always stay.

## Other formats

| File | What to do |
| --- | --- |
| `.bvh` | Convert it. |
| `.fbx`, `.glb`, `.gltf` | Export BVH with the Blender script below, then convert. |
| `.c3d` | Marker data, not a skeleton. Solve it to BVH (Blender C3D add-on, or the BVH export on the CMU site) and then convert. |
| `.npz`, `.npy` | AMASS / SMPL axis-angle poses. `inspect` prints `poses` and `trans` shapes. A body model is required to turn those into joints, and this tool does not ship one. Export BVH from a viewer that has SMPL, then convert. |

FBX to BVH, with Blender on `PATH`:

```powershell
blender --background --python blender/export_bvh.py -- --input walk.fbx --output walk.bvh
```

The script imports the file, takes the first armature, and exports that
action's frame range. C3D is refused there on purpose: Blender will not
import it unless a separate add-on has already produced an armature.

`inspect` on an `.npz` reads only the NumPy header, so a large archive is
not loaded into memory.

## Free CMU walking data

The [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/) is
free for all uses. Subject **07**, trial **01** is a straight walk (317
frames, 120 fps, facing +Z, channel order ZYX). A MotionBuilder BVH mirror
is the file this preset was checked against:

```text
https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/007/07_01.bvh
```

The database's own BVH downloads, and the [cgspeed](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture)
mirror, use the same joint names. Convert a one-stride excerpt the way
`examples/cmu_07_01_walk.rbxmx` was made:

```powershell
python -m mocap2r15 07_01.bvh -o cmu_07_01_walk.rbxmx --fps 30 --start 0.5 --end 1.6 --loop --loop-blend 4 --name "CMU 07_01 walk"
```

That window is one left-foot cycle (0.5 s to 1.6 s). Auto scale fits the
24.6-unit skeleton to 5 studs. The clip in `examples/` is in-place: root XZ
stays 0, the hip bobs by about 0.2 studs, the arms hang, and the knees bend
backward. Add `--root-motion` to keep the path (about 5.4 studs of travel
along Roblox -Z over that stride). The source BVH is not in this repo.

If you redistribute a clip derived from CMU data, keep their acknowledgement:

> The data used in this project was obtained from mocap.cs.cmu.edu.
> The database was created with funding from NSF EIA-0196217.

`examples/synthetic_walk.bvh` is original and does not use CMU data. It has
the same joint names, a T-pose bind about 1.75 m tall, arms hung by the
channels, and one meter of +Z travel with the hips parked at X = 40 so a
missed origin subtraction is obvious. `examples/synthetic_walk.rbxmx` is the
in-place loop (`--loop --loop-blend 4`).

## Tests

From the repo root:

```powershell
python -m unittest discover -s tests -v
```

The suite covers Euler channel order (matched against Blender's ZYX
convention), BVH parsing, preset selection, spine aiming, arm and knee
direction after the facing yaw, rest-frame subtraction, scale, in-place
versus root motion, resampling, foot IK bone lengths, the `.rbxmx` hierarchy,
CLI exit codes, and the checked-in examples.

## Limitations

- The output is `.rbxmx`, the file Studio inserts as a `KeyframeSequence`. This tool does not write FBX. FBX is an **input** path through Blender.
- SMPL / AMASS `.npz` is not converted. There is no body model in the repo, and inventing joint positions from axis-angles without one would be wrong.
- C3D markers are not solved into a skeleton.
- One humanoid is assumed. Props, extra spines, and non-biped roots are dropped or warned.
- Fingers and toes are dropped. Toe and finger offsets only aim the foot and the hand.
- Foot plant is a heuristic. It will not repair a clip whose feet never pass near the ground, and it clamps large corrections instead of stretching the leg.
- `--rest-frame` on a T-pose bakes the arms out. Use it on a standing frame.
- `auto` can pick the wrong family when two presets share most names. SOMA's `LeftLeg` is a thigh and CMU's `LeftLeg` is a shin. Pass `--preset soma` or `--preset cmu` when you know the source.
- Auto height-fit assumes the bind pose is a full body. A file that is only a pair of legs will be stretched to 5 studs. Use `--unit` or `--studs-per-unit` then.
- Translation on `HumanoidRootPart` is correct for the Animation Editor and is what some Humanoid playback ignores. `--root-bone lower-torso` is the switch.
- The retarget is a generalization of the usual BVH-to-R15 pipeline (chain collapse, facing yaw, bind-pose correction). It is not a port of any one project, and it does not do muscle or contact optimization beyond `--foot-lock`.

## License

MIT, Copyright (c) 2026 Paul Beck. See [LICENSE](LICENSE).

CMU clips you download separately stay under the CMU Graphics Lab terms.
Cite the database if you redistribute motions derived from it.
