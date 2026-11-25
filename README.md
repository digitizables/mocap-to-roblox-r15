# mocap-to-roblox-r15

Convert publicly available motion-capture walks and other locomotion into
Roblox **R15** animations (`KeyframeSequence` `.rbxmx` files).

The converter is a pure Python 3.9+ CLI (standard library only). It parses
BVH, retargets the skeleton onto the 15-part R15 rig, scales the stride into
studs, and writes an animation you can insert in Roblox Studio.

## What you can do with it

- Map CMU, Mixamo, LaFAN1, SOMA-style, and many similarly named BVH skeletons onto R15.
- Collapse multi-bone spines onto the Waist, drop fingers and toes, and correct rest-pose differences (T-pose arms versus the R15 arms-down stance).
- Turn the source forward axis onto Roblox **-Z**.
- Fit translation to a standard R15 height, or scale by real-world units.
- Export in-place cycles or root motion, resample FPS, thin keyframes, blend loop seams, and run a basic foot-contact plant.
- Batch-convert a folder of trials.

C3D, FBX, and AMASS/SMPL `.npz` files are detected and reported. A short Blender
script converts those to BVH when Blender can import them. AMASS axis-angle
poses still need a SMPL body to become a skeleton.

## Status

The tool, tests, example walk, and the full usage guide live in this
repository. See the sections below once you have cloned it — they are filled
in alongside the implementation (setup, commands, joint map, limitations,
and where to get free CMU walking data).

## License

MIT. CMU motion-capture clips you download separately stay under the
[CMU Graphics Lab](http://mocap.cs.cmu.edu/) terms (free for all uses). Cite
the database if you redistribute motions derived from it.
