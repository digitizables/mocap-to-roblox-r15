"""Retarget motion-capture locomotion onto Roblox R15 KeyframeSequences.

The package is split so each stage can be tested and replaced on its own:

- ``math3d`` — vectors, matrices, quaternions, axis conversion
- ``bvh`` — BVH hierarchy and motion
- ``presets`` — named skeleton maps (CMU, Mixamo, LaFAN1, SOMA, generic)
- ``retarget`` — joint collapse, rest correction, R15 local transforms
- ``cleanup`` — resample, keyframe reduction, loop blend, foot plant
- ``rbxmx`` — KeyframeSequence XML
- ``pipeline`` — one-shot conversion used by the CLI
"""

__version__ = "1.0.0"
