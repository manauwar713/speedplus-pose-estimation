"""3D keypoints of the Tango spacecraft in the Tango body frame [m].

Source: tango3Dpoints in src/utils/tangoPoints.mat of the SPEED+ baseline
(github.com/tpark94/speedplusbaseline), copied as numbers, not code.
Body origin = centre of the bottom face (z = 0); +z points to the top face.

STATUS: published values. Still to do: check them independently by
triangulating from labelled images (scripts/triangulate_keypoints.py).
"""
import numpy as np

TANGO_KEYPOINTS = np.array([
    # x        y        z
    [-0.3700, -0.3850, 0.3215],  # 0  top face corners
    [-0.3700,  0.3850, 0.3215],  # 1
    [ 0.3700,  0.3850, 0.3215],  # 2
    [ 0.3700, -0.3850, 0.3215],  # 3
    [-0.3700, -0.2640, 0.0000],  # 4  bottom face corners
    [-0.3700,  0.3040, 0.0000],  # 5
    [ 0.3700,  0.3040, 0.0000],  # 6
    [ 0.3700, -0.2640, 0.0000],  # 7
    [-0.5427,  0.4877, 0.2535],  # 8  antenna tips
    [ 0.5427,  0.4877, 0.2591],  # 9
    [ 0.3050, -0.5790, 0.2515],  # 10
])

NUM_KEYPOINTS = len(TANGO_KEYPOINTS)
