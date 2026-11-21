import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.geometry.camera import load_camera, project
from src.geometry.rotations import quat_to_R
from src.datasets.labels import load_labels
from src.utils.vis import draw_axes, make_grid


def project_axes(R, r, K, dist, length=0.5):
    """Return (4, 2) pixels of [origin, X end, Y end, Z end] of the Tango body axes."""
    pts_body = np.vstack([np.zeros(3), length * np.eye(3)])  # origin + 3 axis tips, Tango frame [m]
    return project(pts_body, R, r, K, dist)


ROOT = "data/speedplusv2"
OUT = Path("outputs/overlays")
OUT.mkdir(parents=True, exist_ok=True)
K, dist = load_camera(f"{ROOT}/camera.json")

CANDIDATES = [  # (scalar_first, use_transpose)
    (True, False), (True, True),
    (False, False), (False, True),
]

for split in ["train", "lightbox", "sunlamp"]:
    samples = load_labels(ROOT, split)
    for e in samples[:3]:                      # first 3 images on disk
        img = cv2.imread(str(e["path"]), cv2.IMREAD_GRAYSCALE)
        tiles = []
        for scalar_first, use_T in CANDIDATES:
            R = quat_to_R(e["q"], scalar_first)
            if use_T:
                R = R.T
            uv = project_axes(R, e["r"], K, dist)
            name = f"{'w-first' if scalar_first else 'w-last'}, {'R.T' if use_T else 'R'}"
            tiles.append(draw_axes(img, uv, name))
        out_path = OUT / f"{split}_{e['path'].stem}_grid.jpg"
        cv2.imwrite(str(out_path), make_grid(tiles))
        print("saved", out_path)
