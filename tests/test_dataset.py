import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.speedplus import apply_affine, box_to_affine, square_box  # noqa: E402


def test_box_maps_to_crop_and_back():
    uv = np.array([[900.0, 500], [1100, 700], [1000, 650]])
    A = box_to_affine(*square_box(uv, margin=0.2), 256)
    k = apply_affine(A, uv)
    assert (k >= 0).all() and (k <= 255).all()
    # centre of the box lands at the crop centre
    assert np.allclose(apply_affine(A, np.array([[1000.0, 600]])), [[128, 128]])
    # round trip crop -> original (what PnP will need)
    assert np.allclose(apply_affine(cv2.invertAffineTransform(A), k), uv)
