"""Visualisation helpers (drawing only, no geometry)."""
import cv2
import numpy as np

AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X red, Y green, Z blue


def _pt(p):
    return tuple(int(round(c)) for c in p)


def draw_axes(img, uv, label="", thickness=3):
    """Draw body axes on an image.

    img: grayscale (H, W) or BGR (H, W, 3).
    uv:  (4, 2) pixels in order [origin, X end, Y end, Z end].
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    o = _pt(uv[0])
    for end, col in zip(uv[1:], AXIS_COLORS):
        cv2.line(vis, o, _pt(end), col, thickness, cv2.LINE_AA)
    cv2.circle(vis, o, 2 * thickness, (0, 255, 255), -1)  # yellow origin
    if label:
        cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    return vis


def draw_keypoints(img, uv, label="", radius=6):
    """Draw numbered keypoints (N, 2) plus the top/bottom face outlines (points 0-3, 4-7)."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    for face, col in (([0, 1, 2, 3], (255, 128, 0)), ([4, 5, 6, 7], (0, 200, 255))):
        poly = np.round(uv[face]).astype(np.int32)
        cv2.polylines(vis, [poly], True, col, 2, cv2.LINE_AA)
    for i, p in enumerate(uv):
        cv2.circle(vis, _pt(p), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, str(i), _pt(p + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    return vis


def crop_around(img, uv, margin=0.3):
    """Square crop around points uv (N, 2) with a relative margin, clipped to the image."""
    h, w = img.shape[:2]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    xb, yb = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    return img[ya:yb, xa:xb]


def make_grid(images, ncols=2, scale=0.5):
    """Tile equally sized images row-major into a grid and downscale it."""
    blank = np.zeros_like(images[0])
    rows = []
    for i in range(0, len(images), ncols):
        row = list(images[i:i + ncols])
        row += [blank] * (ncols - len(row))
        rows.append(cv2.hconcat(row))
    grid = cv2.vconcat(rows)
    return cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
