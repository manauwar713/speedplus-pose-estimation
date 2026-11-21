"""Heatmap targets + decoding.  *** YOURS to implement (CLAUDE.md) ***

Shapes / sizes used everywhere:
  input crop : S x S = 256 x 256 pixels      (keypoints arrive in these "crop pixels")
  heatmap    : H x H =  64 x  64 pixels      (network output resolution)
  stride     : S / H = 4  -> 1 heatmap pixel covers 4 x 4 crop pixels
  N = 11 keypoints -> heatmaps array has shape (N, H, H) = (11, 64, 64)

Array indexing reminder: an image array is indexed [row, col] = [y, x].
Keypoints are stored as (x, y). Mixing these two up is the #1 heatmap bug.

Build order (test each before the next):
  crop_to_heatmap / heatmap_to_crop -> gaussian_2d -> make_heatmaps
  -> argmax_2d -> subpixel_refine -> decode_heatmaps
"""
import numpy as np

STRIDE = 4  # 256 / 64


# --------------------------------------------------------------------------- #
# 1. Coordinate conversion (the convention you must be able to defend)
# --------------------------------------------------------------------------- #
def crop_to_heatmap(xy, stride=STRIDE):
    """Crop-pixel coords (..., 2) -> heatmap-pixel coords (..., 2). Float in, float out.

    Two options, pick ONE and write in a comment WHY:
      (a) xy / stride
      (b) (xy + 0.5) / stride - 0.5     <- treats pixel *centres* consistently
    Sanity check for (b): the centre of crop pixels 0..3 is 1.5 -> should land on heatmap pixel 0.
    """
    xy = (xy + 0.5) / stride - 0.5
    return xy


def heatmap_to_crop(xy, stride=STRIDE):
    """Exact inverse of crop_to_heatmap: heatmap_to_crop(crop_to_heatmap(p)) == p."""
    xy = (xy + 0.5) * stride - 0.5
    return xy


# --------------------------------------------------------------------------- #
# 2. Target generation
# --------------------------------------------------------------------------- #
def gaussian_2d(center_xy, size, sigma):
    """One (size, size) float32 map: exp(-((x-x0)^2 + (y-y0)^2) / (2 sigma^2)), peak value 1.

    center_xy is in HEATMAP pixels and may be fractional (e.g. 31.6, 12.25) -> do NOT round it,
    keeping the fractional part is what lets decode recover sub-pixel positions.
    Hint: xs = np.arange(size); then broadcasting xs[None, :] and ys[:, None] gives a
          (size, size) grid without a Python loop. Rows are y, columns are x.
    """
    # TODO (you): ~4 lines
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing='xy')
    return np.exp(-((x_grid - center_xy[0]) ** 2 + (y_grid - center_xy[1]) ** 2) / (2 * sigma ** 2))


def make_heatmaps(kpts, vis, input_size=256, heatmap_size=64, sigma=2.0):
    """kpts (N, 2) crop pixels, vis (N,) 0/1  ->  float32 (N, heatmap_size, heatmap_size).

    For each keypoint: if vis == 0 -> all-zero map; else gaussian_2d at crop_to_heatmap(kpt).
    stride = input_size / heatmap_size.
    Think: a visible keypoint 1 px inside the crop edge -> half its Gaussian is cut off. OK?
    """
    # TODO (you): ~6 lines (np.zeros for the output, loop over the N keypoints)
    stride = input_size / heatmap_size
    heatmaps = np.zeros((len(kpts), heatmap_size, heatmap_size), dtype=np.float32)
    for i, (kpt, v) in enumerate(zip(kpts, vis)):
        if v == 1:
            heatmaps[i] = gaussian_2d(crop_to_heatmap(kpt, stride), heatmap_size, sigma)
    return heatmaps


# --------------------------------------------------------------------------- #
# 3. Decoding
# --------------------------------------------------------------------------- #
def argmax_2d(hm):
    """One (H, W) map -> (x, y) integer heatmap coords of the brightest pixel, and its value.

    Hint: np.argmax works on the flattened array; np.unravel_index turns that back into
          (row, col) = (y, x). Return in (x, y) order!
    """
    # TODO (you): ~3 lines
    idx = np.argmax(hm)
    y, x = np.unravel_index(idx, hm.shape)
    return x, y, hm[y, x]


def subpixel_refine(hm, x, y):
    """Refine integer peak (x, y) to float using the neighbours. Returns (x_f, y_f).

    Simple version (classic SimpleBaseline): move 0.25 px towards the brighter neighbour:
        dx = 0.25 * sign(hm[y, x+1] - hm[y, x-1]),  same for dy with rows.
    Better version (optional): fit a parabola through (left, centre, right):
        dx = 0.5 * (r - l) / (2c - l - r)      (guard: denominator can be 0)
    Edge case: peak on the border (x == 0 or x == W-1) has no neighbour on one side -> skip refine.
    """
    dx = 0.5 * (hm[y, x + 1] - hm[y, x - 1]) / (2 * hm[y, x] - hm[y, x - 1] - hm[y, x + 1]) if 0 < x < hm.shape[1] - 1 else 0
    dy = 0.5 * (hm[y + 1, x] - hm[y - 1, x]) / (2 * hm[y, x] - hm[y - 1, x] - hm[y + 1, x]) if 0 < y < hm.shape[0] - 1 else 0
    return x + dx, y + dy
    


def decode_heatmaps(heatmaps, input_size=256):
    """(N, H, H) -> kpts (N, 2) float in CROP pixels, peaks (N,) max value of each map.

    Per map: argmax_2d -> subpixel_refine -> heatmap_to_crop.
    peaks are later used as a confidence score (stage 6).
    """
    # TODO (you): ~8 lines
    kpts = []
    peaks = []
    for hm in heatmaps:
        x, y, peak = argmax_2d(hm)
        x_f, y_f = subpixel_refine(hm, x, y)
        kpts.append(heatmap_to_crop(np.array([x_f, y_f]), stride=input_size / hm.shape[0]))
        peaks.append(peak)
    return np.array(kpts), np.array(peaks)
