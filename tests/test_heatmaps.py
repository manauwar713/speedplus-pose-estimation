import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.heatmaps import (  # noqa: E402
    argmax_2d, crop_to_heatmap, decode_heatmaps, gaussian_2d, heatmap_to_crop,
    make_heatmaps, subpixel_refine,
)

RNG = np.random.default_rng(0)
N, S, H = 11, 256, 64


def random_kpts(n=N, low=10, high=S - 10):
    """n random keypoints (n, 2) in crop pixels, away from the border."""
    return RNG.uniform(low, high, size=(n, 2))


# --- 1. coordinate conversion ----------------------------------------------- #
def test_coord_round_trip():
    p = random_kpts(1000)
    assert np.allclose(heatmap_to_crop(crop_to_heatmap(p)), p)


def test_coord_known_value():
    # Pixel-centre convention (b): heatmap pixel j covers crop pixels 4j..4j+3, centre 4j + 1.5.
    assert np.allclose(crop_to_heatmap(np.array([1.5, 1.5])), [0.0, 0.0])
    assert np.allclose(crop_to_heatmap(np.array([129.5, 5.5])), [32.0, 1.0])
    # Whole crop [-0.5, 255.5] maps onto whole heatmap [-0.5, 63.5].
    assert np.allclose(crop_to_heatmap(np.array([-0.5, 255.5])), [-0.5, 63.5])


# --- 2. target generation --------------------------------------------------- #
def test_gaussian_peak_and_shape():
    g = gaussian_2d(np.array([20.0, 30.0]), 64, 2.0)
    assert g.shape == (64, 64)
    assert np.isclose(g.max(), 1.0)
    row, col = np.unravel_index(g.argmax(), g.shape)
    assert (row, col) == (30, 20)  # rows are y -> catches an x/y swap


def test_gaussian_fractional_centre_not_rounded():
    # centre 20.3: pixel 20 brightest, but pixel 21 must be brighter than pixel 19.
    g = gaussian_2d(np.array([20.3, 30.0]), 64, 2.0)
    assert g[30, 20] == g.max()
    assert g[30, 21] > g[30, 19]


def test_make_heatmaps_shape_and_invisible():
    kpts = random_kpts()
    vis = np.ones(N)
    vis[3] = 0
    hm = make_heatmaps(kpts, vis)
    assert hm.shape == (N, H, H)
    assert hm.dtype == np.float32
    assert np.all(hm[3] == 0)
    others = np.delete(hm, 3, axis=0)
    assert np.all(others.max(axis=(1, 2)) > 0.9)  # off-grid centre -> peak slightly < 1


def test_make_heatmaps_accepts_bool_vis():
    # speedplus.py passes vis as a bool array
    kpts = random_kpts()
    vis = np.ones(N, dtype=bool)
    vis[0] = False
    hm = make_heatmaps(kpts, vis)
    assert np.all(hm[0] == 0) and hm[1].max() > 0.9


# --- 3. decoding ------------------------------------------------------------ #
def test_argmax_2d_finds_peak():
    hm = np.zeros((H, H), dtype=np.float32)
    hm[10, 40] = 1.0  # row 10 = y, col 40 = x
    x, y, v = argmax_2d(hm)
    assert (x, y) == (40, 10)
    assert v == 1.0


def test_subpixel_moves_towards_brighter_neighbour():
    hm = np.zeros((H, H), dtype=np.float32)
    hm[30, 30] = 1.0
    hm[30, 31] = 0.6
    hm[30, 29] = 0.2
    hm[29, 30] = hm[31, 30] = 0.4
    xf, yf = subpixel_refine(hm, 30, 30)
    assert 30 < xf < 30.5
    assert yf == 30


def test_subpixel_border_is_not_refined():
    hm = np.zeros((H, H), dtype=np.float32)
    hm[0, 63] = 1.0
    xf, yf = subpixel_refine(hm, 63, 0)
    assert (xf, yf) == (63, 0)


def test_decode_round_trip():
    kpts = random_kpts(500)
    decoded, peaks = decode_heatmaps(make_heatmaps(kpts, np.ones(len(kpts))))
    err = np.abs(decoded - kpts).max()
    print(f"decode round-trip max error: {err:.4f} crop px")  # 0.048 on 2026-11-17
    assert decoded.shape == kpts.shape
    assert err < 0.1          # parabola fit; argmax alone would be up to 2 px
    assert np.all(peaks > 0.9)


def test_decode_integer_keypoint_is_exact():
    # A keypoint on a heatmap pixel centre has symmetric neighbours -> zero shift.
    kpt = heatmap_to_crop(np.array([[32.0, 17.0]]))   # = (129.5, 69.5)
    decoded, _ = decode_heatmaps(make_heatmaps(kpt, np.ones(1)))
    assert np.allclose(decoded, kpt, atol=1e-4)


def test_decode_border_keypoint_biased_inward():
    # Keypoint just inside the crop edge: half its Gaussian is cut, no refine possible.
    kpt = np.array([[1.0, 254.0]])
    decoded, _ = decode_heatmaps(make_heatmaps(kpt, np.ones(1)))
    assert np.all(np.abs(decoded - kpt) <= 0.5 + 1e-6)


def test_decode_invisible_has_zero_peak():
    # vis = 0 -> all-zero map -> decodes to a fake point; the peak must flag it.
    _, peaks = decode_heatmaps(make_heatmaps(random_kpts(2), np.array([1, 0])))
    assert peaks[0] > 0.9 and peaks[1] == 0
