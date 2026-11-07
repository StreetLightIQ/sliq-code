"""sRGB to linear light, and sampling of usable road pixels."""
import numpy as np


def srgb_to_linear(c):
    """sRGB-encoded values in [0, 1] -> linear light."""
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_luma(rgb_uint8):
    lin = srgb_to_linear(np.asarray(rgb_uint8, np.float32) / 255.0)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def sample_road(rgb_half, valid_half, rng, n=4000, floor=1e-4):
    """Usable road pixels of a half-resolution frame -> full-resolution (u, v) and linear luma, without pixels at the sensor floor."""
    vv, uu = np.nonzero(valid_half == 1)
    if len(uu) > n:
        pick = rng.choice(len(uu), n, replace=False)
        uu, vv = uu[pick], vv[pick]
    y = linear_luma(rgb_half[vv, uu])
    keep = y >= floor
    return (uu[keep] * 2 + 1.0), (vv[keep] * 2 + 1.0), y[keep]
