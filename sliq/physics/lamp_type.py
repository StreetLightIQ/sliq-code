"""Lamp type from colour: orange (sodium), white (LED) or unclear.

Feature: log(R/B) of the linear-RGB glow ring around the head, averaged over the track. An Otsu split is fitted on training lamps and reused
unchanged; lamps within `margin` of the split are unclear."""
import numpy as np


def feature(ring):
    if not ring:
        return None
    r, g, b = (max(float(c), 1e-6) for c in ring)
    return float(np.log(r / b))


def fit_split(features):
    """Otsu threshold on 1-D features -> dict(split, margin, cluster means)."""
    x = np.sort(np.asarray([f for f in features if f is not None], float))
    if len(x) < 10:
        return None
    best, cut = -1, None
    for i in range(3, len(x) - 3):
        a, b = x[:i], x[i:]
        between = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if between > best:
            best, cut = between, 0.5 * (x[i - 1] + x[i])
    lo, hi = x[x < cut], x[x >= cut]
    margin = 0.25 * (hi.mean() - lo.mean())
    return dict(split=float(cut), margin=float(margin), white_mean=float(lo.mean()), orange_mean=float(hi.mean()), n=len(x),
                n_white=int(len(lo)), n_orange=int(len(hi)))


def classify(ring, split):
    f = feature(ring)
    if f is None or split is None:
        return 'unclear'
    if f >= split['split'] + split['margin']:
        return 'orange'
    if f <= split['split'] - split['margin']:
        return 'white'
    return 'unclear'
