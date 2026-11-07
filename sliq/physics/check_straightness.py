"""How far the GPS path departs from a straight line over 30 m: heading from the chord over the next 15 m, lateral offset of the point
30 m ahead from that heading. Fixes faster than 25 m/s after the last kept fix are dropped.

    python -m sliq.physics.check_straightness"""
import glob
import json

import numpy as np

from sliq.physics import fieldmodel as FM
from sliq.physics.scene import R_EARTH


def deviations(recs, chord=15.0, ahead=30.0):
    ll = np.array([r['latlon'] if r['latlon'] else [np.nan, np.nan] for r in recs], float)
    if not np.isfinite(ll).all():
        return []
    lat0 = np.radians(ll[0, 0])
    xy = np.stack([np.radians(ll[:, 1] - ll[0, 1]) * R_EARTH * np.cos(lat0), np.radians(ll[:, 0] - ll[0, 0]) * R_EARTH], 1)
    t = np.array([r['t'] for r in recs])
    change = np.r_[0, np.nonzero(np.any(np.diff(xy, axis=0) != 0, axis=1))[0] + 1]
    f, ft = xy[change], t[change]
    keep = [0]
    for k in range(1, len(f)):
        if np.linalg.norm(f[k] - f[keep[-1]]) / max(1e-3, ft[k] - ft[keep[-1]]) <= 25:
            keep.append(k)
    f = f[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(f, axis=0), axis=1))]
    out = []
    for a in range(len(f)):
        b, c = np.searchsorted(s, s[a] + chord), np.searchsorted(s, s[a] + ahead)
        if c >= len(f):
            break
        h = (f[b] - f[a]) / np.linalg.norm(f[b] - f[a])
        p = f[c] - f[a]
        out.append(abs(p[0] * h[1] - p[1] * h[0]))
    return out


def main():
    offs = []
    for split in ('train', 'val', 'test'):
        for d in sorted(glob.glob(FM.ROOT + f'cache/{split}/*/')):
            offs += deviations(json.load(open(d + 'frames.json'))['frames'])
    o = np.array(offs)
    print(json.dumps(dict(windows=len(o), median_m=round(float(np.median(o)), 2), p90_m=round(float(np.percentile(o, 90)), 2),
                          share_over_1m=round(float((o > 1).mean()), 3)), indent=1))


if __name__ == '__main__':
    main()
