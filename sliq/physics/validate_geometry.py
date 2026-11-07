"""Geometry checks on the training and validation spans.

1. Height repeatability: lamps with >= 8 usable sightings over >= 16 m get one height from even and one from odd sightings.
2. Known-answer recovery: real camera poses, distances and sighting frames, with each lamp replaced by a synthetic one of known height and
   position, projected with 2 px pixel noise and 0.5 deg per-frame tilt noise, then fitted like real data.

    python -m sliq.physics.validate_geometry"""
import glob
import json
import os

import numpy as np

from sliq.common import paths
from sliq.physics import scene as S

ROOT = paths.RUNS + '/physics/'


def spans():
    return sorted(glob.glob(ROOT + 'cache/train/*/') + glob.glob(ROOT + 'cache/val/*/'))


def setup(d, cfg):
    meta = json.load(open(d + 'frames.json'))
    R = meta['frames']
    cams = [S.camera_for(r, cfg) for r in R]
    dist = S.robust_distance(R)
    tracks = {t: [(i, l['box']) for i, l in obs if not S._edge(l['box'])] for t, obs in S._tracks(R).items()}
    return R, cams, dist, tracks


def repeatability(cfg):
    pairs = []
    for d in spans():
        R, cams, dist, tracks = setup(d, cfg)
        if dist is None:
            continue
        for t, obs in tracks.items():
            if len(obs) < 8 or np.ptp([dist[i] for i, _ in obs]) < 16:
                continue
            a, _ = S.fit_height(obs[0::2], cams, dist, cfg.cam_height_m)
            b, _ = S.fit_height(obs[1::2], cams, dist, cfg.cam_height_m)
            if a is not None and b is not None and 3 <= a <= 18 and 3 <= b <= 18:
                pairs.append((a, b))
    p = np.array(pairs)
    diff = np.abs(p[:, 0] - p[:, 1])
    return dict(lamps=len(p), median_abs_diff_m=float(np.median(diff)), p90_abs_diff_m=float(np.percentile(diff, 90)),
                corr=float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]), median_height_m=float(np.median(p)))


def synthetic(cfg, px_sigma=2.0, tilt_sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    errs, foot_errs, n_fail = [], [], 0
    for d in spans():
        R, cams, dist, tracks = setup(d, cfg)
        if dist is None:
            continue
        for t, obs in tracks.items():
            if len(obs) < 4 or np.ptp([dist[i] for i, _ in obs]) < 8:
                continue
            H = rng.uniform(5, 14); x = rng.choice([-1, 1]) * rng.uniform(3, 10)
            i_last = obs[-1][0]
            z0 = dist[i_last] + rng.uniform(8, 20)
            syn = []
            for i, _ in obs:
                cam = cams[i]
                p = np.array([x, cfg.cam_height_m - H, z0 - dist[i]])
                if p[2] <= 1:
                    continue
                u, v, front = cam.project(p[None])
                if not front[0]:
                    continue
                v = v[0] + cam.f * np.tan(np.radians(rng.normal(0, tilt_sigma))) + rng.normal(0, px_sigma)
                u = u[0] + rng.normal(0, px_sigma)
                syn.append((i, [u - 5, v - 5, u + 5, v + 5]))
            h, _ = S.fit_height(syn, cams, dist, cfg.cam_height_m)
            if h is None:
                n_fail += 1
                continue
            errs.append(h - H)
            # foot position from the nearest sighting, as in fieldmodel
            i, b = min(syn, key=lambda s: z0 - dist[s[0]])
            r = S._sight(cams[i], b)
            if -r[1] > 0.02:
                foot = dist[i] + r[2] * (h - cfg.cam_height_m) / -r[1]
                foot_errs.append(foot - z0)
    e, f = np.abs(errs), np.abs(foot_errs)
    return dict(lamps=len(errs), not_fitted=n_fail, height_median_abs_err_m=float(np.median(e)), height_p90_abs_err_m=float(np.percentile(e, 90)),
                height_bias_m=float(np.median(errs)), within_1m=float(np.mean(e <= 1)), within_2m=float(np.mean(e <= 2)),
                foot_median_abs_err_m=float(np.median(f)), foot_p90_abs_err_m=float(np.percentile(f, 90)))


def main():
    cfg = S.SceneConfig()
    out = dict(repeatability=repeatability(cfg), synthetic=synthetic(cfg), synthetic_no_noise=synthetic(cfg, 0.0, 0.0))
    for k, v in out.items():
        print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))
    os.makedirs(ROOT + 'eval', exist_ok=True)
    json.dump(out, open(ROOT + 'eval/geometry_validation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
