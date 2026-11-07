"""Propagate per-lamp state and height uncertainty into street-level results, on training and validation spans.

Each draw redraws every lamp's state from its classifier probabilities (intensity 1, 0.5 or 0), jitters fitted heights by the repeatability
noise (sigma = 0.38 / 0.674 m) and draws assumed heights from the pooled fitted heights. Reports the change of lit share, dark share and
longest dark stretch, and the street ranking (Spearman) over 200 draws.

    python -m sliq.physics.uncertainty_mc"""
import argparse
import copy
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

from sliq.physics import fieldmodel as FM
from sliq.physics import scene as S

SIGMA = 0.38 / 0.674          # median even/odd height difference 0.38 m (validate_geometry)
METRICS = ('lit_share', 'dark_share', 'longest_dark_m')


def load_spans(cfg, split):
    spans = []
    for d in sorted(glob.glob(FM.ROOT + 'cache/train/*/') + glob.glob(FM.ROOT + 'cache/val/*/')):
        sc = S.build(d)
        lamps = FM.lamps_on_road(sc, cfg, split)
        if not lamps:
            continue
        (x0, x1), _ = FM.road_band(sc, cfg)
        dist = np.asarray(sc.meta['dist'])
        s_lo = min(float(dist.min()), min(L['s'] for L in lamps)) - cfg.margin_m
        s_hi = max(float(dist.max()) + 30, max(L['s'] for L in lamps)) + cfg.margin_m
        spans.append(dict(lamps=lamps, box=(s_lo, s_hi, x0, x1), h_ref=sc.meta['typical_height']))
    return spans


def shares(lamps, box, h_ref, cfg):
    s_lo, s_hi, x0, x1 = box
    s_ax, x_ax = np.arange(s_lo, s_hi, cfg.grid_m), np.arange(x0, x1 + 1e-6, cfg.grid_m)
    Sg, Xg = np.meshgrid(s_ax, x_ax, indexing='ij')
    total = FM.field(lamps, Sg, Xg).sum(0)
    e_ref = 1.0 / h_ref ** 2
    dark = total < cfg.dark_frac * e_ref
    centre = np.argmin(np.abs(x_ax - 0.5 * (x0 + x1)))
    return (total >= cfg.lit_frac * e_ref).mean(), dark.mean(), FM.longest_run(dark[:, centre], cfg.grid_m)


def redraw(lamps, pool, rng, cfg):
    out = copy.deepcopy(lamps)
    for L in out:
        p = np.clip(np.asarray(L['probs'], float), 0, None)
        L['intensity'] = (1.0, cfg.dim_factor, 0.0)[rng.choice(3, p=p / p.sum())]
        if L['height_source'] == 'fitted':
            L['height'] = float(np.clip(L['height'] + rng.normal(0, SIGMA), 4, 16))
        else:
            L['height'] = float(rng.choice(pool))
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--draws', type=int, default=200)
    a.add_argument('--seed', type=int, default=0)
    args = a.parse_args()
    cfg = FM.FieldConfig()
    spans = load_spans(cfg, FM.lamp_type_split())
    pool = np.array([L['height'] for sp in spans for L in sp['lamps'] if L['height_source'] == 'fitted'])
    rng = np.random.default_rng(args.seed)
    base = np.array([shares(sp['lamps'], sp['box'], sp['h_ref'], cfg) for sp in spans])
    draws = np.array([[shares(redraw(sp['lamps'], pool, rng, cfg), sp['box'], sp['h_ref'], cfg) for _ in range(args.draws)] for sp in spans])
    out = {}
    for j, m in enumerate(METRICS):
        dev = np.abs(draws[:, :, j] - base[:, None, j])
        rho = [spearmanr(base[:, j], draws[:, k, j])[0] for k in range(args.draws)]
        out[m] = dict(median_abs_change=float(np.median(dev)), p90_abs_change=float(np.percentile(dev, 90)), median_street_rank_rho=float(np.median(rho)))
        print(m, {k: round(v, 3) for k, v in out[m].items()})
    os.makedirs(FM.ROOT + 'eval', exist_ok=True)
    json.dump(dict(draws=args.draws, height_sigma=SIGMA, spans=len(spans), results=out), open(FM.ROOT + 'eval/uncertainty_mc.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
