"""Sensitivity of the per-lamp results to the model assumptions, on training and validation spans: each reported lamp's coverage and
loss-if-off under an alternative setting vs the default (Spearman across lamps and median absolute change).

    python -m sliq.physics.field_sensitivity"""
import glob
import json
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr

from sliq.physics import fieldmodel as FM

ALTS = {'beam n_white=4': dict(n_white=4.0), 'beam n_white=16': dict(n_white=16.0), 'beam all n=2': dict(n_white=2.0, n_unclear=2.0),
        'lit threshold 0.1': dict(lit_frac=0.1), 'lit threshold 0.4': dict(lit_frac=0.4), 'Dim strength 0.3': dict(dim_factor=0.3),
        'Dim strength 0.7': dict(dim_factor=0.7)}


def collect(cfg, split):
    out = {}
    for d in sorted(glob.glob(FM.ROOT + 'cache/train/*/') + glob.glob(FM.ROOT + 'cache/val/*/')):
        r = FM.analyse(d, cfg, split)
        for L in r['lamps']:
            out[(r['span'], L['lamp'])] = L
    return out


def main():
    split = FM.lamp_type_split()
    base = collect(FM.FieldConfig(), split)
    rep = [k for k, L in base.items() if L['reported']]
    print(f'baseline: {len(base)} lamps, {len(rep)} reported')
    res = dict(lamps=len(base), reported=len(rep), alternatives={})
    for name, ch in ALTS.items():
        alt = collect(replace(FM.FieldConfig(), **ch), split)
        row = {}
        for m in ('coverage', 'loss_if_off_share'):
            a = [base[k][m] for k in rep if base[k][m] is not None and alt[k][m] is not None]
            b = [alt[k][m] for k in rep if base[k][m] is not None and alt[k][m] is not None]
            row[m] = dict(spearman=float(spearmanr(a, b)[0]), median_abs_change=float(np.median(np.abs(np.subtract(a, b)))))
        res['alternatives'][name] = row
        print(f"{name:20s} coverage rank rho {row['coverage']['spearman']:.2f} (median change {row['coverage']['median_abs_change']:.2f}) | "
              f"loss-if-off rank rho {row['loss_if_off_share']['spearman']:.2f} (median change {row['loss_if_off_share']['median_abs_change']:.2f})", flush=True)
    json.dump(res, open(FM.ROOT + 'field/sensitivity.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
