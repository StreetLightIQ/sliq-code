"""Leave-one-line-out test of the light model on published Groton illuminance measurements.

Data: Morante (2008), Mesopic Street Lighting Demonstration and Evaluation, final report for Groton
Utilities, Tables 6-8. Lamp positions are the pole stations. Height, head offset, beam exponent, lamp
strengths and background are fitted on the other lines of the same installation.
    python -m sliq.physics.groton_eval
"""
import collections
import csv
import itertools
import json
import os

import numpy as np
from scipy.optimize import nnls

from sliq.common import paths

CSV = os.path.join(os.path.dirname(__file__), 'data', 'groton_3.csv')
OUT = os.path.join(paths.RUNS, 'physics', 'eval', 'groton.json')
FT = 0.3048
LAMPS = {'Meridian Street': [0.0, 36.6, 67.1], 'Shennecossett Road': [0.0, 79.2]}
EDGES = ('near pavement edge', 'far pavement edge')
GRID = dict(x=np.arange(-4.0, 16.1, 1.0), h=(6.0, 7.5, 9.0, 10.5, 12.0), n=(0.0, 1.0, 2.0, 4.0, 8.0))


def shapes(s, y, lamps, x_l, h, n):
    cols = []
    for p in lamps:
        d2 = (s - p) ** 2 + (y - x_l) ** 2 + h ** 2
        c = h / np.sqrt(d2)
        cols.append(c ** n * c / d2)
    return np.stack(cols + [np.ones_like(s)], 1)


def fit(s, y, lux, lamps):
    best = None
    w = 1 / np.maximum(lux, 0.2)
    for x_l, h, n in itertools.product(GRID['x'], GRID['h'], GRID['n']):
        A = shapes(s, y, lamps, x_l, h, n)
        c, _ = nnls(A * w[:, None], lux * w)
        r = np.log(np.maximum(A @ c, 0.05)) - np.log(np.maximum(lux, 0.05))
        loss = float((r ** 2).sum())
        if best is None or loss < best[0]:
            best = (loss, x_l, h, n, c)
    return best


def rel_error(pred, obs):
    return float(np.mean(np.abs(pred - obs)) / np.mean(obs))


def within(pred, obs, k=1.5):
    return np.maximum(pred / obs, obs / pred) <= k


def load():
    inst = collections.defaultdict(list)
    for r in csv.DictReader(open(CSV)):
        inst[(r['street'], r['lamp_type'])].append(
            (float(r['distance_m']), float(r['offset_ft']) * FT, float(r['lux']), r['line']))
    return inst


def main():
    lines = []
    for (street, lamp), pts in load().items():
        s, y, lux = (np.array([p[i] for p in pts]) for i in range(3))
        line = np.array([p[3] for p in pts])
        for held in sorted(set(line)):
            tr, te = line != held, line == held
            _, x_l, h, n, c = fit(s[tr], y[tr], lux[tr], LAMPS[street])
            pred = shapes(s[te], y[te], LAMPS[street], x_l, h, n) @ c
            interp = np.array([lux[tr & (s == si)].mean() for si in s[te]])
            obs = lux[te]
            lines.append(dict(installation=f'{street} {lamp}', line=held, edge=held in EDGES,
                              model=rel_error(pred, obs), interp=rel_error(interp, obs),
                              model_within=int(within(pred, obs).sum()), interp_within=int(within(interp, obs).sum()),
                              n=int(te.sum()), fit=dict(head_m=float(x_l), height_m=float(h), exponent=float(n))))

    def row(name, sel):
        m = float(np.median([l['model'] for l in sel]))
        i = float(np.median([l['interp'] for l in sel]))
        print(f'{name:30s} {len(sel):3d} lines  model {100 * m:3.0f}%  interpolation {100 * i:3.0f}%')
        return dict(lines=len(sel), model=m, interp=i)

    summary = {}
    for name in dict.fromkeys(l['installation'] for l in lines):
        summary[name] = row(name, [l for l in lines if l['installation'] == name])
    summary['edges'] = row('Edges', [l for l in lines if l['edge']])
    summary['interior'] = row('Interior', [l for l in lines if not l['edge']])
    summary['all'] = row('All', lines)
    n = sum(l['n'] for l in lines)
    summary['within_1.5'] = dict(model=sum(l['model_within'] for l in lines) / n,
                                 interp=sum(l['interp_within'] for l in lines) / n, points=n)
    print(f"Within 1.5x: model {100 * summary['within_1.5']['model']:.0f}%  "
          f"interpolation {100 * summary['within_1.5']['interp']:.0f}%  ({n} points)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(summary=summary, lines=lines), open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
