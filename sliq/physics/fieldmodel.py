"""Per-lamp light map from lamp geometry and state, with no road brightness needed.

Light model: E_i(p) = I_i cos^n(theta) cos(theta) / d^2 on the ground plane, theta from the downward vertical, relative units (On lamp: I = 1).
  I_i = P(On) + dim_factor P(Dim), from the classifier's mean softmax over the track
  n   = beam exponent by colour type (white narrow, orange broad, unclear in between)
Lit: E >= lit_frac E_ref, dark: E < dark_frac E_ref, with E_ref under one On lamp at the span's typical height. The road is a strip along
the drive, as wide as the segmenter's road pixels. Each lamp serves the cells where it contributes most; per lamp we report coverage,
the longest dark gap, loss-if-off and spacing.

    python -m sliq.physics.fieldmodel --split val"""
import csv
import glob
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from sliq.common import paths
from sliq.physics import lamp_type as LT
from sliq.physics import scene as S

ROOT = paths.RUNS + '/physics/'


@dataclass
class FieldConfig:
    n_white: float = 8.0
    n_orange: float = 2.0
    n_unclear: float = 4.0
    dim_factor: float = 0.5
    lit_frac: float = 0.2
    dark_frac: float = 0.05
    grid_m: float = 0.5
    margin_m: float = 15.0
    merge_m: float = 4.0          # tracks whose feet are this close are one lamp
    road_z: tuple = (8.0, 30.0)   # forward range whose road pixels give the road width
    foot_d: tuple = (5.0, 30.0)   # usable sighting distances; never seen within 30 m -> only_seen_far
    min_road_px: int = 2000


def lamp_type_split(train_dir=None):
    """Colour split fitted on training lamps (cached)."""
    path = ROOT + 'field/lamp_type_split.json'
    if os.path.exists(path):
        return json.load(open(path))
    feats = []
    for d in sorted(glob.glob(ROOT + 'cache/train/*/')):
        sc = S.build(d)
        feats += [LT.feature(L['ring']) for L in sc.lamps.values()]
    split = LT.fit_split(feats)
    if split is None:
        raise SystemExit('too few training lamps to fit the lamp-type split')
    os.makedirs(ROOT + 'field', exist_ok=True)
    json.dump(split, open(path, 'w'), indent=1)
    return split


def road_band(sc, cfg):
    xs = []
    for f in sc.frames:
        p, ok = f.cam.ground_points(f.u, f.v)
        sel = ok & (p[:, 2] >= cfg.road_z[0]) & (p[:, 2] <= cfg.road_z[1])
        xs.append(p[sel, 0])
    x = np.concatenate(xs) if xs else np.array([])
    if len(x) < cfg.min_road_px:
        return (-5.0, 5.0), 'default'
    lo, hi = np.percentile(x, [5, 95])
    return (float(max(lo, -12)), float(min(hi, 12))), 'segmenter'


def _seen_apart(ids_a, ids_b, boxes):
    """True if a track of ids_a and one of ids_b were seen in the same frames as separate heads (median centre gap >= 1 box width)."""
    for a in ids_a:
        for b in ids_b:
            common = set(boxes.get(a, {})) & set(boxes.get(b, {}))
            if not common:
                continue
            gaps = []
            for i in common:
                A, B = np.asarray(boxes[a][i], float), np.asarray(boxes[b][i], float)
                w = max(A[2] - A[0], A[3] - A[1], B[2] - B[0], B[3] - B[1], 1.0)
                gaps.append(np.linalg.norm((A[:2] + A[2:]) / 2 - (B[:2] + B[2:]) / 2) / w)
            if np.median(gaps) >= 1.0:
                return True
    return False


def lamps_on_road(sc, cfg, split):
    """Lamp feet in the road frame, merged across broken tracks unless they were seen together."""
    dist = np.asarray(sc.meta['dist'])
    boxes = {}
    for i, rec in enumerate(json.load(open(os.path.join(sc.meta['span'], 'frames.json')))['frames']):
        for l in rec['lamps']:
            boxes.setdefault(l['id'], {})[i] = l['box']
    raw = []
    for tid, L in sc.lamps.items():
        # placed from its nearest sightings, where angle errors and road bends matter least
        sight = sorted(((float(np.linalg.norm(p)), dist[i] + p[2], p[0]) for i, p in L['det'].items()), key=lambda t: t[0])
        ok = [t for t in sight if t[0] >= cfg.foot_d[0]] or sight
        near = ok[0][0] <= cfg.foot_d[1]
        s, x = np.median(np.array([(t[1], t[2]) for t in ok[:3]]), 0)
        pr = np.asarray(L['probs_mean'])
        raw.append(dict(ids=[tid], s=float(s), x=float(x), height=L['height'], height_source=L['height_source'], height_rms=L['height_rms'],
                        n_frames=L['n_frames'], probs=pr, ring=L['ring'], near=near))
    raw.sort(key=lambda r: r['s'])
    merged = []
    for r in raw:
        m = next((q for q in merged if abs(q['s'] - r['s']) <= cfg.merge_m and abs(q['x'] - r['x']) <= cfg.merge_m
                  and not _seen_apart(q['ids'], r['ids'], boxes)), None)
        if m is None:
            merged.append(r)
            continue
        w1, w2 = m['n_frames'], r['n_frames']
        for k in ('s', 'x'):
            m[k] = (m[k] * w1 + r[k] * w2) / (w1 + w2)
        m['probs'] = (m['probs'] * w1 + r['probs'] * w2) / (w1 + w2)
        if r['height_source'] == 'fitted' and m['height_source'] != 'fitted':
            m.update(height=r['height'], height_source='fitted', height_rms=r['height_rms'])
        m['ids'] += r['ids']; m['n_frames'] += r['n_frames']; m['near'] |= r['near']
        m['ring'] = m['ring'] or r['ring']
    for m in merged:
        p = m['probs']
        m['state'] = ('On', 'Dim', 'Off')[int(np.argmax(p))]
        m['state_conf'] = float(np.max(p))
        m['intensity'] = float(p[0] + cfg.dim_factor * p[1])
        m['type'] = LT.classify(m['ring'], split)
        m['n'] = {'white': cfg.n_white, 'orange': cfg.n_orange}.get(m['type'], cfg.n_unclear)
        m['side'] = 'left' if m['x'] < 0 else 'right'
    return merged


def field(lamps, S_grid, X_grid, which=None):
    """Illuminance per lamp on the grid -> array (n_lamps, ns, nx)."""
    out = np.zeros((len(lamps),) + S_grid.shape)
    for k, L in enumerate(lamps):
        if which is not None and k not in which:
            continue
        dz, dx = S_grid - L['s'], X_grid - L['x']
        d2 = dz ** 2 + dx ** 2 + L['height'] ** 2
        c = L['height'] / np.sqrt(d2)
        out[k] = L['intensity'] * c ** (L['n'] + 1) / d2
    return out


def longest_run(mask, step):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * step


def analyse(span_dir, cfg=FieldConfig(), split=None):
    sc = S.build(span_dir)
    split = split if split is not None else lamp_type_split()
    lamps = lamps_on_road(sc, cfg, split)
    (x0, x1), band_src = road_band(sc, cfg)
    dist = np.asarray(sc.meta['dist'])
    out = dict(span=os.path.basename(span_dir.rstrip('/')), road_band=(x0, x1), road_band_source=band_src, dist_source=sc.meta['dist_source'],
               typical_height=sc.meta['typical_height'], travelled_m=sc.meta['travelled_m'], config=asdict(cfg), lamps=[])
    if not lamps:
        return out
    s_lo = min(float(dist.min()), min(L['s'] for L in lamps)) - cfg.margin_m
    s_hi = max(float(dist.max()) + 30, max(L['s'] for L in lamps)) + cfg.margin_m
    out.update(grid_metrics(lamps, s_lo, s_hi, x0, x1, cfg, sc.meta['typical_height']))
    return out


def grid_metrics(lamps, s_lo, s_hi, x0, x1, cfg, h_ref, keep_grid=False):
    """Light map on the strip [s_lo, s_hi] x [x0, x1] and the span and per-lamp metrics."""
    s_ax = np.arange(s_lo, s_hi, cfg.grid_m); x_ax = np.arange(x0, x1 + 1e-6, cfg.grid_m)
    Sg, Xg = np.meshgrid(s_ax, x_ax, indexing='ij')
    per = field(lamps, Sg, Xg)
    total = per.sum(0)
    e_ref = 1.0 / h_ref ** 2
    lit, dark = total >= cfg.lit_frac * e_ref, total < cfg.dark_frac * e_ref
    owner = np.argmax(per, 0)
    cell = cfg.grid_m ** 2
    centre = np.argmin(np.abs(x_ax - 0.5 * (x0 + x1)))
    out = dict(lamps=[])
    out['span_lit_share'] = float(lit.mean())
    out['span_dark_share'] = float(dark.mean())
    out['span_longest_dark_m'] = longest_run(dark[:, centre], cfg.grid_m)
    by_side = {}
    for k, L in enumerate(lamps):
        by_side.setdefault(L['side'], []).append(k)
    for k, L in enumerate(lamps):
        mine = owner == k
        served_s = s_ax[mine.any(1)]
        others = total - per[k]
        lost = (lit & (others < cfg.lit_frac * e_ref)).sum() * cell
        side = sorted(by_side[L['side']], key=lambda j: lamps[j]['s'])
        pos = side.index(k)
        prev_gap = L['s'] - lamps[side[pos - 1]]['s'] if pos > 0 else None
        next_gap = lamps[side[pos + 1]]['s'] - L['s'] if pos + 1 < len(side) else None
        rows = mine.any(1)
        flags = []
        if L['height_source'] != 'fitted': flags.append('height_assumed')
        if L['state_conf'] < 0.6: flags.append('state_uncertain')
        if L['type'] == 'unclear': flags.append('type_unclear')
        if not L['near']: flags.append('only_seen_far')
        if L['n_frames'] < 3: flags.append('few_sightings')
        out['lamps'].append(dict(
            lamp=k, track_ids=L['ids'], s_m=round(L['s'], 1), x_m=round(L['x'], 1), side=L['side'], height_m=round(L['height'], 1),
            height_source=L['height_source'], state=L['state'], state_conf=round(L['state_conf'], 2), intensity=round(L['intensity'], 2),
            type=L['type'], beam_n=L['n'], spacing_prev_m=None if prev_gap is None else round(prev_gap, 1),
            spacing_next_m=None if next_gap is None else round(next_gap, 1),
            served_m=round(float(served_s.max() - served_s.min() + cfg.grid_m), 1) if len(served_s) else 0.0,
            coverage=round(float(lit[mine].mean()), 3) if mine.any() else None,
            longest_dark_m=longest_run(dark[rows, centre], cfg.grid_m) if rows.any() else 0.0,
            loss_if_off_m2=round(float(lost), 1), loss_if_off_share=round(float(lost / max(mine.sum() * cell, cell)), 3),
            reported=not ({'height_assumed', 'state_uncertain', 'only_seen_far'} & set(flags)) and L['n_frames'] >= 3, flags=flags))
    if keep_grid:
        out['_grid'] = dict(s=s_ax, x=x_ax, lit=lit, dark=dark, centre=centre)
    return out


def run(split, cfg=FieldConfig(), name='field'):
    sp = lamp_type_split()
    out_dir = ROOT + f'field/{name}/{split}/'
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(ROOT + f'cache/{split}/*/')):
        res = analyse(d, cfg, sp)
        json.dump(res, open(out_dir + res['span'] + '.json', 'w'), indent=1)
        for L in res['lamps']:
            rows.append(dict(span=res['span'], **{k: (';'.join(map(str, v)) if isinstance(v, list) else v) for k, v in L.items()}))
        print(f"{split}/{res['span']}: {len(res['lamps'])} lamps, lit {res.get('span_lit_share', float('nan')):.2f}, "
              f"longest dark {res.get('span_longest_dark_m', 0):.0f} m, road {res['road_band']}", flush=True)
    if rows:
        with open(out_dir + 'lamps.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--split', nargs='+', default=['train', 'val'])
    for sp in a.parse_args().split:
        run(sp)
