"""Known-answer test of the audit: synthetic streets on the real drives, the full pipeline, scored against the true light map.

Street: poles every U(25, 45) m on one or both sides, 4-9 m from the camera line, 7-12 m high; On 80%, Dim 12%, Off 8%; white beam.
Observation: Off lamps are never detected; a lamp 8-80 m ahead in the image is sighted with probability 0.75; 10% of tracks break in two;
box centres get 2 px and 0.5 deg tilt noise; state probabilities are drawn from real classifier outputs for lamps of that state.
--fps keeps only the sightings on frames the detector would process at that rate (GPS and gravity stay at full rate).
Scores: lamp recovery, position and height error, state accuracy, lit/dark cell agreement, dark-gap error and loss-if-off ranking.

    python -m sliq.physics.audit_synth --fps 5"""
import collections
import glob
import json
import os
import shutil

import numpy as np
from scipy.stats import spearmanr

from sliq.common.boxes import iou
from sliq.physics import fieldmodel as FM
from sliq.physics import scene as S

ROOT = FM.ROOT
TMP = ROOT + 'synth_audit_tmp/'
STATES = (('On', 0.80, 1.0), ('Dim', 0.12, 0.5), ('Off', 0.08, 0.0))


def real_state_probs():
    """True state -> per-lamp mean classifier probabilities of real lamps (train and val spans)."""
    from sliq.detector import ignore_data
    R = ignore_data.frame_records()
    out = collections.defaultdict(list)
    for d in sorted(glob.glob(ROOT + 'cache/train/*/') + glob.glob(ROOT + 'cache/val/*/')):
        recs = json.load(open(d + 'frames.json'))['frames']
        votes, probs = collections.defaultdict(list), collections.defaultdict(list)
        for r in recs:
            for l in r['lamps']:
                probs[l['id']].append(l['probs'])
            rec = R.get(r['labelled_stem']) if r['labelled_stem'] else None
            if not rec:
                continue
            for l in r['lamps']:
                best, st = 0.3, None
                for o in rec['objects']:
                    if o.get('state') in ('on', 'dim'):
                        x, y, w, h = o['bbox_xywh']
                        v = iou(l['box'], [x, y, x + w, y + h])
                        if v >= best:
                            best, st = v, o['state']
                if st:
                    votes[l['id']].append(st)
        for tid, v in votes.items():
            out[collections.Counter(v).most_common(1)[0][0]].append(np.mean(probs[tid], 0).tolist())
    return {'On': out['on'], 'Dim': out['dim']}


def white_ring(split):
    r = float(np.exp(split['white_mean']))
    return [r * 0.1, 0.1, 0.1]


def make_street(rng, travelled):
    lamps = []
    sides = [-1, 1] if rng.random() < 0.5 else [rng.choice([-1, 1])]
    for side in sides:
        s = rng.uniform(-10, 20)
        while s < travelled + 30:
            st = rng.choice(len(STATES), p=[p for _, p, _ in STATES])
            lamps.append(dict(s=s, x=side * rng.uniform(4, 9), height=rng.uniform(7, 12), state=STATES[st][0], intensity=STATES[st][2]))
            s += rng.uniform(25, 45)
    return lamps


def processed_frames(recs, fps):
    """Indices of the frames a detector running at fps would see."""
    t = np.array([r['t'] for r in recs])
    return set(int(np.argmin(np.abs(t - x))) for x in np.arange(t[0], t[-1] + 1e-9, 1.0 / fps))


def observe(rng, lamps, recs, cams, dist, cfg, probs_by_state, ring, fps=5.0):
    per_frame = [[] for _ in recs]
    nid = 1
    for k, L in enumerate(lamps):
        if L['state'] == 'Off':
            continue
        p = probs_by_state[L['state']][rng.integers(len(probs_by_state[L['state']]))]
        ids = [nid, nid + 1] if rng.random() < 0.1 else [nid]
        nid += len(ids)
        seen = []
        for i, cam in enumerate(cams):
            rel = np.array([L['x'], cfg.cam_height_m - L['height'], L['s'] - dist[i]])
            if not (8 <= np.linalg.norm(rel) <= 80) or rel[2] <= 1:
                continue
            u, v, front = cam.project(rel[None])
            u, v = u[0], v[0] + cam.f * np.tan(np.radians(rng.normal(0, 0.5)))
            u, v = u + rng.normal(0, 2), v + rng.normal(0, 2)
            if not front[0] or not (6 <= u <= 1274 and 6 <= v <= 714) or rng.random() < 0.25:
                continue
            seen.append((i, u, v, cam.f * 0.6 / np.linalg.norm(rel)))
        cut = len(seen) // 2
        for j, (i, u, v, w) in enumerate(seen):
            tid = ids[0] if (len(ids) == 1 or j < cut) else ids[1]
            w = max(3.0, w)
            per_frame[i].append(dict(id=int(tid), box=[u - w / 2, v - w / 2, u + w / 2, v + w / 2], score=0.9, probs=list(p), ring=ring, truth=k))
    if fps < 5.0:                       # the cache is already at 5 fps
        keep = processed_frames(recs, fps)
        per_frame = [ls if i in keep else [] for i, ls in enumerate(per_frame)]
    return per_frame


def truth_metrics(lamps, s_lo, s_hi, x0, x1, cfg, h_ref):
    T = []
    for L in lamps:
        T.append(dict(L, ids=[0], n=cfg.n_white, side='left' if L['x'] < 0 else 'right', height_source='fitted', state_conf=1.0, type='white',
                      near=True, n_frames=10))
    return T, FM.grid_metrics(T, s_lo, s_hi, x0, x1, cfg, h_ref, keep_grid=True)


def one(span_dir, seed, probs_by_state, split, cfg, fps=5.0):
    rng = np.random.default_rng(seed)
    meta = json.load(open(span_dir + 'frames.json'))
    recs = meta['frames']
    dist = S.robust_distance(recs)
    if dist is None:
        return None
    scfg = S.SceneConfig()
    cams = [S.camera_for(r, scfg) for r in recs]
    lamps = make_street(rng, float(dist[-1] - dist[0]))
    per = observe(rng, lamps, recs, cams, dist, scfg, probs_by_state, white_ring(split), fps)
    d = TMP + f'{os.path.basename(span_dir.rstrip("/"))}_{seed}/'
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(span_dir):
        if f.endswith('.png') and not os.path.exists(d + f):
            os.symlink(span_dir + f, d + f)
    new = []
    for r, ls, x in zip(recs, per, dist):
        new.append(dict(r, lamps=ls, dist_m=float(x), confounders=[], detections=[]))
    json.dump(dict(meta, frames=new), open(d + 'frames.json', 'w'))
    sc = S.build(d)
    found = FM.lamps_on_road(sc, cfg, split)
    (x0, x1), _ = FM.road_band(sc, cfg)
    s_lo, s_hi = float(dist.min()) - cfg.margin_m, float(dist.max()) + 30 + cfg.margin_m
    h_ref = sc.meta['typical_height']
    T, tm = truth_metrics(lamps, s_lo, s_hi, x0, x1, cfg, h_ref)
    pm = FM.grid_metrics(found, s_lo, s_hi, x0, x1, cfg, h_ref, keep_grid=True) if found else None
    shutil.rmtree(d)
    # true lamp of each found lamp: majority over its sightings
    truth_of = {}
    for r in new:
        for l in r['lamps']:
            truth_of.setdefault(l['id'], []).append(l['truth'])
    res = dict(span=os.path.basename(span_dir.rstrip('/')), seed=seed, true_lamps=len(lamps),
               true_lit=[L for L in lamps if L['state'] != 'Off'].__len__(), found=len(found))
    match = {}
    for k, F in enumerate(found):
        t = collections.Counter(sum((truth_of[i] for i in F['ids']), [])).most_common(1)[0][0]
        if abs(lamps[t]['s'] - F['s']) <= 5 and t not in match.values():
            match[k] = t
    res['matched'] = len(match)
    res['pos_err'] = [float(np.hypot(found[k]['s'] - lamps[t]['s'], found[k]['x'] - lamps[t]['x'])) for k, t in match.items()]
    res['h_err'] = [float(found[k]['height'] - lamps[t]['height']) for k, t in match.items() if found[k]['height_source'] == 'fitted']
    res['state'] = [(found[k]['state'], lamps[t]['state']) for k, t in match.items()]
    if pm is None:
        return res
    tg, pg = tm['_grid'], pm['_grid']
    c = tg['centre']
    # lit and dark agreement in 5 m cells along the road centre line
    step = int(round(5 / cfg.grid_m))
    tl = tg['lit'][:, c]; pl = pg['lit'][:, c]; td = tg['dark'][:, c]; pd = pg['dark'][:, c]
    nb = len(tl) // step
    cell = lambda a: a[:nb * step].reshape(nb, step).mean(1) >= 0.5
    res['cells'] = dict(lit=(cell(tl).tolist(), cell(pl).tolist()), dark=(cell(td).tolist(), cell(pd).tolist()))
    res['lit_share'] = (tm['span_lit_share'], pm['span_lit_share'])
    res['longest_dark'] = (tm['span_longest_dark_m'], pm['span_longest_dark_m'])
    tl_loss = {i: L['loss_if_off_m2'] for i, L in enumerate(tm['lamps'])}
    res['loss'] = [(pm['lamps'][k]['loss_if_off_m2'], tl_loss[t]) for k, t in match.items()]
    return res


def summarise(rows):
    s = {}
    tl = sum(r['true_lit'] for r in rows); m = sum(r['matched'] for r in rows); f = sum(r['found'] for r in rows)
    s['lamps_true_lit'], s['recall_lit_lamps'], s['precision_found_lamps'] = tl, m / tl, m / max(f, 1)
    pe = np.concatenate([r['pos_err'] for r in rows]); he = np.abs(np.concatenate([r['h_err'] for r in rows]))
    s['position_err_median_m'], s['position_err_p90_m'] = float(np.median(pe)), float(np.percentile(pe, 90))
    s['height_err_median_m'], s['height_within_1m'] = float(np.median(he)), float(np.mean(he <= 1))
    st = [x for r in rows for x in r['state']]
    s['state_accuracy'] = float(np.mean([a == b for a, b in st]))
    dims = [(a, b) for a, b in st if b == 'Dim']; ons = [(a, b) for a, b in st if b == 'On']
    s['dim_recall'] = float(np.mean([a == 'Dim' for a, b in dims])) if dims else None
    s['on_recall'] = float(np.mean([a == 'On' for a, b in ons]))
    R = [r for r in rows if 'cells' in r]
    for k in ('lit', 'dark'):
        t = np.concatenate([r['cells'][k][0] for r in R]); p = np.concatenate([r['cells'][k][1] for r in R])
        tp = (t & p).sum(); s[f'{k}_cells_accuracy'] = float((t == p).mean())
        s[f'{k}_cells_precision'] = float(tp / max(p.sum(), 1)); s[f'{k}_cells_recall'] = float(tp / max(t.sum(), 1))
    for k in ('lit_share', 'longest_dark'):
        a = np.array([r[k] for r in R])
        s[f'{k}_spearman'] = float(spearmanr(a[:, 0], a[:, 1])[0]); s[f'{k}_median_abs_err'] = float(np.median(np.abs(a[:, 0] - a[:, 1])))
    loss = [(np.array(r['loss'])) for r in R if len(r['loss']) >= 4]
    rho = [spearmanr(l[:, 0], l[:, 1])[0] for l in loss if np.ptp(l[:, 1]) > 0]
    s['loss_if_off_rank_rho_median_per_street'] = float(np.nanmedian(rho))
    top = []
    for l in loss:
        top.append(int(np.argmax(l[:, 0]) in np.argsort(-l[:, 1])[:3]))
    s['most_critical_lamp_in_true_top3'] = float(np.mean(top))
    s['streets'] = len(R)
    return s


def main():
    import argparse
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--split', nargs='+', default=['test'])
    a.add_argument('--seeds', type=int, default=20)
    a.add_argument('--fps', type=float, default=5.0)
    args = a.parse_args()
    cfg = FM.FieldConfig()
    split = FM.lamp_type_split()
    probs = real_state_probs()
    print(f"real classifier outputs: On lamps {len(probs['On'])}, Dim lamps {len(probs['Dim'])}", flush=True)
    rows = []
    for d in sorted(sum((glob.glob(ROOT + f'cache/{sp}/*/') for sp in args.split), [])):
        for seed in range(args.seeds):
            r = one(d, seed, probs, split, cfg, args.fps)
            if r is not None:
                rows.append(r)
        print(os.path.basename(d.rstrip('/')), 'done', flush=True)
    s = summarise(rows)
    for k, v in s.items():
        print(f'{k:40s} {v:.3f}' if isinstance(v, float) else f'{k:40s} {v}')
    os.makedirs(ROOT + 'eval', exist_ok=True)
    out = ROOT + f"eval/audit_{'_'.join(args.split)}_{args.fps:g}fps.json"
    json.dump(dict(summary=s, args=vars(args), streets=[{k: v for k, v in r.items() if k != 'cells'} for r in rows]), open(out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
