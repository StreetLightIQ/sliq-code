"""Cloud stage for one uploaded phone session: classifier, segmenter, then the light model.

Input is a session folder as uploaded by the app (records.jsonl plus files/). Frames are the full-resolution
frames uploaded per track, deduplicated by frame id. Lamp boxes and track ids are the phone's own.
Per frame: gravity from the device sensors, GPS from the telemetry stream (interpolated), distance from the
GPS speed. The output cache has the same format as physics/data.py, so scene and fieldmodel run unchanged.
    python -m sliq.cloud.process_session <session_dir>
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np
from PIL import Image

from sliq.physics import data as DATA
from sliq.physics import fieldmodel

OUT = os.path.join(DATA.CACHE, 'sessions')


def load_session(sdir):
    recs = [json.loads(l) for l in open(os.path.join(sdir, 'records.jsonl'))]
    obs = [r['payload'] for r in recs if r['kind'] == 'OBSERVATION']
    locs = []
    for f in sorted(glob.glob(os.path.join(sdir, 'files/survey/telemetry-*.jsonl.gz'))):
        for line in gzip.open(f, 'rt'):
            r = json.loads(line)
            if r['kind'] == 'location':
                locs.append(r['data'])
    return obs, locs


def frames_of(obs, sdir):
    """Uploaded frames sorted by sensor time. frame_lamps lists every lamp in the frame."""
    out = {}
    for o in obs:
        for f in (o.get('physics_frames') or {}).get('frames') or []:
            fid = f['frame']['frame_id']
            path = os.path.join(sdir, 'files', f['storage_path'].split(o['session_id'] + '/', 1)[1])
            if fid in out or not os.path.exists(path):
                continue
            ctx = f['context']
            g = ((ctx.get('device_sensors') or {}).get('android.sensor.gravity:gravity') or {}).get('sample', {}).get('values')
            out[fid] = dict(frame=f['frame'], path=path, gravity=g, lamps=ctx.get('frame_lamps') or [])
    return sorted(out.values(), key=lambda f: f['frame']['sensor_timestamp_ns'])


def travelled(locs, utc_ms):
    """Distance (m) at each utc_ms from the integrated GPS speed."""
    t = np.array([l['utc_ms'] for l in locs], float)
    v = np.array([l['speed_mps'] if l.get('speed_mps') is not None else np.nan for l in locs], float)
    o = np.argsort(t)
    t, v = t[o], v[o]
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t) / 1000.0)])
    return np.interp(utc_ms, t, cum)


def process(sdir, dash_row=DATA.H):
    sdir = os.path.abspath(sdir)
    sid = os.path.basename(sdir.rstrip('/'))
    out_dir = os.path.join(OUT, sid[:8])
    os.makedirs(out_dir, exist_ok=True)
    obs, locs = load_session(sdir)
    frames = frames_of(obs, sdir)
    good = [l for l in locs if l['accuracy_m'] <= 25]
    lt = np.array([l['utc_ms'] for l in good], float)
    lla = np.array([[l['latitude'], l['longitude']] for l in good])
    utc = np.array([f['frame']['utc_ms'] for f in frames], float)
    dist = travelled(locs, utc)
    models = DATA.Models(detector=False)
    records = []
    for k, (fr, u, d) in enumerate(zip(frames, utc, dist)):
        raw = np.asarray(Image.open(fr['path']).convert('RGB'))
        assert raw.shape == (DATA.H, DATA.W, 3), raw.shape
        boxes = [[l['bbox']['x1'] * DATA.W, l['bbox']['y1'] * DATA.H, l['bbox']['x2'] * DATA.W, l['bbox']['y2'] * DATA.H] for l in fr['lamps']]
        cb, cs = models.confounders([fr['path']])[0]
        lab = models.segment([raw], [(cb, cs)])[0]
        probs = models.classify(raw, boxes)
        lamps = [dict(id=int(l['track_id']), box=b, score=float(l['score']), probs=p.round(4).tolist(), ring=DATA.ring_colour(raw, b))
                 for l, b, p in zip(fr['lamps'], boxes, probs)]
        Image.fromarray(np.asarray(Image.fromarray(raw).resize((DATA.W // 2, DATA.H // 2), Image.BOX))).save(f'{out_dir}/{k:05d}_rgb.png')
        Image.fromarray(DATA.valid_mask(raw, lab, boxes, dash_row)).save(f'{out_dir}/{k:05d}_valid.png')
        Image.fromarray(lab[::2, ::2]).save(f'{out_dir}/{k:05d}_labels.png')
        records.append(dict(k=k, video_frame=fr['frame']['frame_id'], t=(u - utc[0]) / 1000.0, lead_in=False, labelled_stem=None,
                            gravity=fr['gravity'], latlon=[float(np.interp(u, lt, lla[:, 0])), float(np.interp(u, lt, lla[:, 1]))],
                            dist_m=float(d), lamps=lamps,
                            detections=[dict(box=b, score=l['score']) for b, l in zip(boxes, fr['lamps'])],
                            confounders=[dict(box=b.tolist(), score=float(s)) for b, s in zip(cb, cs)]))
    json.dump(dict(split='session', rec=sid, start=0.0, end=records[-1]['t'], lead=0.0, fps_video=None, step=None,
                   conf_tau=models.conf_tau, frames=records), open(out_dir + '/frames.json', 'w'))
    print(f'{len(records)} frames, {dist[-1] - dist[0]:.0f} m -> {out_dir}')
    return out_dir


def main():
    a = argparse.ArgumentParser(description='Cloud stage for one uploaded session')
    a.add_argument('session_dir')
    a.add_argument('--dash-row', type=int, default=DATA.H, help='first image row hidden by the dashboard')
    args = a.parse_args()
    out_dir = process(args.session_dir, args.dash_row)
    res = fieldmodel.analyse(out_dir)
    json.dump(res, open(out_dir + '/audit.json', 'w'), indent=1)
    print(f"{len(res['lamps'])} lamps, lit share {res.get('span_lit_share', float('nan')):.2f}")


if __name__ == '__main__':
    main()
