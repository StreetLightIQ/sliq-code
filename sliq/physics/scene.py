"""Cached span -> per-frame camera, usable road pixels and lamp positions, with one height per lamp.

Distance driven comes from GPS. Heights come from each lamp's sight lines and the distance driven (fit_height). Without GPS, speed is
taken from the lamps' own motion at the typical height."""
import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from sliq.physics import radiometry
from sliq.physics.geometry import Camera, camera_from_gravity

R_EARTH = 6371000.0


@dataclass
class FrameData:
    key: str
    cam: Camera
    u: np.ndarray                          # usable road pixels
    v: np.ndarray
    lamps: dict                            # lamp id -> (3,) camera-relative position
    glows: list = field(default_factory=list)       # confounding-light boxes (xyxy)


@dataclass
class SceneConfig:
    hfov_deg: float = 70.0
    cam_height_m: float = 1.3
    default_pitch_deg: float = 5.0         # when there is no accelerometer
    pixels: int = 4000
    glow_min_score: float = 0.25
    glow_min_area: float = 300.0
    glow_max: int = 20
    max_range_m: float = 80.0
    behind_m: float = 30.0
    seed: int = 0


@dataclass
class Scene:
    frames: list                           # FrameData of the scored frames (no lead-in)
    lamps: dict                            # lamp id -> dict(height, det, n_frames, probs_mean, ring, frames)
    meta: dict = field(default_factory=dict)
    raw: list = field(default_factory=list)  # frames.json records of the fitted frames, same order


def camera_for(record, cfg):
    if record.get('gravity'):
        cam = camera_from_gravity(1280, 720, record['gravity'], cfg.hfov_deg, cfg.cam_height_m)
        return cam
    return Camera(1280, 720, cfg.hfov_deg, cfg.default_pitch_deg, 0.0, cfg.cam_height_m)


def _edge(box, margin=4):
    x1, y1, x2, y2 = box
    return x1 < margin or y1 < margin or x2 > 1280 - margin or y2 > 720 - margin


def _tracks(records):
    tracks = {}
    for i, r in enumerate(records):
        for l in r['lamps']:
            tracks.setdefault(l['id'], []).append((i, l))
    return tracks


def robust_distance(records, window_s=3.0, max_speed=30.0):
    """Distance driven (m) at each record from GPS: speed between distinct fixes, rolling median over +-window_s, capped, integrated.
    None without GPS."""
    ll = np.array([r['latlon'] if r['latlon'] else [np.nan, np.nan] for r in records], float)
    t = np.array([r['t'] for r in records])
    if not np.isfinite(ll).all():
        return None
    lat0 = np.radians(ll[0, 0])
    xy = np.stack([np.radians(ll[:, 1] - ll[0, 1]) * R_EARTH * np.cos(lat0), np.radians(ll[:, 0] - ll[0, 0]) * R_EARTH], 1)
    change = np.nonzero(np.any(np.diff(xy, axis=0) != 0, axis=1))[0] + 1
    if len(change) < 3:
        return None
    ft, fxy = t[change], xy[change]
    step_t, step_v = 0.5 * (ft[1:] + ft[:-1]), np.linalg.norm(np.diff(fxy, axis=0), axis=1) / np.maximum(np.diff(ft), 0.2)
    speed = np.array([np.median(step_v[np.abs(step_t - ti) <= window_s]) if (np.abs(step_t - ti) <= window_s).any() else np.median(step_v) for ti in t])
    speed = np.minimum(speed, max_speed)
    return np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])


def _sight(cam, box):
    """Unit ray through the box centre, pitch and roll removed (x lateral, z forward)."""
    return cam.rays(np.array((box[0] + box[2]) / 2), np.array((box[1] + box[3]) / 2))


def fit_height(obs, cams, dist, cam_h, min_driven=8.0):
    """Height from sight lines and distance driven: (H - cam_h) * cot_t = Z0 - dist_t, least squares with one round of outlier
    removal. -> (H or None, rms residual in m)."""
    c, d = [], []
    for i, box in obs:
        r = _sight(cams[i], box)
        if -r[1] < 0.02:
            continue
        c.append(r[2] / -r[1]); d.append(dist[i])
    c, d = np.array(c), np.array(d)
    if len(c) < 4 or np.ptp(d) < min_driven:
        return None, None
    A = np.stack([c, -np.ones_like(c)], 1)
    keep = np.ones(len(c), bool)
    for _ in range(2):
        sol, *_ = np.linalg.lstsq(A[keep], -d[keep], rcond=None)
        res = A @ sol + d
        keep = np.abs(res) <= max(1.0, 2.5 * np.median(np.abs(res[keep])))
    return float(sol[0] + cam_h), float(np.sqrt(np.mean(res[keep] ** 2)))


def build(span_dir, cfg=SceneConfig(), lamp_filter=None, heights_prior=9.0, h_range=(4.0, 16.0)):
    """One height per lamp (fitted, else the span's median fitted height, else heights_prior). Where the lamp is detected its position is
    its sight line cut at that height, elsewhere it is shifted from the nearest detection by the distance driven."""
    meta = json.load(open(span_dir + '/frames.json'))
    records = meta['frames']
    cams = [camera_for(r, cfg) for r in records]
    if all('dist_m' in r for r in records):              # phone sessions: distance from GPS speed (cloud.process_session)
        dist = np.array([r['dist_m'] for r in records], float)
    else:
        dist = robust_distance(records)
    dist_source = 'gps'
    tracks = {tid: [(i, l) for i, l in obs] for tid, obs in _tracks(records).items()}
    if lamp_filter is not None:
        tracks = {tid: obs for tid, obs in tracks.items() if lamp_filter(tid, obs)}
    heights, fitted = {}, {}
    if dist is not None:
        for tid, obs in tracks.items():
            h, rms = fit_height([(i, l['box']) for i, l in obs if not _edge(l['box'])], cams, dist, cfg.cam_height_m)
            if h is not None and h_range[0] <= h <= h_range[1]:
                fitted[tid] = (h, rms)
    typical = float(np.median([h for h, _ in fitted.values()])) if len(fitted) >= 3 else heights_prior
    for tid in tracks:
        heights[tid] = fitted[tid][0] if tid in fitted else typical
    if dist is None:
        dist_source = 'lamp_odometry'
        t = np.array([r['t'] for r in records])
        v = []
        for tid, obs in tracks.items():
            obs = [(i, l) for i, l in obs if not _edge(l['box'])]
            for (i, a), (j, b) in zip(obs[:-1], obs[1:]):
                ra, rb = _sight(cams[i], a['box']), _sight(cams[j], b['box'])
                if 0 < j - i <= 3 and -ra[1] > 0.02 and -rb[1] > 0.02:
                    v.append(((heights[tid] - cfg.cam_height_m) * (ra[2] / -ra[1] - rb[2] / -rb[1])) / (t[j] - t[i]))
        speed = float(np.clip(np.median(v), 0, 30)) if v else 0.0
        dist = (t - t[0]) * speed
    lamps = {}
    for tid, obs in tracks.items():
        H = heights[tid]
        det = {}
        for i, l in obs:
            r = _sight(cams[i], l['box'])
            if -r[1] > 0.02:
                det[i] = r * ((H - cfg.cam_height_m) / -r[1])
        if not det:
            continue
        lamps[tid] = dict(height=H, height_source='fitted' if tid in fitted else 'typical', height_rms=fitted.get(tid, (None, None))[1],
                          det=det, n_frames=len(obs), n_fit_frames=sum(not records[i]['lead_in'] for i, _ in obs),
                          probs_mean=np.mean([l['probs'] for _, l in obs], 0).tolist(),
                          ring=np.mean([l['ring'] for _, l in obs if l['ring']], 0).tolist() if any(l['ring'] for _, l in obs) else None,
                          frames=[i for i, _ in obs])
    rng = np.random.default_rng(cfg.seed)
    frames, raw, fit_idx = [], [], []
    for i, r in enumerate(records):
        if r['lead_in']:
            continue
        k = r['k']
        rgb = np.asarray(Image.open(f'{span_dir}/{k:05d}_rgb.png'))
        valid = np.asarray(Image.open(f'{span_dir}/{k:05d}_valid.png'))
        u, v, _ = radiometry.sample_road(rgb, valid, rng, cfg.pixels)
        rel = {}
        for tid, L in lamps.items():
            if i in L['det']:
                p = L['det'][i]
            else:
                j = min(L['det'], key=lambda j: abs(j - i))
                p = L['det'][j].copy()
                p[2] -= dist[i] - dist[j]
            if -cfg.behind_m <= p[2] <= cfg.max_range_m and abs(p[0]) <= 40:
                rel[tid] = p
        glows = sorted((c for c in r['confounders'] if c['score'] >= cfg.glow_min_score and
                        (c['box'][2] - c['box'][0]) * (c['box'][3] - c['box'][1]) >= cfg.glow_min_area), key=lambda c: -c['score'])[:cfg.glow_max]
        frames.append(FrameData(f"{k:05d}", cams[i], u, v, rel, [c['box'] for c in glows]))
        raw.append(r)
        fit_idx.append(i)
    return Scene(frames, lamps, dict(span=span_dir, dist_source=dist_source, typical_height=typical, n_height_fitted=len(fitted),
                                     cfg=cfg.__dict__, travelled_m=float(dist[-1] - dist[0]),
                                     dist=np.asarray(dist).tolist(), fit_idx=fit_idx), raw)
