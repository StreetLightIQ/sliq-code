"""Perception cache for the light model: run the released models over the drive spans once.

Spans: train/val cover each split's labelled frames (merged when gaps are <= 3 s), test spans come from
drives/test_spans.json. Each span gets up to 10 s of lead-in that is tracked but never fitted.
Frames: every 6th video frame (5 fps) plus every labelled frame.
Models: INT8 student at its validation tau, the centre/scale tracker, segmenter with flip and the
confounder rule, classifier on the 1.5x padded crop.

Per span, writes runs/physics/cache/<split>/<rec>_<start>/:
  frames.json    per frame: video time, lead-in flag, gravity, lat/lon, lamp tracks, confounder boxes
  <k>_rgb.png    640x360 RGB
  <k>_valid.png  640x360 mask: 1 usable road, 2 road but unusable, 0 not road
  <k>_labels.png 640x360 segmentation labels
    python -m sliq.physics.data --split val
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil

import cv2
import numpy as np
import torch
from PIL import Image

from sliq.common import paths
from sliq.models import released

CACHE = os.path.join(paths.RUNS, 'physics', 'cache')
W, H = 1280, 720
STEP, LEAD_IN_S, GAP_S = 6, 10.0, 3.0
SAT = 250


def video_map():
    return json.load(open(os.path.join(paths.DRIVES, 'video_map.json')))


def video_time(frame, vm):
    m = re.search(r'__(\d+)_(?:segment(\d+)|full)__f\d+__t(\d+)ms', frame['stem'])
    rec, seg, ms = m.group(1), m.group(2) or '0', int(m.group(3))
    return vm[rec]['segment_offsets'][seg] + ms / 1000.0


def labelled_spans():
    """-> {split: [dict(rec, start, end, lead, labelled={stem: t})]}."""
    from sliq.common.data_index import build_index
    vm = video_map()
    out = {}
    for split in ('train', 'val'):
        by_rec = collections.defaultdict(list)
        for f in build_index(splits=(split,)):
            by_rec[f['recording']].append((video_time(f, vm), f['stem']))
        spans = []
        for rec, items in by_rec.items():
            items.sort()
            cur = [items[0]]
            for t, s in items[1:]:
                if t - cur[-1][0] > GAP_S:
                    spans.append(dict(rec=rec, start=cur[0][0], end=cur[-1][0], labelled={s_: t_ for t_, s_ in cur}))
                    cur = []
                cur.append((t, s))
            spans.append(dict(rec=rec, start=cur[0][0], end=cur[-1][0], labelled={s_: t_ for t_, s_ in cur}))
        out[split] = spans
    test = json.load(open(os.path.join(paths.DRIVES, 'test_spans.json')))
    out['test'] = [dict(rec=d['rec'], start=float(d['start_s']), end=float(d['end_s']), labelled={}) for d in test]
    # lead-in never enters another span of the same recording
    allspans = [(s['rec'], s['start'], s['end']) for v in out.values() for s in v]
    for v in out.values():
        for s in v:
            lead = max(0.0, s['start'] - LEAD_IN_S)
            for rec, a, b in allspans:
                if rec == s['rec'] and lead < b < s['start']:
                    lead = b + 0.5
            s['lead'] = lead
    return out


def load_sensors(rec):
    path = os.path.join(paths.DRIVES, 'sensors', f'{rec}.csv')
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    acc = np.array([[float(r['acc_x']), float(r['acc_y']), float(r['acc_z'])] for r in rows])
    ll = np.array([[float(r['lat']), float(r['lon'])] for r in rows])
    return dict(acc=acc, latlon=ll)


def sensor_at(sensors, frame_index, smooth=15):
    """Gravity = accelerometer mean over +-smooth frames, lat/lon at the frame."""
    if sensors is None:
        return None, None
    n = len(sensors['acc'])
    i = min(max(frame_index, 0), n - 1)
    a, b = max(0, i - smooth), min(n, i + smooth + 1)
    return sensors['acc'][a:b].mean(0).tolist(), sensors['latlon'][i].tolist()


class Models:
    def __init__(self, device='cuda', detector=True):
        from ultralytics import YOLO
        self.device = device
        self.det_tau = released.tau('student')
        if detector:
            from sliq.deploy.tflite_runner import TFLiteDetector
            self.det = TFLiteDetector(released.path('student'), conf=self.det_tau)
        self.conf_tau = released.tau('confounder')
        self.conf = YOLO(released.path('confounder'))
        self.seg = released.segmenter(device)
        self.clf = released.classifier(device)

    def detect(self, path):
        return self.det(path)

    def confounders(self, paths_):
        out = []
        for r in self.conf.predict(paths_, imgsz=1280, conf=self.conf_tau, device=0, verbose=False, max_det=300):
            out.append((r.boxes.xyxy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy().astype(np.float64)))
        return out

    @torch.no_grad()
    def segment(self, images, conf_boxes):
        """Label maps: 0 background, 1 road, 2 building, 3 confounding light."""
        from sliq.segmentation import dataset as sdata
        from sliq.segmentation import diagnose
        from sliq.segmentation import light_from_boxes as lfb
        x = torch.from_numpy(np.stack([((im.astype(np.float32) / 255.0 - sdata.MEAN) / sdata.STD).transpose(2, 0, 1) for im in images]))
        prob = diagnose.probabilities(self.seg, x.to(self.device), flip=True).cpu()
        masks = []
        for im, (boxes, _) in zip(images, conf_boxes):
            g = np.asarray(Image.fromarray(im).convert('L'), np.float32)
            light = np.zeros(W * H, bool)
            for b in boxes:
                light[lfb.box_pixels(g, (b[0], b[1], b[2] - b[0], b[3] - b[1]))] = True
            masks.append(light.reshape(H, W))
        return diagnose.fuse(prob, torch.from_numpy(np.stack(masks))).numpy().astype(np.uint8)

    @torch.no_grad()
    def classify(self, image, boxes):
        from sliq.classifier import dataset as cdata
        if not len(boxes):
            return np.zeros((0, 3))
        pil = Image.fromarray(image)
        x = torch.stack([cdata.to_tensor(cdata.cut(pil, cdata.pad_box([b[0], b[1], b[2] - b[0], b[3] - b[1]], 1.5, W, H)))
                         for b in boxes])
        with torch.autocast('cuda', dtype=torch.float16):
            logits = self.clf(x.to(self.device)).float()
        return torch.softmax(logits, 1).cpu().numpy()


def ring_colour(raw, box):
    """Mean linear RGB of the glow ring between 1.5x and 3x the box, unsaturated pixels only."""
    from sliq.physics.radiometry import srgb_to_linear
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    X1, Y1 = int(max(0, cx - 1.5 * w)), int(max(0, cy - 1.5 * h))
    X2, Y2 = int(min(W, cx + 1.5 * w)), int(min(H, cy + 1.5 * h))
    if X2 <= X1 or Y2 <= Y1:
        return None
    patch = raw[Y1:Y2, X1:X2].astype(np.float32)
    yy, xx = np.mgrid[Y1:Y2, X1:X2]
    inner = (np.abs(xx - cx) <= 0.75 * w) & (np.abs(yy - cy) <= 0.75 * h)
    ok = ~inner & (patch.max(-1) < SAT) & (patch.max(-1) > 8)
    if ok.sum() < 20:
        return None
    return srgb_to_linear(patch[ok] / 255.0).mean(0).tolist()


def valid_mask(raw, labels, lamp_boxes, dash_row):
    """Half-res uint8: 1 usable road, 2 road but unusable, 0 not road."""
    road = labels == 1
    near_light = cv2.dilate((labels == 3).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    bad = near_light | (raw.max(-1) >= SAT)
    bad[dash_row:] = True
    for x1, y1, x2, y2 in lamp_boxes:
        bad[int(max(0, y1)):int(min(H, y2 + 1)), int(max(0, x1)):int(min(W, x2 + 1))] = True
    usable = road & ~bad
    # a half-res pixel is usable only if all 4 sources are
    u4 = usable.reshape(H // 2, 2, W // 2, 2).all((1, 3))
    r4 = road.reshape(H // 2, 2, W // 2, 2).any((1, 3))
    out = np.zeros((H // 2, W // 2), np.uint8)
    out[r4] = 2
    out[u4] = 1
    return out


def process_span(models, span, split, out_dir, batch=16):
    from sliq.segmentation.dataset import DASH
    from sliq.tracker.centre_scale import CentreScaleTracker
    rec = span['rec']
    cap = cv2.VideoCapture(os.path.join(paths.DRIVES, 'videos', video_map()[rec]['video']))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_video = 30.0 if fps_video > 60 else fps_video        # one recording declares a 120 fps time base
    first, last = int(round(span['lead'] * fps_video)), int(round(span['end'] * fps_video))
    start_idx = int(round(span['start'] * fps_video))
    labelled = {int(round(t * fps_video)): s for s, t in span['labelled'].items()}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    sensors = load_sensors(rec)
    tracker = CentreScaleTracker()
    scale = np.array([W, H, W, H], np.float64)
    tmp = out_dir + '/_jpg'
    os.makedirs(tmp, exist_ok=True)
    records, pending = [], []

    def flush():
        if not pending:
            return
        jpg_paths = [p for _, _, p, _ in pending]
        jpgs = [np.asarray(Image.open(p).convert('RGB')) for p in jpg_paths]
        dets = [models.detect(p) for p in jpg_paths]
        confs = models.confounders(jpg_paths)
        labels = models.segment(jpgs, confs)
        for (idx, raw, path, stem), jpg, (boxes, scores), (cb, cs), lab in zip(pending, jpgs, dets, confs, labels):
            t = idx / fps_video
            _, matched, _ = tracker.update([tuple(b / scale) for b in boxes], scores, t)
            tb = [list(np.array(tr.box) * scale) for tr in matched]
            probs = models.classify(jpg, tb)
            lamps = [dict(id=tr.id, box=b, score=tr.score, probs=p.round(4).tolist(), ring=ring_colour(raw, b))
                     for tr, b, p in zip(matched, tb, probs)]
            k = len(records)
            Image.fromarray(cv2.resize(raw, (W // 2, H // 2), interpolation=cv2.INTER_AREA)).save(f'{out_dir}/{k:05d}_rgb.png')
            Image.fromarray(valid_mask(raw, lab, boxes, DASH[rec])).save(f'{out_dir}/{k:05d}_valid.png')
            Image.fromarray(lab[::2, ::2]).save(f'{out_dir}/{k:05d}_labels.png')
            g, ll = sensor_at(sensors, idx)
            records.append(dict(k=k, video_frame=idx, t=t, lead_in=idx < start_idx, labelled_stem=stem, gravity=g, latlon=ll, lamps=lamps,
                                detections=[dict(box=b.tolist(), score=float(s)) for b, s in zip(boxes, scores)],
                                confounders=[dict(box=b.tolist(), score=float(s)) for b, s in zip(cb, cs)]))
            os.remove(path)
        pending.clear()

    idx = first
    while idx <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if (idx - start_idx) % STEP == 0 or idx in labelled:
            raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert raw.shape == (H, W, 3), raw.shape
            path = f'{tmp}/{idx:07d}.jpg'
            Image.fromarray(raw).save(path, quality=95)
            pending.append((idx, raw, path, labelled.get(idx)))
            if len(pending) >= batch:
                flush()
        idx += 1
    flush()
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(dict(split=split, rec=rec, start=span['start'], end=span['end'], lead=span['lead'], fps_video=fps_video, step=STEP,
                   det_tau=models.det_tau, conf_tau=models.conf_tau, frames=records), open(out_dir + '/frames.json', 'w'))
    return len(records)


def main():
    a = argparse.ArgumentParser(description='Perception cache for the light model')
    a.add_argument('--split', required=True, choices=('train', 'val', 'test'))
    a.add_argument('--limit', type=int, default=None, help='first N spans only')
    args = a.parse_args()
    spans = labelled_spans()[args.split][:args.limit]
    models = Models()
    for s in spans:
        out_dir = f"{CACHE}/{args.split}/{s['rec']}_{s['start']:07.1f}"
        if os.path.exists(out_dir + '/frames.json'):
            continue
        os.makedirs(out_dir, exist_ok=True)
        n = process_span(models, s, args.split, out_dir)
        print(args.split, s['rec'], '%.1f-%.1f lead %.1f' % (s['start'], s['end'], s['lead']), n, 'frames', flush=True)
    print('done', args.split, len(spans), 'spans')


if __name__ == '__main__':
    main()
