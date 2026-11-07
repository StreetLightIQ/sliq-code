"""Confounding light (class 3) from the confounder detector's boxes and the label rule (light_rule.py).

select(): end of `detector.yolo_train --target confounder`. Scores each kept epoch on validation (boxes >= tau -> rule -> light IoU) and keeps
the best (epoch, tau). predict_light(): {stem: bool mask} for frames with 'stem' and 'image'."""
import json
import os
import shutil

import numpy as np
from PIL import Image

from sliq.segmentation import dataset as data
from sliq.segmentation import light_rule

TAUS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
H, W = 720, 1280


def box_pixels(g, xywh):
    """Flat pixel indices of one box's rule mask (same as light_rule.light_mask for one box, but local)."""
    x, y, w, h = [int(round(v)) for v in xywh]
    x, y = max(x, 0), max(y, 0)
    if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:            # np.roll wraps at the edge, use the global version
        return np.flatnonzero(light_rule.light_mask(g, [xywh]))
    c = g[y:y + h, x:x + w]
    if c.size == 0:
        return np.zeros(0, np.int64)
    b = light_rule.close_in_box(c >= max(180, 0.7 * c.max()))
    sel = np.ones_like(b) if (b.mean() >= 0.5 or light_rule.dil(b, 2).sum() < 300) else b
    local = np.zeros((sel.shape[0] + 4, sel.shape[1] + 4), bool)
    local[2:-2, 2:-2] = sel
    local = light_rule.dil(local, 2)
    yy, xx = np.nonzero(local)
    return (yy + y - 2) * W + (xx + x - 2)


def frame_boxes(checkpoint, frames, imgsz, device, conf=min(TAUS)):
    """-> per frame [(confidence, flat pixel indices)]."""
    from sliq.detector import validator
    preds = validator.predict_yolo(checkpoint, [dict(path=f['image']) for f in frames], imgsz, conf=conf, device=device)
    out = []
    for f, (xyxy, scores) in zip(frames, preds):
        g = light_rule.gray(f['image'])
        out.append([(float(s), box_pixels(g, (b[0], b[1], b[2] - b[0], b[3] - b[1]))) for b, s in zip(xyxy, scores)])
    return out


def union(boxes, tau):
    m = np.zeros(H * W, bool)
    for s, idx in boxes:
        if s >= tau:
            m[idx] = True
    return m.reshape(H, W)


def score(per_frame, truth):
    """truth: list of (light bool, valid bool). -> {tau: dict(iou, precision, recall)}."""
    out = {}
    for tau in TAUS:
        tp = fp = fn = 0
        for boxes, (gt, valid) in zip(per_frame, truth):
            p = union(boxes, tau) & valid
            tp += int((p & gt).sum())
            fp += int((p & ~gt).sum())
            fn += int((gt & ~p).sum())
        out[tau] = dict(light_iou=tp / max(tp + fp + fn, 1), light_precision=tp / max(tp + fp, 1), light_recall=tp / max(tp + fn, 1))
    return out


def val_truth(frames):
    truth = []
    for f in frames:
        m = np.asarray(Image.open(f['mask']))
        truth.append((m == 3, m != 255))
    return truth


def select(run_dir, imgsz, eval_every, epochs_total, train_seconds=None, device='0', keep_all=False):
    weights_dir = run_dir + '/ultra/weights'
    epochs = sorted(int(f[5:-3]) for f in os.listdir(weights_dir) if f.startswith('epoch') and f.endswith('.pt'))
    scored = [k for k in epochs if k % eval_every == 0 or k > 0.7 * epochs_total or k == epochs[-1]]
    frames = data.list_frames('val')
    truth = val_truth(frames)
    best = None
    with open(run_dir + '/val_metrics.jsonl', 'w') as log:
        for k in scored:
            curve = score(frame_boxes(f'{weights_dir}/epoch{k}.pt', frames, imgsz, device), truth)
            tau = max(curve, key=lambda t: curve[t]['light_iou'])
            row = dict(epoch=k, best_tau=tau, **curve[tau], curve={str(t): v for t, v in curve.items()})
            log.write(json.dumps(row) + '\n')
            log.flush()
            if best is None or row['light_iou'] > best['light_iou']:
                best = row
    shutil.copy(f"{weights_dir}/epoch{best['epoch']}.pt", run_dir + '/selected.pt')
    json.dump(dict(best_checkpoint=f"epoch{best['epoch']}", tau=best['best_tau'], val=best, imgsz=imgsz, checkpoints_scored=scored,
                   train_seconds=None if train_seconds is None else round(train_seconds),
                   rule='validation light IoU after the label rule, over epochs x TAUS'),
              open(run_dir + '/selection.json', 'w'), indent=1)
    if not keep_all:
        for f in os.listdir(weights_dir):
            if f.startswith('epoch') and f != f"epoch{best['epoch']}.pt":
                os.remove(f'{weights_dir}/{f}')
    print('DONE', os.path.basename(run_dir), 'epoch%d tau %.2f light IoU %.4f P %.3f R %.3f' % (best['epoch'], best['best_tau'], best['light_iou'],
                                                                                            best['light_precision'], best['light_recall']))
    return best


def predict_light(checkpoint, tau, frames, imgsz=1280, device='0'):
    """-> {stem: bool mask} at confidence >= tau."""
    per_frame = frame_boxes(checkpoint, frames, imgsz, device, conf=tau)
    return {f['stem']: union(boxes, tau) for f, boxes in zip(frames, per_frame)}


def load_run(run_name):
    """-> (checkpoint, tau, imgsz) of a finished confounder run."""
    from sliq.common import paths
    run_dir = paths.run_dir('detector', run_name)
    sel = json.load(open(run_dir + '/selection.json'))
    return run_dir + '/selected.pt', sel['tau'], sel['imgsz']
