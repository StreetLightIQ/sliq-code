"""Detector scoring: mAP50, mAP50-95, recall, precision, false positives per frame, centre error, tau and checkpoint selection (1280x720 pixels).

Predictions match lamps greedily by confidence at IoU >= threshold. A prediction centred in an ignore region counts neither way."""
import json

import numpy as np

from sliq.common.data_index import build_index, xywh2xyxy

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def iou_matrix(a, b):
    """Pairwise IoU of two xyxy box arrays -> (len(a), len(b))."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    width = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    height = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = width * height
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def centre_in_regions(boxes_xyxy, regions_xywh):
    """Boolean per box: does its centre lie inside any region?"""
    boxes = np.asarray(boxes_xyxy).reshape(-1, 4)
    if len(boxes) == 0 or len(regions_xywh) == 0:
        return np.zeros(len(boxes), bool)
    cx, cy = ((boxes[:, 0] + boxes[:, 2]) / 2)[:, None], ((boxes[:, 1] + boxes[:, 3]) / 2)[:, None]
    r = np.asarray(regions_xywh, np.float64)
    inside = (cx >= r[None, :, 0]) & (cx <= r[None, :, 0] + r[None, :, 2]) & (cy >= r[None, :, 1]) & (cy <= r[None, :, 1] + r[None, :, 3])
    return inside.any(1)


def filter_frame(boxes_xyxy, scores, ignores_xywh):
    """Drop predictions whose centre is inside an ignore region."""
    boxes = np.asarray(boxes_xyxy).reshape(-1, 4)
    keep = ~centre_in_regions(boxes, ignores_xywh)
    return boxes[keep], np.asarray(scores)[keep]


def match_frame(boxes_xyxy, scores, gt_xyxy, iou_threshold):
    """-> (is_true_positive, scores, boxes, (prediction index, lamp index) pairs), all in descending score order."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    boxes, scores = np.asarray(boxes_xyxy).reshape(-1, 4)[order], np.asarray(scores)[order]
    iou = iou_matrix(boxes, gt_xyxy)
    is_tp, used, pairs = np.zeros(len(boxes), bool), set(), []
    for i in range(len(boxes)):
        candidates = [(iou[i, j], j) for j in range(iou.shape[1]) if j not in used and iou[i, j] >= iou_threshold]
        if candidates:
            j = max(candidates)[1]
            used.add(j)
            is_tp[i] = True
            pairs.append((i, j))
    return is_tp, scores, boxes, pairs


def average_precision(is_tp, scores, n_gt):
    """101-point interpolated AP (COCO style)."""
    if n_gt == 0:
        return float('nan')
    if len(is_tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    tp = is_tp[order].astype(float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = cum_tp / n_gt, cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, np.linspace(0, 1, 101), side='left')
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in index]))


def evaluate(frames, predictions, tau=0.25, iou_thresholds=IOU_THRESHOLDS):
    """frames: dicts with 'lamps' and 'ignores' (xywh). predictions: per frame (xyxy boxes, scores).
    AP is computed over all confidences; recall, precision, false positives and centre error at confidence >= tau and IoU 0.5."""
    n_gt = sum(len(f['lamps']) for f in frames)
    tp_lists = {t: ([], []) for t in iou_thresholds}
    tp = fp = 0
    centre_errors, per_frame = [], []
    for frame, (boxes, scores) in zip(frames, predictions):
        gt = xywh2xyxy(frame['lamps'])
        boxes, scores = filter_frame(boxes, scores, frame['ignores'])
        for t in iou_thresholds:
            is_tp, sorted_scores, _, _ = match_frame(boxes, scores, gt, t)
            tp_lists[t][0].append(is_tp)
            tp_lists[t][1].append(sorted_scores)
        keep = scores >= tau
        is_tp, _, matched_boxes, pairs = match_frame(boxes[keep], scores[keep], gt, 0.5)
        tp += int(is_tp.sum())
        fp += int((~is_tp).sum())
        for i, j in pairs:
            pred_centre = np.array([(matched_boxes[i, 0] + matched_boxes[i, 2]) / 2, (matched_boxes[i, 1] + matched_boxes[i, 3]) / 2])
            gt_centre = np.array([(gt[j, 0] + gt[j, 2]) / 2, (gt[j, 1] + gt[j, 3]) / 2])
            distance = np.linalg.norm(pred_centre - gt_centre)
            centre_errors.append((distance, distance / max(np.hypot(gt[j, 2] - gt[j, 0], gt[j, 3] - gt[j, 1]), 1e-9)))
        per_frame.append((len(gt), int(is_tp.sum()), int((~is_tp).sum())))
    aps = [average_precision(np.concatenate(tp_lists[t][0]), np.concatenate(tp_lists[t][1]), n_gt) if tp_lists[t][0] else 0.0 for t in iou_thresholds]
    errors = np.array(centre_errors).reshape(-1, 2)
    nan = float('nan')
    return dict(n_frames=len(frames), n_gt=n_gt, tau=tau, mAP50=aps[0], mAP50_95=float(np.mean(aps)), recall=tp / max(n_gt, 1), precision=tp / max(tp + fp, 1),
                fp_per_frame=fp / max(len(frames), 1), tp=tp, fp=fp, fn=n_gt - tp,
                centre_err_px_median=float(np.median(errors[:, 0])) if len(errors) else nan, centre_err_diag_median=float(np.median(errors[:, 1])) if len(errors) else nan,
                centre_err_diag_mean=float(errors[:, 1].mean()) if len(errors) else nan, per_frame=per_frame)


def group_metrics(frames, predictions, key, tau=0.25):
    """Metrics per value of frames[i][key]."""
    out = {}
    for value in sorted({f[key] for f in frames}):
        indices = [i for i, f in enumerate(frames) if f[key] == value]
        m = evaluate([frames[i] for i in indices], [predictions[i] for i in indices], tau)
        m.pop('per_frame')
        out[value] = m
    return out


def recall_fp_curve(frames, predictions, taus=None):
    """-> [(tau, recall, false positives per frame, precision)] at IoU 0.5."""
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2) if taus is None else taus
    rows = []
    for tau in taus:
        m = evaluate(frames, predictions, tau, iou_thresholds=[0.5])
        rows.append((float(tau), m['recall'], m['fp_per_frame'], m['precision']))
    return rows


def select_tau(frames, predictions, eps_f=0.2, taus=None):
    """Highest-recall threshold with at most eps_f false positives per frame (ties: precision, then tau). None if none qualifies."""
    rows = [r for r in recall_fp_curve(frames, predictions, taus) if r[2] <= eps_f]
    return max(rows, key=lambda r: (r[1], r[3], r[0]))[0] if rows else None


def select_checkpoint(metrics):
    """Lexicographic mAP50 > recall > precision > -FP/frame (> -centre error)."""
    return max(metrics, key=lambda k: (metrics[k]['mAP50'], metrics[k]['recall'], metrics[k]['precision'], -metrics[k]['fp_per_frame'],
                                       -metrics[k]['centre_err_diag_median']))


def predict_yolo(checkpoint, frames, imgsz=1280, conf=0.001, batch=16, device=0):
    """-> per frame (xyxy boxes, confidences) at a low confidence floor."""
    from ultralytics import YOLO
    model = YOLO(checkpoint)
    paths = [f['path'] for f in frames]
    out = []
    for i in range(0, len(paths), batch):
        for result in model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=300):
            out.append((result.boxes.xyxy.cpu().numpy().astype(np.float64), result.boxes.conf.cpu().numpy().astype(np.float64)))
    return out


def run(checkpoint, split='val', imgsz=1280, tau=0.25, groups=('recording',)):
    frames = build_index(splits=(split,))
    predictions = predict_yolo(checkpoint, frames, imgsz)
    result = evaluate(frames, predictions, tau)
    result.pop('per_frame')
    result['groups'] = {g: group_metrics(frames, predictions, g, tau) for g in groups}
    result['curve'] = recall_fp_curve(frames, predictions)
    return result


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(description='Score a YOLO checkpoint on a split: python -m sliq.detector.validator CKPT --split val')
    a.add_argument('ckpt')
    a.add_argument('--split', default='val')
    a.add_argument('--imgsz', type=int, default=1280)
    a.add_argument('--tau', type=float, default=0.25)
    a.add_argument('--out')
    args = a.parse_args()
    from sliq.common import paths
    from sliq.detector import ignore_data
    ignore_data.configure(paths.TEST_FOOTAGE if args.split == 'test_footage' else paths.MAIN)
    summary = run(args.ckpt, args.split, args.imgsz, args.tau)
    if args.out:
        json.dump(summary, open(args.out, 'w'), indent=1)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k not in ('groups', 'curve')})
