"""End-to-end metrics, with detector and classifier passed in as callables (ground-truth or predicted boxes).

  lamp_state_recall      share of annotated lamps that are detected (IoU >= 0.5) and given the right state
  illumination_fraction  bright share of the valid road pixels, away from confounding light"""
import csv

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from sliq.classifier import dataset as classifier_data
from sliq.common.data_index import xywh2xyxy
from sliq.detector import ignore_data, validator


def eligible_lamps(split):
    """(stem, object id) -> (state, box xywh) for the real lamps of classifier/index.csv."""
    records = ignore_data.frame_records()
    out = {}
    for r in csv.DictReader(open(classifier_data.DATASET + '/classifier/index.csv')):
        if r['split'] != split or r['origin'] != 'real':
            continue
        pack, rest = r['name'].split('__', 1)
        stem, obj_id = rest.rsplit('__', 1)
        stem = pack + '__' + stem
        box = next(o['bbox_xywh'] for o in records[stem]['objects'] if o['id'] == obj_id)
        out[(stem, obj_id)] = (r['class'], [float(v) for v in box])
    return out


def make_classifier_fn(model, device='cuda'):
    """-> classify(image_rgb, boxes_xyxy) giving one state per box (1.5x padded crop resized to 224)."""

    @torch.no_grad()
    def classify(image_rgb, boxes_xyxy):
        if len(boxes_xyxy) == 0:
            return []
        image = Image.fromarray(image_rgb)
        crops = [classifier_data.cut(image, classifier_data.pad_box([x0, y0, x1 - x0, y1 - y0], 1.5, 1280, 720)) for x0, y0, x1, y1 in boxes_xyxy]
        with torch.autocast('cuda', dtype=torch.float16):
            predicted = model(torch.stack([classifier_data.to_tensor(c) for c in crops]).to(device)).argmax(1).cpu().tolist()
        return [classifier_data.CLASSES[i] for i in predicted]

    return classify


def lamp_state_recall(frames, split, boxes_fn, classify_fn, iou_threshold=0.5):
    """boxes_fn(frame) -> thresholded xyxy detections. Pairs are matched greedily by IoU."""
    eligible = eligible_lamps(split)
    per_class = {c: dict(lamps=0, detected=0, correct=0) for c in classifier_data.CLASSES}
    for frame in frames:
        lamps = [(oid, v) for (stem, oid), v in eligible.items() if stem == frame['stem']]
        if not lamps:
            continue
        boxes = boxes_fn(frame)
        labels = classify_fn(np.asarray(Image.open(frame['path']).convert('RGB')), boxes) if len(boxes) else []
        iou = validator.iou_matrix(boxes, xywh2xyxy([box for _, (_, box) in lamps])) if len(boxes) else np.zeros((0, len(lamps)))
        matched = {}
        for flat in np.argsort(-iou, axis=None):
            i, j = divmod(int(flat), iou.shape[1])
            if iou[i, j] < iou_threshold:
                break
            if i not in matched.values() and j not in matched:
                matched[j] = i
        for j, (_, (state, _)) in enumerate(lamps):
            per_class[state]['lamps'] += 1
            if j in matched:
                per_class[state]['detected'] += 1
                per_class[state]['correct'] += int(labels[matched[j]] == state)
    total = sum(v['lamps'] for v in per_class.values())
    detected, correct = sum(v['detected'] for v in per_class.values()), sum(v['correct'] for v in per_class.values())
    return dict(per_class=per_class, lamps=total, end_to_end_recall=correct / max(total, 1), detection_recall=detected / max(total, 1),
                classification_accuracy_given_detected=correct / max(detected, 1))


def illumination_fraction(rgb, road, light, valid, luma_threshold=80, dilate=5):
    """Share of valid road pixels with luma >= luma_threshold, not within `dilate` px of confounding light."""
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    excluded = ndimage.binary_dilation(light, iterations=dilate) if light.any() else light
    area = road & valid
    return float((area & ~excluded & (luma >= luma_threshold)).sum() / max(area.sum(), 1))


def load_gt_masks(split, stem):
    """-> (road, light, valid) masks of a frame."""
    mask = np.asarray(Image.open(f'{classifier_data.DATASET}/segmentation/{split}/masks/{stem}.png'))
    return mask == 1, mask == 3, mask != 255
