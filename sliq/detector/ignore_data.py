"""Detector dataset: lamp boxes (class 0) and ignore regions (class 1).

Ignore regions are the ignore objects plus lamps under 300 px2. They go through the Ultralytics pipeline as class-1 boxes, so augmentation
moves them with the image. ignore_loss.IgnoreAwareLoss splits them off again."""
import glob
import json
import os

import numpy as np
from ultralytics.data import build as _ultra_build
from ultralytics.data.dataset import YOLODataset

from sliq.common import paths

DATA = paths.MAIN + '/detector'
ANN = paths.MAIN + '/annotations'
IGN = 1
CONF = 2                          # confounder boxes, loaded only for the distillation masks
WITH_CONFOUNDERS = False
TARGET = 'lamp'                   # or 'confounder': confounder boxes become class 0
MIN_AREA = 300.0                  # px2 at 1280x720: smaller positives are ignore regions
_records = {}


def configure(dataset_dir, boxes='detector'):
    """Point the loader at a dataset. boxes='detector' reads the tight lamp boxes in detector/annotations when the dataset has them."""
    global DATA, ANN
    DATA, ANN = dataset_dir + '/detector', dataset_dir + '/annotations'
    if boxes == 'detector' and os.path.isdir(DATA + '/annotations'):
        ANN = DATA + '/annotations'
    _records.clear()


def frame_records():
    """stem (pack__basename) -> annotation record."""
    if not _records:
        for path in glob.glob(ANN + '/*/annotations.json'):
            pack = path.split('/')[-2]
            for rec in json.load(open(path))['images']:
                _records[pack + '__' + os.path.splitext(os.path.basename(rec['file_name']))[0]] = rec
    return _records


def frame_boxes(rec):
    """-> (lamps[n,4], ignores[m,4]) as xywh pixels."""
    lamps, ignores = [], []
    for o in rec['objects']:
        box, area = o['bbox_xywh'], o['bbox_xywh'][2] * o['bbox_xywh'][3]
        if o['kind'] == 'positive' and area >= MIN_AREA:
            lamps.append(box)
        elif o['kind'] in ('ignore', 'positive'):
            ignores.append(box)
    return np.array(lamps, np.float32).reshape(-1, 4), np.array(ignores, np.float32).reshape(-1, 4)


def frame_confounders(rec):
    """-> confounding-light boxes as xywh pixels."""
    return np.array([o['bbox_xywh'] for o in rec['objects'] if o['kind'] == 'negative'], np.float32).reshape(-1, 4)


class FloorToIgnore:
    """Runs after the geometric augmentation: a lamp that shrank under MIN_AREA becomes an ignore region."""
    stats = {'seen': 0, 'demoted': 0}

    def __init__(self, imgsz):
        self.scale2 = (imgsz / 1280.0) ** 2

    def __call__(self, labels):
        cls = labels['cls'].reshape(-1)
        if len(cls) == 0:
            return labels
        inst = labels['instances']
        b = inst.bboxes
        w, h = (b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) if inst._bboxes.format == 'xyxy' else (b[:, 2], b[:, 3])
        if inst.normalized:
            h_img, w_img = labels['img'].shape[:2]
            w, h = w * w_img, h * h_img
        small = (cls == 0) & (w * h / self.scale2 < MIN_AREA)
        FloorToIgnore.stats['seen'] += int((cls == 0).sum())
        FloorToIgnore.stats['demoted'] += int(small.sum())
        if small.any():
            labels['cls'] = np.where(small, IGN, cls).astype(np.float32).reshape(-1, 1)
        return labels


class IgnoreDataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment and TARGET == 'lamp':
            transforms.transforms.insert(len(transforms.transforms) - 1, FloorToIgnore(self.imgsz))
        return transforms

    def get_labels(self):
        records, labels = frame_records(), []
        for im_file in self.im_files:
            rec = records[os.path.splitext(os.path.basename(im_file))[0]]
            lamps, ignores = frame_boxes(rec)
            if TARGET == 'confounder':
                lamps = frame_confounders(rec)
            confounders = frame_confounders(rec) if WITH_CONFOUNDERS else np.zeros((0, 4), np.float32)
            boxes = np.concatenate([lamps, ignores, confounders]).astype(np.float32).reshape(-1, 4)
            boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / 1280
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / 720
            boxes[:, 2] /= 1280
            boxes[:, 3] /= 720
            cls = np.concatenate([np.zeros(len(lamps)), np.full(len(ignores), IGN), np.full(len(confounders), CONF)]).astype(np.float32).reshape(-1, 1)
            labels.append(dict(im_file=im_file, shape=(720, 1280), cls=cls, bboxes=np.clip(boxes, 0, 1), segments=[], keypoints=None, normalized=True, bbox_format='xywh'))
        return labels


def install():
    """Make Ultralytics build IgnoreDataset instead of YOLODataset."""
    _ultra_build.YOLODataset = IgnoreDataset


def write_yaml(path):
    name = 'lamp_head' if TARGET == 'lamp' else 'confounding_light'
    open(path, 'w').write(f"path: {DATA}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: {name}\n")
