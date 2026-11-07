"""Classifier crops: the box scaled 1.5x about its centre, clipped to the frame and resized to 224x224. Real crops are re-cut from the
frame and synthetic ones from their stored context image, so box jitter can be applied. Brightness and contrast are never changed."""
import csv
import random

import numpy as np
import torch
from PIL import Image, ImageFilter

from sliq.common import paths
from sliq.detector import ignore_data
from sliq.models.builders import STATES

DATASET = paths.MAIN
CLASSES = STATES
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
CLASS_WEIGHTS = torch.tensor([0.55, 1.29, 1.16])           # sqrt-inverse frequency, in CLASSES order
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
AUGMENTATION = dict(flip=0.5, jitter=0.3, scale=0.3, blur=0.1, noise=0.1)       # probabilities
NO_AUGMENTATION = dict(flip=0.0, jitter=0.0, scale=0.0, blur=0.0, noise=0.0)


def pad_box(box_xywh, factor, width, height):
    """Scale a box about its centre by `factor`; returns clipped xyxy."""
    x, y, w, h = box_xywh
    cx, cy, w, h = x + w / 2, y + h / 2, w * factor, h * factor
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(width, cx + w / 2), min(height, cy + h / 2)]


def perturb_box(box_xywh, rng, shift=0.0, scale=(1.0, 1.0)):
    """Random shift (fraction of w and h) and scale of an xywh box, about its centre."""
    x, y, w, h = box_xywh
    s = rng.uniform(*scale)
    cx = x + w / 2 + rng.uniform(-shift, shift) * w
    cy = y + h / 2 + rng.uniform(-shift, shift) * h
    return [cx - w * s / 2, cy - h * s / 2, w * s, h * s]


def cut(image, box_xyxy):
    x0, y0 = int(np.floor(box_xyxy[0])), int(np.floor(box_xyxy[1]))
    x1, y1 = max(int(np.ceil(box_xyxy[2])), x0 + 1), max(int(np.ceil(box_xyxy[3])), y0 + 1)
    return image.crop((x0, y0, x1, y1))


def to_tensor(image, size=224):
    a = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), np.float32) / 255.0
    return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy())


def load_samples(split, dataset=None):
    """Samples of one split from classifier/index.csv."""
    dataset = dataset or DATASET
    records = ignore_data.frame_records()
    rows = list(csv.DictReader(open(dataset + '/classifier/index.csv')))
    samples = []
    for r in rows:
        if r['split'] != split:
            continue
        sample = dict(cls=CLASS_INDEX[r['class']], origin=r['origin'], name=r['name'], pack=r['pack'] or '')
        if r['origin'] == 'real':
            pack, rest = r['name'].split('__', 1)
            stem, obj_id = rest.rsplit('__', 1)
            stem = pack + '__' + stem
            sample.update(frame=f'{dataset}/detector/{split}/images/{stem}.jpg', box=_box(records, stem, obj_id), pack=pack)
        else:                                                      # synth_<class>_<i>__<pack>__<stem>__<object id>
            parts = r['name'].split('__')
            pack, stem, obj_id = parts[1], parts[1] + '__' + '__'.join(parts[2:-1]), parts[-1]
            sample.update(context=f"{dataset}/classifier/train_synthetic_context/{r['class']}/{r['name']}.png", box=_box(records, stem, obj_id), pack=pack)
        samples.append(sample)
    return samples


def _box(records, stem, obj_id):
    return [float(v) for v in next(o for o in records[stem]['objects'] if o['id'] == obj_id)['bbox_xywh']]


class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, samples, augment=False, pad=1.5, seed=0):
        self.samples, self.aug, self.pad, self.seed = samples, AUGMENTATION if augment else NO_AUGMENTATION, pad, seed
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _source(self, d):
        """-> (image, box_xywh in that image)."""
        if d['origin'] != 'synthetic':
            return Image.open(d['frame']).convert('RGB'), list(d['box'])
        image = Image.open(d['context']).convert('RGB')                    # square context around the box
        x, y, w, h = d['box']
        ox = int(max(0, min(1280 - image.width, x + w / 2 - image.width / 2)))
        oy = int(max(0, min(720 - image.width, y + h / 2 - image.width / 2)))
        return image, [x - ox, y - oy, w, h]

    def __getitem__(self, i):
        d = self.samples[i]
        rng = random.Random((self.seed * 1000003 + self.epoch) * 1000003 + i)
        aug = self.aug
        image, box = self._source(d)
        if rng.random() < aug['jitter']:
            box = perturb_box(box, rng, shift=0.10)
        if rng.random() < aug['scale']:
            box = perturb_box(box, rng, scale=(0.8, 1.25))
        crop = cut(image, pad_box(box, self.pad, *image.size))
        flip, blur, noise = rng.random() < aug['flip'], rng.random() < aug['blur'], rng.random() < aug['noise']
        if flip:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if blur:
            crop = crop.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
        t = to_tensor(crop)
        if noise:
            t = t + torch.randn_like(t) * (2.0 / 255.0 / 0.226)                # about 2 grey levels
        return t, d['cls'], int(d['origin'] == 'synthetic')


def sampler_weights(samples, dim_real_share=0.5):
    """Per-sample probabilities: classes equally likely, Dim drawn half from real and half from synthetic crops."""
    weights = np.zeros(len(samples))
    cls = np.array([s['cls'] for s in samples])
    synthetic = np.array([s['origin'] == 'synthetic' for s in samples])
    for c in range(3):
        idx = np.where(cls == c)[0]
        real, fake = idx[~synthetic[idx]], idx[synthetic[idx]]
        if c == CLASS_INDEX['Dim'] and len(real) and len(fake):
            weights[real], weights[fake] = dim_real_share / len(real), (1 - dim_real_share) / len(fake)
        elif len(idx):
            weights[idx] = 1.0 / len(idx)
    return weights / weights.sum()
