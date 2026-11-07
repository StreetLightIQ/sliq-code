"""Segmentation data: frame (1280x720), 4-class mask (0 background, 1 road, 2 building, 3 confounding light, 255 ignore) and label confidence.

Training: scale 0.75-1.15, a 576x1024 crop (half of them centred on a light or building pixel), horizontal flip, no photometric change.
Validation and test use the full frame."""
import os
import random

import numpy as np
import torch
from PIL import Image

from sliq.common import paths

DATASET = paths.MAIN
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP_HW = (576, 1024)
DASH = {'452': 615, '454': 600, '720': 552, '549': 480, '629': 660, '016': 640}     # dashboard row per recording


def pack_of(stem):
    return stem.split('__')[0]


def sparse_building_packs(frames, max_share=0.01):
    """Training packs whose mean building share is below max_share (buildings mostly not traced)."""
    share = {}
    for f in frames:
        m = np.asarray(Image.open(f['mask']))
        valid = max(int((m != 255).sum()), 1)
        share.setdefault(pack_of(f['stem']), []).append((m == 2).sum() / valid)
    return sorted(p for p, v in share.items() if np.mean(v) < max_share)


def list_frames(split, dataset=None):
    folder = f'{dataset or DATASET}/segmentation/{split}'
    return [dict(stem=f[:-4], image=f'{folder}/images/{f[:-4]}.jpg', mask=f'{folder}/masks/{f}', confidence=f'{folder}/confidence/{f}') for f in sorted(os.listdir(folder + '/masks'))]


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, frames, train=True, seed=0, scale=(0.75, 1.15), flip=0.5, rare_p=0.5, crop_hw=CROP_HW, ignore_background_packs=(), lamps_as_background=False):
        self.frames, self.train, self.seed, self.scale, self.flip, self.rare_p, self.crop_hw = frames, train, seed, scale, flip, rare_p, crop_hw
        self.ignore_background_packs = set(ignore_background_packs)     # background -> ignore in these packs
        assert train or not (self.ignore_background_packs or lamps_as_background), 'training frames only'
        self.lamp_regions = self._lamp_regions(frames) if lamps_as_background else {}
        self.epoch = 0

    def __len__(self):
        return len(self.frames)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, i):
        frame = self.frames[i]
        image, mask, confidence = Image.open(frame['image']).convert('RGB'), Image.open(frame['mask']), Image.open(frame['confidence'])
        if pack_of(frame['stem']) in self.ignore_background_packs:
            m = np.asarray(mask).copy()
            m[m == 0] = 255
            mask = Image.fromarray(m)
        if frame['stem'] in self.lamp_regions:                        # lamp heads: ignore -> background
            m = np.asarray(mask).copy()
            m[self.lamp_regions[frame['stem']] & (m == 255)] = 0
            mask = Image.fromarray(m)
        if self.train:
            image, mask, confidence = self._augment(image, mask, confidence, random.Random((self.seed * 1000003 + self.epoch) * 1000003 + i))
        else:
            image, mask, confidence = np.asarray(image, np.float32) / 255.0, np.asarray(mask, np.uint8), np.asarray(confidence, np.uint8)
        x = torch.from_numpy(((image - MEAN) / STD).transpose(2, 0, 1).copy())
        return x, torch.from_numpy(mask.astype(np.int64).copy()), torch.from_numpy(confidence.astype(np.float32).copy() / 100.0)

    @staticmethod
    def _lamp_regions(frames):
        """stem -> bool mask of lamp-head boxes (>= 300 px2) minus ignore boxes, small lamps and the dashboard."""
        import re
        from sliq.detector import ignore_data
        records, out = ignore_data.frame_records(), {}
        for f in frames:
            lamps, ignores = ignore_data.frame_boxes(records[f['stem']])
            if not len(lamps):
                continue
            region = np.zeros((720, 1280), bool)
            for x, y, w, h in np.round(lamps).astype(int):
                region[y:y + h, x:x + w] = True
            for x, y, w, h in np.round(ignores).astype(int):
                region[y:y + h, x:x + w] = False
            region[DASH[re.search(r'__(\d+)_(?:segment\d+|full)__', f['stem']).group(1)]:] = False
            out[f['stem']] = region
        return out

    def _augment(self, image, mask, confidence, rng):
        width, height = image.size
        size = (int(round(width * rng.uniform(*self.scale))), int(round(height * rng.uniform(*self.scale))))
        image = np.asarray(image.resize(size, Image.Resampling.BILINEAR), np.float32) / 255.0
        mask = np.asarray(mask.resize(size, Image.Resampling.NEAREST), np.uint8)
        confidence = np.asarray(confidence.resize(size, Image.Resampling.NEAREST), np.uint8)
        crop_h, crop_w = self.crop_hw
        pad_h, pad_w = max(0, crop_h - size[1]), max(0, crop_w - size[0])
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=255)
            confidence = np.pad(confidence, ((0, pad_h), (0, pad_w)), constant_values=0)
        top, left = rng.randint(0, mask.shape[0] - crop_h), rng.randint(0, mask.shape[1] - crop_w)
        if rng.random() < self.rare_p:                                 # centre on a light pixel, else a building pixel
            for label in (3, 2):
                ys, xs = np.where(mask == label)
                if len(ys):
                    k = rng.randrange(len(ys))
                    top = int(np.clip(ys[k] - crop_h // 2, 0, mask.shape[0] - crop_h))
                    left = int(np.clip(xs[k] - crop_w // 2, 0, mask.shape[1] - crop_w))
                    break
        window = (slice(top, top + crop_h), slice(left, left + crop_w))
        image, mask, confidence = image[window], mask[window], confidence[window]
        if rng.random() < self.flip:
            image, mask, confidence = image[:, ::-1], mask[:, ::-1], confidence[:, ::-1]
        return image, mask, confidence
