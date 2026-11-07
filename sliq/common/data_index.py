"""Frame index: split, pack, recording, ground-truth lamps and ignore rectangles (pixel xywh, 1280x720)."""
import json
import re

import numpy as np

from sliq.common import paths
from sliq.detector import ignore_data

DATASET = paths.MAIN


def build_index(dataset=None, splits=('train', 'val', 'test')):
    """-> frames sorted by stem: dicts with stem, split, path, pack, recording, lamps[n,4], ignores[m,4]."""
    dataset = dataset or DATASET
    split_of = json.load(open(dataset + '/splits.json'))['frames']
    records = ignore_data.frame_records()
    frames = []
    for stem, split in split_of.items():
        if split not in splits:
            continue
        lamps, ignores = ignore_data.frame_boxes(records[stem])
        frames.append(dict(stem=stem, split=split, path=f'{dataset}/detector/{split}/images/{stem}.jpg', pack=stem.split('__')[0],
                           recording=re.search(r'__(\d+)_(?:segment\d+|full)__', stem).group(1), lamps=lamps, ignores=ignores))
    return sorted(frames, key=lambda f: f['stem'])


def xywh2xyxy(boxes):
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
