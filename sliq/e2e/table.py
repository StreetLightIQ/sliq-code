"""Dataset table: frames, lamps, ignore regions and confounders per split, and classifier crops per split, state and origin.

    python -m sliq.e2e.table"""
import collections
import csv
import json

import numpy as np

from sliq.classifier import dataset as classifier_data
from sliq.common import paths
from sliq.common.data_index import build_index
from sliq.detector import ignore_data


def split_table():
    """Lamp heads from the detector boxes. Ignore objects and confounders from the scene annotations."""
    ignore_data.configure(paths.MAIN, 'scene')
    scene = dict(ignore_data.frame_records())
    ignore_data.configure(paths.MAIN, 'detector')
    table = {}
    for split in ('train', 'val', 'test'):
        frames = build_index(splits=(split,))
        c = collections.Counter()
        for f in frames:
            c['lamps'] += len(f['lamps'])
            c['ignores'] += sum(o['kind'] == 'ignore' for o in scene[f['stem']]['objects'])
            c['confounders'] += sum(o['kind'] == 'negative' for o in scene[f['stem']]['objects'])
        table[split] = dict(frames=len(frames), **c, lamp_size_median=float(np.median([np.sqrt(b[2] * b[3]) for f in frames for b in f['lamps']])))
    ignore_data.configure(paths.MAIN, 'scene')
    return table


def crop_table():
    c = collections.Counter((r['split'], r['class'], r['origin']) for r in csv.DictReader(open(classifier_data.DATASET + '/classifier/index.csv')))
    out = collections.defaultdict(dict)
    for (split, state, origin), n in sorted(c.items()):
        out[split][f'{state}_{origin}'] = n
    return {k: dict(v, total=sum(v.values())) for k, v in out.items()}


if __name__ == '__main__':
    print(json.dumps(dict(frames=split_table(), classifier_crops=crop_table()), indent=1))
