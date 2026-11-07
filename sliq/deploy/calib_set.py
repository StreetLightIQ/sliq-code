"""INT8 calibration set: 300 training frames, stratified by recording. Writes runs/deploy/calib/calib_train{N}.{txt,yaml,json}.

    python -m sliq.deploy.calib_set --n 300"""
import argparse
import collections
import json
import os

import numpy as np

from sliq.common import paths
from sliq.common.data_index import build_index
from sliq.detector import ignore_data

OUT = paths.RUNS + '/deploy/calib/'


def sample(frames, n, seed=0):
    rng = np.random.default_rng(seed)
    by_rec = collections.defaultdict(list)
    for f in frames:
        by_rec[f['recording']].append(f)
    quota = {r: max(1, round(n * len(v) / len(frames))) for r, v in by_rec.items()}
    chosen = []
    for r in sorted(by_rec):
        pool = sorted(by_rec[r], key=lambda f: f['stem'])
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), min(quota[r], len(pool)), replace=False))]
    return sorted(chosen, key=lambda f: f['stem'])


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--n', type=int, default=300)
    a.add_argument('--seed', type=int, default=0)
    args = a.parse_args()
    ignore_data.configure(paths.MAIN)
    frames = build_index(dataset=paths.MAIN, splits=('train',))
    chosen = sample(frames, args.n, args.seed)
    assert all(f['split'] == 'train' for f in chosen)
    os.makedirs(OUT, exist_ok=True)
    txt = f'{OUT}calib_train{args.n}.txt'
    open(txt, 'w').write('\n'.join(f['path'] for f in chosen) + '\n')
    open(f'{OUT}calib_train{args.n}.yaml', 'w').write(f'path: {OUT}\ntrain: {txt}\nval: {txt}\nnames:\n  0: lamp_head\n')
    summary = dict(n=len(chosen), seed=args.seed, split='train', by_recording=collections.Counter(f['recording'] for f in chosen),
                   frames_with_lamps=sum(len(f['lamps']) > 0 for f in chosen), lamps=int(sum(len(f['lamps']) for f in chosen)))
    json.dump(summary, open(f'{OUT}calib_train{args.n}.json', 'w'), indent=1)
    print(summary)


if __name__ == '__main__':
    main()
