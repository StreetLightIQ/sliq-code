"""Score frozen detector files once on the test footage, each at the tau chosen for it on validation (from its *_val.json, or a run's
selection.json for a .pt). A .USED flag blocks a second run under the same name.

    python -m sliq.deploy.test_footage --name student_n FILE [FILE ...] --imgsz 960"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from sliq.common import data_index, paths
from sliq.common.data_index import build_index
from sliq.deploy import score
from sliq.detector import ignore_data

OUT = paths.RUNS + '/deploy/test_footage/'


def test_frames():
    ignore_data.configure(paths.TEST_FOOTAGE)
    data_index.DATASET = paths.TEST_FOOTAGE
    return build_index(dataset=paths.TEST_FOOTAGE, splits=('test',))


def val_tau(path):
    if path.endswith('.pt'):
        return json.load(open(os.path.join(os.path.dirname(path), 'selection.json')))['tau_star']
    return json.load(open(path.rsplit('.', 1)[0] + '_val.json'))['own_tau']


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('files', nargs='+')
    a.add_argument('--name', required=True)
    a.add_argument('--imgsz', type=int, default=960)
    args = a.parse_args()
    os.makedirs(OUT, exist_ok=True)
    flag = OUT + args.name + '.USED'
    assert not os.path.exists(flag), f'{flag} exists: already scored'
    from sliq.deploy.export import letterbox_shape
    hw = letterbox_shape(args.imgsz)
    taus = {p: val_tau(p) for p in args.files}                         # fixed before any prediction
    frames = test_frames()
    result = dict(name=args.name, split='test footage', n_frames=len(frames), n_lamps=int(sum(len(f['lamps']) for f in frames)),
                  date=time.strftime('%F %T'), files={})
    for p in args.files:
        preds, ms = score.predict(p, frames, hw)
        tau = taus[p]
        r = dict(file=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(), tau=tau, at_val_tau=score.summarize(frames, preds, tau),
                 size_bins=score.size_bins(frames, preds, tau), cpu_ms_median=float(np.median(ms[5:])))
        result['files'][os.path.basename(p)] = r
        s = r['at_val_tau']
        print('TEST', os.path.basename(p), 'mAP50 %.3f mAP50-95 %.3f tau %s recall %.3f precision %.3f fp/frame %.3f' % (s['mAP50'], s['mAP50_95'], tau, s['recall'], s['precision'], s['fp_per_frame']), flush=True)
    json.dump(result, open(OUT + args.name + '.json', 'w'), indent=1)
    open(flag, 'w').write(time.strftime('%F %T') + '\n')


if __name__ == '__main__':
    main()
