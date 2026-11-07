"""Collect finished runs into per-run tables and mean +- sample SD across seeds.

    python -m sliq.runner.report [--out results.md]"""
import argparse
import collections
import glob
import json
import re

import numpy as np

from sliq.common.paths import RUNS


def _detector(selection):
    """mAP at the selected checkpoint; recall, precision and FP/frame at the run's tau."""
    v = selection['val_at_025']
    tau = selection.get('tau_star')
    row = next((r for r in selection.get('curve', []) if abs(r[0] - tau) < 1e-6), None) if tau is not None else None
    recall, fp, precision = (row[1], row[2], row[3]) if row else (v['recall'], v['fp_per_frame'], v['precision'])
    return dict(mAP50=v['mAP50'], mAP50_95=v['mAP50_95'], recall=recall, precision=precision, fp_per_frame=fp, centre_err=v['centre_err_diag_median'], tau=tau)


def _classifier(selection):
    v = selection['val']
    return dict(val_loss=v['loss'], balanced_on_dim=v['balanced_on_dim'], on=v['recall_counts']['On'], dim=v['recall_counts']['Dim'])


def _segmenter(selection):
    v = selection['val']
    return dict(mean_iou=v['mean_iou'], light_iou=v['light_iou'], road_iou=v['road_iou'], building_iou=v['building_iou'], light_precision=v['light_precision'], light_recall=v['light_recall'])


EXTRACT = {'detector': _detector, 'classifier': _classifier, 'segmentation': _segmenter}


def collect():
    """-> {task: {run name: metrics}} for every finished run."""
    rows = {task: {} for task in EXTRACT}
    for task, extract in EXTRACT.items():
        for path in glob.glob(f'{RUNS}/{task}/*/selection.json'):
            sel = json.load(open(path))
            if task != 'detector' or 'tau_star' in sel:          # confounder runs have no tau_star
                rows[task][path.split('/')[-2]] = extract(sel)
    return rows


def _format(value):
    if isinstance(value, (list, tuple)):
        return f'{value[0]}/{value[1]}'
    return '-' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)


def table(rows):
    out = []
    for task, runs in rows.items():
        if not runs:
            continue
        keys = list(next(iter(runs.values())))
        out += [f'\n### {task}\n', '| run | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 1)]
        out += [f'| {name} | ' + ' | '.join(_format(m[k]) for k in keys) + ' |' for name, m in sorted(runs.items())]
        groups = collections.defaultdict(list)
        for name, m in runs.items():
            groups[re.sub(r'_s\d+$', '', name)].append(m)
        for group, members in sorted(groups.items()):
            if len(members) > 1:
                numeric = [k for k in keys if isinstance(members[0][k], float)]
                out.append(f'\n{group} ({len(members)} seeds): ' + ', '.join(f'{k} {np.mean([m[k] for m in members]):.4f} +- {np.std([m[k] for m in members], ddof=1):.4f}' for k in numeric))
    return '\n'.join(out)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--out')
    args = a.parse_args()
    text = table(collect())
    print(text)
    if args.out:
        open(args.out, 'w').write(text)
