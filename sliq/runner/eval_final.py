"""Detector and classifier evaluation on the test footage (once per name) or on validation (--split val).
Detector: metrics at the tau chosen on validation. Classifier: state accuracy on ground-truth boxes, and end to end on detector boxes.

    python -m sliq.runner.eval_final --name final --detector teacher --classifier classifier"""
import argparse
import json
import os
import time

import torch

from sliq.classifier import dataset as classifier_data
from sliq.common import data_index, paths
from sliq.common.data_index import build_index, xywh2xyxy
from sliq.detector import ignore_data, validator
from sliq.detector.runs import load_detector
from sliq.e2e import metrics as e2e
from sliq.models.builders import LampClassifier


def use_dataset(path):
    ignore_data.configure(path)
    data_index.DATASET = classifier_data.DATASET = path


def confident_boxes(frame, prediction, tau):
    """Boxes with confidence >= tau, centred outside the ignore regions."""
    boxes, scores = validator.filter_frame(*prediction, frame['ignores'])
    return boxes[scores >= tau]


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--name', required=True)
    a.add_argument('--split', default='test', choices=['val', 'test'])
    a.add_argument('--detector', required=True, help='detector run')
    a.add_argument('--classifier', required=True, help='classifier run')
    args = a.parse_args()
    out_dir = f'{paths.RUNS}/final/{args.name}_{args.split}'
    if args.split == 'test' and os.path.exists(out_dir + '/USED'):
        raise SystemExit('the test footage was already scored under this name')
    os.makedirs(out_dir, exist_ok=True)

    predict, tau, _ = load_detector(args.detector)
    use_dataset(paths.TEST_FOOTAGE if args.split == 'test' else paths.MAIN)
    frames = build_index(splits=(args.split,))
    predictions = predict(frames)
    result = dict(split=args.split, detector=args.detector, classifier=args.classifier, tau=tau)
    result['detector'] = validator.evaluate(frames, predictions, tau)
    result['detector'].pop('per_frame')

    model = LampClassifier(pretrained=False)
    model.load_state_dict(torch.load(paths.run_dir('classifier', args.classifier) + '/selected.pt', map_location='cuda'))
    classify = e2e.make_classifier_fn(model.cuda().eval())
    by_stem = {f['stem']: p for f, p in zip(frames, predictions)}
    result['state_gt_boxes'] = e2e.lamp_state_recall(frames, args.split, lambda f: xywh2xyxy(f['lamps']), classify)
    result['state_detector_boxes'] = e2e.lamp_state_recall(frames, args.split, lambda f: confident_boxes(f, by_stem[f['stem']], tau), classify)
    json.dump(result, open(out_dir + '/results.json', 'w'), indent=1)
    if args.split == 'test':
        open(out_dir + '/USED', 'w').write(time.strftime('%F %T'))
    d, g, e = result['detector'], result['state_gt_boxes'], result['state_detector_boxes']
    print('DONE', out_dir, 'detector mAP50 %.3f recall %.3f precision %.3f fp/frame %.3f' % (d['mAP50'], d['recall'], d['precision'], d['fp_per_frame']),
          '| state on gt boxes %.3f | end to end %.3f' % (g['end_to_end_recall'], e['end_to_end_recall']))


if __name__ == '__main__':
    main()
