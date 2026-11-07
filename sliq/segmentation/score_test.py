"""Score the frozen segmenter once on the test footage, with the settings chosen on validation (flip + confounder detector).
A .USED flag blocks a second run under the same name.

    python -m sliq.segmentation.score_test --name final --segmenter segmenter --confounder confounder
    python -m sliq.segmentation.score_test --name baseline --segmenter seg_mbv3"""
import argparse
import json
import os
import time

from sliq.common import paths
from sliq.segmentation import diagnose

OUT = paths.RUNS + '/segmentation/test_footage/'


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--name', required=True)
    a.add_argument('--segmenter', required=True)
    a.add_argument('--arch', default='segformer_b2')
    a.add_argument('--confounder')
    a.add_argument('--confounder-tau', type=float)
    a.add_argument('--no-flip', action='store_true')
    args = a.parse_args()
    os.makedirs(OUT, exist_ok=True)
    flag = OUT + args.name + '.USED'
    assert not os.path.exists(flag), f'{flag} exists: already scored'
    confounder = diagnose.confounder_spec(args.confounder, args.confounder_tau) if args.confounder else None
    model = diagnose.load_segmenter(args.segmenter, args.arch)
    result = diagnose.evaluate(model, split='test', dataset=paths.TEST_FOOTAGE, flip=not args.no_flip, confounder=confounder)
    json.dump(dict(name=args.name, segmenter=args.segmenter, date=time.strftime('%F %T'), **result), open(OUT + args.name + '.json', 'w'), indent=1)
    open(flag, 'w').write(time.strftime('%F %T') + '\n')
    diagnose.report('TEST ' + args.name, result)


if __name__ == '__main__':
    main()
