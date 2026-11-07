"""Data, run and weight locations from configs/paths.yaml (overridable with SLIQ_DATA, SLIQ_RUNS, SLIQ_WEIGHTS)."""
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CFG = yaml.safe_load(open(os.path.join(REPO, 'configs', 'paths.yaml')))


def _root(key, env):
    p = os.environ.get(env, _CFG[key])
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(REPO, p))


DATA = _root('data', 'SLIQ_DATA')
RUNS = _root('runs', 'SLIQ_RUNS')
WEIGHTS = _root('weights', 'SLIQ_WEIGHTS')

MAIN = DATA + '/main'                  # splits.json, annotations/, detector/, classifier/, segmentation/
TEST_FOOTAGE = DATA + '/test_footage'  # same layout, scored once
DRIVES = DATA + '/drives'              # videos/, sensors/, video_map.json, test_spans.json


def run_dir(task, name):
    return f'{RUNS}/{task}/{name}'
