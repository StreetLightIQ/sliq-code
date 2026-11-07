import numpy as np
import pytest

from sliq.common.data_index import build_index, xywh2xyxy
from sliq.e2e import metrics as M
from sliq.e2e import table

pytestmark = pytest.mark.data


def _perfect_classifier(eligible, frame):
    """Knows the state of every ground-truth lamp, looked up by box."""
    states = {tuple(np.round(v[1], 3)): v[0] for (stem, _), v in eligible.items() if stem == frame['stem']}
    return lambda image, boxes: [states.get(tuple(np.round([b[0], b[1], b[2] - b[0], b[3] - b[1]], 3)), 'On') for b in boxes]


def test_ground_truth_boxes_with_a_perfect_classifier_reach_the_ceiling():
    eligible = M.eligible_lamps('val')
    lamps = correct = 0
    for frame in build_index(splits=('val',)):
        result = M.lamp_state_recall([frame], 'val', lambda f: xywh2xyxy(f['lamps']), _perfect_classifier(eligible, frame))
        lamps, correct = lamps + result['lamps'], correct + round(result['end_to_end_recall'] * result['lamps'])
    assert lamps == len(eligible) and correct == lamps


def test_no_detections_give_zero_recall():
    result = M.lamp_state_recall(build_index(splits=('val',))[:60], 'val', lambda f: np.zeros((0, 4)), lambda image, boxes: [])
    assert result['end_to_end_recall'] == 0 and result['lamps'] > 0


def test_a_box_is_matched_to_one_lamp_only():
    eligible = M.eligible_lamps('val')
    frame = next(f for f in build_index(splits=('val',)) if sum(k[0] == f['stem'] for k in eligible) >= 2)
    one_box = xywh2xyxy(frame['lamps'])[:1]
    result = M.lamp_state_recall([frame], 'val', lambda f: one_box, lambda image, boxes: ['On'] * len(boxes))
    assert sum(v['detected'] for v in result['per_class'].values()) == 1


def test_illumination_fraction():
    rgb = np.zeros((20, 20, 3))
    rgb[:, 10:] = 255
    road, light, valid = np.ones((20, 20), bool), np.zeros((20, 20), bool), np.ones((20, 20), bool)
    assert M.illumination_fraction(rgb, road, light, valid) == pytest.approx(0.5)
    light[:, 10] = True
    assert M.illumination_fraction(rgb, road, light, valid, dilate=2) < 0.5


def test_split_table():
    counts = table.split_table()
    assert [counts[s]['lamps'] for s in ('train', 'val', 'test')] == [1894, 397, 237]
    assert [counts[s]['ignores'] for s in ('train', 'val', 'test')] == [8762, 1085, 791]
    assert [counts[s]['confounders'] for s in ('train', 'val', 'test')] == [15489, 3632, 2693]
