import numpy as np
import pytest

from sliq.common import paths
from sliq.common.data_index import build_index, xywh2xyxy
from sliq.detector import ignore_data
from sliq.detector import validator as V


def frame(lamps, ignores=()):
    return dict(lamps=np.array(lamps, float).reshape(-1, 4), ignores=np.array(ignores, float).reshape(-1, 4))


def prediction(boxes_xywh, scores):
    return (xywh2xyxy(boxes_xywh) if len(boxes_xywh) else np.zeros((0, 4))), np.array(scores, float)


def test_perfect_predictions():
    frames = [frame([[10, 10, 20, 20]]), frame([[100, 50, 30, 30], [300, 50, 25, 25]])]
    preds = [prediction([[10, 10, 20, 20]], [0.9]), prediction([[100, 50, 30, 30], [300, 50, 25, 25]], [0.8, 0.7])]
    m = V.evaluate(frames, preds, tau=0.5)
    assert m['mAP50'] == pytest.approx(1.0) and m['mAP50_95'] == pytest.approx(1.0)
    assert m['recall'] == 1 and m['precision'] == 1 and m['fp_per_frame'] == 0


def test_false_positive_and_miss():
    frames = [frame([[10, 10, 20, 20]]), frame([[100, 50, 30, 30]])]
    preds = [prediction([[10, 10, 20, 20], [200, 200, 20, 20]], [0.9, 0.6]), prediction([], [])]
    m = V.evaluate(frames, preds, tau=0.5)
    assert (m['recall'], m['precision'], m['fp_per_frame'], m['fn']) == (0.5, 0.5, 0.5, 1)


def test_prediction_in_an_ignore_region_is_neither_true_nor_false_positive():
    m = V.evaluate([frame([[10, 10, 20, 20]], ignores=[[200, 200, 40, 40]])], [prediction([[10, 10, 20, 20], [210, 210, 10, 10]], [0.9, 0.9])], tau=0.5)
    assert m['fp'] == 0 and m['recall'] == 1


def test_a_ground_truth_lamp_inside_an_ignore_region_still_counts():
    m = V.evaluate([frame([[205, 205, 20, 20]], ignores=[[200, 200, 40, 40]])], [prediction([], [])], tau=0.5)
    assert m['n_gt'] == 1 and m['recall'] == 0


def test_iou_threshold_and_duplicate_prediction():
    frames = [frame([[0, 0, 20, 20]])]
    duplicate = V.evaluate(frames, [prediction([[0, 0, 20, 20], [1, 1, 20, 20]], [0.9, 0.8])], tau=0.5)
    assert duplicate['tp'] == 1 and duplicate['fp'] == 1
    assert V.evaluate(frames, [prediction([[12, 0, 20, 20]], [0.9])], tau=0.5)['recall'] == 0   # IoU 0.25


def test_average_precision_depends_on_the_confidence_ranking():
    frames = [frame([[0, 0, 20, 20]]), frame([[0, 0, 20, 20]])]
    good = [prediction([[0, 0, 20, 20]], [0.9]), prediction([[0, 0, 20, 20], [100, 100, 20, 20]], [0.8, 0.1])]
    bad = [prediction([[0, 0, 20, 20], [100, 100, 20, 20]], [0.5, 0.9]), prediction([[0, 0, 20, 20]], [0.8])]
    assert V.evaluate(frames, good, 0.05)['mAP50'] > V.evaluate(frames, bad, 0.05)['mAP50']


def test_centre_error():
    m = V.evaluate([frame([[0, 0, 20, 20]])], [prediction([[2, 0, 20, 20]], [0.9])], tau=0.5)
    assert m['centre_err_px_median'] == pytest.approx(2.0) and m['centre_err_diag_median'] == pytest.approx(2 / np.hypot(20, 20))


def test_threshold_and_checkpoint_selection():
    frames = [frame([[0, 0, 20, 20]])]
    preds = [prediction([[0, 0, 20, 20], [100, 0, 20, 20], [200, 0, 20, 20]], [0.9, 0.5, 0.2])]
    assert V.select_tau(frames, preds, eps_f=0.0, taus=[0.1, 0.4, 0.6]) == 0.6
    assert V.select_tau(frames, preds, eps_f=1.0, taus=[0.1, 0.4, 0.6]) in (0.1, 0.4, 0.6)
    metrics = {'a': dict(mAP50=.5, recall=.5, precision=.5, fp_per_frame=.1, centre_err_diag_median=.1), 'b': dict(mAP50=.5, recall=.6, precision=.5, fp_per_frame=.1, centre_err_diag_median=.2),
               'c': dict(mAP50=.49, recall=.9, precision=.9, fp_per_frame=0, centre_err_diag_median=.05)}
    assert V.select_checkpoint(metrics) == 'b'


@pytest.mark.data
@pytest.mark.parametrize('split,lamps', [('val', 397), ('test', 237)])
def test_ground_truth_as_prediction_reaches_the_ceiling(split, lamps):
    ignore_data.configure(paths.MAIN, 'detector')
    frames = build_index(splits=(split,))
    ignore_data.configure(paths.MAIN, 'scene')
    preds = [(xywh2xyxy(f['lamps']), np.ones(len(f['lamps']))) for f in frames]
    m = V.evaluate(frames, preds, tau=0.5)
    assert m['n_gt'] == lamps and m['recall'] == 1 and m['precision'] == 1 and m['fp'] == 0 and m['mAP50_95'] == pytest.approx(1.0)
