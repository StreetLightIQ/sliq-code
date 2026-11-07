import numpy as np

from sliq.deploy import calib_set, score
from sliq.deploy.export import letterbox_shape


def test_letterbox_shapes_match_inference():
    assert letterbox_shape(1280) == (736, 1280)
    assert letterbox_shape(960) == (544, 960)
    assert letterbox_shape(640) == (384, 640)


def test_calibration_sample_is_stratified_deterministic_and_never_leaves_its_input():
    frames = [dict(stem=f'{r}_{i}', recording=r, split='train') for r, n in (('720', 64), ('452', 36)) for i in range(n)]
    a, b = calib_set.sample(frames, 25, seed=0), calib_set.sample(frames, 25, seed=0)
    assert [f['stem'] for f in a] == [f['stem'] for f in b]
    assert sum(f['recording'] == '720' for f in a) == 16 and sum(f['recording'] == '452' for f in a) == 9
    assert {f['stem'] for f in a} <= {f['stem'] for f in frames}


def test_size_bins_count_lamps_by_area():
    frame = dict(lamps=np.array([[0, 0, 20, 20], [100, 100, 60, 60]], float), ignores=np.zeros((0, 4)))     # 400 px2 and 3600 px2
    preds = [(np.array([[0, 0, 20, 20]], float), np.array([0.9]))]
    bins = score.size_bins([frame], preds, 0.5)
    assert bins['300-600'] == dict(gt=1, tp=1, recall=1.0) and bins['2500-inf'] == dict(gt=1, tp=0, recall=0.0)


def test_agreement_is_one_for_identical_predictions():
    p = [(np.array([[0, 0, 10, 10], [50, 50, 70, 70]], float), np.array([0.9, 0.4]))]
    assert score.agreement(None, p, p, 0.3) == dict(share_matched=1.0, n=2, mean_abs_conf_diff=0.0)
