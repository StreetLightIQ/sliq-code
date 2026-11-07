import numpy as np
import pytest
import torch

from sliq.segmentation import dataset as D
from sliq.segmentation import losses as L
from sliq.segmentation import train as seg_train


def test_ignored_pixels_do_not_matter():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    y = torch.randint(0, 4, (2, 16, 16))
    y[:, 8:, :] = 255
    for loss in (lambda a, b: L.ce(a, b), lambda a, b: L.ce(a, b, weight=L.CLASS_WEIGHTS), lambda a, b: L.dice(a, b)):
        value = loss(logits, y)
        grad, = torch.autograd.grad(value, logits)
        assert grad[:, :, 8:, :].abs().max() == 0
        shifted = logits.detach().clone()
        shifted[:, :, 8:, :] += 100
        assert torch.allclose(loss(shifted, y), value)


def test_all_ignored_is_finite():
    logits, y = torch.randn(1, 4, 4, 4), torch.full((1, 4, 4), 255)
    assert L.ce(logits, y) == 0 and torch.isfinite(L.dice(logits, y))


def test_ce_matches_torch_and_confidence_weighting():
    logits, y = torch.randn(2, 4, 8, 8), torch.randint(0, 4, (2, 8, 8))
    assert torch.allclose(L.ce(logits, y), torch.nn.functional.cross_entropy(logits, y), atol=1e-6)
    assert torch.allclose(L.ce(logits, y, weight=L.CLASS_WEIGHTS), torch.nn.functional.cross_entropy(logits, y, weight=L.CLASS_WEIGHTS), atol=1e-6)
    q = torch.ones(2, 8, 8)
    q[:, :, :4] = 0   # only the right half counts
    expected = L.ce(logits[:, :, :, 4:], y[:, :, 4:], weight=L.CLASS_WEIGHTS)
    assert torch.allclose(L.ce(logits, y, q=q, weight=L.CLASS_WEIGHTS), expected, atol=1e-6)


def test_dice_is_small_for_a_perfect_and_large_for_a_wrong_prediction():
    y = torch.randint(0, 4, (1, 16, 16))
    good = torch.nn.functional.one_hot(y, 4).permute(0, 3, 1, 2).float() * 30
    assert L.dice(good, y) < 0.05 and L.dice(torch.roll(good, 1, 1), y) > 0.6


def test_make_selects_the_documented_terms():
    logits, y, q = torch.randn(1, 4, 8, 8), torch.randint(0, 4, (1, 8, 8)), torch.rand(1, 8, 8)
    assert torch.allclose(L.make('ce')(logits, y, q), L.ce(logits, y))
    assert torch.allclose(L.make('wce')(logits, y, q), L.ce(logits, y, weight=L.CLASS_WEIGHTS))
    assert torch.allclose(L.make('conf_wce')(logits, y, q), L.ce(logits, y, q=q, weight=L.CLASS_WEIGHTS))
    assert torch.allclose(L.make('wce_dice', 1.0, 2.0)(logits, y, q), L.ce(logits, y, weight=L.CLASS_WEIGHTS) + 2.0 * L.dice(logits, y))
    with pytest.raises(ValueError):
        L.make('nope')


def test_iou_report_ignores_255():
    result = L.iou_report(L.confusion(torch.tensor([[0, 1, 2, 3]]), torch.tensor([[0, 1, 2, 255]])))
    assert result['mean_iou'] == 1.0 and result['pixels'] == 3


def test_selection_rule():
    rows = [dict(epoch=1, mean_iou=.5, light_iou=.30, road_iou=.60, building_iou=.30), dict(epoch=2, mean_iou=.4, light_iou=.20, road_iou=.70, building_iou=.30),
            dict(epoch=3, mean_iou=.6, light_iou=.35, road_iou=.50, building_iou=.30)]
    assert seg_train.select(rows)[0]['epoch'] == 3
    assert seg_train.select(rows, rmin=0.65, bmin=0.2) == (rows[1], True)
    assert seg_train.select(rows, rmin=0.9) == (rows[2], False)   # infeasible: falls back


@pytest.mark.data
def test_dataset_shapes_and_padding():
    frames = D.list_frames('train')
    assert len(frames) == 1507
    x, mask, q = D.SegDataset(frames[:6], train=True, seed=1)[0]
    assert x.shape == (3, 576, 1024) and mask.shape == (576, 1024) and q.shape == (576, 1024)
    assert set(np.unique(mask.numpy())) <= {0, 1, 2, 3, 255} and float(q[mask == 255].max()) == 0.0
    x, mask, q = D.SegDataset(D.list_frames('val')[:2], train=False)[0]
    assert x.shape == (3, 720, 1280) and mask.shape == (720, 720 * 16 // 9)


@pytest.mark.data
def test_rare_class_cropping_finds_lights():
    dataset = D.SegDataset(D.list_frames('train')[:40], train=True, seed=2, rare_p=1.0)
    share = np.mean([(dataset[i][1] == 3).any().item() for i in range(40)])
    assert share > 0.7
