import json
import types

import numpy as np
import pytest
import torch
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import v8DetectionLoss

from sliq.common import paths
from sliq.detector import ignore_data, ignore_loss

MODELS = ['yolo26m', 'yolo11m', 'yolo12m']


# ---- data
@pytest.mark.data
def test_lamp_counts_per_split():
    splits = json.load(open(paths.MAIN + '/splits.json'))['frames']
    ignore_data.configure(paths.MAIN, 'detector')
    records = ignore_data.frame_records()
    counts = {'train': 0, 'val': 0, 'test': 0}
    for stem, split in splits.items():
        if split in counts:
            counts[split] += len(ignore_data.frame_boxes(records[stem])[0])
    ignore_data.configure(paths.MAIN, 'scene')
    assert counts == {'train': 1894, 'val': 397, 'test': 237}


def _instances(boxes):
    return Instances(np.array(boxes, np.float32), segments=np.zeros((0, 1000, 2), np.float32), bbox_format='xyxy', normalized=False)


def test_floor_to_ignore_demotes_small_lamps_after_scaling():
    image = np.zeros((1280, 1280, 3), np.uint8)
    labels = dict(img=image, cls=np.zeros((3, 1), np.float32), instances=_instances([[0, 0, 20, 20], [100, 100, 117, 117], [300, 300, 340, 340]]))   # 400, 289, 1600 px2
    ignore_data.FloorToIgnore.stats.update(seen=0, demoted=0)
    assert ignore_data.FloorToIgnore(1280)(labels)['cls'].reshape(-1).tolist() == [0, 1, 0]
    assert ignore_data.FloorToIgnore.stats == {'seen': 3, 'demoted': 1}
    half = ignore_data.FloorToIgnore(640)   # areas count at 1280 scale
    assert half(dict(img=image, cls=np.zeros((1, 1), np.float32), instances=_instances([[0, 0, 12, 12]])))['cls'].reshape(-1).tolist() == [0]
    assert half(dict(img=image, cls=np.zeros((1, 1), np.float32), instances=_instances([[0, 0, 8, 8]])))['cls'].reshape(-1).tolist() == [1]


# ---- loss
@pytest.fixture(scope='module', params=MODELS)
def criterion(request):
    from ultralytics.nn.tasks import DetectionModel
    ignore_loss.install()
    model = DetectionModel(f'{request.param}.yaml', nc=1, verbose=False).cuda().train()
    model.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=1)
    crit = model.init_criterion()
    return model, crit, getattr(crit, 'one2one', crit)                  # E2ELoss wraps two criteria


def _batch(ignore=None, packs=('pack_05', 'pack_05')):
    cls = [0.0, 0.0] + ([1.0] if ignore else [])
    image = [0.0, 1.0] + ([0.0] if ignore else [])
    boxes = [[0.5, 0.5, 0.04, 0.06], [0.3, 0.4, 0.04, 0.06]] + ([ignore] if ignore else [])
    return dict(cls=torch.tensor(cls).view(-1, 1), batch_idx=torch.tensor(image), bboxes=torch.tensor(boxes), im_file=[f'/x/{p}__a.jpg' for p in packs])


def _loss(criterion, batch, seed=0):
    model, _, one = criterion
    torch.manual_seed(seed)
    preds = model(torch.randn(2, 3, 640, 640, device='cuda'))
    out = one.parse_output(preds)
    out = out['one2one'] if 'one2one' in out else out
    return one.get_assigned_targets_and_loss(out, batch)[1].detach()


@pytest.mark.gpu
def test_ignore_region_removes_classification_loss_only(criterion):
    plain, with_ignore = _loss(criterion, _batch()), _loss(criterion, _batch([0.5, 0.2, 0.9, 0.3]))        # strip away from both lamps
    assert with_ignore[1] < plain[1] - 1e-6 and abs(with_ignore[0] - plain[0]) < 1e-6


@pytest.mark.gpu
def test_ignore_box_is_not_a_target(criterion):
    assert torch.allclose(_loss(criterion, _batch()), _loss(criterion, _batch([0.5, 0.5, 0.0001, 0.0001])), atol=1e-6)      # too small for an anchor centre


@pytest.mark.gpu
def test_sample_weights_change_localisation_loss_only(criterion):
    ignore_loss.set_sample_weights({'pack_05': 1.0, 'pack_18': 0.2})
    mixed = _loss(criterion, _batch(packs=('pack_05', 'pack_18')))
    uniform = _loss(criterion, _batch(packs=('pack_05', 'pack_05')))
    ignore_loss.set_sample_weights({})
    assert torch.isfinite(mixed).all()
    assert not torch.allclose(mixed[0], uniform[0])
    assert torch.allclose(mixed[1], uniform[1])


@pytest.mark.gpu
def test_equals_stock_loss_without_ignores_and_unit_weights(criterion):
    model, _, one = criterion
    ignore_loss.set_sample_weights({})
    torch.manual_seed(3)
    preds = model(torch.randn(2, 3, 640, 640, device='cuda'))
    out = one.parse_output(preds)
    out = out['one2one'] if 'one2one' in out else out
    batch = _batch()
    stock = v8DetectionLoss(model, tal_topk=getattr(one.assigner, 'topk', 10), tal_topk2=getattr(one.assigner, 'topk2', None))
    reference = stock.get_assigned_targets_and_loss(out, {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()})[1].detach()
    assert torch.allclose(one.get_assigned_targets_and_loss(out, batch)[1].detach(), reference, rtol=1e-4, atol=1e-6)


# ---- trainer
def test_lamp_trainer_never_validates():
    from sliq.detector.yolo_train import lamp_trainer
    trainer_class = lamp_trainer()
    assert trainer_class.validate(None) == ({}, 0.0) and trainer_class.final_eval(None) is None
