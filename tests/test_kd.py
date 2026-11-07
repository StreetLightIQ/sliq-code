import types

import numpy as np
import pytest
import torch

from sliq.detector import ignore_data, ignore_loss, kd


# ---- masks and resampling
def _boxes(rows):
    """rows: (image, class, cx, cy, w, h) normalised."""
    t = torch.tensor(rows, dtype=torch.float32)
    return dict(batch_idx=t[:, 0], cls=t[:, 1:2], bboxes=t[:, 2:])


def test_region_maps_mark_lamps_boost_small_ones_and_keep_lamps_over_confounders():
    s = 20 / 1280   # 20 px lamp is small, 40 px is not
    batch = _boxes([[0, 0, 0.5, 0.5, s, s], [1, 0, 0.25, 0.25, 2 * s, 2 * s], [1, 2, 0.25, 0.25, 0.2, 0.2], [1, 1, 0.8, 0.8, 0.1, 0.1]])
    lamp_w, conf = kd.region_maps(batch, 2, 16, 16)
    assert lamp_w[0].max() == kd.SMALL_BOOST and lamp_w[1].max() == 1.0
    assert (lamp_w[0] > 0).sum() >= 1
    assert not (conf & (lamp_w > 0)).any() and conf[1].any() and not conf[0].any()
    assert lamp_w[1, 12:, 12:].sum() == 0 and not conf[1, 12:, 12:].any()


def test_resample_is_identity_on_equal_grids_and_preserves_constants():
    x = torch.rand(2, 3, 16 * 16 + 8 * 8)
    assert torch.equal(kd.resample(x, [(16, 16), (8, 8)], [(16, 16), (8, 8)]), x)
    c = torch.full((1, 1, 16 * 16 + 8 * 8), 0.7)
    assert torch.allclose(kd.resample(c, [(16, 16), (8, 8)], [(12, 12), (6, 6)]), torch.full((1, 1, 144 + 36), 0.7))


def test_decoded_teacher_boxes_scale_with_image_size():
    feats = [torch.zeros(1, 1, 8, 8)]
    raw = torch.zeros(1, 4, 64)
    raw[:, :, 0] = 1.0   # one cell each side at stride 8
    assert torch.allclose(kd.decode_px(raw, feats, torch.tensor([8.0]))[0, 0], torch.tensor([-4.0, -4.0, 12.0, 12.0]))


# ---- criterion
@pytest.fixture(scope='module')
def models():
    from ultralytics.nn.tasks import DetectionModel
    ignore_loss.install()
    kd.install()
    student = DetectionModel('yolo26n.yaml', nc=1, verbose=False).cuda().train()
    student.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=1)
    kd.TEACHER = DetectionModel('yolo26m.yaml', nc=1, verbose=False).cuda().eval()
    kd.build_adapters(student, kd.TEACHER, 'cuda')
    return student


def _batch(ignore=None, confounder=True):
    rows = [[0, 0, 0.5, 0.5, 0.04, 0.06], [1, 0, 0.3, 0.4, 0.04, 0.06]] + ([[0, 2, 0.7, 0.3, 0.1, 0.1]] if confounder else []) + ([[0, 1] + ignore] if ignore else [])
    b = _boxes(rows)
    b['im_file'] = ['/x/pack_05__a.jpg', '/x/pack_18__b.jpg']
    torch.manual_seed(0)
    b['img_teacher'] = torch.rand(2, 3, 640, 640, device='cuda')
    b['img'] = torch.nn.functional.interpolate(b['img_teacher'], size=(480, 480), mode='bilinear', align_corners=False, antialias=True)
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}


def _run(student, mode, batch):
    kd.CFG.update(mode=mode, w_cls=1.0, w_box=1.0, w_feat=1.0, student_imgsz=480, teacher_imgsz=640)
    ignore_loss.set_sample_weights({'pack_18': 0.6})
    crit = kd.KDCriterion(student)
    student.zero_grad()
    kd.ADAPTERS.zero_grad()
    loss, items = crit(student(batch['img']), batch)
    return crit, loss, items


@pytest.mark.gpu
def test_kd_off_is_the_normal_loss(models):
    from ultralytics.utils.loss import E2ELoss
    batch = _batch(confounder=False)
    _, loss, items = _run(models, 'off', batch)
    reference, _ = E2ELoss(models, ignore_loss.IgnoreAwareLoss)(models(batch['img']), batch)
    assert loss.shape == reference.shape and torch.allclose(loss, reference, rtol=1e-4) and 'kd_cls' not in items


@pytest.mark.gpu
def test_confounders_are_not_lamp_targets(models):
    _, with_conf, _ = _run(models, 'off', _batch(confounder=True))
    _, without, _ = _run(models, 'off', _batch(confounder=False))
    assert torch.allclose(with_conf, without, rtol=1e-4)


@pytest.mark.gpu
def test_all_terms_finite_and_reach_student_and_adapters(models):
    _, loss, items = _run(models, 'both', _batch(ignore=[0.5, 0.2, 0.9, 0.3]))
    assert loss.shape[0] == 4 and torch.isfinite(loss).all() and items['kd_cls'] > 0 and items['kd_feat'] > 0
    loss.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in kd.ADAPTERS.parameters())
    assert models.model[0].conv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in kd.TEACHER.parameters())


@pytest.mark.gpu
def test_score_kd_is_masked_in_ignore_regions(models):
    big = [0.5, 0.5, 1.0, 1.0]   # image 0 fully ignored
    _, _, plain = _run(models, 'out', _batch())
    _, _, masked = _run(models, 'out', _batch(ignore=big))
    assert masked['kd_cls'] != plain['kd_cls']
    crit, _, _ = _run(models, 'out', _batch(ignore=big))
    keep, fg = crit.e2e.one2many.last['keep'], crit.e2e.one2many.last['fg_mask']
    assert (keep[0] == fg[0].float()).all()
    assert keep[1].all()


# ---- data
def test_confounders_load_only_when_asked():
    rec = dict(objects=[dict(kind='positive', bbox_xywh=[0, 0, 30, 30]), dict(kind='negative', bbox_xywh=[100, 100, 10, 10]), dict(kind='ignore', bbox_xywh=[5, 5, 5, 5])])
    assert ignore_data.frame_confounders(rec).tolist() == [[100, 100, 10, 10]]
    lamps, ignores = ignore_data.frame_boxes(rec)
    assert len(lamps) == 1 and len(ignores) == 1
    assert ignore_data.WITH_CONFOUNDERS is False
