"""Distillation from the frozen YOLO26m teacher into a YOLO26n/s student (ultralytics 8.4.176).

The batch is augmented at the teacher's size; the student sees an antialiased downscale. Terms on the one-to-many branch:
  score KD:   BCE of the student logit against the teacher probability, focal-weighted by |p_s - p_t|^2, masked like the classification loss.
  box KD:     1 - GIoU to the teacher box on anchors assigned to a lamp where the teacher is confident.
  feature KD: neck maps through a 1x1 adapter, standardised per channel, squared error averaged over lamp, confounder and background cells.
Teacher and adapters live here, so saved checkpoints stay plain Ultralytics detectors."""
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox, make_anchors

from sliq.detector import ignore_loss

BOX_CONF = 0.3
SMALL_AREA, SMALL_BOOST = 600.0, 2.0           # lamps under 600 px2 count double
W_LAMP, W_CONF, W_BG = 1.0, 0.5, 0.1
LEVEL_W = (2.0, 1.0, 1.0)                      # strides 8, 16, 32

CFG = dict(mode='off', w_cls=1.0, w_box=1.0, w_feat=1.0, student_imgsz=960, teacher_imgsz=1280)
TEACHER = None
ADAPTERS = None


def constants():
    return dict(BOX_CONF=BOX_CONF, SMALL_AREA=SMALL_AREA, SMALL_BOOST=SMALL_BOOST, W_LAMP=W_LAMP, W_CONF=W_CONF, W_BG=W_BG, LEVEL_W=LEVEL_W)


def load_teacher(checkpoint, device):
    """Frozen teacher in eval mode."""
    global TEACHER
    from ultralytics import YOLO
    TEACHER = YOLO(checkpoint).model.float().to(device).eval()
    for p in TEACHER.parameters():
        p.requires_grad_(False)
    return TEACHER


def head_channels(model):
    head = model.model[-1]
    return [head.cv2[i][0].conv.in_channels for i in range(head.nl)]


def build_adapters(student, teacher, device):
    global ADAPTERS
    ADAPTERS = nn.ModuleList(nn.Conv2d(cs, ct, 1) for cs, ct in zip(head_channels(student), head_channels(teacher))).to(device)
    return ADAPTERS


def _to_maps(flat, shapes):
    """(bs, C, sum h*w) -> [(bs, C, h, w)] per level."""
    out, start = [], 0
    for h, w in shapes:
        out.append(flat[..., start:start + h * w].reshape(flat.shape[0], flat.shape[1], h, w))
        start += h * w
    return out


def resample(flat, src_shapes, dst_shapes):
    """Bilinearly resample per-level maps stored flat (bs, C, A_src) onto another grid -> (bs, C, A_dst)."""
    if list(src_shapes) == list(dst_shapes):
        return flat
    maps = [F.interpolate(m, size=d, mode='bilinear', align_corners=False) for m, d in zip(_to_maps(flat, src_shapes), dst_shapes)]
    return torch.cat([m.flatten(2) for m in maps], 2)


def decode_px(boxes, feats, stride):
    """raw head boxes (bs, 4, A) -> xyxy pixels (bs, A, 4)."""
    anchors, strides = make_anchors(feats, stride, 0.5)
    return dist2bbox(boxes.permute(0, 2, 1).float(), anchors, xywh=False) * strides


def standardise(x):
    x = x.float()
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-6)


def region_maps(batch, bs, h, w):
    """-> (lamp weight map, confounder mask), each (bs, h, w). A cell belongs to a box if they overlap."""
    device = batch['bboxes'].device
    cls, idx, boxes = batch['cls'].view(-1).to(device), batch['batch_idx'].view(-1).to(device).long(), batch['bboxes'].to(device).float()
    lamp_w, conf = torch.zeros(bs, h, w, device=device), torch.zeros(bs, h, w, dtype=torch.bool, device=device)
    for value, target in ((0, 'lamp'), (2, 'conf')):
        sel = cls == value
        if not sel.any():
            continue
        b, image = boxes[sel], idx[sel]
        x0, x1 = (b[:, 0] - b[:, 2] / 2) * w, (b[:, 0] + b[:, 2] / 2) * w
        y0, y1 = (b[:, 1] - b[:, 3] / 2) * h, (b[:, 1] + b[:, 3] / 2) * h
        cols, rows = torch.arange(w, device=device), torch.arange(h, device=device)
        inside = ((rows[None] + 1 > y0[:, None]) & (rows[None] < y1[:, None]))[:, :, None] & ((cols[None] + 1 > x0[:, None]) & (cols[None] < x1[:, None]))[:, None, :]
        if target == 'lamp':
            weight = torch.where(b[:, 2] * b[:, 3] * 1280 * 1280 < SMALL_AREA, SMALL_BOOST, 1.0)
            values = inside.float() * weight[:, None, None]
            lamp_w.index_reduce_(0, image, values, 'amax', include_self=True)
        else:
            conf = torch.zeros(bs, h, w, device=device).index_reduce_(0, image, inside.float(), 'amax', include_self=True) > 0
    return lamp_w, conf & (lamp_w == 0)


def feature_kd(student_feats, teacher_feats, batch):
    assert ADAPTERS is not None, 'adapters are built in KDTrainer.build_optimizer'
    total = 0.0
    for level, (sf, tf) in enumerate(zip(student_feats, teacher_feats)):
        bs, _, h, w = sf.shape
        tf = tf.float()
        if tf.shape[-2:] != sf.shape[-2:]:
            tf = F.interpolate(tf, size=(h, w), mode='bilinear', align_corners=False, antialias=True)
        d = (standardise(ADAPTERS[level](sf)) - standardise(tf)).pow(2).mean(1)               # (bs, h, w)
        lamp_w, conf = region_maps(batch, bs, h, w)
        background = (lamp_w == 0) & ~conf
        term = W_BG * d[background].mean() if background.any() else d.sum() * 0
        if lamp_w.sum() > 0:
            term = term + W_LAMP * (d * lamp_w).sum() / lamp_w.sum()
        if conf.any():
            term = term + W_CONF * d[conf].mean()
        total = total + LEVEL_W[level] * term
    return total / sum(LEVEL_W)


class KDCriterion:
    """Ignore-aware E2E loss plus the distillation terms, as one extra loss element."""

    def __init__(self, model):
        from ultralytics.utils.loss import E2ELoss
        self.e2e = E2ELoss(model, ignore_loss.IgnoreAwareLoss)
        self.stride = model.stride

    @property
    def updates(self):
        return self.e2e.updates

    @updates.setter
    def updates(self, value):
        self.e2e.updates = value

    def update(self):
        self.e2e.update()

    def __call__(self, preds, batch):
        loss, items = self.e2e(preds, batch)
        if CFG['mode'] == 'off':
            return loss, items
        student = self.e2e.one2many.parse_output(preds)['one2many']
        masks = self.e2e.one2many.last
        bs = student['scores'].shape[0]
        with torch.no_grad():
            teacher = TEACHER(batch.get('img_teacher', batch['img']))[1]['one2many']
        s_shapes = [f.shape[-2:] for f in student['feats']]
        t_shapes = [f.shape[-2:] for f in teacher['feats']]
        zero = loss.new_zeros(())
        l_cls = l_box = l_feat = zero
        use_out, use_feat = CFG['mode'] in ('out', 'both'), CFG['mode'] in ('feat', 'both')
        if use_out:
            t_prob = resample(teacher['scores'].float().sigmoid(), t_shapes, s_shapes)[:, 0].clamp(0, 1)          # (bs, A_s)
            s_logit = student['scores'][:, 0].float()
            keep = masks['keep'].float()
            bce = F.binary_cross_entropy_with_logits(s_logit, t_prob, reduction='none')
            l_cls = (bce * (s_logit.sigmoid() - t_prob).abs().pow(2) * keep).sum() / (t_prob * keep).sum().clamp(min=1)
            scale = CFG['student_imgsz'] / CFG['teacher_imgsz']
            t_box = resample((decode_px(teacher['boxes'], teacher['feats'], self.stride) * scale).permute(0, 2, 1), t_shapes, s_shapes).permute(0, 2, 1)
            s_box = decode_px(student['boxes'], student['feats'], self.stride)
            sel = masks['fg_mask'] & (t_prob > BOX_CONF)
            if sel.any():
                weight = (t_prob * masks['q'].float().view(-1, 1))[sel]
                iou = bbox_iou(s_box[sel], t_box[sel], xywh=False, GIoU=True).view(-1)
                l_box = ((1 - iou) * weight).sum() / weight.sum().clamp(min=1e-6)
        if use_feat:
            l_feat = feature_kd(student['feats'], teacher['feats'], batch)
        kd = CFG['w_cls'] * l_cls + CFG['w_box'] * l_box + CFG['w_feat'] * l_feat
        items = {**items, 'kd_cls': l_cls.detach(), 'kd_box': l_box.detach(), 'kd_feat': l_feat.detach()}
        return torch.cat((loss, (kd * bs).view(1))), items


def install():
    """Use KDCriterion in every DetectionModel built after this call."""
    from ultralytics.nn import tasks
    tasks.DetectionModel.init_criterion = lambda self: KDCriterion(self)
