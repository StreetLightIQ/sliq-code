"""Ignore-aware Ultralytics detection loss (ultralytics 8.4.176).

Ignore regions are split off before target assignment. Anchors centred in an ignore region and not assigned a lamp get no classification loss.
Optional per-pack sample weights scale the box and DFL terms only."""
import json
import os

import torch
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

IGN = 1
Q = {}                                          # pack -> weight, default 1.0


def set_sample_weights(weights):
    Q.clear()
    Q.update(weights)


def load_sample_weights(dataset_dir):
    """<dataset>/sample_weights.json if the dataset has one."""
    path = os.path.join(dataset_dir, 'sample_weights.json')
    set_sample_weights(json.load(open(path)) if os.path.exists(path) else {})
    return dict(Q)


def pack_of(im_file):
    return os.path.basename(im_file).split('__')[0]


class IgnoreAwareLoss(v8DetectionLoss):
    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = preds["boxes"].permute(0, 2, 1).contiguous(), preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype, bs = pred_scores.dtype, pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]       # (h, w)
        scale = imgsz[[1, 0, 1, 0]]
        cls = batch["cls"].view(-1).to(self.device)
        image_idx = batch["batch_idx"].view(-1).to(self.device).long()
        boxes = batch["bboxes"].to(self.device)
        is_ignore = cls == IGN

        # anchors centred inside an ignore region, (bs, anchors)
        centres = anchor_points * stride_tensor
        ignored_anchor = torch.zeros(bs, centres.shape[0], device=self.device)
        if is_ignore.any():
            xywh = boxes[is_ignore] * scale
            x0, y0, x1, y1 = (xywh[:, 0] - xywh[:, 2] / 2)[:, None], (xywh[:, 1] - xywh[:, 3] / 2)[:, None], (xywh[:, 0] + xywh[:, 2] / 2)[:, None], (xywh[:, 1] + xywh[:, 3] / 2)[:, None]
            inside = ((centres[None, :, 0] >= x0) & (centres[None, :, 0] <= x1) & (centres[None, :, 1] >= y0) & (centres[None, :, 1] <= y1)).float()
            ignored_anchor.index_add_(0, image_idx[is_ignore], inside)
        ignored_anchor = ignored_anchor > 0

        # lamps only (class 0) are targets
        lamps = cls == 0
        targets = torch.cat((image_idx[lamps].view(-1, 1).to(boxes.dtype), cls[lamps].view(-1, 1), boxes[lamps]), 1)
        targets = self.preprocess(targets, bs, scale_tensor=scale)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        q = torch.tensor([Q.get(pack_of(f), 1.0) for f in batch["im_file"]], device=self.device, dtype=dtype)
        target_scores_sum = target_scores.sum().clamp_(min=1)
        weighted_scores = target_scores * q.view(-1, 1, 1)
        weighted_sum = weighted_scores.sum().clamp_(min=1)

        keep = (~(ignored_anchor & ~fg_mask)).unsqueeze(-1).to(dtype)
        bce = self.bce(pred_scores, target_scores.to(dtype)) * keep
        if self.class_weights is not None:
            bce = bce * self.class_weights
        loss[1] = bce.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, weighted_scores, weighted_sum, fg_mask, imgsz, stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        self.last = dict(fg_mask=fg_mask, ignored_anchor=ignored_anchor, keep=keep.squeeze(-1), q=q)      # used by kd.py
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, dict(zip(self.loss_names, loss.detach()))


def install():
    """Use IgnoreAwareLoss in every DetectionModel built after this call."""
    from ultralytics.nn import tasks

    def init_criterion(self):
        end_to_end = getattr(self.model[-1], "one2one_cv2", None) is not None
        return E2ELoss(self, IgnoreAwareLoss) if end_to_end else IgnoreAwareLoss(self)

    tasks.DetectionModel.init_criterion = init_criterion
