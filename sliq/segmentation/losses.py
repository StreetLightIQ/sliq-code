"""Segmentation losses. All of them ignore label 255.

y: (B, H, W) int64 labels in {0 background, 1 road, 2 building, 3 confounding light, 255 ignore}; q: (B, H, W) pixel reliability in [0, 1] (the confidence map)."""
import torch
import torch.nn.functional as F

IGNORE = 255
CLASS_WEIGHTS = torch.tensor([0.71, 1.13, 2.06, 3.35])       # sqrt-inverse frequency: background, road, building, confounding light


def ce(logits, y, q=None, weight=None):
    """Cross-entropy over valid pixels:  L = - sum_p q_p w_{y_p} log P(y_p) / sum_p q_p w_{y_p}.
    weight = per-class weights w, q = per-pixel reliability; with both left out this is the plain mean CE."""
    valid = y != IGNORE
    target = torch.where(valid, y, torch.zeros_like(y))
    log_p = F.log_softmax(logits.float(), 1).gather(1, target.unsqueeze(1)).squeeze(1)
    w = torch.ones_like(log_p) if weight is None else weight.to(logits.device)[target]
    if q is not None:
        w = w * q
    w = w * valid
    return -(w * log_p).sum() / w.sum().clamp_min(1e-9)


def dice(logits, y, smooth=1.0):
    """Soft Dice averaged over the 4 classes, valid pixels only, computed over the whole batch."""
    valid = (y != IGNORE).unsqueeze(1)
    p = F.softmax(logits.float(), 1) * valid
    t = F.one_hot(y.clamp(max=3), 4).permute(0, 3, 1, 2).float() * valid
    intersection = (p * t).sum((0, 2, 3))
    total = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    return (1 - (2 * intersection + smooth) / (total + smooth)).mean()


def make(kind, lam_ce=1.0, lam_dice=1.0):
    """-> loss(logits, y, q). kind: ce | wce | wce_dice | conf_wce | conf_wce_dice  (w = class weights, conf = pixel confidence)."""
    use_weights, use_confidence, use_dice = kind != 'ce', kind.startswith('conf'), kind.endswith('dice')
    if kind not in ('ce', 'wce', 'wce_dice', 'conf_wce', 'conf_wce_dice'):
        raise ValueError(kind)

    def loss(logits, y, q):
        value = ce(logits, y, q=q if use_confidence else None, weight=CLASS_WEIGHTS if use_weights else None)
        return lam_ce * value + lam_dice * dice(logits, y) if use_dice else value

    return loss


def confusion(predicted, y, n=4):
    """n x n confusion matrix (rows = label, columns = prediction) over valid pixels."""
    valid = y != IGNORE
    return torch.bincount(y[valid] * n + predicted[valid], minlength=n * n).view(n, n)


def iou_report(confusion_matrix):
    c = confusion_matrix.double()
    tp = c.diag()
    fp, fn = c.sum(0) - tp, c.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp_min(1)
    present = c.sum(1) > 0
    return dict(iou=iou.tolist(), mean_iou=float(iou[present].mean()), light_iou=float(iou[3]), road_iou=float(iou[1]), building_iou=float(iou[2]),
                light_precision=float(tp[3] / (tp[3] + fp[3]).clamp_min(1)), light_recall=float(tp[3] / (tp[3] + fn[3]).clamp_min(1)), pixels=c.sum().item())
