"""Score a segmenter on a split: per-class IoU, confusion matrix and the bright-road share (road pixels with luma >= 80) from predicted vs
ground-truth masks. Options: --flip (average with the flipped frame) and --confounder (class 3 from the confounder detector and the label
rule, classes 0-2 from the segmenter).

    python -m sliq.segmentation.diagnose --segmenter segmenter --flip --confounder confounder --name final_val
--segmenter and --confounder take a run name or a .pt file. Writes runs/segmentation/diag/NAME.json."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sliq.common import paths
from sliq.e2e.metrics import illumination_fraction
from sliq.segmentation import dataset as data
from sliq.segmentation import light_from_boxes, losses
from sliq.segmentation import models as seg_models

OUT = paths.RUNS + '/segmentation/diag/'


def load_segmenter(spec, arch='segformer_b2', device='cuda'):
    if spec.endswith('.pt'):
        return seg_models.load(spec, arch, device)
    return seg_models.load_run(paths.run_dir('segmentation', spec), device)


def confounder_spec(spec, tau=None):
    """-> (checkpoint, tau, imgsz) from a run name or a .pt file (then tau is required)."""
    if spec.endswith('.pt'):
        assert tau is not None, '--confounder-tau'
        return spec, tau, 1280
    return light_from_boxes.load_run(spec)


@torch.no_grad()
def probabilities(model, x, flip=False):
    """Softmax, averaged with the flipped input when flip=True."""
    with torch.autocast('cuda', dtype=torch.float16):
        prob = F.softmax(model(x)['out'].float(), 1)
        if flip:
            prob = (prob + F.softmax(model(x.flip(3))['out'].float().flip(3), 1)) / 2
    return prob


def fuse(prob, light):
    """Classes 0-2 from the segmenter, class 3 from the confounder rule."""
    predicted = prob[:, :3].argmax(1)
    predicted[light] = 3
    return predicted


def evaluate(model, split='val', dataset=None, flip=False, confounder=None):
    """confounder: (checkpoint, tau, imgsz) or None."""
    frames = data.list_frames(split, dataset)
    light_masks = light_from_boxes.predict_light(*confounder[:2], frames, imgsz=confounder[2]) if confounder else None
    loader = torch.utils.data.DataLoader(data.SegDataset(frames, False), batch_size=4, num_workers=8)
    confusion = torch.zeros(4, 4, dtype=torch.long)
    pairs, i = [], 0
    for x, y, _ in loader:
        prob = probabilities(model, x.cuda(), flip).cpu()
        if confounder:
            predicted = fuse(prob, torch.from_numpy(np.stack([light_masks[frames[i + b]['stem']] for b in range(x.shape[0])])))
        else:
            predicted = prob.argmax(1)
        confusion += losses.confusion(predicted, y)
        for b in range(y.shape[0]):
            mask, p = y[b].numpy(), predicted[b].numpy()
            rgb = np.asarray(Image.open(frames[i]['image']).convert('RGB'))
            valid = mask != 255
            pairs.append((illumination_fraction(rgb, mask == 1, mask == 3, valid), illumination_fraction(rgb, p == 1, p == 3, valid)))
            i += 1
    gt, pred = np.array(pairs).T
    c = confusion.double()
    return dict(split=split, flip=flip, confounder=confounder[:2] if confounder else None, n_frames=len(frames), **losses.iou_report(confusion),
                confusion_row_normalised=(c / c.sum(1, keepdim=True).clamp_min(1)).numpy().round(4).tolist(),
                bright_road_share=dict(gt_mean=float(gt.mean()), pred_mean=float(pred.mean()), mean_abs_diff=float(np.abs(gt - pred).mean()),
                                       pearson=float(np.corrcoef(gt, pred)[0, 1])))


def report(name, r):
    print('%s mIoU %.4f road %.4f building %.4f light %.4f | light P %.3f R %.3f | bright road r %.3f mad %.3f' % (
        name, r['mean_iou'], r['road_iou'], r['building_iou'], r['light_iou'], r['light_precision'], r['light_recall'],
        r['bright_road_share']['pearson'], r['bright_road_share']['mean_abs_diff']))


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--segmenter', required=True)
    a.add_argument('--arch', default='segformer_b2', choices=seg_models.ARCHS, help='for a .pt segmenter')
    a.add_argument('--name', required=True)
    a.add_argument('--flip', action='store_true')
    a.add_argument('--confounder')
    a.add_argument('--confounder-tau', type=float)
    a.add_argument('--split', default='val', choices=['train', 'val'])
    args = a.parse_args()
    confounder = confounder_spec(args.confounder, args.confounder_tau) if args.confounder else None
    result = evaluate(load_segmenter(args.segmenter, args.arch), args.split, flip=args.flip, confounder=confounder)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(segmenter=args.segmenter, **result), open(OUT + args.name + '.json', 'w'), indent=1)
    report(args.name, result)


if __name__ == '__main__':
    main()
