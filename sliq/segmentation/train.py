"""Segmenter training: random 576x1024 crops, full frames for validation, label 255 ignored, AdamW, polynomial decay with warmup, AMP.
Selection: the epoch with the highest confounding-light IoU whose road and building IoU reach the floors.

    python -m sliq.segmentation.train --name segmenter --seed 42
    python -m sliq.segmentation.train --config configs/ablations/seg_b5.yaml"""
import argparse
import json
import os
import platform
import shutil
import time

import numpy as np
import torch
import yaml

from sliq.common import paths
from sliq.segmentation import dataset as data
from sliq.segmentation import losses
from sliq.segmentation import models as seg_models


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long, device=device)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x)['out']
        confusion += losses.confusion(out.argmax(1), y)
    return losses.iou_report(confusion.cpu())


def select(rows, rmin=None, bmin=None):
    """-> (chosen row, constraint_met)."""
    if rmin is None and bmin is None:
        return max(rows, key=lambda r: r['mean_iou']), True
    feasible = [r for r in rows if r['road_iou'] >= (rmin or 0) and r['building_iou'] >= (bmin or 0)]
    return max(feasible or rows, key=lambda r: r['light_iou']), bool(feasible)


def parse_args():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--config', help='yaml file of argument defaults (configs/ablations/)')
    a.add_argument('--name')
    a.add_argument('--arch', default='segformer_b2', choices=seg_models.ARCHS)
    a.add_argument('--loss', default='wce_dice', choices=['ce', 'wce', 'wce_dice', 'conf_wce', 'conf_wce_dice'])
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--epochs', type=int, default=30)
    a.add_argument('--batch', type=int, default=8)
    a.add_argument('--lr', type=float, default=6e-5)
    a.add_argument('--wd', type=float, default=1e-4)
    a.add_argument('--warmup', type=int, default=3)
    a.add_argument('--rare-p', type=float, default=0.5, help='share of crops centred on a light or building pixel')
    a.add_argument('--eval-every', type=int, default=2)
    a.add_argument('--rmin', type=float, default=0.6798, help='road IoU floor')
    a.add_argument('--bmin', type=float, default=0.2567, help='building IoU floor')
    a.add_argument('--lam-ce', type=float, default=1.0)
    a.add_argument('--lam-dice', type=float, default=1.0)
    a.add_argument('--data', default=paths.MAIN)
    a.add_argument('--clean-buildings', type=float, default=0.0, help='background -> ignore in train packs with a mean building share below this')
    a.add_argument('--crop', type=int, nargs=2, default=list(data.CROP_HW), metavar=('H', 'W'), help='720 1280 = full frame')
    a.add_argument('--lamps-as-background', action='store_true', help='lamp-head boxes are background instead of ignore')
    a.add_argument('--workers', type=int, default=10)
    a.add_argument('--subset', type=int, default=0, help='smoke runs: use only this many train and val frames')
    args, _ = a.parse_known_args()
    if args.config:
        a.set_defaults(**{k.replace('-', '_'): v for k, v in yaml.safe_load(open(args.config)).items()})
    args = a.parse_args()
    assert args.name, '--name or a config with a name'
    return args


def main():
    args = parse_args()
    run_dir = paths.run_dir('segmentation', args.name)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda'

    model, provenance = seg_models.build(args.arch)
    model = model.to(device)
    dataset = args.data
    train_frames, val_frames = data.list_frames('train', dataset), data.list_frames('val', dataset)
    cleaned = data.sparse_building_packs(train_frames, args.clean_buildings) if args.clean_buildings else []
    if args.subset:
        train_frames, val_frames = train_frames[::len(train_frames) // args.subset][:args.subset], val_frames[:args.subset]
    json.dump(dict(args=vars(args), model=provenance, background_ignored_packs=cleaned, class_weights=losses.CLASS_WEIGHTS.tolist(), dataset=dataset, start=time.strftime('%F %T'),
                   versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
                   selection=f'light IoU s.t. road >= {args.rmin} and building >= {args.bmin}'), open(run_dir + '/run_record.json', 'w'), indent=1)

    train_set = data.SegDataset(train_frames, True, args.seed, rare_p=args.rare_p, crop_hw=tuple(args.crop), ignore_background_packs=cleaned,
                                lamps_as_background=args.lamps_as_background)
    val_loader = torch.utils.data.DataLoader(data.SegDataset(val_frames, False), batch_size=4, num_workers=args.workers)
    loss_fn = losses.make(args.loss, args.lam_ce, args.lam_dice)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler()
    rows, start = [], time.time()
    log = open(run_dir + '/val_metrics.jsonl', 'w')
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, generator=generator, num_workers=args.workers, drop_last=True)
        factor = min(1.0, (epoch + 1) / max(1, args.warmup)) * (1 - epoch / args.epochs) ** 0.9
        for group in optimiser.param_groups:
            group['lr'] = args.lr * factor
        model.train()
        total, batches = 0.0, 0
        for x, y, q in loader:
            x, y, q = x.to(device, non_blocking=True), y.to(device, non_blocking=True), q.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x)['out']
            loss = loss_fn(out, y, q)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total, batches = total + loss.item(), batches + 1
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            row = evaluate(model, val_loader, device)
            row.update(epoch=epoch + 1, train_loss=total / max(batches, 1), seconds=round(time.time() - start),
                       peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            rows.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            torch.save(model.state_dict(), f'{run_dir}/epoch{epoch + 1}.pt')

    best, constraint_met = select(rows, args.rmin, args.bmin)
    shutil.copy(f'{run_dir}/epoch{best["epoch"]}.pt', run_dir + '/selected.pt')
    for row in rows:
        os.remove(f'{run_dir}/epoch{row["epoch"]}.pt')
    json.dump(dict(best_epoch=best['epoch'], val=best, constraint_met=constraint_met, train_seconds=round(time.time() - start)), open(run_dir + '/selection.json', 'w'), indent=1)
    print('DONE', args.name, 'epoch', best['epoch'], 'mIoU %.4f light %.4f road %.4f building %.4f' % (best['mean_iou'], best['light_iou'], best['road_iou'], best['building_iou']),
          '' if constraint_met else 'CONSTRAINT NOT MET', 'seconds', round(time.time() - start))


if __name__ == '__main__':
    main()
