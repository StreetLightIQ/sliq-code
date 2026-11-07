"""MobileNetV3-Small lamp-state classifier (On / Dim / Off).

Five head-only epochs, then full fine-tuning on class-balanced real + synthetic batches with class-weighted cross-entropy (AdamW 1e-4, cosine,
5 warmup epochs, early stop after 10). Selection: lowest weighted loss on real validation crops, ties by (R_on + R_dim) / 2.

    python -m sliq.classifier.train --name classifier"""
import argparse
import copy
import json
import math
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sliq.classifier import dataset as data
from sliq.common import paths
from sliq.models.builders import LampClassifier


def build_model():
    return LampClassifier()


@torch.no_grad()
def evaluate(model, loader, device, weights):
    """Weighted CE and per-class recall counts."""
    model.eval()
    confusion = np.zeros((3, 3), int)
    loss_sum, weight_sum = 0.0, 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x).float()
        loss_sum += F.cross_entropy(out, y, weight=weights, reduction='sum').item()
        weight_sum += weights[y].sum().item()
        for target, predicted in zip(y.tolist(), out.argmax(1).tolist()):
            confusion[target, predicted] += 1
    recall = lambda c: confusion[c, c] / max(confusion[c].sum(), 1)
    return dict(loss=loss_sum / max(weight_sum, 1e-9), balanced_on_dim=float((recall(0) + recall(1)) / 2),
                recall_counts={c: (int(confusion[i, i]), int(confusion[i].sum())) for i, c in enumerate(data.CLASSES)}, confusion=confusion.tolist())


def parse_args():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--name', required=True)
    a.add_argument('--seed', type=int, default=43)
    a.add_argument('--epochs', type=int, default=50, help='total, including the head-only epochs')
    a.add_argument('--head-epochs', type=int, default=5)
    a.add_argument('--batch', type=int, default=64)
    a.add_argument('--lr', type=float, default=1e-4)
    a.add_argument('--wd', type=float, default=1e-4)
    a.add_argument('--warmup', type=int, default=5)
    a.add_argument('--patience', type=int, default=10)
    a.add_argument('--workers', type=int, default=10)
    a.add_argument('--subset', type=int, default=0, help='smoke runs: use only this many train and val samples')
    return a.parse_args()


def main():
    args = parse_args()
    run_dir = paths.run_dir('classifier', args.name)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = 'cuda'

    train_samples, val_samples = data.load_samples('train'), data.load_samples('val')
    if args.subset:
        train_samples, val_samples = train_samples[::max(1, len(train_samples) // args.subset)][:args.subset], val_samples[:args.subset]
    weights = data.CLASS_WEIGHTS.to(device)
    json.dump(dict(args=vars(args), class_weights=weights.tolist(), dataset=data.DATASET, start=time.strftime('%F %T'),
                   versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
                   selection='lowest real-validation weighted CE; ties -> higher (R_on + R_dim) / 2'), open(run_dir + '/run_record.json', 'w'), indent=1)

    model = build_model().to(device)
    val_loader = torch.utils.data.DataLoader(data.ClassifierDataset(val_samples), batch_size=128, num_workers=args.workers)

    def train_loader(samples, epoch):
        dataset = data.ClassifierDataset(samples, augment=True, seed=args.seed)
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(data.sampler_weights(samples)), len(samples), replacement=True, generator=generator)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True)

    def run_phase(name, samples, epochs, first_epoch, train_backbone):
        """Yields the validation metrics after every epoch."""
        params = [p for n, p in model.named_parameters() if train_backbone or n.startswith('head')]
        for p in model.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True
        optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        scaler = torch.amp.GradScaler()
        for e in range(epochs):
            factor = min(1.0, (e + 1) / max(1, args.warmup)) * 0.5 * (1 + math.cos(math.pi * e / epochs))
            for group in optimiser.param_groups:
                group['lr'] = args.lr * factor
            model.train()
            if not train_backbone:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()                                           # keep ImageNet BatchNorm statistics
            total, batches = 0.0, 0
            for x, y, _ in train_loader(samples, first_epoch + e):
                x, y = x.to(device, non_blocking=True), y.to(device)
                with torch.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                loss = F.cross_entropy(out.float(), y, weight=weights)
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                total, batches = total + loss.item(), batches + 1
            metrics = evaluate(model, val_loader, device, weights)
            metrics.update(phase=name, epoch=first_epoch + e + 1, train_loss=total / max(batches, 1))
            yield metrics

    best = {'key': None, 'epoch': 0, 'state': None}
    stale = 0
    log = open(run_dir + '/val_metrics.jsonl', 'w')

    def consider(m):
        """Keep the checkpoint if (loss, -balanced recall) improves."""
        nonlocal stale
        key = (round(m['loss'], 9), -m['balanced_on_dim'])
        log.write(json.dumps(m) + '\n')
        log.flush()
        if best['key'] is None or key < best['key']:
            best.update(key=key, epoch=m['epoch'], state=copy.deepcopy(model.state_dict()))
            torch.save(best['state'], run_dir + '/selected.pt')
            stale = 0
            return True
        stale += 1
        return False

    start, epoch = time.time(), 0
    for m in run_phase('head', train_samples, args.head_epochs, epoch, train_backbone=False):
        consider(m)
        epoch += 1
    stale = 0
    for m in run_phase('full', train_samples, args.epochs - args.head_epochs, epoch, train_backbone=True):
        consider(m)
        epoch += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best['state'])
    final = evaluate(model, val_loader, device, weights)
    json.dump(dict(best_epoch=best['epoch'], val=final, train_seconds=round(time.time() - start)), open(run_dir + '/selection.json', 'w'), indent=1)
    print('DONE', args.name, 'best_epoch', best['epoch'], 'val_loss %.4f' % final['loss'], final['recall_counts'], 'balanced_on_dim %.3f' % final['balanced_on_dim'], 'seconds', round(time.time() - start))


if __name__ == '__main__':
    main()
