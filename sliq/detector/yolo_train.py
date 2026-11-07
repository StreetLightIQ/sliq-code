"""YOLO detector training: ignore-aware loss, fixed epoch budget, checkpoints scored by detector.validator, lexicographic selection and tau.

    python -m sliq.detector.yolo_train --name teacher --seed 44
    python -m sliq.detector.yolo_train --name confounder --target confounder --epochs 30 --seed 42

Differences from stock Ultralytics: nbs = batch (no gradient accumulation), warmup_bias_lr = 0, and Ultralytics' own validation is off
because it cannot read the ignore class."""
import argparse
import json
import os
import platform
import shutil
import time

from sliq.common import paths
from sliq.common.data_index import build_index
from sliq.detector import ignore_data, ignore_loss, validator

RUNS = paths.RUNS + '/detector/'
AUGMENTATION = dict(fliplr=0.3, scale=0.1, translate=0.0, mosaic=0.0, close_mosaic=0)      # geometric only


def lamp_trainer():
    """DetectionTrainer that never validates."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class LampTrainer(DetectionTrainer):
        def validate(self):
            return {}, 0.0

        def final_eval(self):
            pass

    return LampTrainer


def parse_args():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--name', required=True)
    a.add_argument('--model', default='yolo26m', choices=['yolo26m', 'yolo11m', 'yolo12m', 'yolo26s', 'yolo26n'])
    a.add_argument('--imgsz', type=int, default=1280)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--epochs', type=int, default=60)
    a.add_argument('--batch', type=int, default=8)
    a.add_argument('--data', default=paths.MAIN)
    a.add_argument('--eval-every', type=int, default=2, help='score every k-th checkpoint and all of the last 30%%')
    a.add_argument('--eps-f', type=float, default=0.2, help='false positives per frame allowed at tau')
    a.add_argument('--device', default='0')
    a.add_argument('--workers', type=int, default=8)
    a.add_argument('--target', default='lamp', choices=['lamp', 'confounder'], help='confounder: scored by light IoU (segmentation.light_from_boxes)')
    a.add_argument('--keep-all', action='store_true', help='keep every epoch checkpoint')
    return a.parse_args()


def main():
    args = parse_args()
    run_dir = RUNS + args.name
    os.makedirs(run_dir, exist_ok=True)
    ignore_data.configure(args.data, 'scene' if args.target == 'confounder' else 'detector')
    ignore_data.TARGET = args.target
    ignore_loss.install()
    ignore_data.install()
    sample_weights = ignore_loss.load_sample_weights(args.data)

    import torch
    import ultralytics
    from ultralytics import YOLO
    json.dump(dict(
        args=vars(args), sample_weights=sample_weights, augmentation=AUGMENTATION, start=time.strftime('%F %T'),
        versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, ultralytics=ultralytics.__version__),
        init_weights=f'{args.model}.pt',
        selection='lexicographic mAP50 > recall > precision > -FP/frame; tau = max recall with FP/frame <= eps_f'),
        open(run_dir + '/run_record.json', 'w'), indent=1)
    data_yaml = run_dir + '/data.yaml'
    ignore_data.write_yaml(data_yaml)

    start = time.time()
    model = YOLO(f'{args.model}.pt')
    model.train(trainer=lamp_trainer(), data=data_yaml, imgsz=args.imgsz, batch=args.batch, nbs=args.batch, epochs=args.epochs, optimizer='AdamW', lr0=1e-4, weight_decay=5e-4,
                cos_lr=True, warmup_epochs=max(1, round(0.1 * args.epochs)), warmup_bias_lr=0.0, amp=True, workers=args.workers, device=args.device,
                flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, auto_augment=None, **AUGMENTATION,
                val=False, plots=False, seed=args.seed, deterministic=True, cache=False, patience=0, save_period=1,
                project=RUNS.rstrip('/'), name=args.name + '/ultra', exist_ok=True, verbose=False)
    train_seconds = time.time() - start
    if args.target == 'confounder':
        from sliq.segmentation import light_from_boxes
        light_from_boxes.select(run_dir, args.imgsz, args.eval_every, args.epochs, train_seconds, device=args.device, keep_all=args.keep_all)
        return

    frames = build_index(dataset=args.data, splits=('val',))
    weights_dir = run_dir + '/ultra/weights'
    epochs = sorted(int(f[5:-3]) for f in os.listdir(weights_dir) if f.startswith('epoch') and f.endswith('.pt'))
    scored = [k for k in epochs if k % args.eval_every == 0 or k > 0.7 * args.epochs or k == epochs[-1]]
    metrics = {}
    with open(run_dir + '/val_metrics.jsonl', 'w') as log:
        for k in scored:
            preds = validator.predict_yolo(f'{weights_dir}/epoch{k}.pt', frames, args.imgsz, device=args.device)
            m = validator.evaluate(frames, preds, 0.25)
            m.pop('per_frame')
            m['epoch'] = k
            metrics[f'epoch{k}'] = m
            log.write(json.dumps(m) + '\n')
            log.flush()
    best = validator.select_checkpoint(metrics)
    shutil.copy(f'{weights_dir}/{best}.pt', run_dir + '/selected.pt')
    preds = validator.predict_yolo(run_dir + '/selected.pt', frames, args.imgsz, device=args.device)
    tau = validator.select_tau(frames, preds, args.eps_f)
    json.dump(dict(best_checkpoint=best, tau_star=tau, eps_f=args.eps_f, val_at_025=metrics[best], curve=validator.recall_fp_curve(frames, preds),
                   train_seconds=round(train_seconds), checkpoints_scored=scored), open(run_dir + '/selection.json', 'w'), indent=1)
    if not args.keep_all:
        for f in os.listdir(weights_dir):
            if f.startswith('epoch') and f != f'{best}.pt':
                os.remove(f'{weights_dir}/{f}')
    m = metrics[best]
    print('DONE', args.name, best, 'mAP50 %.4f recall %.4f precision %.4f fp/frame %.3f tau* %s' % (m['mAP50'], m['recall'], m['precision'], m['fp_per_frame'], tau), 'train_seconds', round(train_seconds))


if __name__ == '__main__':
    main()
