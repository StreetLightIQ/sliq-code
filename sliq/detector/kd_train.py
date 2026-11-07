"""Student detector training (YOLO26n/s), with or without distillation from the teacher. A baseline (--kd off) and a distilled run share the
loader, recipe, scoring and selection of yolo_train.py. The batch is augmented at --load-imgsz and downscaled to --imgsz on the GPU.

    python -m sliq.detector.kd_train --name student_n                  # deployed student: YOLO26n, 960, KD weights 4/4/1
    python -m sliq.detector.kd_train --name base_n --kd off"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time

from sliq.common import paths
from sliq.common.data_index import build_index
from sliq.detector import ignore_data, ignore_loss, kd, validator
from sliq.detector.yolo_train import AUGMENTATION, RUNS, lamp_trainer


def kd_trainer():
    import torch.nn.functional as F
    base = lamp_trainer()

    class KDTrainer(base):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            size = kd.CFG['student_imgsz']
            if kd.CFG['mode'] != 'off':
                batch['img_teacher'] = batch['img']
            if batch['img'].shape[-1] != size:
                batch['img'] = F.interpolate(batch['img'], size=(size, size), mode='bilinear', align_corners=False, antialias=True)
            return batch

        def build_optimizer(self, model, *args, **kwargs):
            optimizer = super().build_optimizer(model, *args, **kwargs)
            if kd.CFG['mode'] in ('feat', 'both'):
                adapters = kd.build_adapters(model, kd.TEACHER, next(model.parameters()).device)
                optimizer.add_param_group(dict(params=list(adapters.parameters()), weight_decay=kwargs.get('decay', 0.0)))
            return optimizer

    return KDTrainer


def parse_args():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--name', required=True)
    a.add_argument('--model', default='yolo26n', choices=['yolo26n', 'yolo26s', 'yolo26m'])
    a.add_argument('--imgsz', type=int, default=960, help='student size')
    a.add_argument('--load-imgsz', type=int, default=1280, help='loader size (= teacher input)')
    a.add_argument('--teacher', default='teacher_s44', help='finished teacher run (best validation seed)')
    a.add_argument('--kd', default='both', choices=['off', 'out', 'feat', 'both'])
    a.add_argument('--w-cls', type=float, default=4.0)
    a.add_argument('--w-box', type=float, default=4.0)
    a.add_argument('--w-feat', type=float, default=1.0)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--epochs', type=int, default=60)
    a.add_argument('--batch', type=int, default=8)
    a.add_argument('--data', default=paths.MAIN)
    a.add_argument('--eval-every', type=int, default=2)
    a.add_argument('--eps-f', type=float, default=0.2)
    a.add_argument('--device', default='0')
    a.add_argument('--workers', type=int, default=8)
    a.add_argument('--keep-all', action='store_true')
    return a.parse_args()


def main():
    args = parse_args()
    assert args.imgsz <= args.load_imgsz
    run_dir = RUNS + args.name
    assert not os.path.exists(run_dir + '/selection.json'), 'finished run exists; pick a new name'
    os.makedirs(run_dir, exist_ok=True)
    ignore_data.configure(args.data)
    ignore_data.WITH_CONFOUNDERS = args.kd in ('feat', 'both')
    ignore_loss.install()
    ignore_data.install()
    sample_weights = ignore_loss.load_sample_weights(args.data)
    kd.install()
    kd.CFG.update(mode=args.kd, w_cls=args.w_cls, w_box=args.w_box, w_feat=args.w_feat, student_imgsz=args.imgsz, teacher_imgsz=args.load_imgsz)

    import torch
    import ultralytics
    from ultralytics import YOLO
    teacher_ckpt = RUNS + args.teacher + '/selected.pt'
    teacher_sha = None
    if args.kd != 'off':
        teacher_sha = hashlib.sha256(open(teacher_ckpt, 'rb').read()).hexdigest()
        teacher_args = json.load(open(RUNS + args.teacher + '/run_record.json'))['args']
        assert teacher_args['imgsz'] == args.load_imgsz, 'the teacher must see the size it was trained at'
        kd.load_teacher(teacher_ckpt, f'cuda:{args.device}')
    init_weights = f'{args.model}.pt'
    json.dump(dict(
        args=vars(args), sample_weights=sample_weights, augmentation=AUGMENTATION, start=time.strftime('%F %T'),
        kd=dict(mode=args.kd, teacher=args.teacher if args.kd != 'off' else None, teacher_sha256=teacher_sha, weights=dict(cls=args.w_cls, box=args.w_box, feat=args.w_feat),
                constants=kd.constants(), branch='one2many', confounders_loaded=ignore_data.WITH_CONFOUNDERS),
        versions=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, ultralytics=ultralytics.__version__),
        init_weights=init_weights,
        selection='lexicographic mAP50 > recall > precision > -FP/frame; tau = max recall with FP/frame <= eps_f'),
        open(run_dir + '/run_record.json', 'w'), indent=1)
    data_yaml = run_dir + '/data.yaml'
    ignore_data.write_yaml(data_yaml)

    start = time.time()
    model = YOLO(init_weights)

    def report(trainer):
        try:
            items = getattr(trainer, 'tloss', None)
            with open(run_dir + '/train_loss.jsonl', 'a') as f:
                f.write(json.dumps(dict(epoch=trainer.epoch, **{k: float(v) for k, v in items.items()}) if isinstance(items, dict) else {}) + '\n')
        except Exception:
            pass

    model.add_callback('on_train_epoch_end', report)
    model.train(trainer=kd_trainer(), data=data_yaml, imgsz=args.load_imgsz, batch=args.batch, nbs=args.batch, epochs=args.epochs, optimizer='AdamW', lr0=1e-4, weight_decay=5e-4,
                cos_lr=True, warmup_epochs=max(1, round(0.1 * args.epochs)), warmup_bias_lr=0.0, amp=True, workers=args.workers, device=args.device,
                flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, auto_augment=None, **AUGMENTATION,
                val=False, plots=False, seed=args.seed, deterministic=True, cache=False, patience=0, save_period=1,
                project=RUNS.rstrip('/'), name=args.name + '/ultra', exist_ok=True, verbose=False)
    train_seconds = time.time() - start

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
    json.dump(dict(best_checkpoint=best, tau_star=tau, eps_f=args.eps_f, val_at_025=metrics[best], val_at_tau=validator.evaluate(frames, preds, tau) if tau else None,
                   curve=validator.recall_fp_curve(frames, preds), train_seconds=round(train_seconds), checkpoints_scored=scored),
              open(run_dir + '/selection.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    if not args.keep_all:
        for f in os.listdir(weights_dir):
            if f.startswith('epoch') and f != f'{best}.pt':
                os.remove(f'{weights_dir}/{f}')
    m = metrics[best]
    print('DONE', args.name, best, 'mAP50 %.4f recall %.4f precision %.4f fp/frame %.3f tau* %s' % (m['mAP50'], m['recall'], m['precision'], m['fp_per_frame'], tau), 'train_seconds', round(train_seconds))


if __name__ == '__main__':
    main()
