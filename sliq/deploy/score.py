"""Score a detector file (.pt or .tflite) on validation: mAP50, mAP50-95, the file's own validation tau, recall by lamp size and CPU time.
Optionally compares predictions against a reference .pt.

    python -m sliq.deploy.score runs/deploy/student_n/student_n_int8_544x960.tflite --imgsz 960"""
import argparse
import json
import time

import numpy as np

from sliq.common import paths
from sliq.common.data_index import build_index, xywh2xyxy
from sliq.detector import ignore_data, validator

SIZE_BINS = [(300, 600), (600, 1200), (1200, 2500), (2500, 1e9)]


def val_frames():
    ignore_data.configure(paths.MAIN)
    return build_index(dataset=paths.MAIN, splits=('val',))


THREADS = 4


def predict(path, frames, hw, device='cpu'):
    """-> (per-frame (xyxy, conf), per-frame ms incl. pre/post-processing). .tflite goes through tflite_runner, .pt through Ultralytics."""
    out, ms = [], []
    if path.endswith('.tflite'):
        from sliq.deploy.tflite_runner import TFLiteDetector
        model = TFLiteDetector(path, threads=THREADS)
        assert model.hw == tuple(hw), (model.hw, hw)
        for f in frames:
            t = time.perf_counter()
            out.append(model(f['path']))
            ms.append(1000 * (time.perf_counter() - t))
        return out, ms
    from ultralytics import YOLO
    model = YOLO(path, task='detect')
    for f in frames:
        t = time.perf_counter()
        r = model.predict(f['path'], imgsz=list(hw), conf=0.001, max_det=300, device=device, verbose=False)[0]
        ms.append(1000 * (time.perf_counter() - t))
        out.append((r.boxes.xyxy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy().astype(np.float64)))
    return out, ms


def size_bins(frames, preds, tau):
    out = {}
    for lo, hi in SIZE_BINS:
        tp = gt = 0
        for f, (b, s) in zip(frames, preds):
            lamps = np.asarray(f['lamps']).reshape(-1, 4)
            area = lamps[:, 2] * lamps[:, 3]
            g = xywh2xyxy(lamps)
            b2, s2 = validator.filter_frame(b, s, f['ignores'])
            keep = s2 >= tau
            _, _, _, pairs = validator.match_frame(b2[keep], s2[keep], g, 0.5)
            matched = {j for _, j in pairs}
            sel = [j for j in range(len(g)) if lo <= area[j] < hi]
            gt += len(sel)
            tp += sum(j in matched for j in sel)
        out[f'{lo}-{int(hi) if hi < 1e9 else "inf"}'] = dict(gt=gt, tp=tp, recall=tp / max(gt, 1))
    return out


def summarize(frames, preds, tau):
    m = validator.evaluate(frames, preds, tau)
    m.pop('per_frame')
    return {k: m[k] for k in ('mAP50', 'mAP50_95', 'recall', 'precision', 'fp_per_frame', 'tp', 'fp', 'fn', 'tau')}


def score(path, hw, device='cpu'):
    frames = val_frames()
    preds, ms = predict(path, frames, hw, device)
    tau = validator.select_tau(frames, preds)
    return dict(file=path, split='val', input_hw=list(hw), n_frames=len(frames), n_lamps=int(sum(len(f['lamps']) for f in frames)),
                at_own_tau=summarize(frames, preds, tau) if tau is not None else None, own_tau=tau,
                size_bins_at_own_tau=size_bins(frames, preds, tau) if tau is not None else None,
                cpu_ms=dict(median=float(np.median(ms[5:])), p90=float(np.percentile(ms[5:], 90)), n=len(ms) - 5)), preds


def agreement(frames, a, b, tau):
    """Share of boxes >= tau in a with an IoU >= 0.5 partner >= tau in b, and the mean confidence difference of the pairs."""
    hits = total = 0
    diffs = []
    for (ba, sa), (bb, sb) in zip(a, b):
        ka, kb = sa >= tau, sb >= tau
        iou = validator.iou_matrix(ba[ka], bb[kb])
        total += int(ka.sum())
        if iou.size:
            best = iou.argmax(1)
            ok = iou.max(1) >= 0.5
            hits += int(ok.sum())
            diffs += list(np.abs(sa[ka][ok] - sb[kb][best[ok]]))
    return dict(share_matched=hits / max(total, 1), n=total, mean_abs_conf_diff=float(np.mean(diffs)) if diffs else None)


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('file')
    a.add_argument('--imgsz', type=int, default=960, help='long side')
    a.add_argument('--threads', type=int, default=4)
    a.add_argument('--reference', help='a .pt to compare predictions against')
    a.add_argument('--out')
    args = a.parse_args()
    import torch
    global THREADS
    THREADS = args.threads
    torch.set_num_threads(args.threads)
    from sliq.deploy.export import letterbox_shape
    hw = letterbox_shape(args.imgsz)
    result, preds = score(args.file, hw)
    if args.reference:
        frames = val_frames()
        ref_preds, _ = predict(args.reference, frames, hw)
        tau = result['own_tau']
        result['agreement'] = dict(reference=args.reference, tau=tau, file_vs_ref=agreement(frames, preds, ref_preds, tau), ref_vs_file=agreement(frames, ref_preds, preds, tau))
    out = args.out or args.file.rsplit('.', 1)[0] + '_val.json'
    json.dump(result, open(out, 'w'), indent=1)
    s = result['at_own_tau']
    print('VAL', args.file, 'mAP50 %.3f mAP50-95 %.3f tau %s recall %.3f precision %.3f fp/frame %.3f' % (s['mAP50'], s['mAP50_95'], s['tau'], s['recall'], s['precision'], s['fp_per_frame']),
          '| cpu median %.0f ms' % result['cpu_ms']['median'], '->', out)


if __name__ == '__main__':
    main()
