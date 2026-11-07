"""Export a detector run to LiteRT: fp32, fp16, int8dyn (INT8 weights, float activations) or int8 (static, see int8.py).
The input is the letterboxed inference shape (960 -> 544x960). NMS runs outside the model. Use the export environment (README).

    python -m sliq.deploy.export --run kd_n_960_s42 --quant fp32
    python -m sliq.deploy.export --run kd_n_960_s42 --quant int8      # needs the fp32 file and deploy.calib_set
Writes runs/deploy/<run>/<run>_<quant>_<h>x<w>.tflite and a .json record."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time

from sliq.common import paths

CALIB = paths.RUNS + '/deploy/calib/calib_train300'
QUANTS = ('fp32', 'fp16', 'int8dyn', 'int8')


def letterbox_shape(imgsz, width=1280, height=720, stride=32):
    """(h, w) input for a 1280x720 frame at long side imgsz."""
    h = round(height * imgsz / width)
    return -(-h // stride) * stride, imgsz


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _versions():
    from importlib import metadata
    out = dict(python=platform.python_version())
    for p in ('torch', 'ultralytics', 'ai-edge-litert', 'ai-edge-quantizer', 'tensorflow', 'onnx2tf', 'onnx'):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def export_ultralytics(ckpt, out_dir, quant, hw):
    """fp32 and int8dyn through Ultralytics' LiteRT export, fp16 through the saved_model route."""
    from ultralytics import YOLO
    fmt = 'saved_model' if quant == 'fp16' else 'litert'
    work = f'{out_dir}/work_{quant}'
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    shutil.copy(ckpt, work + '/model.pt')
    quantize = {'fp32': 32, 'int8dyn': 'w8a32', 'fp16': None}[quant]
    produced = YOLO(work + '/model.pt').export(format=fmt, imgsz=list(hw), batch=1, quantize=quantize, device='cpu')
    if fmt == 'saved_model':
        produced = [os.path.join(produced, f) for f in os.listdir(produced) if f.endswith('float16.tflite')][0]
    return produced


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--run', required=True)
    a.add_argument('--quant', default='fp32', choices=QUANTS)
    a.add_argument('--imgsz', type=int, help="long side (default: the run's size)")
    args = a.parse_args()
    run_dir = paths.run_dir('detector', args.run)
    ckpt = run_dir + '/selected.pt'
    hw = letterbox_shape(args.imgsz or json.load(open(run_dir + '/run_record.json'))['args']['imgsz'])
    out_dir = paths.run_dir('deploy', args.run)
    os.makedirs(out_dir, exist_ok=True)
    name = f'{out_dir}/{args.run}_{args.quant}_{hw[0]}x{hw[1]}{"_tf" if args.quant == "fp16" else ""}.tflite'   # _tf: pixel outputs
    start = time.time()
    record = dict(run=args.run, checkpoint_sha256=sha(ckpt), quant=args.quant, input_hw=hw, head='one2many, NMS outside the model')
    if args.quant == 'int8':
        from sliq.deploy import int8
        fp32 = f'{out_dir}/{args.run}_fp32_{hw[0]}x{hw[1]}.tflite'
        assert os.path.exists(fp32), 'export --quant fp32 first'
        int8.quantize(fp32, CALIB + '.txt', name)
        record.update(source_fp32_sha256=sha(fp32), calibration=json.load(open(CALIB + '.json')), float_ops=list(int8.FLOAT_OPS), kept_float=list(int8.KEEP_FLOAT))
    else:
        shutil.copy(export_ultralytics(ckpt, out_dir, args.quant, hw), name)
    record.update(file_sha256=sha(name), bytes=os.path.getsize(name), export_seconds=round(time.time() - start, 1), date=time.strftime('%F %T'), versions=_versions())
    json.dump(record, open(name.replace('.tflite', '.json'), 'w'), indent=1)
    print('EXPORTED', name, record['bytes'], 'bytes')


if __name__ == '__main__':
    main()
