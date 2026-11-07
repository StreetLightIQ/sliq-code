"""Static INT8 quantisation of an FP32 .tflite with ai_edge_quantizer.

Activation ranges are the mean over the calibration frames of each frame's min/max. The box/score decode, the final head convolutions and
concatenations stay in float."""
import cv2
import numpy as np

from sliq.deploy.tflite_runner import TFLiteDetector

KEEP_FLOAT = (r'.*_NormalizeCoords(;.*)?$',
              r'.*head\.Detect_\d+(;.*)?$',
              r'.*head\.Detect_\d+/torch\.nn\.modules\.container\.Sequential_\d+/torch\.nn\.modules\.conv\.Conv2d_\d+(;.*)?$')
FLOAT_OPS = ('CONCATENATION',)


def calibration_samples(fp32, calib_txt):
    det = TFLiteDetector(fp32)
    out = []
    for p in (l.strip() for l in open(calib_txt) if l.strip()):
        padded, _, _, _ = det.letterbox(cv2.imread(p))
        out.append({'args_0': padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0})
    return out


def _mean_update():
    count = {}

    def update(qsv, new):
        if not qsv:
            return new
        n = count.pop(id(qsv), 1)
        out = {k: (qsv[k] * n + new[k]) / (n + 1) for k in ('min', 'max')}
        count[id(out)] = n + 1
        return out
    return update


def make_quantizer(fp32):
    from ai_edge_quantizer import qtyping, quantizer, recipe
    qt = quantizer.Quantizer(fp32)
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    no_q = recipe.AlgorithmName.NO_QUANTIZE
    for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
        qt.update_quantization_recipe(regex='.*', operation_name=op, algorithm_key=no_q)
    for rx in KEEP_FLOAT:
        qt.update_quantization_recipe(regex=rx, operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, algorithm_key=no_q)
    for op in FLOAT_OPS:
        qt.update_quantization_recipe(regex='.*', operation_name=getattr(qtyping.TFLOperationName, op), algorithm_key=no_q)
    return qt


def calibrate(fp32, calib_txt):
    from ai_edge_quantizer import calibrator
    qt = make_quantizer(fp32)
    calib = calibrator.Calibrator(qt._float_model_buffer, num_threads=16, qsv_update_func=_mean_update())
    calib.calibrate({'serving_default': calibration_samples(fp32, calib_txt)}, qt._recipe_manager)
    return calib.get_model_qsvs()


def quantize(fp32, calib_txt, out_path):
    qt = make_quantizer(fp32)
    qt.quantize(calibration_result=calibrate(fp32, calib_txt), enable_progress_report=False).export_model(out_path, overwrite=True)
    return out_path
