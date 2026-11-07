"""Load a finished detector run: prediction function, threshold and checkpoint.

    predict, tau, checkpoint = load_detector('teacher')"""
import json

from sliq.common import paths


def load_detector(run_name, device=0):
    from sliq.detector import validator
    run_dir = paths.run_dir('detector', run_name)
    selection = json.load(open(run_dir + '/selection.json'))
    imgsz = json.load(open(run_dir + '/run_record.json'))['args']['imgsz']
    checkpoint = run_dir + '/selected.pt'
    return (lambda frames, **kw: validator.predict_yolo(checkpoint, frames, imgsz, device=device, **kw)), selection['tau_star'], checkpoint
