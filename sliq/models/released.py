"""The released models in weights/ and their validation thresholds (weights/thresholds.json)."""
import json
import os

import torch

from sliq.common import paths

FILES = dict(student='student_n_int8.tflite', teacher='teacher_m.pt', confounder='confounder_m.pt', classifier='classifier.pt', segmenter='segmenter_b2.pt')


def path(name):
    return os.path.join(paths.WEIGHTS, FILES[name])


def tau(name):
    return json.load(open(os.path.join(paths.WEIGHTS, 'thresholds.json')))[name]


def segmenter(device='cuda'):
    from sliq.segmentation import models
    return models.load(path('segmenter'), 'segformer_b2', device)


def classifier(device='cuda'):
    from sliq.models.builders import LampClassifier
    model = LampClassifier(pretrained=False)
    model.load_state_dict(torch.load(path('classifier'), map_location=device))
    return model.to(device).eval()
