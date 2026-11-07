"""Confounding-light label rule. For each confounder box: bright = gray >= max(180, 0.7 x box max), closed inside the box; the whole box is
filled when the bright share is >= 0.5 or the dilated bright area is < 300 px; then a 2 px dilation. On ground-truth boxes it gives the class-3
labels, on detected boxes it turns the confounder detector into a class-3 predictor."""
import numpy as np
from PIL import Image


def gray(path):
    return np.asarray(Image.open(path).convert('L')).astype(int)


def dil(m, k=1):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


def close_in_box(b, k=3):
    p = np.pad(b, k + 1)
    d = dil(p, k)
    e = ~dil(~d, k)
    return e[k + 1:-(k + 1), k + 1:-(k + 1)] | b


def light_mask(g, boxes_xywh, shape=(720, 1280)):
    """g: gray image, boxes: xywh pixel boxes -> bool mask of confounding-light pixels."""
    out = np.zeros(shape, bool)
    for box in boxes_xywh:
        x, y, w, h = [int(round(v)) for v in box]
        x, y = max(x, 0), max(y, 0)
        c = g[y:y + h, x:x + w]
        if c.size == 0:
            continue
        b = close_in_box(c >= max(180, 0.7 * c.max()))
        sel = np.ones_like(b) if (b.mean() >= 0.5 or dil(b, 2).sum() < 300) else b
        full = np.zeros(shape, bool)
        full[y:y + sel.shape[0], x:x + sel.shape[1]] = sel
        out |= dil(full, 2)
    return out
