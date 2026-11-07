"""Reference LiteRT detector runner, the same steps as the phone app.

Letterbox to the model input (pad 114), RGB in [0, 1], output (1, 5, A) = cx, cy, w, h, score, then NMS at IoU 0.7 (max 300) and boxes mapped
back to the 1280x720 frame. Files from the saved_model route (*_tf.tflite) output pixels instead of normalised coordinates."""
import cv2
import numpy as np
import torch
import torchvision


class TFLiteDetector:
    def __init__(self, path, threads=4, conf=0.001, iou=0.7, max_det=300):
        from ai_edge_litert.interpreter import Interpreter
        self.interpreter = Interpreter(model_path=path, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = self.input['shape']
        self.nchw = shape[1] == 3
        self.hw = (int(shape[2]), int(shape[3])) if self.nchw else (int(shape[1]), int(shape[2]))
        self.normalised = not path.endswith('_tf.tflite')
        self.conf, self.iou, self.max_det = conf, iou, max_det

    def letterbox(self, image):
        h0, w0 = image.shape[:2]
        H, W = self.hw
        r = min(H / h0, W / w0)
        new_w, new_h = round(w0 * r), round(h0 * r)
        dw, dh = (W - new_w) / 2, (H - new_h) / 2
        if (new_w, new_h) != (w0, h0):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)), r, left, top

    def raw(self, x):
        """x: float32 (1, 3, H, W) RGB in [0, 1] -> output (5, A) as float32."""
        if not self.nchw:
            x = x.transpose(0, 2, 3, 1)
        self.interpreter.set_tensor(self.input['index'], np.ascontiguousarray(x, dtype=np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output['index'])[0].astype(np.float32)

    def __call__(self, path):
        """-> (xyxy float64 boxes in frame pixels, scores)."""
        bgr = cv2.imread(path)
        h0, w0 = bgr.shape[:2]
        padded, r, left, top = self.letterbox(bgr)
        x = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        y = self.raw(x)
        H, W = self.hw
        xywh, scores = y[:4].T.copy(), y[4]
        if self.normalised:
            xywh *= np.array([W, H, W, H], np.float32)
        keep = scores >= self.conf
        xywh, scores = xywh[keep], scores[keep]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        idx = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.iou).numpy()[:self.max_det]
        boxes, scores = boxes[idx], scores[idx]
        boxes = (boxes - np.array([left, top, left, top], np.float32)) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        return boxes.astype(np.float64), scores.astype(np.float64)
