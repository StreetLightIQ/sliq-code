# Models

The files in `weights/` are stored with Git LFS (`git lfs pull`). `weights/thresholds.json` holds the validation-chosen confidence thresholds.

| File | Model | Paper | Threshold | Size | SHA-256 |
|---|---|---|---|---|---|
| `student_n_int8.tflite` | YOLO26n student, KD from the teacher, static INT8, 544x960 input. Runs in the app | Tables 2, 3, 5 | 0.10 | 3.0 MB | `daff225760e96f7606973aec28105fef703a24fc2f350d5d810e21a90f5e3653` |
| `teacher_m.pt` | YOLO26m teacher at 1280, seed 44 | Tables 1, 2 | 0.15 | 44 MB | `a709ce74d7f4a6b403fe44fb02b333d6f1451270eac88a576be2dbf57eb70794` |
| `confounder_m.pt` | YOLO26m confounder detector for the confounding-light class | Table 4 | 0.03 | 44 MB | `5d3188989a3e0b34b4694779f5a4be8e7c5729f78e51067e61afab28fb13bb03` |
| `classifier.pt` | MobileNetV3-Small lamp state classifier (On, Dim, Off), seed 43 | classifier results | | 3.8 MB | `46b71d2045439c78152df5efe01138c0ae5242af3c0d3bdf51313336ce559730` |
| `segmenter_b2.pt` | SegFormer-B2 segmenter (background, road, building, confounding light), seed 42 | Table 4 | | 110 MB | `1b6187b8571fdd63e639d14bc3a594426fe2cd75352384950f4ceca9333a5b0e` |

The two `.pt` detectors are Ultralytics checkpoints with the optimiser state removed. They load with `ultralytics.YOLO`. The classifier and segmenter are PyTorch state dicts and load with `sliq.models.released`:

```python
from sliq.models import released
classifier = released.classifier()
segmenter = released.segmenter()
```

The `.tflite` student runs with `sliq.deploy.tflite_runner.TFLiteDetector`, which repeats the app's letterbox, decoding and NMS.

## Licences

The detectors are trained with Ultralytics (AGPL-3.0). The segmenter is fine-tuned from `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`, released under the NVIDIA Source Code License for SegFormer, which allows research and non-commercial use only. The classifier starts from torchvision's ImageNet MobileNetV3-Small weights.
