# StreetLightIQ (SLIQ)

Code, models and Android app for the paper *StreetLightIQ: A Mobile Sensing System for City-Scale Streetlight Monitoring and Roadway Illumination Estimation*.

SLIQ turns a phone on a car dashboard into a streetlight survey tool. The phone detects and tracks lamp heads while it records. A cloud stage classifies each lamp as On, Dim or Off and segments the road. A physics-based light model then estimates relative road illumination and ranks lamps by how much light would be lost if they failed. No lux meter is needed.

## Pipeline

| Stage | Where | What |
|---|---|---|
| Phone | `app/` | YOLO26n student (INT8 LiteRT, 960x544, at most 5 fps), centre/scale tracker, upload of track records and up to five full-resolution frames per track |
| Cloud | `sliq/cloud/` | MobileNetV3-Small state classifier on a 1.5x padded crop, SegFormer-B2 segmenter with flip and a confounder detector |
| Light model | `sliq/physics/` | camera pose from gravity, distance from GPS, lamp height and position from tracks, light map, per-lamp coverage and loss if off |

## Repository

| Path | Contents |
|---|---|
| `sliq/detector` | teacher training with ignore regions, knowledge distillation, validation and checkpoint selection |
| `sliq/deploy` | LiteRT export, static INT8 quantisation, scoring, device report |
| `sliq/classifier`, `sliq/segmentation` | lamp state classifier and scene segmenter |
| `sliq/tracker` | Python port of the app tracker, tested against the app |
| `sliq/physics` | perception cache, scene geometry, light model, audits and checks |
| `sliq/runner`, `sliq/e2e` | final evaluation and tables |
| `configs/` | data paths and the segmentation ablation configs |
| `scripts/reproduce.sh` | every run behind the paper tables, in order |
| `weights/` | the released models (Git LFS), see [MODELS.md](MODELS.md) |
| `app/`, `backend/` | Android app and the Supabase schema it uploads to |

## Install

Python 3.12 and one CUDA GPU.

```
pip install -r requirements.txt
pip install -e .
git lfs pull
```

LiteRT export, INT8 quantisation and `.tflite` scoring use a separate CPU environment from `requirements-export.txt`. The FP16 export goes through TensorFlow and needs `requirements-export-tf.txt`.

## Data

The dataset is not included. It is available from the authors on request. [DATA.md](DATA.md) describes the expected layout. Point `configs/paths.yaml` or the `SLIQ_DATA` variable at it. Outputs go to `runs/` (`SLIQ_RUNS`).

## Reproducing the paper

`scripts/reproduce.sh` runs everything in order. The main entry points are:

| Paper | Command |
|---|---|
| Table 1, detector selection | `python -m sliq.detector.yolo_train --name teacher_s44 --seed 44` (and `--model yolo11m`, `--model yolo12m`, `--imgsz`) |
| Table 2, distillation | `python -m sliq.detector.kd_train --name kd_n_960_s42 --seed 42` |
| Table 3, quantisation | `python -m sliq.deploy.export --run kd_n_960_s42 --quant int8`, then `python -m sliq.deploy.score FILE` |
| Classifier and test footage | `python -m sliq.classifier.train --name classifier`, then `python -m sliq.runner.eval_final` |
| Table 4, segmentation | `python -m sliq.segmentation.train --config configs/ablations/<row>.yaml`, then `python -m sliq.segmentation.diagnose` |
| Table 5, device | `python -m sliq.deploy.device_report SESSION_DIR` |
| Table 6, physics summary | `python -m sliq.physics.data --split train`, then `python -m sliq.physics.fieldmodel` |
| Table 7, known-answer audit | `python -m sliq.physics.audit_synth --fps 5` (also 2.65 and 1.5) |
| Table 8, sensitivity | `python -m sliq.physics.field_sensitivity` |
| Table 9, Groton measurements | `python -m sliq.physics.groton_eval` |

The RF-DETR row in Table 1 was trained with the official `rfdetr` package at the same resolution. The failure categories in the appendix are manual labels.

The test footage is scored once. `sliq.deploy.test_footage` and `sliq.segmentation.score_test` leave a `.USED` flag and refuse to run again.

## App

```
cd app
./gradlew assembleDebug
```

This needs JDK 17 and Android SDK 35. The build copies `weights/student_n_int8.tflite` into the APK. Create a Supabase project, run `backend/setup.sql` in its SQL editor and enter the project URL and key in the app settings. Uploaded sessions are processed with `python -m sliq.cloud.process_session SESSION_DIR`.

## Tests

```
pytest tests
```

Tests that need the dataset or a GPU are skipped without them. `tests/fixtures/tracker_parity.json` is written by the app test `TrackerParityTest` and checks that the Python tracker matches the app.

## Licence

AGPL-3.0, see [LICENSE](LICENSE). The segmenter is fine-tuned from NVIDIA's SegFormer, whose weights are under the NVIDIA Source Code License for research and non-commercial use. The Groton measurements in `sliq/physics/data/` are transcribed from Morante (2008), *Mesopic Street Lighting Demonstration and Evaluation*, final report for Groton Utilities.
