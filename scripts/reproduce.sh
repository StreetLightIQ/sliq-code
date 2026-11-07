#!/bin/sh
# Every run behind the paper's tables, in order. Needs the dataset (DATA.md) and one GPU.
# Export and LiteRT steps need the export environment (requirements-export.txt). Each block can be run on its own once its inputs exist.
set -e
py() { python -m "$@"; }

# Dataset table
py sliq.e2e.table

# Teacher: three seeds; seed 44 is the best on validation and teaches the students
for s in 42 43 44; do py sliq.detector.yolo_train --name teacher_s$s --seed $s; done

# Architecture comparison (seed 42). The RF-DETR row used the official rfdetr package.
py sliq.detector.yolo_train --name yolo11m --model yolo11m
py sliq.detector.yolo_train --name yolo12m --model yolo12m
py sliq.detector.yolo_train --name yolo26m_640 --imgsz 640
py sliq.detector.yolo_train --name yolo26m_1600 --imgsz 1600

# Distillation table: plain and KD students, three seeds at 960, one at 1280
for m in n s; do
  for s in 42 43 44; do
    py sliq.detector.kd_train --name base_${m}_960_s$s --model yolo26$m --kd off --seed $s
    py sliq.detector.kd_train --name kd_${m}_960_s$s --model yolo26$m --seed $s
  done
  py sliq.detector.kd_train --name base_${m}_1280 --model yolo26$m --imgsz 1280 --kd off
  py sliq.detector.kd_train --name kd_${m}_1280 --model yolo26$m --imgsz 1280
done
py sliq.deploy.collect

# Quantisation table: the deployed student is kd_n_960_s42, the YOLO26s rows use kd_s_960_s43
py sliq.deploy.calib_set --n 300
for q in fp32 fp16 int8dyn int8; do py sliq.deploy.export --run kd_n_960_s42 --quant $q; done
for q in fp32 int8; do py sliq.deploy.export --run kd_s_960_s43 --quant $q; done
for f in runs/deploy/kd_n_960_s42/*.tflite runs/deploy/kd_s_960_s43/*.tflite; do py sliq.deploy.score "$f"; done
py sliq.deploy.score runs/detector/kd_n_960_s42/selected.pt
py sliq.deploy.test_footage --name student_n runs/detector/kd_n_960_s42/selected.pt runs/deploy/kd_n_960_s42/*.tflite
py sliq.deploy.collect

# Classifier, and the detector and classifier on the test footage
py sliq.classifier.train --name classifier
py sliq.runner.eval_final --name final --detector teacher_s44 --classifier classifier

# Segmentation table: ablations, the final segmenter (three seeds), the confounder detector and the inference rows
for s in 42 43 44; do py sliq.segmentation.train --config configs/ablations/seg_baseline_mbv3.yaml --name seg_mbv3_s$s --seed $s; done
for c in seg_b2 seg_b5 seg_full_frame seg_lamps_background seg_sparse_buildings; do py sliq.segmentation.train --config configs/ablations/$c.yaml; done
for s in 42 43 44; do py sliq.segmentation.train --name segmenter_s$s --seed $s; done
py sliq.detector.yolo_train --name confounder --target confounder --epochs 30
py sliq.segmentation.diagnose --segmenter segmenter_s42 --name seed
py sliq.segmentation.diagnose --segmenter segmenter_s42 --flip --name flip
py sliq.segmentation.diagnose --segmenter segmenter_s42 --flip --confounder confounder --name final
py sliq.segmentation.score_test --name final --segmenter segmenter_s42 --confounder confounder
py sliq.segmentation.score_test --name baseline --segmenter seg_mbv3_s42 --arch mbv3 --no-flip
py sliq.runner.report

# Physics: perception cache with the released models (weights/), then the light map, audits and checks
for split in train val test; do py sliq.physics.data --split $split; done
py sliq.physics.fieldmodel --split train val test
py sliq.physics.field_sensitivity
py sliq.physics.uncertainty_mc
for fps in 5 2.65 1.5; do py sliq.physics.audit_synth --fps $fps; done
py sliq.physics.validate_geometry
py sliq.physics.check_straightness
py sliq.physics.groton_eval

# Device table (survey sessions are not part of the dataset) and the light map schematic
py sliq.deploy.device_report --drives
py sliq.deploy.device_report --memory weights/student_n_int8.tflite
py sliq.figures.lightmap_schematic
