# Data

The dataset is available from the authors on request. It comes in the layout below. Set `data:` in `configs/paths.yaml` or the `SLIQ_DATA` variable to its root.

```
data/
  main/                      principal dataset (train, val, test splits)
    splits.json              frame stem -> split
    annotations/<pack>/annotations.json
                             lamp heads with On, Dim, Off state, confounders, ignore regions
    detector/<split>/        images/, labels/ (YOLO format), ignore_masks/
    detector/annotations/<pack>/annotations.json
                             the same frames with lamp boxes tightened to the visible lamp head, used by the detector
    classifier/              index.csv, <split>/<state>/ crops, train_synthetic_context/
    segmentation/<split>/    images/, masks/ (0 background, 1 road, 2 building, 3 confounding light, 255 ignored), confidence/
    sample_weights.json      optional per-pack weights for the box loss
  test_footage/              separately recorded test footage, same layout, scored once
  drives/
    videos/                  the six source recordings
    sensors/<rec>.csv        accelerometer, gyroscope and GPS per video frame
    video_map.json           recording -> video file and segment offsets
    test_spans.json          the test spans used by the light model
```

The detector reads `detector/annotations`. The classifier, segmenter, confounder detector and dataset table read `annotations`, so classifier crops are counted separately from detector boxes.

Frame stems have the form `<pack>__<rec>_segment<k>__f<frame>__t<ms>ms`, so every frame can be traced back to its recording and time.

The splits are blocks of time within each recording. Frames within 2 s of a block edge are left out (112 frames).

The Groton measurements used by `sliq.physics.groton_eval` are in the repository (`sliq/physics/data/groton_3.csv`). They are transcribed from Tables 6 to 8 of Morante (2008) and cover the two high-pressure sodium installations and the induction installation.
