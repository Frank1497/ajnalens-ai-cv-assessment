# Task 2 — Hard-hat detector

YOLOv8n fine-tuned for 30 epochs on Roboflow Universe **Hard Hat Workers** (joseph-nelson/hard-hat-workers, v10, Public Domain). The final model is the 2-class run `imgsz1088_box_dfl` (`head`, `helmet`), deployed as a **TensorRT FP16 engine**. The `person` class was dropped. Every number below is read from a file in `task2/outputs/`, named next to it.

## Where things are


| spec requirement                          | folder                                                                                                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 fine-tune, hyperparameters              | `outputs/1_training/` (results.csv, results.png, final run args, logs). Hyperparameters and reasons are in the table below.                                                          |
| 2 val metrics, confusion matrix, PR curve | `outputs/2_evaluation/`                                                                                                                                                              |
| 3 five failure cases                      | `outputs/3_failure_cases/`                                                                                                                                                           |
| 4 ONNX and FP16 / TensorRT export         | `outputs/4_export/` (parity table). The exported models are in this folder: `best.onnx`, `best_fp16.onnx`, `best.engine`. `best.pt` is in `outputs/runs/imgsz1088_box_dfl/weights/`. |
| 5 benchmark on 100 images                 | `outputs/5_benchmark/`                                                                                                                                                               |
| 6 PPE_VIOLATION in the Task 1 pipeline    | `outputs/6_ppe_violation/` (events JSON, run log; the video is regenerated, not committed)                                                                                           |
| supporting evidence for the model choice  | `outputs/extras/model_comparison/` (runs A to E scored and compared)                                                                                                                 |
| extra check                               | `outputs/extras/generalization_test/`                                                                                                                                                |
| training runs                             | `outputs/runs/` (final run with `best.pt`, plus runs A to D)                                                                                                                         |


`outputs/.scratch/` holds Ultralytics validation output that commands regenerate. It is not committed.

## Run

One command runs the whole pipeline on existing weights and data, then the PPE check:

```bash
source .venv/bin/activate
python task2/src/hardhat_pipeline.py evaluate && python task2/src/zone_monitor.py --config task2/configs/zone_cctv_cuplock_ppe.json
```

First run only: `python task2/src/hardhat_pipeline.py download` fetches the dataset and `prepare-weights` builds the start weights. The full sequence, one step per requirement:

```bash
python task2/src/hardhat_pipeline.py train --init hardhat --epochs 30 --batch 32 --imgsz 1088 --box 10 --dfl 2.0 --name imgsz1088_box_dfl
python task2/src/hardhat_pipeline.py evaluate                 # metrics.md, curves, confusion matrices (conf 0.25)
python task2/src/hardhat_pipeline.py confusion-all              # confusion_by_run/, one matrix per kept run
python task2/src/hardhat_pipeline.py compare                  # model_selection.md, every kept run on the same test labels
python task2/src/hardhat_pipeline.py export --fp16 --engine   # Ultralytics: best.onnx, best_fp16.onnx, best.engine, all at the run's imgsz 1088
python task2/src/hardhat_pipeline.py parity                   # export_parity_test.json, every exported file on the test split
python task2/src/hardhat_pipeline.py benchmark --n 100        # benchmark.md
python task2/src/hardhat_pipeline.py failures                 # failure_cases/
python task2/src/zone_monitor.py --config task2/configs/zone_cctv_cuplock_ppe.json   # requirement 6
```

Environment: RTX 5070 Ti 16 GB, torch 2.11.0+cu128, ultralytics 8.4.202, onnx 1.18.0, onnxruntime-gpu 1.22.0, TensorRT 11.4. The dataset is not in git (3.4 GB). `hardhat_pipeline.py download` fetches it.

## Hyperparameters and why


| setting       | value                           | reason                                                                                                                                                                                                         |
| ------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| epochs        | 30                              | The brief allows 20 to 30. Best epoch was the last, so the budget was the limit.                                                                                                                               |
| imgsz         | 1088                            | Best mAP50-95 and the fewest head/helmet mix-ups of the runs tried (see model selection and the confusion table). Costs 4.46 ms against 3.45 ms in PyTorch at 640. Heads and helmets are small in CCTV frames. |
| batch         | 32                              | Fits 16 GB at 1088 with `cache=disk`.                                                                                                                                                                          |
| box / dfl     | 10 / 2.0 (defaults 7.5 / 1.5)   | mAP50 was already 0.97 while mAP50-95 sat near 0.67. The gap is box tightness, so localization loss was raised.                                                                                                |
| start weights | keremberke hard-hat YOLOv8n     | Required: pretrained weights. Its two classes map to ours (Hardhat to helmet, NO-Hardhat to head). 355 of 355 tensors copied (`outputs/1_training/keremberke_transfer.json`).                                  |
| classes       | head, helmet                    | The brief asks for two. `person` is labelled on 154 of 4916 training images and scored mAP50 0.03, so it was dropped.                                                                                          |
| augmentation  | Ultralytics defaults, mosaic on | Kept on purpose for blur and lighting changes in open sites.                                                                                                                                                   |
| checkpoint    | chosen on valid                 | The test split was not used to choose it.                                                                                                                                                                      |


Starting from the hard-hat weights (B) reached mAP50-95 0.40 at epoch 2. Starting from COCO (A) took 7 epochs. The final scores are the same within noise.

## Dataset


| split | images | head | helmet | person (dropped) |
| ----- | ------ | ---- | ------ | ---------------- |
| train | 4916   | 4612 | 13919  | 450              |
| valid | 1413   | 1339 | 3913   | 101              |
| test  | 706    | 726  | 1915   | 64               |


Helmet boxes outnumber head boxes about 3 to 1. Counts come from the label files.

## Model selection

All runs re-scored on the same 706 test images, head and helmet only, at their own training size and at 640 (`outputs/extras/model_comparison/model_selection.md`).

`default` = Ultralytics loss gains (box 7.5, dfl 1.5). `box+dfl` = box 10, dfl 2.0. A is the only run started from COCO weights. B to E start from the hard-hat weights. A and B have 3 classes. C drops helmet-only images. C and D train at 1088, so they are scored at 1088 and at 640.


| model                             | eval imgsz | P     | R     | mAP50 | mAP50-95  |
| --------------------------------- | ---------- | ----- | ----- | ----- | --------- |
| A imgsz=640 default COCO-init     | 640        | 0.955 | 0.925 | 0.972 | 0.670     |
| B imgsz=640 default               | 640        | 0.953 | 0.934 | 0.971 | 0.671     |
| C imgsz=1088 box+dfl undersampled | 640        | 0.941 | 0.924 | 0.964 | 0.662     |
| C imgsz=1088 box+dfl undersampled | 1088       | 0.947 | 0.942 | 0.976 | 0.672     |
| D imgsz=640 box+dfl               | 640        | 0.944 | 0.929 | 0.967 | 0.664     |
| **D imgsz=1088 box+dfl (FINAL)**  | 1088       | 0.946 | 0.945 | 0.973 | **0.676** |
| E imgsz=640 box+dfl               | 640        | 0.941 | 0.940 | 0.973 | 0.674     |


What this supports and what it does not:

- Spread across A, B, D at its own size and E is 0.006 mAP50-95, from single runs with one seed. That is too small to rank the models by accuracy alone.
- D (final) leads mAP50-95 at 0.676, 0.002 above E. mAP50 is level at 0.973. Run C scores higher mAP50 (0.976) with looser boxes (mAP50-95 0.672).
- The case for D rests on the head versus helmet confusion matrix (next section), not on mAP. On the test split D has 14 head/helmet mix-ups against 24 for E, and 12 bare heads called helmet against 17. E has 11 fewer missed objects.
- D costs more per frame: PyTorch predict takes 4.46 ms at 1088 against 3.45 ms at 640 (`outputs/extras/model_comparison/model_selection_latency.json`). The TensorRT engine of D runs at 2.15 ms. A 640 TensorRT engine was built earlier and deleted, so the engine speed of the 640 model is not compared here.
- Earlier text said 1088 lowered mAP50-95. That scored D at 640, which is not the size it was trained for. It was wrong and is corrected.
- Undersampling helmet-only images (C) did not help. Full data was restored.



### Confusion matrix of every run

**Open** `outputs/extras/model_comparison/confusion_by_run.md` **to see every matrix as an image**: the side-by-side grid, the count table, and one count plus one normalized matrix per run.

Same procedure for all: test split, IoU 0.7, head and helmet only (`outputs/extras/model_comparison/confusion_by_run.md` and `.json`). Confusion counts are at conf 0.25, the operating threshold. mAP is at conf 0.001, the mAP protocol, and comes from `model_selection.json`. Figure with all panels and mAP in each title: `outputs/extras/model_comparison/confusion_by_run/ALL_RUNS.png`. Runs A and B are 3-class, so their `person` predictions are dropped from the 3x3 view.


| model                                                              | imgsz | mAP50 | mAP50-95 | head ok /726 | head as helmet | head missed | helmet ok /1915 | helmet as head | helmet missed | bg FP head | bg FP helmet |
| ------------------------------------------------------------------ | ----- | ----- | -------- | ------------ | -------------- | ----------- | --------------- | -------------- | ------------- | ---------- | ------------ |
| A imgsz=640 default COCO-init (imgsz640_default_coco)              | 640   | 0.972 | 0.670    | 689          | 14             | 23          | 1830            | 5              | 80            | 66         | 136          |
| B imgsz=640 default (imgsz640_default)                             | 640   | 0.971 | 0.671    | 691          | 12             | 23          | 1829            | 5              | 81            | 68         | 132          |
| C imgsz=1088 box+dfl undersampled (imgsz1088_box_dfl_undersampled) | 640   | 0.964 | 0.662    | 689          | 10             | 27          | 1798            | 10             | 107           | 64         | 136          |
| C imgsz=1088 box+dfl undersampled (imgsz1088_box_dfl_undersampled) | 1088  | 0.976 | 0.672    | 702          | 12             | 12          | 1834            | 9              | 72            | 65         | 135          |
| D imgsz=1088 box+dfl (FINAL) (imgsz1088_box_dfl)                   | 640   | 0.967 | 0.664    | 685          | 14             | 27          | 1817            | 7              | 91            | 81         | 150          |
| D imgsz=1088 box+dfl (FINAL) (imgsz1088_box_dfl)                   | 1088  | 0.973 | 0.676    | 689          | 12             | 25          | 1842            | 2              | 71            | 55         | 142          |
| E imgsz=640 box+dfl (imgsz640_box_dfl)                             | 640   | 0.973 | 0.674    | 689          | 17             | 20          | 1843            | 7              | 65            | 72         | 139          |


Reading it:

- The costly error is a bare head called helmet, because it hides a violation. D at 1088 has 12 of 726 (1.7%). E has 17 (2.3%). A, B and C have 14, 12 and 10 to 12.
- D at 1088 has the fewest helmets called head (2) and the fewest background false positives on head (55). E has the fewest missed helmets (65 against 71) and the fewest missed heads (20 against 25).
- Head and helmet mix-ups in total: D 14, E 24. Missed objects: D 96, E 85. All errors: D 307, E 320.
- Every run has 130 to 150 background false positives on helmet. Run C, which undersampled helmet-only images, did not reduce them.
- One seed per run. The gaps are 10 mix-ups and 11 misses on 2,641 objects. They point to D but are not proven.



## Valid split (requirement 2)

1413 images, 5252 instances, scored at imgsz 1088. Source: `outputs/2_evaluation/metrics.md`. Curves: `outputs/2_evaluation/val_PR_curve.png`, `val_F1_curve.png`, `val_P_curve.png`, `val_R_curve.png`. Confusion matrices: `outputs/2_evaluation/val_confusion_matrix.png` and `val_confusion_matrix_normalized.png`.


| class  | instances | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | --------- | ------ | ----- | -------- |
| all    | 5252      | 0.938     | 0.943  | 0.969 | 0.671    |
| head   | 1339      | 0.919     | 0.942  | 0.955 | 0.664    |
| helmet | 3913      | 0.956     | 0.945  | 0.983 | 0.677    |




## Held-out test

706 images, 2641 instances, not used to pick the checkpoint. Curves: `outputs/2_evaluation/test_PR_curve.png` and the other `test_*` files.


| class  | instances | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | --------- | ------ | ----- | -------- |
| all    | 2641      | 0.946     | 0.945  | 0.973 | 0.676    |
| head   | 726       | 0.945     | 0.939  | 0.962 | 0.671    |
| helmet | 1915      | 0.947     | 0.950  | 0.983 | 0.681    |


**Confusion matrix on test at conf 0.25** (`outputs/2_evaluation/test_confusion_matrix.png`). Drawn at the operating threshold. At the mAP threshold of 0.001 the matrix counted thousands of background false positives and was unreadable.


| true \ predicted | head | helmet | missed |
| ---------------- | ---- | ------ | ------ |
| head (726)       | 689  | 12     | 25     |
| helmet (1915)    | 2    | 1842   | 71     |


Background false positives: 55 as head, 142 as helmet. The costly error is the 12 bare heads called helmet.

## Failure cases (requirement 3)

Images are in `outputs/3_failure_cases/`, boxes drawn in the original image's pixel frame. Blue is a true positive, red a false positive, green a missed label, orange and purple a class mix-up.

1. `failure_1` — A yellow rounded object on the equipment is scored as helmet at 0.77. It is not a head.
2. `failure_2` — The labelled head is only the back of a head behind the man in plaid. The best same-class score on that box is 0.001.
3. `failure_3` — A wide-brimmed felt hat, labelled head, is called helmet at 0.83, so a soft hat would be treated as compliant.
4. `failure_4` — The helmet is about 10 px tall on a person far up the slope. Foreground helmets are detected and this one scores 0.013.
5. `failure_5` — Three helmets of 9–14 px on the far workers are missed. The larger helmets closer to the camera are detected.



## Export and benchmark (requirements 4 and 5)

Exported with Ultralytics from `best.pt` of `imgsz1088_box_dfl`, all at imgsz 1088, static shape, batch 1: `python task2/src/hardhat_pipeline.py export --fp16 --engine`. The files are in `outputs/4_export/`: `best.onnx` (FP32, kept for portability), `best_fp16.onnx` and `best.engine` (TensorRT FP16). `best.pt` is in `outputs/runs/imgsz1088_box_dfl/weights/`. Each format on the full test split (`outputs/4_export/export_parity_test.json`):


| format             | file           | MB  | P     | R     | mAP50 | mAP50-95 |
| ------------------ | -------------- | --- | ----- | ----- | ----- | -------- |
| pytorch            | best.pt        | 6.3 | 0.945 | 0.944 | 0.973 | 0.6758   |
| onnx_fp16_cuda     | best_fp16.onnx | 6.3 | 0.950 | 0.941 | 0.972 | 0.6749   |
| tensorrt_fp16_cuda | best.engine    | 9.5 | 0.950 | 0.942 | 0.972 | 0.6753   |


Same 100 sorted test images, batch 1, imgsz 1088, RTX 5070 Ti. Latency is end-to-end `predict` at conf 0.25 after 10 untimed warm-up calls. mAP50 is `val` on those 100 images (352 instances), so it differs from the 706-image figure (`outputs/5_benchmark/benchmark.md`).


| backend                | mean ms | FPS   | model-only ms | mAP50 |
| ---------------------- | ------- | ----- | ------------- | ----- |
| pytorch                | 4.31    | 232.2 | 2.75          | 0.974 |
| onnx_fp16_cuda         | 3.67    | 272.5 | 1.96          | 0.978 |
| **tensorrt_fp16_cuda** | 2.15    | 465.3 | 0.57          | 0.977 |




### Why the TensorRT engine is the final deployed format

- Accuracy is the same. The three formats are within 0.0009 mAP50-95 on the 706 test images.
- It is the fastest. End to end it runs 2.0 times PyTorch (4.31 ms) and 1.7 times ONNX FP16 on CUDA (3.67 ms). The network forward alone is 0.57 ms against 2.75 ms and 1.96 ms. End to end gains less because preprocessing and NMS do not shrink.
- It is small (9.5 MB) and the PPE video runs on it.
- Cost: an engine is tied to one GPU and one TensorRT version, so it is rebuilt on each target device. The ONNX file is the portable fallback.
- The engine is the Ultralytics export as written. An earlier attempt to wrap it in a second metadata header made it unloadable, and that wrapper was removed.
- Timing varies by about 5% between runs on this GPU.



## PPE on the CCTV clip (requirement 6)

`task2/src/zone_monitor.py` runs the Task 1 tracker and adds the helmet engine at imgsz 1088. A head or helmet box is bound to a tracked person only when its centre lies inside that person's box and in the top half of it, where a head sits. A box bound to nobody, such as a helmet-shaped part of the crane, is discarded and never drawn. A person with a bound helmet is green and labelled helmet. Otherwise the ID is red, labelled no helmet, and a `PPE_VIOLATION` interval opens. The label is held for 1 second before it changes (`ppe.debounce_sec`). Without that hold the same clip gave 8 violations from detector dropouts.

`outputs/6_ppe_violation/events_cctv_cuplock_ppe.json` has 32 records: 18 IDLE_ALERT, 13 ZONE_ENTRY and 1 PPE_VIOLATION. The 31 zone and idle records match `task1/outputs/events_cctv_cuplock.json` exactly. The run covered 2391 frames at 26.14 FPS including tracking and drawing. The latest run, with the live preview window open, logged 17.88 FPS (`outputs/6_ppe_violation/run_log_ppe.txt`).

The one violation is track 3, 17.47 s to 17.57 s, on a worker in dark clothes bent over under the crane hook. At 15.0 s the worker is green with a helmet box on a yellow hard hat. At 17.45 s and 17.55 s the head is turned down and hidden and the box reads no helmet. That is a short, ambiguous event: the worker does wear a hard hat earlier, and the model cannot see it at the end. It is a likely false alarm, and the README does not claim otherwise.

**Ultralytics CLI.** The same engine also runs from the CLI on the video, with head and helmet boxes only and no person tracker:

```bash
yolo detect predict model=task2/outputs/4_export/best.engine source=<clip>.mp4 imgsz=1088 conf=0.35 device=0 save=True
```

This ran on all 2391 frames at 0.6 ms inference per frame (3.2 ms preprocess, 0.7 ms postprocess) and wrote an annotated video. Frames at 16.5 s and 60 s show helmet boxes on the two workers' heads. Without a person box the CLI also draws a helmet box on a round object on the crane at 60 s, which is why the tracked pipeline binds boxes to people.

Limits. The check is a helmet box whose centre sits in the top half of a person box. A bent-over worker has the head low in the box and can fall outside that zone. The debounce delays every event start by up to 1 second. The 260 MB output video is not in git. The command above regenerates it.

## Answers

**More helmet boxes than head boxes.** Averaged mAP moves toward the larger class, so a model can score high while missing bare heads. Keep per-class precision, recall and mAP. On test, head mAP50-95 is 0.667 against 0.680 for helmet. Two fixes: oversample images that contain `head`, and set the operating threshold from the head PR curve for a recall target instead of from the averaged mAP. I tried a third, undersampling helmet-only images (run C), and it did not help.

**30 FPS on a Jetson Orin Nano or a phone.** Change the runtime first. Use the FP16 TensorRT engine on the Jetson, built on the device, and measure speed and mAP before trying INT8. This model already runs at 465.3 FPS on an RTX 5070 Ti at imgsz 1088, but the Jetson figure is not measured here, so the 30 FPS claim is unverified. Small heads lose recall first under a lower input size or INT8 (failure cases 4 and 5). On a phone, run the ONNX file on fewer frames and let the tracker hold the helmet or no-helmet label between detections.

## Generalization

Ten Wikimedia images, licences in `outputs/extras/generalization_test/inputs/SOURCES.json`, scored with the final model at conf 0.25 (`outputs/extras/generalization_test/detections.json`, 18 boxes in total). Two images have no detections: wm_05.jpg, wm_09.jpg. wm_05 is a dark close-up of a leather helmet with no worker, so empty is correct. wm_09 is a foreground back-of-head under a cap and scarf, outside the hard-hat training look, plus small far-away scaffold figures. wm_08 has two workers high on a scaffold against the sky. Its one detection is a helmet on the left worker, who wears a head cloth and not a hard hat, so it is a false compliance.