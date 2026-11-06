# Task 2 — Hard-hat detector

YOLOv8n fine-tuned for 30 epochs on Roboflow Universe **Hard Hat Workers** (joseph-nelson/hard-hat-workers, v10, Public Domain). The final model is the 2-class run `imgsz640_box_dfl` (`head`, `helmet`), deployed as a **TensorRT FP16 engine**. The `person` class was dropped. Every number below is read from a file in `task2/outputs/`, named next to it.

## Where things are

| spec requirement | folder |
|---|---|
| 1 fine-tune, hyperparameters | `outputs/1_training/` (results.csv, results.png, final run args, logs). Hyperparameters and reasons are in the table below. |
| 2 val metrics, confusion matrix, PR curve | `outputs/2_evaluation/` |
| 3 five failure cases | `outputs/3_failure_cases/` |
| 4 ONNX and FP16 / TensorRT export | `outputs/4_export/` (parity table). The exported models are in this folder: `best.onnx`, `best_fp16.onnx`, `best.engine`, `best_labeled.engine`. `best.pt` is in `outputs/runs/imgsz640_box_dfl/weights/`. |
| 5 benchmark on 100 images | `outputs/5_benchmark/` |
| 6 PPE_VIOLATION in the Task 1 pipeline | `outputs/6_ppe_violation/` (events JSON, run log; the video is regenerated, not committed) |
| supporting evidence for the model choice | `outputs/extras/model_comparison/` (runs A to E scored and compared) |
| extra check | `outputs/extras/generalization_test/` |
| training runs | `outputs/runs/` (final run with `best.pt`, plus runs A to D) |

`outputs/.scratch/` holds Ultralytics validation output that commands regenerate. It is not committed.

## Run

```bash
source .venv/bin/activate
python task2/src/hardhat_pipeline.py train --init hardhat --epochs 30 --batch 32 --imgsz 640 --box 10 --dfl 2.0 --name imgsz640_box_dfl
python task2/src/hardhat_pipeline.py evaluate                 # metrics.md, curves, confusion matrices (conf 0.25)
python task2/src/hardhat_pipeline.py confusion-all              # confusion_by_run/, one matrix per kept run
python task2/src/hardhat_pipeline.py compare                  # model_selection.md, every kept run on the same test labels
python task2/src/hardhat_pipeline.py export --imgsz 640 --fp16 --engine   # also writes best_labeled.engine (Ultralytics header)
python task2/src/hardhat_pipeline.py parity                   # export_parity_test.json, every exported file on the test split
python task2/src/hardhat_pipeline.py benchmark --n 100        # benchmark.md
python task2/src/hardhat_pipeline.py failures                 # failure_cases/
python task2/src/zone_monitor.py --config task2/configs/zone_cctv_cuplock_ppe.json   # requirement 6
```

Environment: RTX 5070 Ti 16 GB, torch 2.11.0+cu128, ultralytics 8.4.202, onnx 1.18.0, onnxruntime-gpu 1.22.0, TensorRT 11.4. The dataset is not in git (3.4 GB). `hardhat_pipeline.py download` fetches it.

## Hyperparameters and why

| setting | value | reason |
|---|---|---|
| epochs | 30 | The brief allows 20 to 30. Best epoch was the last, so the budget was the limit. |
| imgsz | 640 | See the model selection table. 1088 did not beat 640 by more than noise and costs more per frame. |
| batch | 32 | Fits 16 GB at 640. |
| box / dfl | 10 / 2.0 (defaults 7.5 / 1.5) | mAP50 was already 0.97 while mAP50-95 sat near 0.67. The gap is box tightness, so localization loss was raised. |
| start weights | keremberke hard-hat YOLOv8n | Required: pretrained weights. Its two classes map to ours (Hardhat to helmet, NO-Hardhat to head). 355 of 355 tensors copied (`outputs/1_training/keremberke_transfer.json`). |
| classes | head, helmet | The brief asks for two. `person` is labelled on 154 of 4916 training images and scored mAP50 0.03, so it was dropped. |
| augmentation | Ultralytics defaults, mosaic on | Kept on purpose for blur and lighting changes in open sites. |
| checkpoint | chosen on valid | The test split was not used to choose it. |

Starting from the hard-hat weights (B) reached mAP50-95 0.40 at epoch 2. Starting from COCO (A) took 7 epochs. The final scores are the same within noise.

## Dataset

| split | images | head | helmet | person (dropped) |
|---|---:|---:|---:|---:|
| train | 4916 | 4612 | 13919 | 450 |
| valid | 1413 | 1339 | 3913 | 101 |
| test | 706 | 726 | 1915 | 64 |

Helmet boxes outnumber head boxes about 3 to 1. Counts come from the label files.

## Model selection

All runs re-scored on the same 706 test images, head and helmet only, at their own training size and at 640 (`outputs/extras/model_comparison/model_selection.md`).

`default` = Ultralytics loss gains (box 7.5, dfl 1.5). `box+dfl` = box 10, dfl 2.0. A is the only run started from COCO weights. B to E start from the hard-hat weights. A and B have 3 classes. C drops helmet-only images. C and D train at 1088, so they are scored at 1088 and at 640.

| model | eval imgsz | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| A imgsz=640 default COCO-init | 640 | 0.955 | 0.925 | 0.972 | 0.670 |
| B imgsz=640 default | 640 | 0.953 | 0.934 | 0.971 | 0.671 |
| C imgsz=1088 box+dfl undersampled | 640 | 0.941 | 0.924 | 0.964 | 0.662 |
| C imgsz=1088 box+dfl undersampled | 1088 | 0.947 | 0.942 | 0.976 | 0.672 |
| D imgsz=1088 box+dfl | 640 | 0.944 | 0.929 | 0.967 | 0.664 |
| D imgsz=1088 box+dfl | 1088 | 0.946 | 0.945 | 0.973 | 0.676 |
| **E imgsz=640 box+dfl (FINAL)** | 640 | 0.941 | 0.940 | 0.973 | 0.674 |

What this supports and what it does not:
- Spread across A, B, E and D at its own size is 0.006 mAP50-95, from single runs with one seed. That is too small to rank the models by accuracy.
- D scores 0.676 at 1088, 0.002 above E. Earlier I reported that 1088 lowered mAP50-95. That came from scoring D at 640, which is not the size it was trained for. It was wrong and is corrected here.
- E was chosen for cost, not for a measurable accuracy win. On the RTX 5070 Ti, PyTorch predict takes 3.45 ms at 640 and 4.38 ms at 1088 (`outputs/extras/model_comparison/model_selection_latency.json`), and 1088 pixels is 2.9 times the pixel count. The edge cost was not measured.
- Undersampling helmet-only images (C) did not help. Full data was restored.

### Confusion matrix of every run

Same procedure for all: test split, IoU 0.7, head and helmet only (`outputs/extras/model_comparison/confusion_by_run.md` and `.json`). Confusion counts are at conf 0.25, the operating threshold. mAP is at conf 0.001, the mAP protocol, and comes from `model_selection.json`. Figure with all panels and mAP in each title: `outputs/extras/model_comparison/confusion_by_run/ALL_RUNS.png`. Runs A and B are 3-class, so their `person` predictions are dropped from the 3x3 view.

| model | imgsz | mAP50 | mAP50-95 | head ok /726 | head as helmet | head missed | helmet ok /1915 | helmet as head | helmet missed | bg FP head | bg FP helmet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A imgsz=640 default COCO-init (imgsz640_default_coco) | 640 | 0.972 | 0.670 | 689 | 14 | 23 | 1830 | 5 | 80 | 66 | 136 |
| B imgsz=640 default (imgsz640_default) | 640 | 0.971 | 0.671 | 691 | 12 | 23 | 1829 | 5 | 81 | 68 | 132 |
| C imgsz=1088 box+dfl undersampled (imgsz1088_box_dfl_undersampled) | 640 | 0.964 | 0.662 | 689 | 10 | 27 | 1798 | 10 | 107 | 64 | 136 |
| C imgsz=1088 box+dfl undersampled (imgsz1088_box_dfl_undersampled) | 1088 | 0.976 | 0.672 | 702 | 12 | 12 | 1834 | 9 | 72 | 65 | 135 |
| D imgsz=1088 box+dfl (imgsz1088_box_dfl) | 640 | 0.967 | 0.664 | 685 | 14 | 27 | 1817 | 7 | 91 | 81 | 150 |
| D imgsz=1088 box+dfl (imgsz1088_box_dfl) | 1088 | 0.973 | 0.676 | 689 | 12 | 25 | 1842 | 2 | 71 | 55 | 142 |
| E imgsz=640 box+dfl (FINAL) (imgsz640_box_dfl) | 640 | 0.973 | 0.674 | 689 | 17 | 20 | 1843 | 7 | 65 | 72 | 139 |

Reading it:
- The costly error is a bare head called helmet, because it hides a violation. Run E has the highest count, 17 of 726 (2.3%). The others have 10 to 14. The gap is 3 to 7 boxes on 726 and one seed per run, so it does not separate the models.
- E has the fewest missed helmets at 640 (65). D at 1088 is close (71) and has the fewest helmets called head (2) and the fewest background false positives on head (55).
- Every run has 130 to 150 background false positives on helmet. Run C, which undersampled helmet-only images, did not reduce them. It added 27 missed heads at 640.
- The two COCO-start and hard-hat-start 3-class runs match the 2-class runs closely. The class change moved no cell by more than a few percent.
- D at 1088 is not worse than E on any safety-relevant cell. E remains the choice on speed (3.45 ms against 4.38 ms) and on one training size serving the engine, not on a measured accuracy gain.

## Valid split (requirement 2)

1413 images, 5252 instances. Source: `outputs/2_evaluation/metrics.md`. Curves: `outputs/2_evaluation/val_PR_curve.png`, `val_F1_curve.png`, `val_P_curve.png`, `val_R_curve.png`. Confusion matrices: `outputs/2_evaluation/val_confusion_matrix.png` and `val_confusion_matrix_normalized.png`.

| class | instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| all | 5252 | 0.948 | 0.933 | 0.970 | 0.671 |
| head | 1339 | 0.930 | 0.928 | 0.960 | 0.662 |
| helmet | 3913 | 0.966 | 0.938 | 0.980 | 0.680 |

## Held-out test

706 images, 2641 instances, not used to pick the checkpoint. Curves: `outputs/2_evaluation/test_PR_curve.png` and the other `test_*` files.

| class | instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| all | 2641 | 0.941 | 0.940 | 0.973 | 0.674 |
| head | 726 | 0.933 | 0.937 | 0.962 | 0.667 |
| helmet | 1915 | 0.949 | 0.944 | 0.983 | 0.680 |

**Confusion matrix on test at conf 0.25** (`outputs/2_evaluation/test_confusion_matrix.png`). The matrix is drawn at the operating threshold. At the mAP threshold of 0.001 it counted 6,380 background false positives and was unreadable.

| true \ predicted | head | helmet | missed |
|---|---:|---:|---:|
| head (726) | 689 | 17 | 20 |
| helmet (1915) | 7 | 1843 | 65 |

Background false positives: 72 as head, 139 as helmet. The costly error is the 17 bare heads called helmet.

## Failure cases (requirement 3)

Images are in `outputs/3_failure_cases/`, boxes drawn in the original image's pixel frame. Blue is a true positive, red a false positive, green a missed label, orange and purple a class mix-up.

1. `failure_1` — A yellow rounded object on the equipment is scored as helmet at 0.84. It is not a head.
2. `failure_2` — The labelled head is only the back of a head behind the man in plaid. The best same-class score on that box is 0.001.
3. `failure_3` — A wide-brimmed felt hat, labelled head, is called helmet at 0.82, so a soft hat would be treated as compliant.
4. `failure_4` — The helmet is about 10 px tall on a person far up the slope. Foreground helmets are detected and this one scores 0.002.
5. `failure_5` — Three helmets of 9–14 px on the far workers are missed. The larger helmets closer to the camera are detected.

## Export and benchmark (requirements 4 and 5)

Exported models are in `outputs/4_export/`: `best.onnx` (FP32), `best_fp16.onnx`, `best.engine` (raw TensorRT plan) and `best_labeled.engine` (the same plan with the Ultralytics header, used by the benchmark and the PPE video). `best.pt` is in `outputs/runs/imgsz640_box_dfl/weights/`. Full test split, per format (`outputs/4_export/export_parity_test.json`):

| format | file | MB | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| pytorch | best.pt | 6.2 | 0.945 | 0.937 | 0.973 | 0.6732 |
| onnx_fp16_cuda | best_fp16.onnx | 6.2 | 0.939 | 0.942 | 0.973 | 0.6734 |
| tensorrt_fp16_cuda | best_labeled.engine | 7.8 | 0.939 | 0.942 | 0.973 | 0.6735 |

Same 100 sorted test images, batch 1, imgsz 640, RTX 5070 Ti. Latency is end-to-end `predict` at conf 0.25. mAP50 is `val` on those 100 images (352 instances), so it differs from the 706-image figure (`outputs/5_benchmark/benchmark.md`).

| backend | mean ms | FPS | model-only ms | mAP50 |
|---|---:|---:|---:|---:|
| pytorch | 3.39 | 295.1 | 2.39 | 0.975 |
| onnx_fp16_cuda | 2.20 | 455.0 | 1.20 | 0.974 |
| **tensorrt_fp16_cuda** | 1.34 | 746.0 | 0.33 | 0.973 |

### Why the TensorRT engine is the final model
- Accuracy is the same. The three formats are within 0.0003 mAP50-95 on the 706 test images.
- It is the fastest. End to end it runs 2.5 times PyTorch (3.39 ms) and 1.6 times ONNX FP16 on CUDA (2.20 ms). The network forward alone is 0.33 ms against 2.39 ms and 1.20 ms. End to end gains less because preprocessing and NMS, about 1 ms, do not shrink.
- It is small (7.8 MB) and the PPE video already runs on it.
- Cost: an engine is tied to one GPU and one TensorRT version, so it is rebuilt on each target device. The ONNX file is the portable fallback.
- File format: `best.engine` is the raw TensorRT plan. `best_labeled.engine` is the same plan behind a 246-byte Ultralytics header carrying the class names `head` and `helmet`. Ultralytics needs the header, raw TensorRT rejects it.

## PPE on the CCTV clip (requirement 6)

`task2/src/zone_monitor.py` runs the Task 1 tracker and adds the helmet engine. A head or helmet box is bound to a tracked person only when its centre lies inside that person's box and in the top half of it, where a head sits. Overlap alone is not enough. A box bound to nobody, such as a helmet-shaped part of the crane, is discarded and never drawn. A person with a bound helmet is green and labelled helmet. Otherwise the ID is red, labelled no helmet, and a `PPE_VIOLATION` interval opens.

`outputs/6_ppe_violation/events_cctv_cuplock_ppe.json` has 32 records: 18 IDLE_ALERT, 13 ZONE_ENTRY and 1 PPE_VIOLATION. The 31 zone and idle records match `task1/outputs/events_cctv_cuplock.json` exactly. The run covered 2391 frames at 25.94 FPS including tracking and drawing.

The helmet label is debounced: a track has no label until one value holds for 1 second (25 frames, `ppe.debounce_sec` in the config). Without it the same clip gave 8 violations, because helmet detections drop out for a few frames and each dropout opened and closed an event. Worker ID 3 alone had four short intervals.

The one violation is track 3, opened at 16.16 s and closed at 17.57 s, on a worker in dark clothes walking in front of the crane. At 16.3 s the box and the no helmet label sit on that worker and no hard hat is visible, so the event is correct. At 16.8 s and 17.2 s the worker is still in view but carries no box, because the person tracker (conf 0.5) stopped detecting them. The event close time is the last frame the track was seen. The other workers on the clip wear helmets and stay green.

Limits. The check is a helmet box whose centre sits in the top half of a person box. A worker bent over, as at 100 s, has the head low in the box and may fall outside that zone. The debounce delays every event start by up to 1 second. The 260 MB output video is not in git. The command above regenerates it.

## Answers

**More helmet boxes than head boxes.** Averaged mAP moves toward the larger class, so a model can score high while missing bare heads. Keep per-class precision, recall and mAP. On test, head mAP50-95 is 0.667 against 0.680 for helmet. Two fixes: oversample images that contain `head`, and set the operating threshold from the head PR curve for a recall target instead of from the averaged mAP. I tried a third, undersampling helmet-only images (run C), and it did not help.

**30 FPS on a Jetson Orin Nano or a phone.** Change the runtime first. Use the FP16 TensorRT engine on the Jetson, built on the device, and measure speed and mAP before trying INT8. This model already runs at 746.0 FPS on an RTX 5070 Ti, but the Jetson figure is not measured here, so the 30 FPS claim is unverified. Small heads lose recall first under a lower input size or INT8 (failure cases 4 and 5). On a phone, run the ONNX file on fewer frames and let the tracker hold the helmet or no-helmet label between detections.

## Generalization

Ten Wikimedia images, licences in `outputs/extras/generalization_test/inputs/SOURCES.json`, scored with the final model at conf 0.25 (`outputs/extras/generalization_test/detections.json`, 15 boxes in total). Two images have no detections. wm_05 is a dark close-up of a leather helmet with no worker, so empty is correct. wm_09 is a foreground back-of-head under a cap and scarf, outside the hard-hat training look, plus small far-away scaffold figures. wm_08 has two workers high on a scaffold against the sky. Its one detection is a helmet at 0.295 on the left worker, who wears a head cloth and not a hard hat, so it is a false compliance that barely clears the threshold. The right worker is missed. Their heads are a few pixels wide at 640. A file written earlier recorded 3 empty images because it came from an older model run at conf 0.35. It was regenerated.
