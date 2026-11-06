TASK 2 HANDOVER — Hard-Hat Detector
Status: DONE and committed (be53706 on main, after 3c2c629). All 6 requirements in the assessment PDF are met.
Final model: imgsz640_box_dfl, deployed as a TensorRT FP16 engine.
Every number below was read from a file in task2/outputs/ on 6 Nov 2026 IST. README.md carries the full tables and the written answers.

FINAL MODEL
  Task: 2 classes (head, helmet). YOLOv8n, 30 epochs, batch 32, imgsz 640, box 10, dfl 2.0.
  Started from the keremberke hard-hat weights (pretrained, as the brief requires).
  task2/outputs/runs/imgsz640_box_dfl/weights/best.pt   PyTorch FP32, 6.2 MB
  task2/outputs/4_export/
    best.onnx             ONNX FP32, 12.3 MB, opset 17, static 1x3x640x640 (kept for portability, not in the comparison)
    best_fp16.onnx        ONNX FP16, 6.2 MB
    best.engine           raw TensorRT plan, 7.8 MB
    best_labeled.engine   same plan behind a 246-byte Ultralytics header (class names). Used by the PPE video and the benchmark.

REQUIREMENT CHECKLIST (PDF Task 2)
  1 fine-tune, hyperparameters and why      README "Hyperparameters and why"; outputs/1_training/ (results.csv, results.png, args, logs)
  2 val metrics, confusion matrix, PR       outputs/2_evaluation/ (metrics.md, val_* and test_* curves, confusion matrices at conf 0.25)
  3 five failure cases with a reason each   outputs/3_failure_cases/ (5 images, failure_cases.json, README.md). All 5 checked by eye.
  4 ONNX plus FP16 or TensorRT              outputs/4_export/ (models, export record, export_parity_test.json)
  5 benchmark on the same 100 images        outputs/5_benchmark/benchmark.md (pytorch, onnx_fp16_cuda, tensorrt_fp16_cuda)
  6 helmet / no helmet + PPE_VIOLATION      src/zone_monitor.py; outputs/6_ppe_violation/events_cctv_cuplock_ppe.json has 1 PPE_VIOLATION
  README answers (class imbalance, 30 FPS edge)   README "Answers"

RESULTS
Test split, 706 images, 2641 instances (2_evaluation/metrics.md): P 0.941, R 0.940, mAP50 0.973, mAP50-95 0.674. Head 0.667, helmet 0.680.
Valid split, 1413 images: mAP50 0.970, mAP50-95 0.671.

Three formats on the full test split (4_export/export_parity_test.json)
  pytorch              best.pt              mAP50 0.973  mAP50-95 0.6732
  onnx_fp16_cuda       best_fp16.onnx       mAP50 0.973  mAP50-95 0.6734
  tensorrt_fp16_cuda   best_labeled.engine  mAP50 0.973  mAP50-95 0.6735
  The three agree within 0.0003.

Benchmark, same 100 sorted test images, batch 1, imgsz 640, RTX 5070 Ti (5_benchmark/benchmark.md)
  pytorch              3.39 ms   295.1 FPS   mAP50 0.975
  onnx_fp16_cuda       2.20 ms   455.0 FPS   mAP50 0.974   (CUDAExecutionProvider confirmed active)
  tensorrt_fp16_cuda   1.34 ms   746.0 FPS   mAP50 0.973
  TensorRT is 2.5x PyTorch and 1.6x ONNX end to end. Network forward alone: 0.33 ms vs 2.39 and 1.20.
  Timing varies run to run: TensorRT read 787.3 FPS on an earlier run, 746.0 on the saved one (about 5%).
  Ten untimed predictions run before timing so the first slow GPU calls do not skew the mean.

WHY THE ENGINE, AND WHY THIS TRAINING RUN (extras/model_comparison/)
  Run names (folders in outputs/runs/). default = box 7.5 / dfl 1.5. box+dfl = box 10 / dfl 2.0.
  Test split, same labels for every run, mAP50-95:
    A imgsz640_default_coco             640 0.670   COCO start, 3-class, 25 epochs
    B imgsz640_default                  640 0.671   hard-hat start, 3-class, 25 epochs
    C imgsz1088_box_dfl_undersampled    640 0.662 / 1088 0.672   2-class, helmet-only images dropped, 25 epochs
    D imgsz1088_box_dfl                 640 0.664 / 1088 0.676   2-class, full data, 30 epochs
    E imgsz640_box_dfl (FINAL)          640 0.674   2-class, full data, 30 epochs
  Spread is 0.006 from single-seed runs. Accuracy does not separate the models.
  E was chosen on cost. D at 1088 scores 0.002 higher but takes 4.50 ms vs 3.45 ms in PyTorch on this GPU (2.9x the pixels).
  The engine was chosen because accuracy is unchanged, it is the fastest, and it is 7.8 MB. Its cost is portability: rebuild per GPU, keep ONNX as the fallback.

CONFUSION MATRICES (test, conf 0.25)
  Final model (2_evaluation/test_confusion_matrix.png): head 689 right, 17 called helmet, 20 missed. Helmet 1843 right, 7 called head, 65 missed. Background false positives: 72 head, 139 helmet.
  Every run: open extras/model_comparison/confusion_by_run.md. It embeds the grid image (ALL_RUNS.png), the count table, and one count and one normalized matrix per run.
    bare head called helmet (the costly error): A 14, B 12, C 10 / 12, D 14 / 12, E 17.   Missed helmets: E 65, D@1088 71, others 72 to 107.
  E is highest on the costly error, by 3 to 7 boxes of 726. One seed per run, so not significant. D@1088 matches or beats E on the safety cells. E was chosen on cost.

PPE VIDEO (requirement 6)
  A head or helmet box counts only when its centre is inside a tracked person's box, in the top half. Boxes that fit no person are dropped, so nothing draws on the crane.
  The label is held for 1 second before it changes (ppe.debounce_sec in the config). Without it the same clip gave 8 violations from detector dropouts.
  Result: 32 events (18 IDLE_ALERT, 13 ZONE_ENTRY, 1 PPE_VIOLATION). The 31 zone and idle events match task1/outputs/events_cctv_cuplock.json exactly.
  The one violation is track 3, 16.16 s to 17.57 s. Checked by eye at 16.3 s: no hard hat on that worker.

OUTPUT FOLDER LAYOUT
  outputs/1_training  2_evaluation  3_failure_cases  4_export  5_benchmark  6_ppe_violation   one folder per spec requirement
  outputs/extras/model_comparison    runs A to E scored and compared (justifies the final model)
  outputs/extras/generalization_test   extra check on 10 outside images (2 empty: wm_05 correct, wm_09 small and unusual)
  outputs/runs                       training runs: final best.pt plus args and results for A to D
  outputs/.scratch                   Ultralytics val output, regenerated, git-ignored
  Task 1's config and README point at outputs/4_export/best_labeled.engine.

GIT
  Committed: 134 files, 70.1 MB. No video, no dataset, no .env, no file over 20 MB.
  Tracked weights: best.pt of the final run, the four files in 4_export, weights/keremberke_remapped_3cls.pt, weights/yolo11n.pt.
  Not tracked: dataset (3.4 GB, `download` restores it), the 260 MB PPE video (`zone_monitor.py` regenerates it), last.pt, and the weights of runs A to D. Their args.yaml and results.csv are tracked.

CORRECTIONS MADE (what was wrong, so the review does not hit it)
  - Run names carried "warmstart". Renamed by image size and loss setting. Pipeline command warmstart is now prepare-weights, --init warmstart is --init hardhat.
  - Exported models sat in runs/. Moved to 4_export/. The export command now writes there.
  - Confusion matrices were drawn at conf 0.001 and showed 12,648 background false positives on val. Now drawn at conf 0.25.
  - An early command used the wrong default run and overwrote the final model's curves and matrices with the old 3-class run. Restored. DEFAULT_RUN points at the final run.
  - Earlier text said 1088 hurt mAP50-95. That scored D at 640, not at its own size. Corrected.
  - The 544.7 FPS TensorRT figure came from an ad hoc script and was never saved. Replaced by the pipeline benchmark.
  - "TensorRT version mismatch" was the Ultralytics metadata header, not a version problem.
  - Failure-case images were drawn with labels multiplied by 640 on an unresized frame. Redrawn in original pixel coordinates.
  - generalization_test/detections.json came from an older model at conf 0.35. Regenerated at conf 0.25.
  - PPE video drew helmets on the crane. Fixed with the person-box rule above, then the 1 second debounce.

OPEN (none block the deliverable)
  1. Jetson or phone 30 FPS is not measured. No device here. README says so.
  2. wm_08 in the generalization set has one false helmet at 0.295 on a head cloth. Not changed.
  3. A bent-over worker may fall outside the top-half head zone in the PPE video. Not changed.
  4. Tasks 1, 3 and 4, the root README, the demo video and docs/system_design.md are not committed. Outside Task 2.

REPRODUCE
  See README "Run". Commands: download, prepare-weights, train, evaluate, compare, confusion-all, export, label-engine, parity, benchmark, failures, generalize.
  `hardhat_pipeline.py download` restores the dataset.
