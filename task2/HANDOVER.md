TASK 2 HANDOVER — Hard-Hat Detector
Status: DONE. All 6 requirements in the assessment PDF are met.
Final model: imgsz1088_box_dfl, exported with Ultralytics to ONNX FP16 and a TensorRT FP16 engine.
Every number below was read from a file in task2/outputs/ on 6 Nov 2026 IST. README.md carries the full tables and the written answers.

FINAL MODEL
  YOLOv8n, 2 classes (head, helmet), 30 epochs, batch 32, imgsz 1088, box 10, dfl 2.0, cache=disk, seed 0.
  Started from the keremberke hard-hat weights (pretrained, as the brief requires).
  task2/outputs/runs/imgsz1088_box_dfl/weights/best.pt   PyTorch FP32, 6.3 MB
  task2/outputs/4_export/ (Ultralytics export, imgsz 1088, static shape, batch 1)
    best.onnx             ONNX FP32, 12.6 MB (kept for portability, not in the comparison)
    best_fp16.onnx        ONNX FP16, 6.3 MB
    best.engine           TensorRT FP16, 9.5 MB. Used by the PPE video and the CLI.
  The 640 model's exports (ONNX, FP16 ONNX, engines) were deleted. Only its best.pt and results are kept for the comparison.

WHY THIS MODEL (changed from imgsz640_box_dfl at the user's request, after the head/helmet confusion matrices)
  Test split, head versus helmet, conf 0.25 (extras/model_comparison/confusion_by_run.md):
                                  imgsz1088_box_dfl @1088    imgsz640_box_dfl @640
    heads correct /726                    689                       689
    helmets correct /1915                1842                      1843
    head called helmet (costly)            12                        17
    helmet called head                      2                         7
    mix-ups total                          14                        24
    missed objects                         96                        85
    background false positives            197                       211
    all errors                            307                       320
  mAP50-95 0.676 against 0.674. mAP50 level at 0.973.
  All evidence points to the 1088 model, but one seed per run and gaps of about 10 boxes on 2,641 objects mean it is not proven.
  Cost: PyTorch predict 4.46 ms at 1088 against 3.45 ms at 640. The 640 TensorRT engine was deleted, so the engine speed of the 640 model is not compared.

REQUIREMENT CHECKLIST (PDF Task 2)
  1 fine-tune, hyperparameters and why      README "Hyperparameters and why"; outputs/1_training/ (results.csv, results.png, args, logs)
  2 val metrics, confusion matrix, PR       outputs/2_evaluation/ (metrics.md, val_* and test_* curves, confusion matrices at conf 0.25)
  3 five failure cases with a reason each   outputs/3_failure_cases/ (5 images, failure_cases.json, README.md). Same 5 images fail on the 1088 model; scores in the reasons updated.
  4 ONNX plus FP16 or TensorRT              outputs/4_export/ (models, export record, export_parity_test.json)
  5 benchmark on the same 100 images        outputs/5_benchmark/benchmark.md (pytorch, onnx_fp16_cuda, tensorrt_fp16_cuda)
  6 helmet / no helmet + PPE_VIOLATION      src/zone_monitor.py; outputs/6_ppe_violation/events_cctv_cuplock_ppe.json has 1 PPE_VIOLATION
  README answers (class imbalance, 30 FPS edge)   README "Answers"

RESULTS (final model, scored at imgsz 1088)
Valid split, 1413 images, 5252 instances: P 0.938, R 0.943, mAP50 0.969, mAP50-95 0.671.
Test split, 706 images, 2641 instances: P 0.946, R 0.945, mAP50 0.973, mAP50-95 0.676. Head 0.671, helmet 0.681.
Test confusion (conf 0.25): head 689 right, 12 called helmet, 25 missed. Helmet 1842 right, 2 called head, 71 missed. Background false positives: 55 head, 142 helmet.

Three formats on the full test split (4_export/export_parity_test.json)
  pytorch              best.pt              mAP50 0.973  mAP50-95 0.6758
  onnx_fp16_cuda       best_fp16.onnx       mAP50 0.972  mAP50-95 0.6749
  tensorrt_fp16_cuda   best.engine          mAP50 0.972  mAP50-95 0.6753
  The three agree within 0.0009.

Benchmark, same 100 sorted test images, batch 1, imgsz 1088, RTX 5070 Ti (5_benchmark/benchmark.md)
  pytorch              4.31 ms   232.2 FPS   mAP50 0.974
  onnx_fp16_cuda       3.67 ms   272.5 FPS   mAP50 0.978   (CUDAExecutionProvider confirmed active)
  tensorrt_fp16_cuda   2.15 ms   465.3 FPS   mAP50 0.977
  TensorRT is 2.0x PyTorch and 1.7x ONNX end to end. Network forward alone: 0.57 ms vs 2.75 and 1.96.
  Timing varies about 5% between runs. Ten untimed predictions run before timing so the first slow GPU calls do not skew the mean.

ALL RUNS (extras/model_comparison/; run folders in outputs/runs/). default = box 7.5 / dfl 1.5, box+dfl = box 10 / dfl 2.0.
  A imgsz640_default_coco           640 mAP50-95 0.670   COCO start, 3-class, 25 epochs
  B imgsz640_default                640 0.671            hard-hat start, 3-class, 25 epochs
  C imgsz1088_box_dfl_undersampled  640 0.662 / 1088 0.672   2-class, helmet-only images dropped, 25 epochs
  D imgsz1088_box_dfl (FINAL)       640 0.664 / 1088 0.676   2-class, full data, 30 epochs
  E imgsz640_box_dfl                640 0.674            2-class, full data, 30 epochs
  Spread is 0.006 from single-seed runs. Every run is scored at its own training size; scoring at the wrong size shifts mAP by up to 0.012.
  Confusion matrix of every run, as images: extras/model_comparison/confusion_by_run.md

PPE VIDEO (requirement 6)
  A head or helmet box counts only when its centre is inside a tracked person's box, in the top half. Boxes that fit no person are dropped, so nothing draws on the crane.
  The label is held for 1 second before it changes (ppe.debounce_sec). Without it the clip gave 8 violations from detector dropouts.
  imgsz of the helmet engine in both configs is 1088. A 640 value with a 1088 engine would be wrong.
  Result: 32 events (18 IDLE_ALERT, 13 ZONE_ENTRY, 1 PPE_VIOLATION), 26.14 FPS with tracking. The 31 zone and idle events match task1/outputs/events_cctv_cuplock.json exactly.
  The one violation is track 3, 17.47 s to 17.57 s. The worker is bent under the crane hook and the head is hidden. Earlier at 15.0 s the same worker is green with a helmet box on a yellow hat. So this is a short ambiguous event and a likely false alarm.
  (Before the model change the violation ran 16.16 s to 17.57 s. It shortened because the 1088 model sees the helmet longer.)
  Ultralytics CLI also run on the same video with the engine: all 2391 frames at 0.6 ms inference, imgsz 1088, conf 0.35. Boxes on the workers' heads. At 60 s it also draws a helmet box on a round object on the crane, which the tracked pipeline avoids by binding boxes to people.
  CLI output was a 1.9 GB avi, checked at 16.5 s and 60 s, then deleted. It is not in the repo.

OUTPUT FOLDER LAYOUT
  outputs/1_training  2_evaluation  3_failure_cases  4_export  5_benchmark  6_ppe_violation   one folder per spec requirement
  outputs/extras/model_comparison   all runs scored and compared; confusion matrices as images
  outputs/extras/generalization_test   extra check on 10 outside images (2 empty: wm_05 correct, wm_09 small and unusual; wm_08 one false helmet on a head cloth)
  outputs/runs                      training runs: best.pt of the final model and of the others, args and results
  outputs/.scratch                  Ultralytics val output, regenerated, git-ignored
  Task 1's config and README point at outputs/4_export/best.engine.

CORRECTIONS MADE
  - Final model switched from imgsz640_box_dfl to imgsz1088_box_dfl. evaluation, failure cases, export, parity, benchmark, generalization and PPE events were all regenerated for it.
  - The pipeline scored everything at 640. It now reads each run's training imgsz from its args.yaml and uses it for evaluation, export, parity, benchmark, failure cases and generalization.
  - A wrapper that added a second Ultralytics header (best_labeled.engine) made the new engine unloadable ("different version"). The wrapper and the label-engine command are removed. Ultralytics' own best.engine is used everywhere.
  - Failure-case reasons quoted 640 scores (0.84, 0.82, 0.002). Updated to the 1088 scores (0.77, 0.83, 0.013).
  - Run names carried "warmstart". Renamed by image size and loss setting.
  - Exported models sat in runs/. Moved to 4_export/.
  - Confusion matrices were drawn at conf 0.001 with thousands of background false positives. Now conf 0.25.
  - An early command overwrote the final model's curves with the old 3-class run. Restored; DEFAULT_RUN follows the final run.
  - Earlier text said 1088 hurt mAP50-95. That scored D at 640, not at its own size.
  - PPE video drew helmets on the crane. Fixed with the person-box rule, then the 1 second debounce.

OPEN (none block the deliverable)
  1. Jetson or phone 30 FPS is not measured. No device here. README says so.
  2. One seed per run. The preference for imgsz1088_box_dfl rests on about 10 fewer mix-ups on 2,641 objects. A second seed would test it.
  3. The one PPE_VIOLATION is likely a false alarm (hidden head). Not changed.
  4. A bent-over worker may fall outside the top-half head zone. Not changed.
  5. Tasks 1, 3 and 4, the root README, the demo video and docs/system_design.md are outside Task 2.

REPRODUCE
  See README "Run". Commands: download, prepare-weights, train, evaluate, compare, confusion-all, export, parity, benchmark, failures, generalize.
  `hardhat_pipeline.py download` restores the dataset.
