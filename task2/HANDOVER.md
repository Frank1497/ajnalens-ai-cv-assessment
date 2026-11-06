TASK 2 HANDOVER — Hard-Hat Detector
Status: complete against the 6 requirements in the assessment PDF. Final model: warmstart_e30_2cls_640_fulldata, deployed as TensorRT FP16.
Every number below was read from a file in task2/outputs/ on 6 Nov 2026 IST. README.md carries the full tables and the written answers.

FINAL MODEL
  task2/outputs/runs/warmstart_e30_2cls_640_fulldata/weights/
    best.pt               PyTorch FP32, 6.2 MB
    best.onnx             ONNX FP32, 12.3 MB, opset 17, static 1x3x640x640
    best_fp16.onnx        ONNX FP16, 6.2 MB
    best.engine           raw TensorRT plan, 7.8 MB
    best_labeled.engine   same plan behind a 246-byte Ultralytics header (class names). Used by the PPE video and by the benchmark.
  2 classes (head, helmet), 30 epochs, batch 32, imgsz 640, box 10, dfl 2.0, warm-start from keremberke hard-hat weights.

REQUIREMENT CHECKLIST
  1 fine-tune, hyperparameters and why     README "Hyperparameters and why"; results.csv (30 epochs)
  2 val metrics, confusion matrix, PR      outputs/2_evaluation/ (metrics.md, val_* and test_* curves, confusion matrices at conf 0.25)
  3 five failure cases                     outputs/3_failure_cases/ (5 images, failure_cases.json, README.md); images 1, 2, 3 checked by eye
  4 ONNX plus FP16 or TensorRT             three formats (pytorch, onnx_fp16_cuda, tensorrt_fp16_cuda); 4_export/export_parity_test.json
  5 benchmark on the same 100 images       outputs/5_benchmark/benchmark.md (pytorch, onnx_fp16_cuda, tensorrt_fp16_cuda)
  6 helmet / no helmet + PPE_VIOLATION     task2/src/zone_monitor.py; 6_ppe_violation/events_cctv_cuplock_ppe.json has 1 PPE_VIOLATION (1 s debounce, checked by eye)
  README answers (class imbalance, 30 FPS edge)   task2/README.md

RESULTS
Test split, 706 images, 2641 instances (2_evaluation/metrics.md): P 0.941, R 0.940, mAP50 0.973, mAP50-95 0.674. Head 0.667, helmet 0.680.
Valid split, 1413 images: mAP50 0.970, mAP50-95 0.671.

Exports on the full test split (4_export/export_parity_test.json)
  pytorch              best.pt              mAP50 0.973  mAP50-95 0.6732
  onnx_fp16_cuda       best_fp16.onnx       mAP50 0.973  mAP50-95 0.6734
  tensorrt_fp16_cuda   best_labeled.engine  mAP50 0.973  mAP50-95 0.6735
  The three agree within 0.0003.

Benchmark, same 100 sorted test images, batch 1, imgsz 640, RTX 5070 Ti (5_benchmark/benchmark.md)
  pytorch              3.36 ms   297.2 FPS   mAP50 0.975
  onnx_fp16_cuda       2.13 ms   470.1 FPS   mAP50 0.974   (CUDAExecutionProvider confirmed active)
  tensorrt_fp16_cuda   1.27 ms   787.3 FPS   mAP50 0.973
  TensorRT is 2.6x PyTorch and 1.7x ONNX end to end. Network forward alone: 0.32 ms vs 2.37 and 1.11 ms.

WHY THE ENGINE, AND WHY THIS TRAINING RUN (extras/model_comparison/model_selection.md, model_selection_latency.json)
  Test split, same labels for every run. default = box 7.5 / dfl 1.5, box+dfl = box 10 / dfl 2.0:
    A imgsz=640 default, COCO start                 640   mAP50-95 0.670
    B imgsz=640 default                             640   0.671
    C imgsz=1088 box+dfl, undersampled              640 0.662 / 1088 0.672
    D imgsz=1088 box+dfl                            640 0.664 / 1088 0.676
    E imgsz=640 box+dfl (FINAL)                     640   0.674
  Spread is 0.006 from single-seed runs. Accuracy does not separate the models.
  E was chosen on cost. D at 1088 scores 0.002 higher but takes 4.38 ms vs 3.45 ms in PyTorch on this GPU (2.9x the pixels).
  The engine was chosen because accuracy is unchanged, it is the fastest, and it is 7.8 MB. Its cost is portability: rebuild per GPU, keep ONNX as the fallback.

CONFUSION MATRIX (test, conf 0.25, test_confusion_matrix.png)
  head 689 right, 17 called helmet, 20 missed. helmet 1843 right, 7 called head, 65 missed. Background false positives: 72 head, 139 helmet.

CONFUSION MATRIX OF EVERY RUN (extras/model_comparison/confusion_by_run.md and confusion_by_run/ALL_RUNS.png; test, conf 0.25)
  bare head called helmet (costly): A 14, B 12, C 10 / 12, D 14 / 12, E 17.  Missed helmets: E 65, D@1088 71, others 72 to 107.
  E is highest on the costly error, by 3 to 7 boxes of 726. One seed per run, so not significant. D@1088 matches or beats E on the safety cells. E was chosen on cost.

OUTPUT FOLDER LAYOUT (reorganised 6 Nov 2026 IST; every command re-run afterwards and writes here)
  outputs/1_training  2_evaluation  3_failure_cases  4_export  5_benchmark  6_ppe_violation   one folder per spec requirement
  outputs/extras/model_comparison   runs A to E scored and compared (justifies the final model)
  outputs/extras/generalization_test   extra check on 10 outside images
  outputs/runs   training runs and weights (task1 config points at runs/warmstart_e30_2cls_640_fulldata/weights/best_labeled.engine)
  outputs/.scratch   Ultralytics val output, regenerated, git-ignored
  Benchmark reduced to the three formats on request and re-run: TensorRT 787.3 FPS. Accuracy values reproduced exactly.

CORRECTIONS MADE THIS PASS
  - generalization_test/detections.json came from an older model and conf 0.35. 8 of 10 images differed from the final model. Regenerated at conf 0.25. Now 2 empty images, not 3.
  - confusion_by_run.json had no mAP fields. Added from model_selection.json.
  - I overwrote the stored curves and confusion matrices with the old 3-class run by accident (default run was warmstart_e25). Restored from the final run. DEFAULT_RUN now points at the final run.
  - Confusion matrices were drawn at conf 0.001 and showed 12,648 background false positives on val. They are now drawn at conf 0.25.
  - Earlier text said 1088 hurt mAP50-95. That scored D at 640, not at its own size. Corrected in README and model_selection.md.
  - Benchmark now includes TensorRT. The earlier 544.7 FPS row came from an ad hoc script and was never saved. The saved 787.3 FPS is measured by `hardhat_pipeline.py benchmark`.
  - The earlier "TensorRT serialization version mismatch" was the Ultralytics metadata header, not a version problem.
  - PPE video drew helmets on the crane because any box merely touching a person box counted. Now a box binds to a person only by centre inside the top half of the person box; unbound boxes are dropped. Rule covered by 7 assertion cases; frames at 20 s, 60 s, 100 s viewed. Video regenerated at 15:44 IST.
  - Failure-case images had been drawn with labels multiplied by 640 on an unresized frame. Redrawn in original pixel coordinates.

CLEANUP DONE (5.7 GB to 737 MB in task2/)
  Removed: .npy image caches (6,329 files), 1.9 GB CLI avi, empty bench/eval dirs, duplicate best.fp16.onnx, warmstart_e25 best.onnx, old onnx_parity_test.json, failure_match_cache.json, benchmark_100 symlink dir, empty notebooks/, __pycache__, repo-root runs/.
  Kept: every training run folder (args.yaml, results.csv, plots, weights) and every eval folder, so the selection table can be reproduced and explained.
  task2/.gitignore added: data splits, videos, last.pt, caches are ignored. The final weights and exports are un-ignored (149 MB staged, no file over 50 MB).
  Weights and training logs of runs A to D stay on disk and are ignored by git.

OPEN (none block the deliverable)
  1. Jetson or phone 30 FPS is not measured. README says so.
  2. wm_08 in the generalization set has one false helmet at 0.295 on a head cloth. Not changed.
  3. Bent-over workers may fall outside the top-half head zone in the PPE video. Not changed.
  4. Nothing is committed yet. Demo video, root README and docs/system_design.md belong to other tasks.

CLOSED THIS PASS
  - PPE_VIOLATION flicker: 1 s label debounce, 8 violations to 1, the 1 left verified by eye.
  - export JSON: now lists pytorch, onnx_fp32, onnx_fp16, tensorrt_fp16_raw, tensorrt_fp16_labeled. best_labeled.engine is built by `label-engine` and by `export`, tested byte-identical to the file the benchmark used.
  - model_selection_latency.json is written by `compare`, no ad hoc script.
  - failure images 1 to 5 checked by eye; numbers in all JSONs cross-checked.

REPRODUCE
  See README "Run". `hardhat_pipeline.py download` restores the dataset. `compare` and `parity` are new commands added this pass.
