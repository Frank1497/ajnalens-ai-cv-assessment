# Data sources

No assessment data was supplied. Each task uses public or self-collected material. Nothing here is confidential company data. Files over GitHub's 100 MB limit stay outside the repo and are named in the task README.

## Task 1

Real workshop CCTV, first 120 seconds of a Cuplock recording: 2560×1440, about 19.93 fps, 2391 frames. The file is outside this repo at `raw_footage/task1_cctv/Cuplock-1_CCTV_CCTV_20250823100919_20250823101719_919368_first120s.mp4`. `configs/task1_zone_cctv_cuplock.json` points `video.source` at that absolute path. It is not redistributed.

## Task 2

Roboflow Universe `joseph-nelson/hard-hat-workers`, version 10 (`raw_AllClasses`), Public Domain. YOLO labels, classes `head`, `helmet`, `person`. Downloaded into `task2/data/` with the Roboflow split: 4916 train, 1413 valid, 706 test images. The final model drops `person` and trains on `head` and `helmet` only. Provenance notes from the download are `task2/data/README.dataset.txt` and `task2/data/README.roboflow.txt`.

The ten generalization photos are Wikimedia Commons images listed with page and licence in `task2/outputs/generalization_test/inputs/SOURCES.json`.

## Task 3

20 motor nameplate photos from the Hugging Face dataset [`sergiudanstan/motor-nameplate`](https://huggingface.co/datasets/sergiudanstan/motor-nameplate) (MIT). The photos live in `task3/data/nameplates_v2/`. Ground truth was read off the plates into `labels_ground_truth.csv` in that folder. Twelve of the original twenty frames were replaced from the same photo set; the map is `sources.json`. The companion dataset `sergiudanstan/motor-nameplate-labels` is an automatic OCR table and is not the answer key.

## Task 4

Three pump operation manuals stored in `task4/app/manuals/`:

- `pump_manual_centrifugal.pdf`
- `pump_manual_desmi_modular.pdf`
- `pump_manual_finishthompson_db.pdf`

They are chunked into the FAISS index under `task4/app/index/`. The five test questions are `task4/questions.json`.
