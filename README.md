# AjnaLens AI/ML & CV Engineer — Practical Assessment

Four tasks, one folder each. Every task has a `README.md` (what it does, how to run it, results) and a `HANDOVER.md` (working state and open items).

| Task | Topic | Start here |
|---|---|---|
| 1 | Object detection and zone monitoring: YOLO person detection, BoT-SORT tracking, zone-entry and idle rules on video | [task1/README.md](task1/README.md) |
| 2 | Hard-hat detector: YOLOv8n fine-tuned on Hard Hat Workers, evaluated and compared across runs | [task2/README.md](task2/README.md) |
| 3 | OCR with structured extraction: motor nameplate photos to five JSON fields | [task3/README.md](task3/README.md) |
| 4 | Maintenance assistant: RAG over pump manuals with tool calling, served by FastAPI | [task4/README.md](task4/README.md) |

Numbers, run commands and answers to the "Answer briefly" questions live in each task's own README. They are not repeated here.

## Hardware used

All runs and timings: NVIDIA GeForce RTX 5070 Ti (16 GB), Intel i5-13600K, 30 GB RAM, Ubuntu 26.04, Python 3.11.16. A GPU is optional for Tasks 3 and 4 (PaddleOCR runs on CPU, Gemini runs remotely). Tasks 1 and 2 were only timed on this GPU.

## Results at a glance

Every number below is copied from the task README that owns it. Open that README for the file the number was read from.

| Task | Result | Detail |
|---|---|---|
| 1 | 2391 frames of 2560x1440 CCTV tracked. 31 events (13 ZONE_ENTRY, 18 IDLE_ALERT), none left open. 29.08 FPS in the earlier verified run, 18.24 FPS in the latest run with the live window open (`task1/outputs/run_log.txt`). The cause of the gap is not measured. | [task1/README.md](task1/README.md) |
| 2 | Test split, 706 images: mAP50 0.973, mAP50-95 0.676. TensorRT FP16 engine 2.15 ms (465 FPS), PyTorch 4.31 ms (232 FPS), 100 images, batch 1. PPE run adds 1 PPE_VIOLATION to the 31 Task 1 events. | [task2/README.md](task2/README.md) |
| 3 | PaddleOCR 87% overall (64/77 printed fields, serial CER 0.188) against EasyOCR 36% and Tesseract 29%. In-sample, 20 photos. | [task3/README.md](task3/README.md) |
| 4 | 5 of 5 test questions answered through `POST /ask` with citations. Q1 and Q4 call `get_equipment_status`. Q3 correctly declines. | [task4/README.md](task4/README.md) |

## Answers to the "Answer briefly" questions

| Task | Question | Where |
|---|---|---|
| 1 | Why the foot point instead of the box centre? | [task1/README.md](task1/README.md#why-use-the-foot-point-instead-of-the-box-centre) |
| 1 | Stopping an ID swap after a 2 second occlusion | [task1/README.md](task1/README.md#how-would-you-stop-one-persons-id-swapping-after-they-are-hidden-for-2-seconds) |
| 2 | Many more helmet boxes than head boxes; 30 FPS on a Jetson Orin Nano | [task2/README.md](task2/README.md#answers) |
| 3 | Reducing 0/O and 1/I confusion without retraining | [task3/README.md](task3/README.md#answer-briefly-reducing-0o-and-1i-confusion-without-retraining) |
| 4 | Measuring retrieval quality; camera-on-a-part extension | [task4/README.md](task4/README.md#answers-to-required-questions) |

## Assumptions

- No data was supplied. Task 1 uses a real 120 s workshop CCTV clip kept outside the repo (over 100 MB). Tasks 2, 3 and 4 use public datasets and manuals, listed with licences in [data/README.md](data/README.md).
- Task 1 events are start/stop intervals, not one record per frame. Every field the brief asks for is present as a start and stop pair.
- Task 2 trains two classes, `head` and `helmet`. The `person` class in the dataset was dropped. The checkpoint was chosen on the valid split, never on test.
- Task 3 ground truth was read by hand from the plates. A field the plate does not print is `null`, never a guess.
- Task 4 uses Google Gemini for the LLM and tool calling, not Anthropic. Equipment status is mock data held in `task4/app/tools.py`.
- Large videos are on Google Drive and linked from `.txt` files in the folders that would hold them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python==4.10.0.84
```

Python 3.11 (tested on 3.11.16; the pinned numpy 2.4.6 needs 3.11 or later). Each task README lists any extra step it needs.

`easyocr` and `roboflow` depend on `opencv-python-headless`. That wheel overwrites `cv2` and drops `imshow`. The uninstall above puts the GUI build back. `uv pip install -r requirements.txt` skips the headless package because `uv.toml` excludes it, so the extra uninstall is only needed for pip.

## Layout

```
configs/   Task 1 zone configs
data/      README listing every data source (no data was supplied; each task uses public or self-collected material)
docs/      system_design.md, architecture diagrams
demo/      demo video or demo_link.txt
task1/ ... task4/
requirements.txt
```

## AI tool assistance disclosure

Built with AI coding assistance: Claude, via the Hermes agent harness (code,
tuning, debugging, and documentation across all four tasks). Each task's own
README has a per-task disclosure with specifics on what was AI-assisted for
that task. Every claim in this repo's documentation was checked against real
command output, re-runs of the actual pipelines, or direct inspection of the
source data before being written down — not assumed from general knowledge.

## What I would improve with more time

- **Task 1**: a hysteresis-band zone polygon (separate inner/outer boundary)
  to stop re-fires when a person's foot point straddles the zone edge; a
  pose/keypoint model to fix the foot-point heuristic's failure on seated or
  occluded-leg subjects.
- **Task 2**: a second training seed to confirm the 1088-vs-640 model
  preference statistically (current case rests on ~10 fewer mix-ups out of
  2,641 objects, single-seed); oversampling head-containing images to test
  whether it closes the head-vs-helmet class-imbalance gap; real Jetson/edge
  hardware to measure the 30 FPS claim instead of leaving it unverified.
- **Task 3**: a second independent human pass over the ground-truth CSV (one
  reader's transcription only, no cross-check beyond one hand-verified
  serial); a held-out split once more than 20 photos are available, since
  87% is currently in-sample.
- **Task 4**: a proper precision@k/recall@k evaluation set instead of manual
  citation spot-checks; tightening citation lists to only chunks the final
  answer text actually references (sibling-page expansion currently pads
  citations with same-page neighbors that may be unused); a second LLM
  backend (Claude or a self-hosted Ollama model) wired in as a fallback once
  a usable key/working endpoint is available, so a single provider's quota or
  outage does not block the assistant.
- **Cross-task**: a shared `docker-compose` setup (Triton/DeepStream per the
  deployment plan in `docs/system_design.md`) so all four tasks could be
  demoed as live services rather than one-off scripts.
