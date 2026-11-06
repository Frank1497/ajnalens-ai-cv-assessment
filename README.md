# AjnaLens AI/ML & CV Engineer — Practical Assessment

Four tasks, one folder each. Every task has a `README.md` (what it does, how to run it, results) and a `HANDOVER.md` (working state and open items).

| Task | Topic | Start here |
|---|---|---|
| 1 | Object detection and zone monitoring: YOLO person detection, BoT-SORT tracking, zone-entry and idle rules on video | [task1/README.md](task1/README.md) |
| 2 | Hard-hat detector: YOLOv8n fine-tuned on Hard Hat Workers, evaluated and compared across runs | [task2/README.md](task2/README.md) |
| 3 | OCR with structured extraction: motor nameplate photos to five JSON fields | [task3/README.md](task3/README.md) |
| 4 | Maintenance assistant: RAG over pump manuals with tool calling, served by FastAPI | [task4/README.md](task4/README.md) |

Numbers for Tasks 1, 2 and 4 live in their own READMEs. They are not repeated here.

## Task 3 in brief

Pipeline: label photo, OpenCV preprocess, OCR engine, field extraction, JSON and accuracy report. 20 public nameplate photos, ground truth written by hand, three engines compared on identical input.

| Reader | Overall (100 cells) | Printed fields | Serial CER |
|---|---:|---:|---:|
| EasyOCR | 36% | 16/77 | 0.586 |
| Tesseract (psm 4) | 29% | 7/77 | 0.771 |
| PaddleOCR | 87% | 64/77 | 0.188 |

Preprocessing raises PaddleOCR from 69% to 87%. It leaves EasyOCR (37% to 36%) and Tesseract (29% to 29%) flat on field accuracy, though both improve on serial CER. The 87% is in-sample: 20 photos, no held-out split. Details, per-image boards and the 0/O answer are in [task3/README.md](task3/README.md).

```bash
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
python3 task3/src/run_pipeline.py
```

Task 3 needs two things outside the main environment: a Tesseract binary (user-local build in `.tools/tesseract/`) and a separate Paddle environment `.venv_paddle_cpu/` (`task3/requirements_paddle.txt`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python==4.10.0.84
```

Python 3.10 or later. A GPU is optional. Each task README lists any extra step it needs.

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
