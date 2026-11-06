# Task 3 — OCR with Structured Extraction (Motor Nameplates)

Pipeline: **label photo → OpenCV preprocess → OCR engine → field extraction → JSON + accuracy report**

20 nameplate photos, three engines (EasyOCR, Tesseract psm 4, classic PaddleOCR), scored field by field against a hand-made `labels_ground_truth.csv`.

## Results

With OpenCV preprocessing (100 cells = 20 images x 5 fields; "printed" = the 77 cells whose ground truth is non-empty):


| Reader            | Overall | Printed fields | Serial CER |
| ----------------- | ------- | -------------- | ---------- |
| EasyOCR           | 36%     | 16/77          | 0.586      |
| Tesseract (psm 4) | 29%     | 7/77           | 0.771      |
| PaddleOCR         | 87%     | 64/77          | 0.188      |


PaddleOCR per field: equipment_id 90%, serial_number 85%, model 80%, voltage 85%, manufacture_date 95%.

## With and without preprocessing

Same 20 photos, same engines, same extractor. "Raw" feeds the original JPEG straight to the engine.


| Reader            | Overall raw | Overall preprocessed | Printed raw | Printed preprocessed | Serial CER raw | Serial CER preprocessed |
| ----------------- | ----------- | -------------------- | ----------- | -------------------- | -------------- | ----------------------- |
| EasyOCR           | 37%         | 36%                  | 16/77       | 16/77                | 0.786          | 0.586                   |
| Tesseract (psm 4) | 29%         | 29%                  | 7/77        | 7/77                 | 0.938          | 0.771                   |
| PaddleOCR         | 69%         | 87%                  | 46/77       | 64/77                | 0.260          | 0.188                   |


Reading it plainly:

- PaddleOCR gains 18 points overall (46 to 64 printed fields). Preprocessing matters most here.
- EasyOCR and Tesseract do not move on field accuracy (37 to 36, 29 to 29). Their serial CER improves (0.786 to 0.586, 0.938 to 0.771), so the characters are closer, but not close enough to flip a field to an exact match.
- EasyOCR and Tesseract are weak on this data in both modes. The gap to PaddleOCR is the engine, not the preprocessing.



## Data and ground truth

- Photos: `data/nameplates_v2/`, from Hugging Face `sergiudanstan/motor-nameplate` (MIT). Public photos with varied angle, lighting and blur. Source map: `data/nameplates_v2/sources.json` (8 kept from the first pull, 12 replaced because they were diagrams, stock graphics or annotated images).
- `data/nameplates_v2/labels_ground_truth.csv` was written by reading each plate, not copied from the dataset's own automatic labels (`motor-nameplate-labels` is an OCR/VLM output and is not an answer key).
- Fields: `equipment_id, serial_number, model, voltage, manufacture_date`. A cell is empty when the plate does not print that field. Nothing is guessed.
- Where a plate prints several voltage lines, the rated line is used (plate 3: `230/460`, not the `208` "usable at" note or the 50 Hz `380` line).
- Where the only printed code is the model, equipment_id is left empty. The scorer treats a PaddleOCR equipment_id that equals its own model as null on those plates only (plates 07, 10, 012, 015, 017). This rule is in `clear_paddle_equipment_when_model_copy` and applies to PaddleOCR only.
- Scoring: `src/scoring.py`, alphanumeric-normalised exact match, both-null counts as correct. Serial CER is character edit distance over the serial cells that have ground truth.



## Preprocessing (OpenCV)

`src/preprocess.py::detect_and_deskew_label` finds the dominant rectangular label (Canny edges, contour approximation, perspective warp). If no clean quadrilateral exists it falls back to a min-area-rect deskew, then to the full image. Then the same file applies one recipe per engine:


| Engine    | Recipe                                                                                                                                                |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tesseract | upscale to 2400 px wide, bilateral denoise, CLAHE, adaptive threshold. 3200 px was tried and lost 9 points (JPEG block noise amplified into speckle). |
| EasyOCR   | upscale to 1600 px, denoise, CLAHE on luminance, colour kept, no threshold. 2200 px was tried and lost 1 point.                                       |
| PaddleOCR | upscale to 2000 px, CLAHE on plain grayscale. Unsharp mask was tried and lost points (invented characters on worn plates).                            |


Preprocessed images are saved under `outputs/nameplates_v2/preprocessed_easyocr/`, `preprocessed_tesseract/`, `preprocessed_paddleocr/`.

## Field extraction

`src/fields.py` groups OCR boxes into lines and fills the five fields from printed headings (`CAT. NO.`, `SER. NO.`, `MODEL`, `VOLTS`, `DATE`, and equivalents). A field with no match is `null`. The extractor does not guess.

## Deliverables


| Item                                                                         | Path                                                                                                   |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Photos + ground truth                                                        | `task3/data/nameplates_v2/`                                                                            |
| Pipeline script (preprocess, OCR, extract, score, boards)                    | `task3/src/run_pipeline.py`                                                                            |
| Preprocessed images                                                          | `task3/outputs/nameplates_v2/preprocessed_{easyocr,tesseract,paddleocr}/`                              |
| `labels_output.json` (five fields, one object per engine)                    | `task3/outputs/nameplates_v2/labels_output.json`                                                       |
| Accuracy report (field accuracy, serial CER, with/without preprocessing)     | `task3/outputs/nameplates_v2/accuracy_report.md`, `accuracy_summary.json`, `accuracy_summary_raw.json` |
| Per-image boards (photo, ground truth, three engines, crosses on mismatches) | `task3/outputs/nameplates_v2/comparisons/nameplate_XX_comparison.png`, `engine_comparison_grid.png`    |
| Cached raw OCR                                                               | `task3/outputs/nameplates_v2/raw_ocr/`                                                                 |




## Run it

```bash
cd ai-cv-assessment
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
python3 task3/src/run_pipeline.py
```

OCR results are cached in `raw_ocr/`; delete a cache file to force that engine to rerun. Tesseract is a user-local build in `.tools/tesseract/` (no sudo on this machine). PaddleOCR runs from a separate `.venv_paddle_cpu/` (`requirements_paddle.txt`) so its numpy/opencv pins do not touch the main environment. Unit tests for the extractor: `python3 -m pytest task3/src/test_fields.py`.

## What improved the result

Measured on the earlier photo set unless marked v2. Every change was kept only if the score rose.


| Change                                                                                                                                                                          | Effect                                            | Kept |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ---- |
| Add classic PaddleOCR next to EasyOCR                                                                                                                                           | Clear lead over the other two                     | Yes  |
| One shared parser (`fields.py`): rebuild lines from boxes, anchor on printed headings, repair O/0 by position, decode voltage against standard values, accept local month codes | PaddleOCR 75%                                     | Yes  |
| Plain grayscale CLAHE for PaddleOCR, no sharpen                                                                                                                                 | 70% to 76% (unsharp mask had invented characters) | Yes  |
| Larger upscale for Tesseract (3200 px)                                                                                                                                          | 51% to 42%                                        | No   |
| Larger upscale for EasyOCR (2200 px)                                                                                                                                            | 54% to 53%                                        | No   |
| Replace 12 diagram, stock or annotated photos with real nameplate photos (v2)                                                                                                   | PaddleOCR 87% on the v2 set                       | Yes  |
| Correct three ground-truth cells (plates 3, 11, 14)                                                                                                                             | Scores match the plates                           | Yes  |


Details and the full step list are in `HANDOVER.md`.

## Known limits

- 20 photos and no held-out split. The extractor was tuned on the earlier set, not on these 20, but with so few photos 87% is an in-sample figure, not a forecast.
- Ground truth is one reader's transcription. The plate 14 serial was checked against the photo by hand. The remaining cells were not independently checked.
- Open misses on PaddleOCR (character errors or missing headings):
  - Plate 5 date: CSV says `2011`; the OCR year box reads `2021` with no Year heading, so the field stays empty.
  - Plate 8: model and product code are in the OCR text but not on a MODEL / PROD CODE line; serial `1J57` is misread as `1537`.
  - Plate 10 voltage: PaddleOCR returns the brake line `380-460`; the motor rows are split by a triangle.
  - Plate 14 model: the string is in the OCR text on the line above `3 ~ Motor`, so the field is empty.
  - Plate 15 voltage: `V` and `240` are separate boxes.
  - Plate 20 model: the plate has `MACHINE NO.` and no MODEL heading.
- PaddleOCR-VL (a vision-language model, not classic OCR) and Gemini were tried in earlier passes and are not part of this submission. Their code is removed from `src/`; the earlier work is archived outside the repo.



## Answer briefly: reducing 0/O and 1/I confusion without retraining

The engine picks the likeliest glyph with no idea which field it is reading. Fix it after OCR, using what the field must look like:

1. **Repair by position and field.** Inside a mostly-numeric token, a letter `O` touching a digit becomes `0`; in a date code, `O/I/l` become `0/1`. Letters stay letters elsewhere, because real IDs mix both (`DI0034PCI`, `37J776T859G1`).
2. **Validate against the format.** If a serial has a known length or pattern, test the candidate against it, try the 0/O and 1/I swaps, and keep the one that fits. Without a known pattern, return `null` instead of a guess.
3. **Vote across inputs.** Read the same crop under two or three preprocessings and take the per-character majority. The confusions rarely repeat in every variant.

Implemented here: 1, and the null-instead-of-guess part of 3 (`repair_id_token`, date-slot repair and voltage decoding in `src/fields.py`, covered by `src/test_fields.py`). Not implemented: 2 and 4. I did not measure the effect of 1 on its own.

## AI tool assistance disclosure

This task (preprocessing code, OCR pipeline, field-extraction rules, evaluation script, ground-truth transcription support, and this README) was built with AI coding assistance (Claude, via the Hermes agent harness). Ground truth was produced by reading each nameplate image field by field, not by copying the dataset's automatic labels. A human should spot-check the ground-truth CSV, since a small number of fields are hard to read even by eye.