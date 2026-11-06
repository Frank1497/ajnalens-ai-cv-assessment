# Task 3 accuracy report — nameplates v2

20 nameplate photos in `task3/data/nameplates_v2`.
Scored against `labels_ground_truth.csv` with `scoring.py`
(alnum-normalized exact match; both-null counts as correct).
A cross on a comparison board means the cell differs from that CSV.

This run uses OpenCV preprocessing (deskew/crop, then the per-engine recipe).
Preprocessed images: `preprocessed_easyocr/`, `preprocessed_tesseract/`, `preprocessed_paddleocr/`.

## Field accuracy and serial CER (with preprocessing)

| Reader | Overall | Printed fields | equipment_id | serial_number | model | voltage | manufacture_date | Serial CER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EasyOCR | 36.0% | 16/77 | 25.0% | 35.0% | 5.0% | 40.0% | 75.0% | 0.586 |
| Tesseract (psm 4) | 29.0% | 7/77 | 25.0% | 35.0% | 10.0% | 5.0% | 70.0% | 0.771 |
| PaddleOCR (PP-OCRv6 classic) | 87.0% | 64/77 | 90.0% | 85.0% | 80.0% | 85.0% | 95.0% | 0.188 |

## Without preprocessing (raw photos, same engines, same extractor)

Raw run: the original JPEG goes straight to each engine. No crop, deskew, denoise or threshold.

| Reader | Overall raw | Overall preprocessed | Delta (pp) | Printed raw | Printed preprocessed | Serial CER raw | Serial CER preprocessed |
|---|---:|---:|---:|---:|---:|---:|---:|
| EasyOCR | 37.0% | 36.0% | -1.0 | 16/77 | 16/77 | 0.786 | 0.586 |
| Tesseract (psm 4) | 29.0% | 29.0% | +0.0 | 7/77 | 7/77 | 0.938 | 0.771 |
| PaddleOCR (PP-OCRv6 classic) | 69.0% | 87.0% | +18.0 | 46/77 | 64/77 | 0.260 | 0.188 |

Raw OCR cache: `raw_ocr/*_rawimg.json`. Summary: `accuracy_summary_raw.json`.
