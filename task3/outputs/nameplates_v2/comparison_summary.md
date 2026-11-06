# Nameplates v2 — comparison against ground truth

Images: `task3/data/nameplates_v2` (8 original keepers + 12 replacements).
Ground truth: `labels_ground_truth.csv` in that folder. Keeper rows are the existing
transcription. The 12 replacement rows were read off the photos for this run.
Scorer: `scoring.py` (alnum-normalized exact match, both-null counts as correct).

| Reader | Overall | Non-null | Serial CER |
|---|---:|---:|---:|
| EasyOCR | 36.0% | 16/77 (20.8%) | 0.586 |
| Tesseract (psm 4) | 29.0% | 7/77 (9.1%) | 0.771 |
| PaddleOCR (PP-OCRv6 classic) | 87.0% | 64/77 (83.1%) | 0.188 |
