"""Paths shared by every Task 3 module. Everything lives under task3/."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "task3", "data", "nameplates_v2")
OUT_DIR = os.path.join(REPO_ROOT, "task3", "outputs", "nameplates_v2")
RAW_DIR = os.path.join(OUT_DIR, "raw_ocr")
CMP_DIR = os.path.join(OUT_DIR, "comparisons")
PADDLE_WORKER = os.path.join(REPO_ROOT, "task3", "src", "paddle_worker.py")
TESS_ROOT = os.path.join(REPO_ROOT, ".tools", "tesseract", "root")
