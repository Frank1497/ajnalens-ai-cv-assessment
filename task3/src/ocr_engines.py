"""OCR engine runners for Task 3: EasyOCR, Tesseract, classic PaddleOCR.

Each runner reads the engine's own preprocessed image (preprocess.out_path)
and returns {image_name: [{"text", "conf", "box"}, ...]}. Field extraction happens
elsewhere (fields.extract_fields), identically for every engine.

PaddleOCR runs in a separate interpreter (.venv_paddle_cpu) through paddle_worker.py so its
numpy/opencv pins stay out of the main environment.
"""
import json
import os
import shutil
import subprocess
import time

import cv2

from config import PADDLE_WORKER, RAW_DIR, REPO_ROOT, TESS_ROOT
from preprocess import out_path as prep_path


def cached(path, fn, force=False):
    """Run fn() once and keep its JSON result; later runs read the file."""
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    t = time.time()
    data = fn()
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} in {time.time() - t:.1f}s")
    return data


def paddle_python():
    """Isolated Paddle interpreter: $PADDLE_VENV, else .venv_paddle (GPU), else .venv_paddle_cpu."""
    for venv in (os.environ.get("PADDLE_VENV"), ".venv_paddle", ".venv_paddle_cpu"):
        if not venv:
            continue
        py = os.path.join(REPO_ROOT, venv, "bin", "python")
        if os.path.exists(py) and subprocess.run([py, "-c", "import paddle, paddleocr"],
                                                 capture_output=True).returncode == 0:
            return py
    raise SystemExit("No working Paddle venv (.venv_paddle / .venv_paddle_cpu); see requirements_paddle.txt")


def tesseract():
    """pytesseract, pointed at the user-local build in .tools/tesseract when it exists."""
    import pytesseract
    lib = os.path.join(TESS_ROOT, "usr", "lib", "x86_64-linux-gnu")
    if os.path.isdir(lib):  # user-space install from extracted .debs (no sudo on this machine)
        os.environ["LD_LIBRARY_PATH"] = lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["TESSDATA_PREFIX"] = os.path.join(TESS_ROOT, "usr", "share", "tesseract-ocr", "5", "tessdata")
        pytesseract.pytesseract.tesseract_cmd = os.path.join(TESS_ROOT, "usr", "bin", "tesseract")
    elif not shutil.which("tesseract"):
        raise SystemExit("Tesseract binary not found: install tesseract-ocr or build it under .tools/tesseract")
    return pytesseract


def box_of(pts):
    return [float(min(p[0] for p in pts)), float(min(p[1] for p in pts)),
            float(max(p[0] for p in pts)), float(max(p[1] for p in pts))]


def run_easyocr(names):
    import easyocr
    import torch
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return {n: [{"text": t, "conf": float(c), "box": box_of(b)}
                for b, t, c in reader.readtext(prep_path("easyocr", n), detail=1, paragraph=False)]
            for n in names}


def run_tesseract(names, psm):
    pt = tesseract()
    out = {}
    for n in names:
        img = cv2.imread(prep_path("tesseract", n))
        d = pt.image_to_data(img, config=f"--oem 1 --psm {psm}", output_type=pt.Output.DICT)
        words = []
        for i, t in enumerate(d["text"]):
            c = float(d["conf"][i])
            if t.strip() and c >= 0:
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                words.append({"text": t, "conf": c / 100.0, "box": [x, y, x + w, y + h]})
        out[n] = words
    return out


def run_paddle(names):
    py = paddle_python()
    print(f"  PaddleOCR via subprocess: {os.path.relpath(py, REPO_ROOT)} (OCR only)")
    os.makedirs(RAW_DIR, exist_ok=True)
    job_file = os.path.join(RAW_DIR, "paddle_tuned_jobs.json")
    res_file = os.path.join(RAW_DIR, "paddle_tuned_results.json")
    with open(job_file, "w") as f:
        json.dump([{"key": n, "path": prep_path("paddleocr", n)} for n in names], f)
    subprocess.run([py, PADDLE_WORKER, job_file, res_file, "--ocr-only"], check=True)
    with open(res_file) as f:
        return {k: r["ocr"] for k, r in json.load(f).items()}
