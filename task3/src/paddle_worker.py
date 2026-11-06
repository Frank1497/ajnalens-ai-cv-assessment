"""
PaddleOCR worker -- runs inside the ISOLATED .venv_paddle_cpu (paddlepaddle + paddleocr 3.x),
never imported by the main .venv (keeps torch/ultralytics untouched).

    .venv_paddle_cpu/bin/python task3/src/paddle_worker.py jobs.json results.json

jobs.json:    [{"key": "nameplate_03.jpg", "path": "/abs/img.png"}, ...]
results.json: {key: {"ocr": [{text, conf, box:[x0,y0,x1,y1]}]}}

OCR:    PaddleOCR 3.x default OCR pipeline (paddleocr 3.7.0 ships PP-OCRv6_medium det+rec; the
        model names are printed in the "Creating model" log lines), doc-orientation/unwarp off
        (images are already deskewed by the shared OpenCV step).
"""
import os
import sys
import json

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2  # noqa: E402
import paddle  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402


def as_dict(r):
    j = r.json if hasattr(r, "json") else r
    return j.get("res", j) if isinstance(j, dict) else j


def main(job_file, res_file):
    jobs = json.load(open(job_file))
    device = "gpu:0" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
    print(f"[paddle_worker] paddle {paddle.__version__} on {device}, {len(jobs)} jobs", flush=True)
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, device=device)
    out = {}
    for job in jobs:
        img = cv2.imread(job["path"], cv2.IMREAD_COLOR)
        recs = []
        for r in ocr.predict(img):
            d = as_dict(r)
            for t, s, b in zip(d["rec_texts"], d["rec_scores"], d["rec_boxes"]):
                b = [float(x) for x in b]
                recs.append({"text": t, "conf": float(s), "box": b[:4]})
        out[job["key"]] = {"ocr": recs}
        print(f"[paddle_worker] {job['key']}: {len(recs)} lines", flush=True)
    json.dump(out, open(res_file, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
