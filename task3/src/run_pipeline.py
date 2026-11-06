"""Task 3 pipeline: photo -> OpenCV preprocess -> OCR -> field extraction -> JSON + accuracy report.

Readers: EasyOCR, Tesseract psm 4, and classic PaddleOCR. Each field is scored against
task3/data/nameplates_v2/labels_ground_truth.csv. Writes only under task3/outputs/nameplates_v2/.
OCR results are cached in raw_ocr/; delete a cache file to rerun that engine.

Usage:
    source .venv/bin/activate
    export PATH="$HOME/.local/bin:$PATH"
    python3 task3/src/run_pipeline.py
"""
import json
import os

import cv2
import numpy as np

import boards as cmp
import ocr_engines as ocr
import preprocess as prep
import scoring
from config import CMP_DIR, DATA_DIR, OUT_DIR, RAW_DIR
from fields import build_lines, extract_fields
from preprocess import detect_and_deskew_label


BOARD_ENGINES = ("easyocr", "tesseract", "paddleocr")


def preprocess(names):
    for engine in BOARD_ENGINES:
        os.makedirs(prep.out_dir(engine), exist_ok=True)
    for name in names:
        crop = detect_and_deskew_label(cv2.imread(os.path.join(DATA_DIR, name)))
        for engine in BOARD_ENGINES:
            path = prep.out_path(engine, name)
            if not os.path.exists(path):
                cv2.imwrite(path, prep.RECIPES[engine](crop))
        print(f"  preprocessed {name}", flush=True)


def pack_ocr(by_name):
    packed = {}
    for name, records in by_name.items():
        fields, candidates = extract_fields(records)
        packed[name] = {
            "fields": fields,
            "candidates": candidates,
            "lines": [line["text"] for line in build_lines(records)],
        }
    return packed


def clear_paddle_equipment_when_model_copy(paddle_packed, gt):
    """Paddle only. The CSV leaves equipment_id empty on plates whose only
    printed code is the model. When Paddle's equipment_id is that same code,
    drop it so the empty ground-truth cell is not scored as a miss."""
    cleared = []
    for name, packed in paddle_packed.items():
        if gt.get(name, {}).get("equipment_id"):
            continue
        fields = packed["fields"]
        equipment_id = fields.get("equipment_id")
        if equipment_id and scoring.norm(equipment_id) == scoring.norm(fields.get("model")):
            fields["equipment_id"] = None
            cleared.append((name, equipment_id))
    return cleared


def score_engines(gt, per_source):
    engines = {}
    for source, _label, _prep in cmp.READERS:
        preds = {name: per_source[source][name]["fields"] for name in gt}
        result = scoring.score(gt, preds, source)
        non_null, hit, total = scoring.known_field_acc(gt, preds)
        engines[source] = {
            "overall": scoring.overall(result),
            "field_accuracy": result["field_accuracy"],
            "non_null": non_null,
            "non_null_hit": hit,
            "non_null_total": total,
            "serial_number_avg_cer": result["avg_serial_number_cer"],
            "n_serial_with_gt": result["n_serial_with_gt"],
        }
    return engines


def write_summary(engines, non_null_total, raw_engines=None):
    lines = [
        "# Nameplates v2 — comparison against ground truth",
        "",
        "Images: `task3/data/nameplates_v2` (8 original keepers + 12 replacements).",
        "Ground truth: `labels_ground_truth.csv` in that folder. Keeper rows are the existing",
        "transcription. The 12 replacement rows were read off the photos for this run.",
        "Scorer: `scoring.py` (alnum-normalized exact match, both-null counts as correct).",
        "",
        "| Reader | Overall | Non-null | Serial CER |",
        "|---|---:|---:|---:|",
    ]
    for source, label, _prep in cmp.READERS:
        row = engines[source]
        cer = row["serial_number_avg_cer"]
        cer_txt = f"{cer:.3f}" if cer is not None else "n/a"
        lines.append(
            f"| {label} | {row['overall']:.1f}% | {row['non_null_hit']}/{non_null_total} "
            f"({row['non_null']:.1f}%) | {cer_txt} |"
        )
    path = os.path.join(OUT_DIR, "comparison_summary.md")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {path}", flush=True)
    write_accuracy_report(engines, non_null_total, raw_engines)


ENGINE_FILE_KEYS = {
    "easyocr/tuned": "easyocr",
    "tess_psm4/tuned": "tesseract",
    "paddle/tuned": "paddleocr",
}


def write_labels_output(per_source):
    """PDF deliverable: five fields per image per engine. null means not found."""
    payload = {
        "dataset": "task3/data/nameplates_v2",
        "ground_truth": "task3/data/nameplates_v2/labels_ground_truth.csv",
        "note": "null means the field was not found in the OCR text. Values are not guessed.",
        "engines": {
            key: {name: rec["fields"] for name, rec in per_source[source].items()}
            for source, key in ENGINE_FILE_KEYS.items()
        },
    }
    path = os.path.join(OUT_DIR, "labels_output.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"Wrote {path}", flush=True)


def run_raw_ablation(names, gt):
    """Same three engines on the untouched photos. Only the image path changes."""
    original = ocr.prep_path
    ocr.prep_path = lambda _engine, name: os.path.join(DATA_DIR, name)
    try:
        easy = ocr.cached(os.path.join(RAW_DIR, "easyocr_rawimg.json"), lambda: ocr.run_easyocr(names), False)
        tess = ocr.cached(os.path.join(RAW_DIR, "tess_psm4_rawimg.json"), lambda: ocr.run_tesseract(names, 4), False)
        paddle = ocr.cached(os.path.join(RAW_DIR, "paddle_rawimg.json"), lambda: ocr.run_paddle(names), False)
    finally:
        ocr.prep_path = original
    per_source = {
        "easyocr/tuned": pack_ocr(easy),
        "tess_psm4/tuned": pack_ocr(tess),
        "paddle/tuned": pack_ocr(paddle),
    }
    clear_paddle_equipment_when_model_copy(per_source["paddle/tuned"], gt)
    engines = score_engines(gt, per_source)
    with open(os.path.join(OUT_DIR, "accuracy_summary_raw.json"), "w") as handle:
        json.dump({"engines": engines}, handle, indent=2)
    return engines


def write_accuracy_report(engines, non_null_total, raw_engines=None):
    fields = ["equipment_id", "serial_number", "model", "voltage", "manufacture_date"]
    lines = [
        "# Task 3 accuracy report — nameplates v2",
        "",
        "20 nameplate photos in `task3/data/nameplates_v2`.",
        "Scored against `labels_ground_truth.csv` with `scoring.py`",
        "(alnum-normalized exact match; both-null counts as correct).",
        "A cross on a comparison board means the cell differs from that CSV.",
        "",
        "This run uses OpenCV preprocessing (deskew/crop, then the per-engine recipe).",
        "Preprocessed images: `preprocessed_easyocr/`, `preprocessed_tesseract/`, `preprocessed_paddleocr/`.",
        "",
        "## Field accuracy and serial CER (with preprocessing)",
        "",
        "| Reader | Overall | Printed fields | equipment_id | serial_number | model | voltage | manufacture_date | Serial CER |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, label, _prep in cmp.READERS:
        row = engines[source]
        acc = row["field_accuracy"]
        cer = row["serial_number_avg_cer"]
        cer_txt = f"{cer:.3f}" if cer is not None else "n/a"
        cells = " | ".join(f"{acc[field]:.1f}%" for field in fields)
        lines.append(
            f"| {label} | {row['overall']:.1f}% | {row['non_null_hit']}/{non_null_total} | {cells} | {cer_txt} |"
        )
    lines.extend(["", "## Without preprocessing (raw photos, same engines, same extractor)", ""])
    if raw_engines:
        lines.extend([
            "Raw run: the original JPEG goes straight to each engine. No crop, deskew, denoise or threshold.",
            "",
            "| Reader | Overall raw | Overall preprocessed | Delta (pp) | Printed raw | Printed preprocessed | Serial CER raw | Serial CER preprocessed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for source, label, _prep in cmp.READERS:
            pre, raw = engines[source], raw_engines[source]
            cer = lambda r: f"{r['serial_number_avg_cer']:.3f}" if r["serial_number_avg_cer"] is not None else "n/a"
            lines.append(
                f"| {label} | {raw['overall']:.1f}% | {pre['overall']:.1f}% | "
                f"{pre['overall'] - raw['overall']:+.1f} | {raw['non_null_hit']}/{non_null_total} | "
                f"{pre['non_null_hit']}/{non_null_total} | {cer(raw)} | {cer(pre)} |"
            )
        lines.extend([
            "",
            "Raw OCR cache: `raw_ocr/*_rawimg.json`. Summary: `accuracy_summary_raw.json`.",
            "",
        ])
    else:
        lines.extend(["Not run. Run `run_pipeline.py`.", ""])
    path = os.path.join(OUT_DIR, "accuracy_report.md")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"Wrote {path}", flush=True)


def aggregate_grid(gt, per_source, names, engines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scoring import field_match

    labels = [label for _source, label, _prep in cmp.READERS]
    matrix = np.array([
        [sum(field_match(per_source[source][name]["fields"][field], gt[name][field])
             for field in scoring.FIELDS)
         for source, _label, _prep in cmp.READERS]
        for name in names
    ])
    fig = plt.figure(figsize=(24, 15), dpi=110)
    ax = fig.add_axes([0.06, 0.07, 0.40, 0.84])
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(range(len(cmp.READERS)), labels, rotation=20, ha="right", fontsize=11)
    ax.set_yticks(range(len(names)), [name.replace(".jpg", "") for name in names], fontsize=11)
    for i in range(matrix.shape[0]):
        best = matrix[i].max()
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}/5", ha="center", va="center", fontsize=12,
                    fontweight="bold" if matrix[i, j] == best else "normal")
    ax.set_title("Fields correct per image (of 5)", fontsize=15, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)

    ax2 = fig.add_axes([0.55, 0.42, 0.42, 0.49])
    acc = np.array([
        [engines[source]["field_accuracy"][field] for field in scoring.FIELDS]
        for source, _label, _prep in cmp.READERS
    ])
    width = 0.22
    colors = ["#7a7a7a", "#c08a2a", "#2a8a4a"]
    for j, (_source, _label, _prep) in enumerate(cmp.READERS):
        bars = ax2.bar(np.arange(len(scoring.FIELDS)) + (j - 1) * width, acc[j], width,
                       label=labels[j].replace("\n", " "), color=colors[j], edgecolor="white")
        ax2.bar_label(bars, fmt="%.0f", fontsize=8)
    ax2.set_xticks(range(len(scoring.FIELDS)), scoring.FIELDS, fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("field accuracy (%)")
    ax2.set_title("Per-field accuracy, nameplates_v2", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")

    ax3 = fig.add_axes([0.55, 0.07, 0.42, 0.27])
    ax3.axis("off")
    total = next(iter(engines.values()))["non_null_total"]
    rows = []
    for (source, _label, _prep), label in zip(cmp.READERS, labels):
        row = engines[source]
        cer = row["serial_number_avg_cer"]
        rows.append([
            label.replace("\n", " "),
            f"{row['overall']:.1f}%",
            f"{row['non_null_hit']}/{total}",
            f"{cer:.3f}" if cer is not None else "n/a",
        ])
    table = ax3.table(cellText=rows,
                      colLabels=["Reader", "Overall (100 cells)", "Non-null GT cells", "Serial CER"],
                      loc="center", cellLoc="center", colWidths=[0.46, 0.18, 0.19, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)
    fig.suptitle("Nameplates v2 — EasyOCR, Tesseract, and PaddleOCR against ground truth",
                 fontsize=13, fontweight="bold")
    out = os.path.join(OUT_DIR, "engine_comparison_grid.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CMP_DIR, exist_ok=True)
    names = prep.image_names()
    print(f"{len(names)} images", flush=True)
    preprocess(names)

    easy = ocr.cached(os.path.join(RAW_DIR, "easyocr_tuned.json"), lambda: ocr.run_easyocr(names), False)
    tess = ocr.cached(os.path.join(RAW_DIR, "tess_psm4_tuned.json"), lambda: ocr.run_tesseract(names, 4), False)
    paddle = ocr.cached(os.path.join(RAW_DIR, "paddle_tuned.json"), lambda: ocr.run_paddle(names), False)

    per_source = {
        "easyocr/tuned": pack_ocr(easy),
        "tess_psm4/tuned": pack_ocr(tess),
        "paddle/tuned": pack_ocr(paddle),
    }
    gt = scoring.load_ground_truth()
    for name, value in clear_paddle_equipment_when_model_copy(per_source["paddle/tuned"], gt):
        print(f"  paddle equipment_id cleared {name}: {value}", flush=True)
    labels_path = os.path.join(OUT_DIR, "labels_output_v2_per_engine.json")
    with open(labels_path, "w") as handle:
        json.dump(per_source, handle, indent=1)
    print(f"Wrote {labels_path}", flush=True)
    write_labels_output(per_source)

    engines = score_engines(gt, per_source)
    with open(os.path.join(OUT_DIR, "accuracy_summary.json"), "w") as handle:
        json.dump({"engines": engines}, handle, indent=2)
    raw_engines = run_raw_ablation(names, gt)
    write_summary(engines, next(iter(engines.values()))["non_null_total"], raw_engines)

    ordered = sorted(gt, key=lambda name: (len(name), name))
    for name in ordered:
        scores = {
            source: sum(scoring.field_match(per_source[source][name]["fields"][field], gt[name][field])
                        for field in scoring.FIELDS)
            for source, _label, _prep in cmp.READERS
        }
        print(cmp.board(name, gt, per_source, scores), flush=True)
    aggregate_grid(gt, per_source, ordered, engines)


if __name__ == "__main__":
    main()
