"""
Task 3 comparison boards.

For each nameplate writes a board: original photo and ground truth on top, then one
column per reader (EasyOCR, Tesseract psm 4, classic PaddleOCR). Each field is ticked
against labels_ground_truth.csv. A cross means the value differs from ground truth.

Called by run_pipeline.py, which also writes the aggregate engine_comparison_grid.png.
"""
import os

from PIL import Image, ImageDraw, ImageFont

from config import CMP_DIR, DATA_DIR
from preprocess import out_path as prep_path
from scoring import FIELDS, field_match
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

W, MARGIN, GAP = 2800, 30, 20
READERS = [
    ("easyocr/tuned", "EasyOCR", "per-engine: 1600px, colour NLM denoise, CLAHE, no threshold"),
    ("tess_psm4/tuned", "Tesseract (psm 4)", "per-engine: 2400px, bilateral, CLAHE, adaptive threshold"),
    ("paddle/tuned", "PaddleOCR (PP-OCRv6 classic)", "per-engine: 2000px, grayscale CLAHE, no sharpen, no threshold"),
]
COL_W = (W - 2 * MARGIN - (len(READERS) - 1) * GAP) // len(READERS)
GREEN, RED, GREY, INK = (22, 135, 60), (200, 30, 30), (120, 120, 120), (25, 25, 25)
PREP_KEY = {"easyocr/tuned": "easyocr", "tess_psm4/tuned": "tesseract", "paddle/tuned": "paddleocr"}
SHORT = {"equipment_id": "equipment_id", "serial_number": "serial_number", "model": "model",
         "voltage": "voltage", "manufacture_date": "manuf._date"}


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(os.path.join(FONT_DIR, f"{name}{'-Bold' if bold else ''}.ttf"), size)


F_TITLE, F_H, F_B, F_S = font(40, True), font(26, True), font(22), font(19)
F_MONO, F_MONO_B, F_MARK = font(21, mono=True), font(21, True, True), font(30, True)


def wrap(draw, text, fnt, width):
    """Greedy character wrap (values have no spaces to break on, e.g. long serials)."""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=fnt) > width and cur:
            lines.append(cur)
            cur = ""
        cur += ch
    return lines + [cur] if cur else lines or [""]


def wrap_words(draw, text, fnt, width):
    """Word wrap for prose; falls back to character wrap for a single over-long token."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
        if draw.textlength(word, font=fnt) > width:
            *head, cur = wrap(draw, word, fnt, width)
            lines += head
    return lines + [cur] if cur else lines or [""]


def fit(img, w, h):
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)


def short(v, n=60):
    return v if v is None or len(v) <= n else v[: n - 1] + "…"


def board(name, gt, results, scores):
    stem = name.replace(".jpg", "")
    truth = gt[name]
    canvas = Image.new("RGB", (W, 2400), "white")
    d = ImageDraw.Draw(canvas)

    d.rectangle([0, 0, W, 80], fill=(30, 40, 60))
    d.text((MARGIN, 18), f"{stem}  —  OCR comparison against ground truth",
           font=F_TITLE, fill="white")
    y = 100

    photo = Image.open(os.path.join(DATA_DIR, name)).convert("RGB")
    ph = fit(photo, 1150, 520)
    canvas.paste(ph, (MARGIN, y))
    d.rectangle([MARGIN - 2, y - 2, MARGIN + ph.width + 1, y + ph.height + 1], outline=GREY, width=2)
    d.text((MARGIN, y + ph.height + 8), f"Original photo ({photo.width}x{photo.height} px)", font=F_S, fill=GREY)

    gx = MARGIN + max(ph.width, 700) + 60
    d.text((gx, y), "Ground truth (labels_ground_truth.csv)", font=F_H, fill=INK)
    ty = y + 45
    for f in FIELDS:
        d.text((gx, ty), f"{f}:", font=F_B, fill=GREY)
        for i, ln in enumerate(wrap(d, str(truth[f]) if truth[f] else "null (not on plate)", F_MONO_B, W - gx - 330)):
            d.text((gx + 290, ty + i * 27), ln, font=F_MONO_B, fill=INK)
            ty += 27 if i else 0
        ty += 40
    ty += 10
    d.text((gx, ty), "Fields correct on this image (of 5):", font=F_H, fill=INK)
    ty += 42
    best = max(scores[s] for s, _, _ in READERS)
    for s, label, _ in READERS:
        d.text((gx, ty), label, font=F_B, fill=INK)
        n = scores[s]
        bx = gx + 560
        for k in range(5):
            d.rectangle([bx + k * 44, ty + 2, bx + k * 44 + 38, ty + 26], fill=GREEN if k < n else (225, 225, 225))
        d.text((bx + 230, ty), f"{n}/5" + ("  ← best" if n == best else ""), font=F_B,
               fill=GREEN if n == best else INK)
        ty += 36
    d.text((gx, ty + 10), "✓/✗ = matches ground truth (alnum-normalized, both-null counts as correct).",
           font=F_S, fill=GREY)

    y = max(y + ph.height + 50, ty + 80)
    bottoms = []
    for ci, (s, label, prep) in enumerate(READERS):
        x = MARGIN + ci * (COL_W + GAP)
        d.rectangle([x, y, x + COL_W, y + 54], fill=(55, 55, 55))
        d.text((x + 12, y + 12), label, font=F_H, fill="white")
        cy = y + 62
        for ln in wrap_words(d, "Input: " + prep, F_S, COL_W - 16):
            d.text((x + 8, cy), ln, font=F_S, fill=GREY)
            cy += 24
        src_img = prep_path(PREP_KEY[s], name)
        th = fit(Image.open(src_img).convert("RGB"), COL_W - 16, 300)
        canvas.paste(th, (x + 8 + (COL_W - 16 - th.width) // 2, cy + 4))
        d.rectangle([x + 6, cy + 2, x + COL_W - 6, cy + 308], outline=(210, 210, 210), width=1)
        cy += 320

        preds = results[s][name]["fields"]
        for f in FIELDS:
            ok = field_match(preds[f], truth[f])
            d.rectangle([x, cy, x + COL_W, cy + 2], fill=(230, 230, 230))
            cy += 8
            d.text((x + 8, cy), SHORT[f], font=F_S, fill=GREY)
            d.text((x + COL_W - 40, cy - 4), "✓" if ok else "✗", font=F_MARK, fill=GREEN if ok else RED)
            cy += 28
            for ln in wrap(d, short(preds[f], 80) if preds[f] is not None else "null", F_MONO, COL_W - 20)[:3]:
                d.text((x + 10, cy), ln, font=F_MONO, fill=(GREEN if ok else RED) if preds[f] is not None else GREY)
                cy += 26
            cy += 6
        n = scores[s]
        d.rectangle([x, cy + 4, x + COL_W, cy + 46], fill=(235, 245, 238) if n == best else (245, 245, 245))
        d.text((x + 10, cy + 12), f"{n}/5 fields correct vs ground truth", font=F_H, fill=GREEN if n == best else INK)
        cy += 60

        d.text((x + 8, cy), "Raw OCR text (first lines):", font=F_S, fill=GREY)
        cy += 26
        raw = [short(t, 120) for t in results[s][name]["lines"] if t.strip()]
        lines = []
        for t in raw:
            lines += wrap_words(d, t, F_S, COL_W - 20)
            if len(lines) >= 9:
                break
        lines = lines[:9] or ["(no text detected)"]
        for ln in lines:
            d.text((x + 10, cy), ln, font=F_S, fill=INK)
            cy += 23
        bottoms.append(cy)

    H = max(bottoms) + 70
    d.rectangle([0, H - 50, W, H], fill=(240, 240, 240))
    d.text((MARGIN, H - 40), "Each field is compared with labels_ground_truth.csv. A cross means that value differs from ground truth.",
           font=F_S, fill=INK)
    out = os.path.join(CMP_DIR, f"{stem}_comparison.png")
    canvas.crop((0, 0, W, H)).save(out, optimize=True)
    return out
