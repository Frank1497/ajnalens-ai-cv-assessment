"""
Task 3: OpenCV preprocessing.

Step 1 (shared): detect_and_deskew_label finds the largest quadrilateral (Canny edges,
contour approximation), perspective-warps it to a front-on crop, else deskews by the
min-area-rect angle, else returns the photo unchanged.
Step 2 (per engine): each engine gets an input shaped for how it reads.

    tesseract     -> outputs/.../preprocessed_tesseract/<stem>.png   binarized, ~300 dpi equiv, white border
    easyocr       -> outputs/.../preprocessed_easyocr/<stem>.png     colour, denoised, CLAHE on L, no threshold
    paddleocr     -> outputs/.../preprocessed_paddleocr/<stem>.png   grayscale, CLAHE, no sharpen, no threshold

Usage:  python3 task3/src/preprocess.py [--force]
"""
import argparse
import glob
import os

import cv2
import numpy as np

from config import DATA_DIR, OUT_DIR, REPO_ROOT

ENGINES = ("tesseract", "easyocr", "paddleocr")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_deskew_label(img):
    """Try to find the dominant rectangular label region and warp it to a
    front-on crop. Falls back to the full image (just deskewed by min-area
    rect angle) if no good quadrilateral is found.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area:  # ignore small contours, label should be a large chunk of frame
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is not None:
        pts = best.reshape(4, 2).astype("float32")
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth > 50 and maxHeight > 50:
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                             [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            return warped

    # fallback: deskew via min-area-rect of largest contour, else return original
    if contours:
        c = max(contours, key=cv2.contourArea)
        rect_mar = cv2.minAreaRect(c)
        angle = rect_mar[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5 and abs(angle) < 45:
            (h2, w2) = img.shape[:2]
            center = (w2 // 2, h2 // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w2, h2), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    return img


def _resize_width(img, width, max_scale=None):
    s = width / img.shape[1]
    if max_scale:
        s = min(s, max_scale)
    if abs(s - 1.0) < 1e-3:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def _clahe_lab(img, clip):
    """CLAHE on the L channel only: lifts local contrast without shifting hue."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prep_tesseract(img):
    """Tesseract's LSTM is trained on clean dark-on-light page text, and its own
    Otsu binarizer is a weak point on photos (ImproveQuality docs). It wants:
      * ~300 dpi: x-height >= ~20 px. Plates here are 485-738 px wide at roughly 100-150 dpi,
        so the crop is upscaled to 2400 px wide (~4x, about 300 dpi equivalent). A 3200px
        (~5.3x) variant was tried to help the dense WEG table plates (08/09/10) but
        regressed overall 51%->42%: the heavier upscale amplifies JPEG block noise into
        salt speckle across the WHOLE plate, not just the dense tables, which fragments
        Tesseract's layout analysis even on previously-clean plates (05, 013 lost every
        field). Reverted; this is a measured negative result, not silently dropped.
      * denoise before thresholding (bilateral d=9 keeps stroke edges, kills sensor noise),
      * CLAHE clip 2.0 to even out uneven lighting,
      * adaptive Gaussian threshold, block 41 px (~2 stroke-heights at this scale), C=12. Otsu
        was tried first and rejected on visual inspection: one global threshold plus a
        global-mean polarity guess inverted whole plates whose crop kept a large dark
        background (nameplate_10 came out white-on-black). The local threshold keeps dark
        text black regardless of background; the high C keeps flat regions white,
      * a 2x2 open to drop salt noise, border-touching ink blobs whitened, and a 20 px
        white border (docs: text touching the edge gets dropped).
    Page-segmentation mode is an engine flag, not an image property; the driver
    runs psm 4 on this image."""
    gray = cv2.cvtColor(_resize_width(img, 2400), cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12)
    bw = 255 - cv2.morphologyEx(255 - bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.copyMakeBorder(_clear_border_ink(bw), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _clear_border_ink(bw):
    """Whiten large black blobs touching the frame (photo background / plate edge left
    after the crop). Tesseract's page layout treats them as giant glyphs or image regions."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((bw == 0).astype(np.uint8), connectivity=8)
    h, w = bw.shape
    out = bw.copy()
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        if touches and area > 0.002 * h * w:
            out[lab == i] = 255
    return out


def prep_easyocr(img):
    """EasyOCR = CRAFT detector + CRNN recognizer trained on natural-scene colour/gray text.
    CRAFT scores character-region heatmaps from stroke texture, so hard thresholding hurts it.
      * upscale to 1600 px wide (EasyOCR's canvas_size=2560 / mag_ratio=1 handle the rest;
        bigger mostly adds detector false positives on rivets/scratches). A 2200px variant
        was tried to help the dense WEG table plates (08/09/10) but measured -1pp overall
        (54%->53%): CRAFT's detector picked up more false-positive boxes on scratches/rivets
        at the larger canvas than it gained on the dense tables. Reverted; measured negative,
        not silently dropped,
      * fastNlMeansDenoisingColored h=5: removes JPEG/sensor noise, keeps stroke texture,
      * CLAHE clip 2.0 on L (colour kept: the recognizer was trained on colour crops),
      * NO threshold, no morphology."""
    img = _resize_width(img, 1600)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    return _clahe_lab(img, 2.0)


def prep_paddleocr(img):
    """PP-OCR = DB (differentiable binarization) detector + SVTR recognizer. DB learns its
    own threshold map, so pre-binarizing removes the signal it was built to binarize.
    An earlier colour + unsharp-mask recipe measured -5pp against plain grayscale CLAHE
    (hallucinated IDs on a worn plate, dropped a digit in one model, lost a voltage). Reverted
    to plain grayscale CLAHE, the measured-better input for this engine:
      * upscale to 2000 px wide,
      * CLAHE clip 2.0 on plain grayscale (plain grayscale, no colour),
      * NO unsharp mask, NO threshold."""
    gray = cv2.cvtColor(_resize_width(img, 2000), cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


RECIPES = {"tesseract": prep_tesseract, "easyocr": prep_easyocr,
           "paddleocr": prep_paddleocr}


def out_dir(engine):
    return os.path.join(OUT_DIR, f"preprocessed_{engine}")


def out_path(engine, name):
    return os.path.join(out_dir(engine), name.replace(".jpg", ".png"))


def image_names():
    return [os.path.basename(p) for p in sorted(glob.glob(os.path.join(DATA_DIR, "nameplate_*.jpg")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    names = image_names()
    for e in ENGINES:
        os.makedirs(out_dir(e), exist_ok=True)
    for name in names:
        crop = detect_and_deskew_label(cv2.imread(os.path.join(DATA_DIR, name)))
        for e in ENGINES:
            p = out_path(e, name)
            if args.force or not os.path.exists(p):
                cv2.imwrite(p, RECIPES[e](crop))
    for e in ENGINES:
        print(f"{os.path.relpath(out_dir(e), REPO_ROOT)}: {len(glob.glob(os.path.join(out_dir(e), '*.png')))} images")


if __name__ == "__main__":
    main()
