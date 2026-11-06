"""Task 3 scoring: field-level accuracy and serial-number CER against labels_ground_truth.csv.

A prediction matches when it equals the truth after upper-casing and dropping everything
that is not a letter or digit. Both-null counts as correct (nothing printed, nothing found).
"""
import csv
import os

from config import DATA_DIR

FIELDS = ["equipment_id", "serial_number", "model", "voltage", "manufacture_date"]


def load_ground_truth():
    gt = {}
    path = os.path.join(DATA_DIR, "labels_ground_truth.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["image_filename"].strip()
            gt[name] = {f: (row[f].strip() if row[f] and row[f].strip() else None) for f in FIELDS}
    return gt


def norm(s):
    if s is None:
        return None
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def field_match(pred, truth):
    """Exact match after normalization (strip punctuation/case/whitespace).
    Both-None counts as correct (nothing to find, nothing found).
    """
    np_, nt = norm(pred), norm(truth)
    return np_ == nt


def cer(pred, truth):
    """Character Error Rate = edit_distance(pred, truth) / len(truth).
    Returns None if truth is None/empty (can't compute CER without a reference).
    """
    if truth is None or truth == "":
        return None
    p = pred if pred is not None else ""
    t = truth
    # Levenshtein distance via difflib SequenceMatcher-based dynamic programming
    m, n = len(p), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1].upper() == t[j - 1].upper() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    return dist / len(t)


def score(gt, preds, label):
    per_field_correct = {f: 0 for f in FIELDS}
    per_field_total = {f: 0 for f in FIELDS}
    cer_values = []
    rows = []

    for name, truth in gt.items():
        pred = preds.get(name, {f: None for f in FIELDS})
        row = {"image": name}
        for f in FIELDS:
            per_field_total[f] += 1
            match = field_match(pred.get(f), truth.get(f))
            if match:
                per_field_correct[f] += 1
            row[f] = {"pred": pred.get(f), "truth": truth.get(f), "match": match}
        sn_cer = cer(pred.get("serial_number"), truth.get("serial_number"))
        if sn_cer is not None:
            cer_values.append(sn_cer)
        row["serial_number_cer"] = sn_cer
        rows.append(row)

    field_accuracy = {
        f: (per_field_correct[f] / per_field_total[f] * 100.0 if per_field_total[f] else 0.0)
        for f in FIELDS
    }
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else None

    return {
        "label": label,
        "field_accuracy": field_accuracy,
        "avg_serial_number_cer": avg_cer,
        "n_images": len(gt),
        "n_serial_with_gt": len(cer_values),
        "rows": rows,
    }


def overall(res):
    """Mean of the five per-field accuracies, in percent."""
    return sum(res["field_accuracy"].values()) / len(FIELDS)


def known_field_acc(gt, preds):
    """Accuracy over only the (image, field) cells whose ground truth is non-null --
    the stricter view (both-null 'free' matches excluded)."""
    hit = tot = 0
    for name, truth in gt.items():
        for f in FIELDS:
            if truth[f] is not None:
                tot += 1
                hit += field_match(preds.get(name, {}).get(f), truth[f])
    return 100.0 * hit / tot, hit, tot
