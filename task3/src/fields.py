"""
Task 3: engine-agnostic field extraction shared by every OCR engine.

Every engine adapter (EasyOCR, Tesseract, PaddleOCR) emits the same record shape:
    {"text": str, "conf": float in [0,1], "box": [x0, y0, x1, y1]}
`build_lines()` regroups those boxes into physical text lines (same vertical band,
sorted left-to-right), so the extractor sees the same structure regardless of
whether the engine returned word fragments (EasyOCR, Tesseract) or whole lines
(PaddleOCR). That is the only engine-specific step; extraction logic is identical.

`extract_candidates()` returns, per field, a list of candidate dicts:
    {"value": str, "conf": float, "anchored": bool, "rule": str}
`extract_fields()` returns the best candidate per field.

Rules are generic nameplate conventions, not per-image hacks:
  * label anchors (SER NO, S/N, MODEL, CAT NO, VOLTS, DATE, YEAR ... WEEK ...)
  * position-aware character repair: letter O -> digit 0 only inside ID tokens
    where it touches a digit; O/I/l -> 0/1 only in digit slots of date codes
  * voltage lexicon decoding: digit runs are segmented into IEC/NEMA standard
    voltages, tolerating one junk glyph where a '/' was misread (e.g. 2307460)
  * multilingual month codes (WEG prints FEV/ABR, Siemens prints MAI/OKT, ...)
"""
import re

FIELDS = ["equipment_id", "serial_number", "model", "voltage", "manufacture_date"]

STANDARD_VOLTAGES = {
    100, 110, 115, 120, 127, 190, 200, 208, 220, 230, 240, 254, 265, 277, 280,
    380, 400, 415, 420, 440, 460, 480, 500, 525, 550, 575, 600, 660, 690,
}

MONTHS = {
    "JAN", "FEB", "FEV", "MAR", "APR", "ABR", "AVR", "MAY", "MAI", "MAG", "MAJ",
    "JUN", "JUL", "AUG", "AGO", "AOU", "SEP", "SET", "OCT", "OUT", "OKT", "NOV",
    "DEC", "DEZ", "DIC",
}

DIGIT_LIKE = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "S": "5", "Z": "2", "B": "8"}


# ---------------------------------------------------------------------------
# Line reconstruction
# ---------------------------------------------------------------------------

def build_lines(records):
    """Group word/line boxes into physical lines. Returns list of
    {"text", "conf", "tokens": [(text, conf)], "y"} sorted top-to-bottom."""
    recs = [r for r in records if r.get("text", "").strip()]
    if not recs:
        return []
    heights = sorted(max(1.0, r["box"][3] - r["box"][1]) for r in recs)
    med_h = heights[len(heights) // 2]
    recs.sort(key=lambda r: (r["box"][1] + r["box"][3]) / 2)
    lines = []
    for r in recs:
        cy = (r["box"][1] + r["box"][3]) / 2
        if lines and abs(cy - lines[-1]["cy"]) <= 0.5 * med_h:
            ln = lines[-1]
            ln["items"].append(r)
            ln["cy"] = sum((i["box"][1] + i["box"][3]) / 2 for i in ln["items"]) / len(ln["items"])
        else:
            lines.append({"cy": cy, "items": [r]})
    out = []
    for ln in lines:
        items = sorted(ln["items"], key=lambda i: i["box"][0])
        toks = [(i["text"].strip(), float(i["conf"])) for i in items]
        out.append({
            "text": " ".join(t for t, _ in toks),
            "conf": sum(c for _, c in toks) / len(toks),
            "tokens": toks,
            "y": ln["cy"],
        })
    return out


# ---------------------------------------------------------------------------
# Character repair helpers
# ---------------------------------------------------------------------------

def norm(s):
    if s is None:
        return None
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def repair_id_token(tok):
    """Letter O touching a digit inside a mostly-numeric ID becomes 0
    (1LAO2864SE41 -> 1LA02864SE41). Other letters are left alone because real
    IDs mix letters and digits freely (DI0034PCI, 37J776T859G1)."""
    alnum = [c for c in tok if c.isalnum()]
    if not alnum or sum(c.isdigit() for c in alnum) / len(alnum) < 0.4:
        return tok
    chars = list(tok)
    for i, c in enumerate(chars):
        if c == "O":
            left = chars[i - 1] if i > 0 else ""
            right = chars[i + 1] if i + 1 < len(chars) else ""
            if left.isdigit() or right.isdigit():
                chars[i] = "0"
    return "".join(chars)


def to_digits(s):
    return "".join(DIGIT_LIKE.get(c, c) for c in s)


# ---------------------------------------------------------------------------
# Voltage lexicon decoding
# ---------------------------------------------------------------------------

def decode_voltage_run(run):
    """Segment a digit/separator run into standard voltages.
    Returns the decoded groups as a list of strings like ['208-230', '460'],
    or None if the run cannot be fully explained by the lexicon.
    Separators kept: '/', '-'. A single junk digit/glyph between two standard
    voltages is treated as a misread '/' (OCR reads '/' as 1, 7, l, I, |)."""
    s = run.upper().strip(" '\"`~.,;:!)(|[]_")
    s = re.sub(r"V$", "", s)
    s = re.sub(r"\s+", " ", s)
    s = "".join(DIGIT_LIKE[c] if c in "OQIL|" else c for c in s)
    if not re.fullmatch(r"[0-9/\- \u0394YX]+", s or "#"):
        return None

    def solve(t):
        """Greedy-with-backtracking segmentation. After each standard voltage the
        next char is either a real separator (/ - space) or ONE junk glyph (a '/'
        misread as 1/7/2/4, or a delta/star symbol) that is dropped."""
        if len(t) < 3 or not t[:3].isdigit() or int(t[:3]) not in STANDARD_VOLTAGES:
            return None
        head, rest = t[:3], t[3:]
        if not rest:
            return head
        options = []
        if rest[0] in "/- ":
            options.append(("-" if rest[0] == "-" else "/", rest[1:]))
        else:
            options.append(("/", rest))
            options.append(("/", rest[1:]))
            if len(rest) > 1 and rest[1] in "/- ":
                options.append(("-" if rest[1] == "-" else "/", rest[2:]))
        for sep, tail in options:
            if not tail:
                return head
            sub = solve(tail)
            if sub:
                return head + sep + sub
        return None

    for t in (s, s[1:]):
        out = solve(t) if t else None
        if out:
            return out
    return None


VOLT_RUN = r"[0-9OQIl|][0-9OQIl|/\-]{1,18}[0-9OQIl|]V?"
VOLT_RUN_ANCHORED = r"[0-9OQIl|](?:[0-9OQIl|/\-\u0394]| (?=[0-9]))+[0-9OQIl|][\u0394YXyx]?"
# Number printed before the unit, as on "230 V" and "115/230V". The unit letter is not
# itself the label; the lexicon still has to accept the digits, so "450 V" (capacitor)
# and "1400" (rpm) do not qualify.
VOLT_BEFORE_UNIT = re.compile(
    r"(?<![A-Z0-9/])([0-9]{3}(?:\s*[/\-]\s*[0-9]{3})*)\s*V\b", re.I)
CAPACITOR_BEFORE = re.compile(r"(?:UF|µF|ΜF|MFD|CAP)\b", re.I)


def _conn_base(tok):
    """415D+/-10%, 690Y and 400D are connection markings, not part of the number."""
    marked = tok.upper().replace("±", "+/-")
    m = re.fullmatch(r"([0-9]{3})D(?:\+/-)\d+%", marked)
    if m:
        return m.group(1)
    m = re.fullmatch(r"([0-9]{3})[YD]", marked)
    if m:
        return m.group(1)
    return tok


# "V230 460" / "V230/460": the V label is glued to the number, and a lost slash
# comes back as a space between two standard voltages.
GLUED_V = re.compile(
    r"(?:^|[^A-Z0-9])V\s*([0-9]{3})(?:\s*([/\-])\s*|\s+)([0-9]{3})(?![0-9])", re.I)
USABLE_BEFORE = re.compile(r"(?:USABLE\s*@?|@)\s*$", re.I)


def _voltage_glued_label(text):
    m = GLUED_V.search(text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(3))
    if a not in STANDARD_VOLTAGES or b not in STANDARD_VOLTAGES:
        return None
    return f"{m.group(1)}{'-' if m.group(2) == '-' else '/'}{m.group(3)}"


def _conn_kind(tok):
    """(voltage, 'D' or 'Y') for a delta/star connection mark, including 400△|50."""
    marked = tok.upper().replace("±", "+/-").replace("Δ", "△")
    m = re.fullmatch(r"([0-9]{3})D(?:\+/-)\d+%", marked)
    if m and int(m.group(1)) in STANDARD_VOLTAGES:
        return m.group(1), "D"
    m = re.match(r"([0-9]{3})([YD△])", marked)
    if m and int(m.group(1)) in STANDARD_VOLTAGES:
        return m.group(1), "D" if m.group(2) in "D△" else "Y"
    return None


def _voltage_before_unit(text):
    """Standard voltages written in front of V. Capacitor ratings (µF / 400V)
    and a 'usable at' note (@208V) are skipped."""
    found = []
    for m in VOLT_BEFORE_UNIT.finditer(text):
        window = text[max(0, m.start() - 16):m.start()]
        if CAPACITOR_BEFORE.search(window) or USABLE_BEFORE.search(window):
            continue
        dec = decode_voltage_run(m.group(1))
        if dec:
            found.append(dec)
    if not found:
        return None
    return found[0] if len(found) == 1 else "/".join(found)


def voltage_candidates_from_lines(lines):
    cands = []
    anchor = re.compile(r"(?:(?:\b|(?<=[0-9]))(?:VOLTS?|VOLTAGE|VOLT\.|VOLS)|\bV)\b\s*[:\-.~=]?\s*(" + VOLT_RUN_ANCHORED + ")", re.I)
    for ln in lines:
        for m in anchor.finditer(ln["text"]):
            raw = m.group(1)
            dec = decode_voltage_run(raw) or decode_voltage_run(raw.split()[0])
            if dec:
                cands.append({"value": dec, "conf": ln["conf"], "anchored": True, "rule": "VOLT-anchor+lexicon"})
        glued = _voltage_glued_label(ln["text"])
        if glued:
            cands.append({"value": glued, "conf": ln["conf"], "anchored": True, "rule": "VOLT-anchor+lexicon"})
        for tok, _conf in ln["tokens"]:
            base = _conn_base(tok.upper().strip(" ,"))
            if base != tok.upper().strip(" ,") and base.isdigit() and int(base) in STANDARD_VOLTAGES:
                cands.append({"value": base, "conf": ln["conf"], "anchored": True, "rule": "VOLT-conn"})
        before = _voltage_before_unit(ln["text"])
        if before and not any(c["value"] == before for c in cands if c["rule"] == "VOLT-anchor+lexicon"):
            cands.append({"value": before, "conf": ln["conf"], "anchored": True, "rule": "VOLT-before-unit"})
    # Multi-voltage table rows: consecutive lines whose first token decodes to a
    # standard voltage pair or single standard voltage. Needs >=3 rows to count
    # as a table, so a lone stray '400' never becomes a voltage.
    rows, best_rows = [], []
    for ln in lines:
        toks = [re.sub(r"^[^0-9A-Z]+|[^0-9A-Z/]+$", "", t.upper()) for t, _ in ln["tokens"]]
        first = _conn_base(re.sub(r"[/\-]+$", "", toks[0])) if toks else ""
        dec = decode_voltage_run(first) if re.fullmatch(VOLT_RUN, first or "x") or re.fullmatch(r"\d{3}", first or "x") else None
        # Engines that box each table cell separately (PaddleOCR) give '380' '660' as two
        # tokens. Merge them when the pair is physically a dual-voltage rating:
        # star/delta ratio sqrt(3)~1.73 (380/660, 400/690) or NEMA series/parallel 2.0 (230/460).
        if dec and dec.isdigit() and len(toks) > 1 and toks[1].isdigit() and int(toks[1]) in STANDARD_VOLTAGES \
                and 1.6 <= int(toks[1]) / int(dec) <= 2.1:
            dec = f"{dec}/{toks[1]}"
        if dec:
            rows.append((dec, ln["conf"]))
        else:
            if len(rows) > len(best_rows):
                best_rows = rows
            rows = []
    if len(rows) > len(best_rows):
        best_rows = rows
    if len(best_rows) >= 3 and no_duplicate_rows(",".join(r for r, _ in best_rows)):
        cands.append({
            "value": ",".join(r for r, _ in best_rows),
            "conf": sum(c for _, c in best_rows) / len(best_rows),
            "anchored": True,
            "rule": "voltage-column(>=3 rows)",
        })
    # One delta mark and one star mark are a single dual-voltage rating (400△ / 690Y).
    # A table with several of each is left to the column rule above.
    deltas, stars = [], []
    for ln in lines:
        for tok, _conf in ln["tokens"]:
            kind = _conn_kind(tok.strip(" ,"))
            if not kind:
                continue
            (deltas if kind[1] == "D" else stars).append((kind[0], ln["conf"]))
    if len(deltas) == 1 and len(stars) == 1:
        lo, hi = sorted((int(deltas[0][0]), int(stars[0][0])))
        if lo and 1.6 <= hi / lo <= 2.1:
            cands.append({
                "value": f"{lo}/{hi}",
                "conf": (deltas[0][1] + stars[0][1]) / 2,
                "anchored": True,
                "rule": "VOLT-conn-pair",
            })
    return cands


# ---------------------------------------------------------------------------
# Generic anchored ID / date extraction
# ---------------------------------------------------------------------------

ID_VAL = r"([A-Z0-9][A-Z0-9\-\./\$]{2,})"
# "MY 7124", "FM 140156", and two-token codes "FH57B DV112M4/BMG" / "M3AA 80MC2".
ID_VAL_JOIN = (
    r"([A-Z]{1,4}\s+[0-9]{2,}"
    # Second token must contain a digit and must not be a date (17ABR2019) or a bare
    # serial. That keeps "FH57B DV112M4/BMG" and drops a following word such as ENCL.
    r"|[A-Z0-9][A-Z0-9\-]{2,}\s+(?!\d{1,2}[A-Z]{3}\d{2,4}\b)(?!\d{6,}\b)(?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]{2,}"
    r"|[A-Z]{1,3}\s[0-9]{6,}"
    r"|[A-Z0-9][A-Z0-9\-\./\$]{2,})"
)

ANCHORS = {
    "serial_number": [
        # Letter + long digits keeps "Ser No.: FM 14015624040001". A second token is not
        # joined, so a rivet fragment does not absorb the next line.
        (r"\bSER(?:IAL)?\.?\s*N[O0Q]\.?\s*[:;\-.]*\s*([A-Z]{1,3}\s[0-9]{6,}|[A-Z0-9][A-Z0-9\-\./\$]{2,})", "SER NO"),
        (r"\bS\s*/\s*N\.?\s*[:;\-.]*\s*" + ID_VAL, "S/N"),
        (r"\bSL\.?\s*N[O0]\.?\s*[:;\-.]*\s*" + ID_VAL, "SL NO"),
        (r"\bSN\s*[:;\-.]*\s*([A-Z0-9][A-Z0-9\-\./]{3,})", "SN"),
        (r"\bNR\.?\s*([0-9][0-9\.,]{8,})", "NR"),
        # Lookbehind sits on the N, so MODEL NO / FRAME NO / CAT NO are not serials.
        (r"(?<!CAT\. )(?<!CAT )(?<!CAT\.)(?<!ORD\. )(?<!ORD )(?<!PART )"
         r"(?<!MODEL )(?<!MODEL\.)(?<!FRAME )(?<!FRAME\.)(?<!MACHINE )(?<!MACHINE\.)"
         r"(?<!SPEC-)(?<!SPEC )(?<!SPEC\.)(?<![A-Z])"
         r"N[O0]\.?\s*[:;\-.]*\s*([A-Z0-9][A-Z0-9\-\./]{5,})", "NO:(id)"),
    ],
    "model": [
        (r"\bMODEL\s*(?:N[O0]\.?)?\s*[:;\-.]*\s*" + ID_VAL_JOIN, "MODEL"),
        (r"\bMOD\.\s*" + ID_VAL_JOIN, "MOD."),
        (r"\bSPEC\b\.?\s*[-:;\.]*\s*(?:N[O0]\.?)?\s*" + ID_VAL_JOIN, "SPEC"),
        (r"\bTYPE\s*[:;\-.]*\s*" + ID_VAL_JOIN, "TYPE"),
        (r"\bTYP\b\.?\s*" + ID_VAL_JOIN, "TYPE"),
        (r"\bMOT\.\s*" + ID_VAL_JOIN, "TYPE"),
    ],
    "equipment_id": [
        (r"\bCAT(?:ALOG)?\b\.?\s*(?:N[O0]\.?|#)?\s*[:;\-._]*\s*" + ID_VAL, "CAT NO"),
        (r"\bORD\.?\s*N[O0]\.?\s*[:;\-.H]*\s*" + ID_VAL, "ORD NO"),
        (r"\bMACHINE\s*N[O0]\.?\s*[:;\-.]*\s*" + ID_VAL, "MACHINE NO"),
        (r"\bPROD(?:UCT)?\.?\s*CODE\s*[:;\-.]*\s*([A-Z0-9][A-Z0-9\-]+(?:\s[A-Z0-9][A-Z0-9\-]+)?)", "PROD CODE"),
        (r"(?<!SER )(?<!SER\.)\bNO\.?\s+([A-Z][A-Z0-9]{2,4})\b", "NO-short"),
    ],
}

LABEL_WORDS = {
    "NO", "TYPE", "MODEL", "SPEC", "VOLTS", "FRAME", "CODE", "DATE", "SER", "SERIAL", "CAT",
    "CATALOG", "HP", "RPM", "NUMBER", "OTHER", "PART", "PHASE", "DUTY", "AMPS", "HERTZ",
    "MANUFACTURER", "DESIGN", "DESIGNATION", "ENCLOSURE", "INSULATION", "FREQUENCY",
    "SERVICE", "FACTOR", "RATING", "CLASS", "EFFICIENCY", "AMBIENT", "TEMPERATURE",
}
# OCR drops the space before the next label, gluing it onto the value
# (MMG132S-2-38FF265-ECAIN0340333060 = model + "CAT NO" + catalog number).
EMBEDDED_LABEL = re.compile(r"(?<=[A-Z0-9])(?:J?CA[TI]\.?N[O0]|PARTN[O0Q]|SERN[O0])")


def _edit1(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(a == b[:i] + b[i + 1:] for i in range(len(b)))


LABEL_FUZZY = ("TYPE", "DATE", "MODEL", "SPEC", "CAT")


def fuzzy_relabel(text):
    """One-edit OCR of a label word (TIPE, DOTE, COT) back to the real label.
    Same length only, so a value token is not rewritten into a shorter label."""
    def repl(match):
        word = match.group(0)
        for label in LABEL_FUZZY:
            if word != label and len(word) == len(label) and _edit1(word, label):
                return label
        return word
    return re.sub(r"\b[A-Z]{3,5}\b", repl, text)


def is_label_word(v):
    """Pure-letter value within 1 edit of a nameplate label word (HODEL -> MODEL)."""
    n = norm(v) or ""
    return n.isalpha() and any(_edit1(n, w) for w in LABEL_WORDS)


def _clean_id(v):
    v = v.strip().strip(".:-_/")
    v = re.sub(r"\s+", "", v) if re.fullmatch(r"[A-Z]{1,3}\s[0-9]{6,}", v) else v
    m = EMBEDDED_LABEL.search(v)
    if m and m.start() >= 3:
        v = v[:m.start()]
    return repair_id_token(v)


# "3~ Motor M3AA 80MC2" on one line. The first token must contain a digit, so
# "INDUCTION MOTOR IS 12615" (a standard number) and a product code that starts
# with a digit ("3-Motor 3GBA113001-CHB") are not models. Not applied across a
# line break, where the next line is a different field.
MOTOR_MODEL = re.compile(
    r"\bMOTORS?\b\.?\s+(?!TYPE\b|MODEL\b|SPEC\b|FRAME\b|COMPANY\b|DUTY\b|NO\b)"
    r"((?=[A-Z0-9]*\d)[A-Z][A-Z0-9]{1,5}\s+(?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{2,})"
)


def motor_model_candidates(text_lines, conf_lines):
    cands = []
    for text, conf in zip(text_lines, conf_lines):
        m = MOTOR_MODEL.search(fuzzy_relabel(text.upper()))
        if not m:
            continue
        v = _clean_id(m.group(1))
        if len(norm(v) or "") < 3 or is_label_word(v):
            continue
        cands.append({"value": v, "conf": conf, "anchored": True, "rule": "TYPE"})
    return cands


def id_candidates(field, text_lines, conf_lines):
    cands = []
    for text, conf in zip(text_lines, conf_lines):
        t = fuzzy_relabel(text.upper())
        for pat, rule in ANCHORS[field]:
            for m in re.finditer(pat, t):
                v = _clean_id(m.group(1))
                if rule == "NR":
                    v = v.replace(",", ".")
                if len(norm(v) or "") < 3 or is_label_word(v):
                    continue
                # Three digits is a rivet fragment (SER.NO. 000), not a nameplate serial.
                if field == "serial_number" and sum(ch.isdigit() for ch in v) < 4:
                    continue
                cands.append({"value": v, "conf": conf, "anchored": True, "rule": rule})
    return cands


DATE_RX = re.compile(r"\b([0-9OQIl]{1,2})\s?([A-Z]{3})\s?([0-9OQIl]{4}|[0-9OQIl]{2})\b")


def date_candidates(text_lines, conf_lines):
    cands = []
    for text, conf in zip(text_lines, conf_lines):
        t = fuzzy_relabel(text.upper())
        for m in DATE_RX.finditer(t):
            d, mon, y = m.groups()
            if mon not in MONTHS:
                continue
            d, y = to_digits(d), to_digits(y)
            if not (d.isdigit() and y.isdigit()) or not 1 <= int(d) <= 31:
                continue
            cands.append({"value": f"{d.zfill(2)}{mon}{y}", "conf": conf, "anchored": True, "rule": "DDMMMYY"})
        m = re.search(r"\bYEAR\s*[:\-]?\s*((?:19|20)[0-9]{2})\b.*?\bWEEK\s*[:\-]?\s*([0-9]{1,2})\b", t)
        if m:
            cands.append({"value": f"{m.group(1)} WEEK{m.group(2)}", "conf": conf, "anchored": True, "rule": "YEAR+WEEK"})
        for m in re.finditer(r"\b(?:DATE|YEAR|MFG|MFD)\b\.?\s*(?:CODE)?\s*[:\-]?\s*((?:19|20)[0-9]{2}(?:[.\-/][0-9]{2})?)\b", t):
            cands.append({"value": m.group(1), "conf": conf, "anchored": True, "rule": "DATE/YEAR-anchor"})
    return cands


def extract_candidates(records):
    """records -> {field: [candidate, ...]} using physical-line text."""
    lines = build_lines(records)
    texts = [ln["text"] for ln in lines]
    confs = [ln["conf"] for ln in lines]
    # Two-line window so a label at the end of one line can reach a value that
    # wrapped onto the next (common on EasyOCR fragment output).
    pairs = [texts[i] + " " + texts[i + 1] for i in range(len(texts) - 1)]
    pair_confs = [(confs[i] + confs[i + 1]) / 2 for i in range(len(texts) - 1)]

    out = {f: [] for f in FIELDS}
    for f in ("serial_number", "model", "equipment_id"):
        out[f] = id_candidates(f, texts, confs)
        if f == "model":
            out[f].extend(motor_model_candidates(texts, confs))
        if not out[f]:
            out[f] = [dict(c, rule=c["rule"] + "(2-line)") for c in id_candidates(f, pairs, pair_confs)]
    out["voltage"] = voltage_candidates_from_lines(lines)
    out["manufacture_date"] = date_candidates(texts, confs)
    # WEG-style plates print the serial as a bare digit run on the same line as the date.
    if not any(not c["rule"].startswith("NO:") for c in out["serial_number"]):
        for text, conf in zip(texts, confs):
            if not date_candidates([text], [conf]):
                continue
            for m in re.finditer(r"\b([0-9]{8,12})\b", text.upper()):
                out["serial_number"].append(
                    {"value": m.group(1), "conf": conf, "anchored": True, "rule": "NO:date-line"})
    return out


MODEL_RULE_PRIORITY = ["MODEL", "MOD.", "SPEC", "TYPE"]


def rule_tier(field, c):
    """Lower tier = stronger label evidence. Used identically by the per-engine pick
    and the ensemble: an explicit label (MODEL, SER NO, S/N) always outranks a weaker
    fallback (TYPE for model, a bare 'NO: 123456' for serial) from any engine."""
    base = c["rule"].replace("(2-line)", "")
    if field == "model":
        return MODEL_RULE_PRIORITY.index(base) if base in MODEL_RULE_PRIORITY else 99
    if field == "serial_number":
        return 1 if base.startswith("NO:") else 0
    return 0


def no_duplicate_rows(value):
    rows = value.split(",")
    return len(rows) == len(set(rows))


def pick_within_engine(cands, field):
    """Single-engine best value (used for the per-engine rows of the report)."""
    if not cands:
        return None
    top = min(rule_tier(field, c) for c in cands)
    cands = [c for c in cands if rule_tier(field, c) == top]
    if field == "voltage":
        table = [c for c in cands if c["rule"].startswith("voltage-column")]
        if table:
            return max(table, key=lambda c: (n_rows(c), c["conf"]))["value"]
        # "VOLTS 230/460" is a real label. A bare "380 V" elsewhere on the plate
        # must not outrank it just because that box has higher OCR confidence.
        labelled = [c for c in cands if c["rule"] == "VOLT-anchor+lexicon"]
        if labelled:
            return max(labelled, key=lambda c: (n_rows(c), c["conf"]))["value"]
        pair = [c for c in cands if c["rule"] == "VOLT-conn-pair"]
        if pair:
            return pair[0]["value"]
    return max(cands, key=lambda c: c["conf"])["value"]


def n_rows(c):
    return c["value"].count(",") + 1


def resolve_cross_field(fields, cands):
    """Schema conventions of this dataset's ground truth, applied identically to
    every engine and to the ensemble:
      * a TYPE value is the model only when no MODEL/MOD/SPEC label exists;
        when a MODEL label exists, TYPE is the equipment identifier
      * equipment_id falls back to the model/catalog number when the plate has
        no separate CAT/ORD/MACHINE number."""
    model_rules = {c["rule"].replace("(2-line)", "") for c in cands.get("model", [])}
    if fields.get("equipment_id") is None:
        type_vals = [c for c in cands.get("model", []) if c["rule"].startswith("TYPE")]
        if type_vals and model_rules - {"TYPE"}:
            fields["equipment_id"] = max(type_vals, key=lambda c: c["conf"])["value"]
        elif fields.get("model") is not None:
            fields["equipment_id"] = fields["model"]
    return fields


def extract_fields(records):
    cands = extract_candidates(records)
    fields = {f: pick_within_engine(cands[f], f) for f in FIELDS}
    return resolve_cross_field(fields, cands), cands
