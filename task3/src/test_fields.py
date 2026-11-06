"""Unit tests for the shared field extractor (run: python -m pytest task3/src/test_fields.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fields import decode_voltage_run, repair_id_token, extract_fields, build_lines


def rec(text, y, x=0, conf=0.9, h=20):
    return {"text": text, "conf": conf, "box": [x, y, x + 10 * len(text), y + h]}


def test_voltage_lexicon():
    assert decode_voltage_run("230/460") == "230/460"
    assert decode_voltage_run("2307460") == "230/460"          # '/' misread as 7
    assert decode_voltage_run("72301460") == "230/460"         # leading junk + '/' as 1
    assert decode_voltage_run("208-2307460") == "208-230/460"
    assert decode_voltage_run("'208-2307460") == "208-230/460"
    assert decode_voltage_run("115/23OV") == "115/230"
    assert decode_voltage_run("2401415") == "240/415"
    assert decode_voltage_run("415") == "415"
    assert decode_voltage_run("723") is None                   # not a standard voltage
    assert decode_voltage_run("1765") is None                  # rpm, not a voltage
    assert decode_voltage_run("2802480") == "280/480"          # '/' misread as 2
    assert decode_voltage_run("380-415\u0394/660-690Y") == "380-415/660-690"
    assert decode_voltage_run("380 4154/660-690x") == "380/415/660-690"
    assert decode_voltage_run("400/690!") == "400/690"


def test_label_words_and_glued_labels():
    from fields import is_label_word, _clean_id
    assert is_label_word("HODEL") and is_label_word("NUMBER")
    assert not is_label_word("SHDC") and not is_label_word("RGZESD")
    assert _clean_id("MMG132S-2-38FF265-ECAIN0340333060") == "MMG132S-2-38FF265-E"


def test_id_repair():
    assert repair_id_token("1LAO2864SE41") == "1LA02864SE41"
    assert repair_id_token("MODEL") == "MODEL"
    assert repair_id_token("DI0034PCI") == "DI0034PCI"


def test_line_grouping_and_anchors():
    recs = [rec("SER", 100, 0), rec("NO.", 100, 50), rec("W1012031368", 100, 100),
            rec("VOLTS", 140, 0), rec("230/460", 140, 80),
            rec("MODEL:", 180, 0), rec("FM 140156", 180, 80), rec("TYPE: SHDC", 220, 0)]
    lines = build_lines(recs)
    assert [ln["text"] for ln in lines][0] == "SER NO. W1012031368"
    fields, _ = extract_fields(recs)
    assert fields["serial_number"] == "W1012031368"
    assert fields["voltage"] == "230/460"
    assert fields["model"] == "FM140156"
    assert fields["equipment_id"] == "SHDC"


def test_voltage_table_column():
    recs = [rec("380/660", 100), rec("400/690", 130), rec("415/ -", 160), rec("440/ -", 190), rec("460/ -", 220)]
    fields, _ = extract_fields(recs)
    assert fields["voltage"] == "380/660,400/690,415,440,460"
    # per-cell boxes (PaddleOCR style): star/delta pairs merged by sqrt(3) ratio, rpm never merged
    recs = [rec("380", 100, 0), rec("660", 100, 60), rec("1460", 100, 120),
            rec("400", 130, 0), rec("690", 130, 60), rec("1465", 130, 120),
            rec("415", 160, 0), rec("1470", 160, 120), rec("440", 190, 0), rec("1760", 190, 120),
            rec("460", 220, 0), rec("1765", 220, 120)]
    fields, _ = extract_fields(recs)
    assert fields["voltage"] == "380/660,400/690,415,440,460"


def test_value_before_label():
    """Plates print the value in front of the unit or with a space in the code:
    '230 V', not 'V 230'; 'TYPE MY 7124', not 'TYPE MY7124'."""
    fields, _ = extract_fields([
        rec("TYPE MY 7124", 0), rec("IP.", 0, 140), rec("54", 0, 180),
        rec("230 V 2.82 A", 40), rec("20", 40, 160), rec("450", 40, 200),
        rec("No. 1202190103", 80),
    ])
    assert fields["model"] == "MY 7124"
    assert fields["equipment_id"] == "MY 7124"
    assert fields["voltage"] == "230"
    assert fields["serial_number"] == "1202190103"
    fields, _ = extract_fields([rec("115V 230v", 5)])
    assert fields["voltage"] == "115/230"
    fields, _ = extract_fields([rec("16 µF / 400V", 5)])
    assert fields["voltage"] is None
    fields, _ = extract_fields([rec("450 V", 5)])
    assert fields["voltage"] is None
    fields, _ = extract_fields([rec("VOLTS 230/460", 0, conf=0.8), rec("380 V", 40, conf=0.99)])
    assert fields["voltage"] == "230/460"


def test_label_variants_from_raw_ocr():
    fields, _ = extract_fields([rec("TIPE GSA632-4", 0), rec("Dote 2013.03", 30),
                                   rec("Cot. No. HHI150-12-447T", 60), rec("No. 2W057K40-006", 90)])
    assert fields["model"] == "GSA632-4"
    assert fields["manufacture_date"] == "2013.03"
    assert fields["equipment_id"] == "HHI150-12-447T"
    assert fields["serial_number"] == "2W057K40-006"
    fields, _ = extract_fields([rec("Typ FH57B DV112M4/BMG", 0), rec("Nr. 96.01012530.01.0001.05.06", 30)])
    assert fields["model"] == "FH57B DV112M4/BMG"
    assert fields["serial_number"] == "96.01012530.01.0001.05.06"
    fields, _ = extract_fields([rec("No.3G3C2120397154", 0), rec("Prod. Code 3GAA081330-ADK, 008,436", 30),
                                   rec("690Y", 60), rec("400D", 90), rec("415D", 120)])
    assert fields["serial_number"] == "3G3C2120397154"
    assert fields["equipment_id"] == "3GAA081330-ADK"
    assert fields["voltage"] == "690,400,415"
    fields, _ = extract_fields([rec("17ABR2019 1048518705", 0)])
    assert fields["serial_number"] == "1048518705"
    fields, _ = extract_fields([rec("HP75VOLTS 230/460 171/85.6", 0)])
    assert fields["voltage"] == "230/460"
    fields, _ = extract_fields([rec("415D+/-10%", 0)])
    assert fields["voltage"] == "415"
    fields, _ = extract_fields([rec("MODEL 5KS184BCT205B HP 5 PHASE NO E829", 0)])
    assert fields["model"] == "5KS184BCT205B"
    assert fields["equipment_id"] == "E829"
    fields, _ = extract_fields([rec("3~MOT. 1LG4 206-2AA60-Z 200L", 0)])
    assert fields["model"] == "1LG4 206-2AA60-Z"
    fields, _ = extract_fields([rec("No:20190906385", 0)])
    assert fields["serial_number"] == "20190906385"
    fields, _ = extract_fields([rec("MODEL NO. DI0034PCI", 0)])
    assert fields["model"] == "DI0034PCI"
    assert fields["serial_number"] is None
    fields, _ = extract_fields([rec("Nr. 96.01012530.01,0001,05.06", 0)])
    assert fields["serial_number"] == "96.01012530.01.0001.05.06"
    fields, _ = extract_fields([rec("SERNO. 0OO", 0), rec("2554, ONP NETWORKS", 30)])
    assert fields["serial_number"] is None
    fields, _ = extract_fields([rec("Ser No.:FM 14015624040001", 0), rec("MODEL:FM 140156", 30)])
    assert fields["serial_number"] == "FM14015624040001"
    assert fields["model"] == "FM140156"
    fields, _ = extract_fields([rec("MACHINE NO: 0.75KNE4FLG", 0)])
    assert fields["equipment_id"] == "0.75KNE4FLG"
    assert fields["serial_number"] is None
    fields, _ = extract_fields([rec("SPEC-NO. A36-8063-4076", 0)])
    assert fields["model"] == "A36-8063-4076"
    assert fields["serial_number"] is None
    fields, _ = extract_fields([rec("V230 460 A 11.8/5.90", 0), rec("USABLE @208V.13.1 A", 30),
                                   rec("50Hz 380V 7.14A", 60)])
    assert fields["voltage"] == "230/460"
    fields, _ = extract_fields([rec("3~ Motor M3AA 80MC2 IMB34/IM2101 2021", 0),
                                   rec("Prod. Code 3GAA081330-ADK", 30)])
    assert fields["model"] == "M3AA 80MC2"
    assert fields["equipment_id"] == "3GAA081330-ADK"
    fields, _ = extract_fields([rec("3-Motor 3GBA113001-CHB Amb.50", 0)])
    assert fields["model"] is None
    fields, _ = extract_fields([rec("400△|50", 0), rec("690Y", 30)])
    assert fields["voltage"] == "400/690"
    fields, _ = extract_fields([rec("VOLTS 230/460", 0), rec("400△|50", 30), rec("690Y", 60)])
    assert fields["voltage"] == "230/460"


def test_dates():
    fields, _ = extract_fields([rec("O3FEV1O 0000000000", 10)])
    assert fields["manufacture_date"] == "03FEV10"
    fields, _ = extract_fields([rec("YEAR 2001 WEEK 28", 10)])
    assert fields["manufacture_date"] == "2001 WEEK28"
