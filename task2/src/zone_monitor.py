#!/usr/bin/env python3
"""task2/src/zone_monitor.py

Task 1 zone pipeline with the Task 2 helmet engine.
Video -> YOLO person detection -> BoT-SORT -> foot-point zone and idle rules.
A second model finds head and helmet boxes. Each box is bound to a tracked
person only when its centre sits inside that person's head zone (top half of
the person box). Boxes bound to nobody, such as a helmet-shaped object on the
crane, are discarded and never drawn. A person with a bound helmet is green.
Anyone else gets a red ID and a PPE_VIOLATION. A live helmet / no-helmet
count is drawn on the right, after the zone and idle banner, so that banner
cannot cover it.

Usage:
    python zone_monitor.py --config ../configs/zone_cctv_cuplock_ppe.json
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


def foot_point(xyxy):
    """Bottom-centre of a bounding box: (x1,y1,x2,y2) -> (cx, y2)."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


def boxes_intersect(a, b):
    """True when two xyxy boxes overlap on both axes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def assign_ppe_to_persons(persons, detections, head_zone_frac=0.5):
    """Give each head/helmet detection to at most one tracked person.

    persons: list of (track_id, xyxy). detections: list of (name, conf, xyxy).
    A detection belongs to a person only when its CENTRE lies inside the person
    box and in the top head_zone_frac of that box height, where a head sits.
    Touching or overlapping a person box is not enough. Detections that fit no
    person (a helmet-shaped object on the crane, for example) are dropped and
    never drawn. If several person boxes contain the centre, the smallest wins.
    Returns {track_id: [detections]}.
    """
    assigned = {tid: [] for tid, _ in persons}
    for det in detections:
        _name, _conf, box = det
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        best_tid, best_area = None, None
        for tid, p in persons:
            if not (p[0] <= cx <= p[2] and p[1] <= cy <= p[1] + head_zone_frac * (p[3] - p[1])):
                continue
            area = (p[2] - p[0]) * (p[3] - p[1])
            if best_area is None or area < best_area:
                best_tid, best_area = tid, area
        if best_tid is not None:
            assigned[best_tid].append(det)
    return assigned


class LabelDebounce:
    """Hold a track's helmet / no helmet label until a change has lasted `frames` frames in a row.

    The helmet detector drops out for a few frames at a time. Without this, every dropout opens and closes
    a PPE_VIOLATION. A new track has no label (None) until one label has held for `frames` frames, so a track is
    never called no helmet on the strength of a single missed detection. Events therefore start and stop up to
    `frames` frames after the raw change.
    """

    def __init__(self, frames):
        self.frames = max(1, int(frames))
        self.state = {}  # tid -> [stable label or None, candidate label, consecutive count]

    def update(self, tid, raw):
        s = self.state.setdefault(tid, [None, raw, 0])
        if raw == s[0]:
            s[1], s[2] = raw, 0
        else:
            s[2] = s[2] + 1 if raw == s[1] else 1
            s[1] = raw
            if s[2] >= self.frames:
                s[0], s[2] = raw, 0
        return s[0]


def ppe_label_for_person(assigned_detections):
    """helmet if any helmet is assigned to this person, otherwise no helmet.

    A person with only a head box assigned, or nothing assigned, is no helmet.
    """
    return "helmet" if any(name == "helmet" for name, _c, _b in assigned_detections) else "no helmet"


def load_config(path):
    with open(path) as f:
        return json.load(f)


def resolve_from_config(base_dir, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def draw_zone(frame, polygon_pts, color=(0, 0, 255), alpha=0.18):
    overlay = frame.copy()
    pts = np.array(polygon_pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    return frame


def fit_text(text, max_width, font, scale, thickness):
    if max_width <= 0:
        return ""
    while text and cv2.getTextSize(text, font, scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text


COUNTER_FONT = cv2.FONT_HERSHEY_SIMPLEX
COUNTER_SCALE = 1.05
COUNTER_THICK = 2
COUNTER_GAP = 14
COUNTER_MARGIN = 18


def counter_lines(helmet_n, no_helmet_n):
    return [
        (f"HELMET {helmet_n}", (0, 220, 0)),
        (f"NO HELMET {no_helmet_n}", (0, 0, 255)),
    ]


def counter_panel_size(helmet_n, no_helmet_n):
    lines = counter_lines(helmet_n, no_helmet_n)
    sizes = [cv2.getTextSize(text, COUNTER_FONT, COUNTER_SCALE, COUNTER_THICK) for text, _color in lines]
    text_w = max(size[0][0] for size in sizes)
    line_h = max(size[0][1] for size in sizes)
    panel_w = text_w + COUNTER_MARGIN * 2
    panel_h = COUNTER_MARGIN * 2 + line_h * len(lines) + COUNTER_GAP * (len(lines) - 1)
    return panel_w, panel_h, line_h


def draw_right_counter(frame, helmet_n, no_helmet_n):
    """Live counts on the right. Drawn after the zone/idle banner so it stays visible."""
    lines = counter_lines(helmet_n, no_helmet_n)
    panel_w, panel_h, line_h = counter_panel_size(helmet_n, no_helmet_n)
    frame_w = frame.shape[1]
    x0 = frame_w - panel_w
    cv2.rectangle(frame, (x0, 0), (frame_w, panel_h), (0, 0, 0), -1)
    y = COUNTER_MARGIN + line_h
    for text, color in lines:
        cv2.putText(frame, text, (x0 + COUNTER_MARGIN, y), COUNTER_FONT, COUNTER_SCALE, color, COUNTER_THICK, cv2.LINE_AA)
        y += line_h + COUNTER_GAP
    return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../configs/zone_cctv_cuplock_ppe.json")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent

    source = resolve_from_config(base_dir, cfg["video"]["source"])
    task_root = Path(__file__).resolve().parent.parent
    out_dir = task_root / "outputs" / "6_ppe_violation"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = (task_root / cfg["video"]["output"]).resolve()
    events_path = (task_root / cfg["video"]["events_json"]).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    zone_poly_pts = cfg["zone"]["polygon"]
    zone_poly = Polygon(zone_poly_pts)
    idle_px = cfg["idle"]["pixel_threshold"]
    idle_sec = cfg["idle"]["seconds_threshold"]
    idle_radius_mode = cfg["idle"].get("radius_mode", "fixed")
    idle_box_width_fraction = cfg["idle"].get("box_width_fraction", 0.25)
    zone_exit_grace_sec = cfg["zone"].get("exit_grace_sec", 0.5)
    conf_th = cfg["detection"]["conf_threshold"]
    classes = cfg["detection"]["classes"]
    tracker_cfg = cfg["tracker"]["type"]
    model_name = cfg["detection"]["model"]
    imgsz = cfg["detection"].get("imgsz", 640)
    nms_iou = cfg["detection"].get("iou", 0.7)

    tracker_candidate = resolve_from_config(base_dir, tracker_cfg)
    if tracker_candidate.is_file():
        tracker_cfg = str(tracker_candidate)

    model_candidate = resolve_from_config(base_dir, model_name)
    if model_candidate.is_file():
        model_name = str(model_candidate)

    ppe_cfg = cfg.get("ppe")
    ppe_model = None
    if ppe_cfg:
        ppe_path = resolve_from_config(base_dir, ppe_cfg["model"])
        print(f"[task2] loading PPE engine {ppe_path} ...")
        ppe_model = YOLO(str(ppe_path), task="detect")
        print(f"[task2] PPE classes: {ppe_model.names}")

    print(f"[task2] loading model {model_name} ...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps_in, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer: {output_video}")

    track_in_zone = {}
    track_last_seen_frame = {}
    track_last_box = {}
    track_pos_history = {}
    track_idle_fired = {}
    open_zone_event = {}
    open_idle_event = {}
    open_ppe_event = {}
    events = []

    zone_exit_grace_frames = max(0, round(zone_exit_grace_sec * fps_in))

    debounce = LabelDebounce(round(ppe_cfg.get("debounce_sec", 1.0) * fps_in)) if ppe_cfg else None

    frame_idx = 0
    t_start = time.time()
    hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    results_gen = model.track(
        source=str(source),
        conf=conf_th,
        classes=classes,
        imgsz=imgsz,
        iou=nms_iou,
        tracker=tracker_cfg,
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results_gen:
        frame = result.orig_img.copy()
        t_sec = frame_idx / fps_in

        ppe_dets = []
        if ppe_model is not None:
            ppe_result = ppe_model.predict(
                result.orig_img,
                imgsz=ppe_cfg.get("imgsz", 640),
                conf=ppe_cfg.get("conf", 0.35),
                device=0,
                verbose=False,
            )[0]
            if ppe_result.boxes is not None and len(ppe_result.boxes):
                for cls_id, score, box in zip(ppe_result.boxes.cls, ppe_result.boxes.conf, ppe_result.boxes.xyxy):
                    name = ppe_model.names[int(cls_id)]
                    ppe_dets.append((name, float(score), [float(v) for v in box]))

        frame = draw_zone(frame, zone_poly_pts)

        boxes = result.boxes
        active_ids_this_frame = set()
        helmet_n = 0
        no_helmet_n = 0

        # Bind head/helmet detections to tracked persons. Unbound ones are discarded, not drawn.
        ppe_assigned = {}
        if ppe_model is not None and boxes is not None and boxes.id is not None:
            persons = list(zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist()))
            ppe_assigned = assign_ppe_to_persons(persons, ppe_dets, ppe_cfg.get("head_zone_frac", 0.5))
            for dets in ppe_assigned.values():
                for name, _score, bx in dets:
                    cv2.rectangle(frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])),
                                  (0, 0, 255) if name == "head" else (255, 180, 0), 2)

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for tid, xyxy, conf in zip(ids, xyxys, confs):
                active_ids_this_frame.add(tid)
                fx, fy = foot_point(xyxy)
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                box_w = x2 - x1

                inside = zone_poly.contains(Point(fx, fy))
                was_inside = track_in_zone.get(tid, False)
                if inside and not was_inside:
                    ev = {
                        "event": "ZONE_ENTRY",
                        "track_id": tid,
                        "start_frame": frame_idx,
                        "start_sec": round(t_sec, 2),
                        "stop_frame": None,
                        "stop_sec": None,
                        "start_box": [x1, y1, x2, y2],
                        "stop_box": None,
                    }
                    events.append(ev)
                    open_zone_event[tid] = ev
                elif not inside and was_inside:
                    ev = open_zone_event.pop(tid, None)
                    if ev is not None:
                        ev["stop_frame"] = frame_idx
                        ev["stop_sec"] = round(t_sec, 2)
                        ev["stop_box"] = [x1, y1, x2, y2]
                track_in_zone[tid] = inside
                track_last_seen_frame[tid] = frame_idx
                track_last_box[tid] = [x1, y1, x2, y2]

                hist = track_pos_history.setdefault(tid, deque())
                hist.append((frame_idx, fx, fy))
                while len(hist) > 1 and (frame_idx - hist[1][0]) / fps_in >= idle_sec:
                    hist.popleft()

                anchor_frame, ax, ay = hist[0]
                window_covered = (frame_idx - anchor_frame) / fps_in >= idle_sec
                radius = idle_box_width_fraction * box_w if idle_radius_mode == "box_width_fraction" else idle_px
                total_disp = ((fx - ax) ** 2 + (fy - ay) ** 2) ** 0.5
                is_idle = window_covered and total_disp < radius

                if is_idle and not track_idle_fired.get(tid, False):
                    ev = {
                        "event": "IDLE_ALERT",
                        "track_id": tid,
                        "start_frame": frame_idx,
                        "start_sec": round(t_sec, 2),
                        "stop_frame": None,
                        "stop_sec": None,
                        "start_box": [x1, y1, x2, y2],
                        "stop_box": None,
                    }
                    events.append(ev)
                    open_idle_event[tid] = ev
                    track_idle_fired[tid] = True
                elif not is_idle and track_idle_fired.get(tid, False):
                    ev = open_idle_event.pop(tid, None)
                    if ev is not None:
                        ev["stop_frame"] = frame_idx
                        ev["stop_sec"] = round(t_sec, 2)
                        ev["stop_box"] = [x1, y1, x2, y2]
                    track_idle_fired[tid] = False

                ppe_label = None
                if ppe_model is not None:
                    raw_label = ppe_label_for_person(ppe_assigned.get(tid, []))
                    ppe_label = debounce.update(tid, raw_label)
                    violating = ppe_label == "no helmet"
                    if violating and tid not in open_ppe_event:
                        ev = {
                            "event": "PPE_VIOLATION",
                            "track_id": tid,
                            "label": ppe_label,
                            "start_frame": frame_idx,
                            "start_sec": round(t_sec, 2),
                            "stop_frame": None,
                            "stop_sec": None,
                            "start_box": [x1, y1, x2, y2],
                            "stop_box": None,
                        }
                        events.append(ev)
                        open_ppe_event[tid] = ev
                    elif not violating and tid in open_ppe_event:
                        ev = open_ppe_event.pop(tid)
                        ev["stop_frame"] = frame_idx
                        ev["stop_sec"] = round(t_sec, 2)
                        ev["stop_box"] = [x1, y1, x2, y2]
                    if ppe_label == "helmet":
                        helmet_n += 1
                    elif ppe_label == "no helmet":
                        no_helmet_n += 1

                if ppe_label == "helmet":
                    box_color = (0, 180, 0)
                elif ppe_label == "no helmet":
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 140, 255) if inside else (60, 220, 60)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                id_text = f"ID {tid}" if ppe_label is None else f"ID {tid} {ppe_label}"
                cv2.putText(frame, id_text, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                cv2.circle(frame, (int(fx), int(fy)), 5, (255, 0, 255), -1)

        for tid in set(track_in_zone.keys()) | set(open_idle_event.keys()) | set(open_ppe_event.keys()):
            if tid not in active_ids_this_frame:
                last_seen = track_last_seen_frame.get(tid, -1)
                if frame_idx - last_seen > zone_exit_grace_frames:
                    last_box = track_last_box.get(tid)
                    ev = open_zone_event.pop(tid, None)
                    if ev is not None:
                        ev["stop_frame"] = last_seen
                        ev["stop_sec"] = round(last_seen / fps_in, 2)
                        ev["stop_box"] = last_box
                    ev = open_idle_event.pop(tid, None)
                    if ev is not None:
                        ev["stop_frame"] = last_seen
                        ev["stop_sec"] = round(last_seen / fps_in, 2)
                        ev["stop_box"] = last_box
                    ev = open_ppe_event.pop(tid, None)
                    if ev is not None:
                        ev["stop_frame"] = last_seen
                        ev["stop_sec"] = round(last_seen / fps_in, 2)
                        ev["stop_box"] = last_box
                    track_in_zone.pop(tid, None)
                    track_last_seen_frame.pop(tid, None)
                    track_last_box.pop(tid, None)
                    track_pos_history.pop(tid, None)
                    track_idle_fired.pop(tid, None)

        panel_w, _panel_h, _line_h = counter_panel_size(helmet_n, no_helmet_n)
        panel_left = w - panel_w

        status_bits = []
        for tid in sorted(set(open_zone_event) | set(open_idle_event)):
            tags = []
            if tid in open_zone_event:
                tags.append("ZONE_ENTRY")
            if tid in open_idle_event:
                tags.append("IDLE_ALERT")
            status_bits.append(f"id={tid}:{'+'.join(tags)}")
        if status_bits:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            thick = 2
            status_text = fit_text(" | ".join(status_bits), panel_left - 20, font, scale, thick)
            cv2.rectangle(frame, (0, 0), (panel_left, 36), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (10, 24), font, scale, (0, 255, 255), thick)

        draw_right_counter(frame, helmet_n, no_helmet_n)

        counter_text = f"t={t_sec:.1f}s  frame={frame_idx}/{n_frames_total}"
        counter_font_scale = 1.0
        counter_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            counter_text, cv2.FONT_HERSHEY_SIMPLEX, counter_font_scale, counter_thickness)
        margin = 14
        text_x = w - text_w - margin
        text_y = h - margin
        cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10),
                      (w, text_y + baseline + 6), (0, 0, 0), -1)
        cv2.putText(frame, counter_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, counter_font_scale, (255, 255, 255),
                    counter_thickness, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    t_elapsed = time.time() - t_start
    avg_fps = frame_idx / t_elapsed if t_elapsed > 0 else 0.0

    last_frame = frame_idx - 1
    last_sec = round(last_frame / fps_in, 2) if last_frame >= 0 else 0.0
    for tid, ev in list(open_zone_event.items()):
        ev["stop_frame"] = last_frame
        ev["stop_sec"] = last_sec
        ev["stop_box"] = track_last_box.get(tid, ev["start_box"])
    for tid, ev in list(open_idle_event.items()):
        ev["stop_frame"] = last_frame
        ev["stop_sec"] = last_sec
        ev["stop_box"] = track_last_box.get(tid, ev["start_box"])
    for tid, ev in list(open_ppe_event.items()):
        ev["stop_frame"] = last_frame
        ev["stop_sec"] = last_sec
        ev["stop_box"] = track_last_box.get(tid, ev["start_box"])

    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"[task2] frames processed: {frame_idx}")
    print(f"[task2] avg processing FPS: {avg_fps:.2f}  (hardware: {hw_name})")
    print(f"[task2] events: {len(events)} -> {events_path}")
    print(f"[task2] output video -> {output_video}")

    log_path = out_dir / "run_log_ppe.txt"
    with open(log_path, "w") as f:
        f.write(f"frames={frame_idx} avg_fps={avg_fps:.2f} hw={hw_name} "
                f"events={len(events)} source={source.name}\n")


if __name__ == "__main__":
    main()
