#!/usr/bin/env python3
"""task1/src/zone_monitor.py

Object detection + multi-object tracking + zone/idle rule engine on video.
Pipeline: video -> YOLO person detection -> BoT-SORT tracking (stable IDs) ->
foot-point zone-entry rule + idle rule -> annotated output.mp4 + events.json.

Usage:
    python zone_monitor.py --config ../../configs/task1_zone_cctv_cuplock.json
"""
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


def foot_point(xyxy):
    """Bottom-centre of a bounding box: (x1,y1,x2,y2) -> (cx, y2)."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


def box_center_inside(inner, outer):
    """True when the centre of `inner` lies inside `outer`. Both are xyxy."""
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    x1, y1, x2, y2 = outer
    return x1 <= cx <= x2 and y1 <= cy <= y2


def load_config(path):
    with open(path) as f:
        return json.load(f)


def live_preview_enabled(task_label):
    """True when a display exists and this OpenCV build implements highgui.

    opencv-python-headless (and a missing DISPLAY) make cv2.imshow raise.
    Callers should keep writing the output video either way.
    """
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    gui_backend = ""
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            gui_backend = stripped.split(":", 1)[1].strip()
            break
    if has_display and gui_backend.upper() not in ("", "NONE"):
        return True
    print(
        f"[{task_label}] live preview skipped "
        "(no display or OpenCV highgui unavailable)"
    )
    return False


def draw_zone(frame, polygon_pts, color=(0, 0, 255), alpha=0.18):
    overlay = frame.copy()
    pts = np.array(polygon_pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/task1_zone_cctv_cuplock.json")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent  # configs/ or task1/

    source = (base_dir / cfg["video"]["source"]).resolve()
    task_root = Path(__file__).resolve().parent.parent
    out_dir = task_root / "outputs"
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

    # Resolve tracker yaml path relative to config file's directory
    tracker_candidate = (base_dir / tracker_cfg).resolve()
    if tracker_candidate.is_file():
        tracker_cfg = str(tracker_candidate)

    print(f"[task1] loading model {model_name} ...")
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

    # per-track state
    track_in_zone = {}
    track_last_seen_frame = {}
    track_last_box = {}
    track_pos_history = {}
    track_idle_fired = {}
    open_zone_event = {}
    open_idle_event = {}
    events = []

    zone_exit_grace_frames = max(0, round(zone_exit_grace_sec * fps_in))

    frame_idx = 0
    t_start = time.time()
    hw_name = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            hw_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

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

    show_live = live_preview_enabled("task1")
    if show_live:
        try:
            cv2.namedWindow("Task 1 Live", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_live = False
            print(
                "[task1] live preview skipped "
                "(no display or OpenCV highgui unavailable)"
            )

    for result in results_gen:
        frame = result.orig_img.copy()
        t_sec = frame_idx / fps_in

        frame = draw_zone(frame, zone_poly_pts)

        boxes = result.boxes
        active_ids_this_frame = set()

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for tid, xyxy, conf in zip(ids, xyxys, confs):
                active_ids_this_frame.add(tid)
                fx, fy = foot_point(xyxy)
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                box_w = x2 - x1

                # ZONE_ENTRY rule
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

                # IDLE_ALERT rule
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

                # Draw box, id, foot point
                box_color = (0, 140, 255) if inside else (60, 220, 60)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                cv2.circle(frame, (int(fx), int(fy)), 5, (255, 0, 255), -1)

        # Evict stale tracks
        for tid in set(track_in_zone.keys()) | set(open_idle_event.keys()):
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
                    track_in_zone.pop(tid, None)
                    track_last_seen_frame.pop(tid, None)
                    track_last_box.pop(tid, None)
                    track_pos_history.pop(tid, None)
                    track_idle_fired.pop(tid, None)

        # Live status banner
        status_bits = []
        for tid in sorted(set(open_zone_event) | set(open_idle_event)):
            tags = []
            if tid in open_zone_event:
                tags.append("ZONE_ENTRY")
            if tid in open_idle_event:
                tags.append("IDLE_ALERT")
            status_bits.append(f"id={tid}:{'+'.join(tags)}")
        if status_bits:
            status_text = " | ".join(status_bits)
            cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)

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

        # LIVE WINDOW FOR DEMO
        if show_live:
            try:
                cv2.imshow("Task 1 Live", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                show_live = False
                print(
                    "[task1] live preview skipped "
                    "(no display or OpenCV highgui unavailable)"
                )

        writer.write(frame)
        frame_idx += 1

    writer.release()
    if show_live:
        cv2.destroyAllWindows()

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

    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"[task1] frames processed: {frame_idx}")
    print(f"[task1] avg processing FPS: {avg_fps:.2f}  (hardware: {hw_name})")
    print(f"[task1] events: {len(events)} -> {events_path}")
    print(f"[task1] output video -> {output_video}")

    log_path = out_dir / "run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"frames={frame_idx} avg_fps={avg_fps:.2f} hw={hw_name} "
                f"events={len(events)} source={source.name}\n")


if __name__ == "__main__":
    main()