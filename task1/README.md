# Task 1 -- Object Detection and Zone Monitoring

Video analytics pipeline: YOLO person detection -> BoT-SORT tracking (stable
IDs) -> configurable zone-entry and idle rules -> annotated video + events.json.

## Pipeline

```
video -> YOLO11n (person, class 0) -> BoT-SORT tracker (ReID-capable) ->
foot-point-in-polygon rule (ZONE_ENTRY) + fixed-window-displacement rule
(IDLE_ALERT) -> outputs/output_cctv_cuplock.mp4 + outputs/events_cctv_cuplock.json
```

## Data

**Real CCTV footage**, not stock video: `Cuplock-1_CCTV_...mp4`, 120s clip
(first 2 minutes of a 480s recording), 2560x1440, HEVC, ~20fps, overhead
static camera in a workshop/fabrication floor with a mobile crane. Shows 2-3
workers moving, standing still, and working near the crane -- satisfies the
assessment's "at least 1 minute, several people, industrial setting" data
requirement.

**Source is not in this repo** (115MB, over the 100MB-per-file limit): it
lives at an absolute path configured in `task1_zone_cctv_cuplock.json`
(`video.source`). To run this on another machine, edit that one path to
point at your own local copy of the clip, or any other clip meeting the
"1 minute, several people, industrial setting" requirement -- nothing else
in the pipeline is hardcoded to this specific video.

An earlier attempt used 5 concatenated Mixkit stock clips (required camera
stabilization, still panned) plus a static Pixabay forklift clip. Both were
replaced once real CCTV footage became available -- it is genuinely static,
has no licensing ambiguity for a CCTV-use-case demo, and surfaced a real bug
(see HANDOVER.md) that the stock footage never would have. Those files have
been deleted from the repo; the finding they produced is preserved in writing.

## How to run

```bash
cd ai-cv-assessment
source .venv/bin/activate
cd task1/src
python zone_monitor.py --config ../../configs/task1_zone_cctv_cuplock.json
```

Outputs: `task1/outputs/output_cctv_cuplock.mp4` (annotated),
`task1/outputs/events_cctv_cuplock.json` (events), console + overwritten
`task1/outputs/run_log.txt` for FPS/hardware (one line, latest run only).

Last verified run: 2391 frames, 29.08 avg FPS, NVIDIA GeForce RTX 5070 Ti,
31 events written. The latest run, with the live preview window open, logged 18.24 avg FPS
(`task1/outputs/run_log.txt`); the gap is not investigated.

PPE (Task 2 requirement 6) is run entirely from Task 2, against the Task 2
TensorRT engine -- see
[task2/README.md](../task2/README.md#ppe-on-the-cctv-clip-requirement-6) for
the command, config and outputs. The config used is
`task2/configs/zone_cctv_cuplock_ppe.json`, writing to
`task2/outputs/6_ppe_violation/` (video, events.json, run log) -- not
duplicated here to avoid two copies of a 250MB video drifting out of sync.

## Config (`configs/task1_zone_cctv_cuplock.json`)

Everything the spec requires to be configurable, is:


| Key                        | Value                                       | Purpose                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `zone.polygon`             | 4-point polygon, auto-generated (see below) | Restricted-area definition                                                                                                                                                                                                     |
| `zone.exit_grace_sec`      | 2.0                                         | Seconds a track can be briefly undetected before its zone/idle state is force-closed. Frame-count based (`round(exit_grace_sec * source_fps)`), never wall-clock `time.time()` -- deterministic regardless of processing speed |
| `idle.pixel_threshold`     | 25                                          | Flat-pixel idle radius, used when `radius_mode="fixed"`                                                                                                                                                                        |
| `idle.seconds_threshold`   | 5.0                                         | IDLE_ALERT window length                                                                                                                                                                                                       |
| `idle.radius_mode`         | `"box_width_fraction"`                      | Idle radius scales with the person's current box width instead of a flat pixel count (see HANDOVER.md for why)                                                                                                                 |
| `idle.box_width_fraction`  | 0.25                                        | Radius = 0.25 x box width when `radius_mode="box_width_fraction"`                                                                                                                                                              |
| `detection.imgsz`          | 1088                                        | YOLO inference resolution (native clip res is 2560x1440)                                                                                                                                                                       |
| `detection.iou`            | 0.5                                         | NMS IoU threshold -- lowered from Ultralytics' 0.7 default to kill duplicate overlapping boxes on the same person (see HANDOVER.md)                                                                                            |
| `detection.conf_threshold` | 0.5                                         | Detection confidence floor                                                                                                                                                                                                     |
| `tracker.type`             | `botsort_tuned.yaml`                        | Tuned BoT-SORT config (`task1/src/botsort_tuned.yaml`), raised thresholds + `track_buffer: 150` frames + ReID enabled                                                                                                          |




### Zone polygon: data-driven, not hand-picked

The polygon in the config (`[[0,528],[992,528],[992,1440],[0,1440]]`) was
generated by a one-time script (`detect_crane_zone.py`, since removed from
the repo): it HSV-segmented the crane's yellow/orange paintwork across 126
sampled frames (every 1.0s over the 120s clip), unioned every detected
bounding box to capture the crane's full swept/rotated working envelope,
then padded 40px. The script was a generation tool, not a pipeline
dependency -- `zone_monitor.py` only ever reads the resulting numbers from
the config, never re-derives them at runtime. Removing the script does not
affect the pipeline; it only means the polygon can no longer be
regenerated automatically from a new clip without rewriting that logic (or
drawing a new polygon by hand).

## events.json schema

**Deviation from the literal spec, done deliberately:** the spec's
requirement #7 describes one flat record per event (`event`, `track_id`,
`timestamp`, `frame`, `box`). This implementation instead emits
**interval** records: `start_frame`/`start_sec`/`start_box` and
`stop_frame`/`stop_sec`/`stop_box` per event, with every field the spec
asked for still present (just as a pair, not a single snapshot).

Why: a literal one-record-per-instant implementation means an IDLE_ALERT
that holds for 45 real seconds either re-writes a new record every frame
(the first version of this pipeline did exactly that, and the on-screen
event banner did too -- both were bugs, fixed this session) or only
captures the first instant and loses how long the condition actually lasted.
The start/stop interval format is strictly more useful for the "when did
this happen and for how long" question a safety reviewer actually asks, and
was built in direct response to live project feedback. Every event is
guaranteed closed (`stop_frame`/`stop_sec` never `null`) -- verified
directly against the JSON, not just by log output.

Example record:

```json
{
  "event": "ZONE_ENTRY",
  "track_id": 5,
  "start_frame": 622,
  "start_sec": 31.22,
  "stop_frame": 644,
  "stop_sec": 32.32,
  "start_box": [903, 509, 1078, 848],
  "stop_box": [922, 491, 1069, 836]
}
```



## Answers to required questions



### Why use the foot point instead of the box centre?

The foot point (bottom-centre of the box) is the closest proxy available to
where a person's feet actually touch the ground. A restricted zone is a
*floor-plane* polygon -- it defines ground that must not be stepped on, not
a volume in 3D space. The box centre is roughly torso height; for a person
standing upright that point floats somewhere above the floor and does not
correspond to any real ground location, so testing it against a
ground-plane polygon is testing the wrong thing geometrically. The foot
point directly approximates "where is this person standing," which is what
the zone rule needs.

This project surfaced a concrete failure mode of the heuristic, not just a
textbook justification: on the original forklift clip, a seated driver was
correctly classified "person," but because his real feet were never visible
(elevated in the cab), the detector's box stretched to include the seat,
steering wheel, and cab pillar. The box's bottom edge (`y2`) then landed
near the floor behind/below the vehicle -- nowhere near his actual feet --
and incorrectly fired ZONE_ENTRY. Foot-point-as-ground-contact is a
reasonable default, but it fails whenever a subject's real feet are not
visible or not where the box's bottom edge implies (seated, crouched, or
partially out of frame). This is a documented, verified limitation, not a
reason to prefer box-centre instead (which would be worse in every case
where the person is actually standing on the floor and visible).

### How would you stop one person's ID swapping after they are hidden for 2 seconds?

This was built and verified this session, not just designed on paper. Three
separate levers were tested on a confirmed real ID swap (track 4 -> 8,
person continuously visible the whole time, never actually occluded):

1. `track_buffer` (BoT-SORT/ByteTrack parameter): the tracker keeps a
  "lost" track alive for this many frames before assigning a brand-new ID
   on re-detection. Default was 30 frames (1.5s @20fps) -- too short for a
   2+ second gap. Raised to 150 frames (7.5s @20fps). This alone fixed the
   confirmed swap.
2. **Using other strong REID trackers  lke DeepSORT or BOXMOT libraries**  
**or even train trackers for custom different camera angles** 

**For a true 2-second full occlusion** (the person genuinely behind an
opaque object, not just a confidence dip), the correct fix is specifically
#1 (`track_buffer` sized to exceed the occlusion duration) and #3 (ReID
*would* help here, since on re-appearance the tracker has a real
appearance-matching decision to make against whichever track IDs are still
alive in the buffer). This clip's actual swap was a confidence problem
dressed up as an occlusion problem -- both are fixed, but for different
reasons, and that distinction was only found by checking, not assumed.

## Known limitation, found and left as a documented trade-off

A person standing with their foot point resting almost exactly on the zone
polygon's boundary line can generate several real ZONE_ENTRY/exit pairs in
a few seconds from natural micro-movement (weight shifting, reaching for
material) -- each one is a genuine physical boundary crossing, verified
frame-by-frame against the video, not a detector-flicker bug. Raising
`exit_grace_sec` does not fix this (grace period only covers *missing*
detections, not continuously-detected boundary-straddling). The correct
fix is a hysteresis band -- separate inner/outer polygons so "enter" and
"exit" use different boundaries -- which is new scope beyond this
assessment's time budget and is listed as a documented limitation rather
than silently left broken.

## AI tool assistance disclosure

Built with AI coding assistance (Claude via Hermes): the detection/tracking
pipeline, zone/idle rule logic, the data-driven zone-polygon generator, all
tracker/detection tuning, and this documentation. Every claim in this
README and HANDOVER.md was checked against real command output, real
re-runs of the pipeline, and visual verification of specific video frames
before being written down -- see HANDOVER.md for the itemized verification
trail.