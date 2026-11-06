# Task 2 failure cases

Model `best.pt` from `warmstart_e30_2cls_640_fulldata`. Detections kept at conf ≥ 0.25, matched to ground truth at IoU 0.5 in the original image's pixel frame.

The JPEGs are enlarged so the long side is 1600 px. The test photos themselves are about 400–500 px wide. Boxes in `failure_cases.json` are still original-image pixels.

Blue is a true positive, red a false positive, green a missed ground-truth box, orange/purple a class mix-up.

1. `failure_1_005791_jpg.rf.def7f267b5154bc17620455038c68e04.jpg` — A yellow rounded object on the equipment is scored as helmet at 0.84; it is not a head.
2. `failure_2_005902_jpg.rf.c5a95be3adc01849004a84e8aa22873b.jpg` — The labeled head is only the back of a head behind the man in plaid; the best same-class score on that box is 0.001.
3. `failure_3_005639_jpg.rf.4783a19b458cb90f8c6204403be8f54d.jpg` — A wide-brimmed felt hat, labeled head, is called helmet at 0.82, so a non-hard-hat would be treated as compliant.
4. `failure_4_005853_jpg.rf.17371fc987e9260d0b1210c602692684.jpg` — The helmet is about 10 px tall on a person far up the slope; foreground helmets are detected and this one scores 0.002.
5. `failure_5_005742_jpg.rf.9f634e2411cd258071d2e0fbe25bca33.jpg` — Three helmets of 9-14 px on the far workers are missed, while the larger helmets closer to the camera are detected.
