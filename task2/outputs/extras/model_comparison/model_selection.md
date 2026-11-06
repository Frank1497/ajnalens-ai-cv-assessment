Test split (706 images, head+helmet only), batch 32, conf 0.001, IoU 0.7.

| model | eval imgsz | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| A imgsz=640 default COCO-init | 640 | 0.955 | 0.925 | 0.972 | 0.670 |
| B imgsz=640 default | 640 | 0.953 | 0.934 | 0.971 | 0.671 |
| C imgsz=1088 box+dfl undersampled | 640 | 0.941 | 0.924 | 0.964 | 0.662 |
| C imgsz=1088 box+dfl undersampled | 1088 | 0.947 | 0.942 | 0.976 | 0.672 |
| D imgsz=1088 box+dfl | 640 | 0.944 | 0.929 | 0.967 | 0.664 |
| D imgsz=1088 box+dfl | 1088 | 0.946 | 0.945 | 0.973 | 0.676 |
| E imgsz=640 box+dfl (FINAL) | 640 | 0.941 | 0.940 | 0.973 | 0.674 |
