| backend | mean ms | median ms | FPS | model-only ms | mAP50 |
|---|---:|---:|---:|---:|---:|
| pytorch | 4.31 | 3.37 | 232.2 | 2.75 | 0.974 |
| onnx_fp16_cuda | 3.67 | 3.58 | 272.5 | 1.96 | 0.978 |
| tensorrt_fp16_cuda | 2.15 | 2.07 | 465.3 | 0.57 | 0.977 |

First 100 sorted test images, batch 1, imgsz 1088, NVIDIA GeForce RTX 5070 Ti. Mean ms is end-to-end predict at conf 0.25. Model-only is the network forward. mAP50 is Ultralytics val on the same images.
