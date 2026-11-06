| backend | mean ms | median ms | FPS | model-only ms | mAP50 |
|---|---:|---:|---:|---:|---:|
| pytorch | 3.36 | 2.73 | 297.2 | 2.37 | 0.975 |
| onnx_fp16_cuda | 2.13 | 2.04 | 470.1 | 1.11 | 0.974 |
| tensorrt_fp16_cuda | 1.27 | 1.25 | 787.3 | 0.32 | 0.973 |

First 100 sorted test images, batch 1, imgsz 640, NVIDIA GeForce RTX 5070 Ti. Mean ms is end-to-end predict at conf 0.25. Model-only is the network forward. mAP50 is Ultralytics val on the same images.
