| backend | mean ms | median ms | FPS | model-only ms | mAP50 |
|---|---:|---:|---:|---:|---:|
| pytorch | 3.39 | 2.69 | 295.1 | 2.39 | 0.975 |
| onnx_fp16_cuda | 2.20 | 2.14 | 455.0 | 1.20 | 0.974 |
| tensorrt_fp16_cuda | 1.34 | 1.27 | 746.0 | 0.33 | 0.973 |

First 100 sorted test images, batch 1, imgsz 640, NVIDIA GeForce RTX 5070 Ti. Mean ms is end-to-end predict at conf 0.25. Model-only is the network forward. mAP50 is Ultralytics val on the same images.
