# YOLO26 Apples-to-Apples Benchmark

Generated: `2026-05-24T14:59:46.187379`

## Throughput

| Model | Backend | FPS | Mean ms | Median ms | Std ms | Min ms | Max ms | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| yolo26n | mlx | 155.2 | 6.44 | 6.25 | 0.70 | 5.96 | 8.47 |  |
| yolo26n | pytorch_mps | 99.7 | 10.03 | 10.04 | 0.26 | 9.60 | 10.46 |  |
| yolo26n | pytorch_cpu | 42.5 | 23.51 | 23.18 | 0.96 | 22.37 | 25.88 |  |
| yolo26n | coreml_ane | 448.4 | 2.23 | 2.25 | 0.31 | 1.85 | 2.90 | cache-bust, letterbox |
| yolo26n | coreml_all | 363.2 | 2.75 | 2.76 | 0.05 | 2.68 | 2.84 | cache-bust, letterbox |
| yolo26n | coreml_gpu | 334.9 | 2.99 | 2.83 | 0.39 | 2.72 | 4.05 | cache-bust, letterbox |
| yolo26n | coreml_cpu | 97.5 | 10.26 | 10.23 | 0.27 | 9.93 | 10.74 | cache-bust, letterbox |
