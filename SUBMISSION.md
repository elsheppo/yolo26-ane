# YOLO26-ANE Submission

## Track

Enterprise

## Project

YOLO26-ANE: Swift/CoreML backend for YOLO26 on Apple Neural Engine.

## Short Description

YOLO26-ANE extends the webAI YOLO26-MLX Apple Silicon baseline with a second on-device execution path: Swift/CoreML on Apple Neural Engine. The benchmark compares CoreML/ANE against webAI's MLX/Metal path using the same `bus.jpg`, 640x640 input, 3 warmups, 10 timed runs, and webAI's own benchmark script for the MLX baseline.

## Results

| Backend | FPS | Mean ms | Median ms |
|---|---:|---:|---:|
| CoreML/ANE | 448.4 | 2.23 | 2.25 |
| CoreML/GPU | 334.9 | 2.99 | 2.83 |
| webAI MLX/Metal | 155.2 | 6.44 | 6.25 |
| PyTorch MPS | 99.7 | 10.03 | 10.04 |
| CoreML CPU | 97.5 | 10.26 | 10.23 |
| PyTorch CPU | 42.5 | 23.51 | 23.18 |

Focused MLX-vs-ANE run: CoreML/ANE reached 513.8 FPS versus webAI MLX at 153.4 FPS.

Detection parity: CoreML/ANE matched 5/5 Ultralytics detections on `bus.jpg`, with 0.951 mean IoU.

## Enterprise Angle

Enterprise camera systems need low-latency local perception, but the GPU is often needed for rendering, LLM inference, UI work, or additional model workloads. YOLO26-ANE moves the detector onto Apple's Neural Engine so the GPU remains available.

> ANE sees. GPU thinks.
