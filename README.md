# YOLO26 on Apple Neural Engine

**Apple Neural Engine backend for YOLO26. 448-514 FPS on M4 Pro. Zero Python at runtime.**

YOLO26-ANE extends the webAI YOLO26-MLX Apple Silicon baseline with a second on-device execution path: Swift/CoreML on Apple Neural Engine. On the included `bus.jpg` benchmark, the CoreML/ANE backend reaches 448 FPS in the full multi-backend run and 514 FPS in a focused MLX-vs-ANE run.

## What it does

A native Swift + Rust pipeline that runs YOLO26 object detection on Apple Neural Engine through public CoreML APIs. Includes:

- **Live webcam detection** with bounding box overlay and real-time FPS counter
- **Benchmark suite** comparing ANE vs GPU vs CPU performance
- **Rust wrapper** for programmatic integration

## Why we built it

YOLO26-MLX provides a strong native Apple Silicon path through MLX and Metal. This project asks a complementary systems question: can the same YOLO26 workload also run efficiently on Apple's dedicated Neural Engine?

YOLO26's NMS-free export path makes it a good fit for CoreML/ANE because the detector graph can be represented as a static model with simple tensor outputs. That opens a split-compute architecture for local vision systems: camera perception can run on the Neural Engine while the graphics processor remains available for other application work.

## Performance

### webAI Benchmark Shape

The table below uses the same `bus.jpg`, 640x640 input, 3 warmup runs, 10 timed runs, and webAI's own MLX/MPS/CPU benchmark script for those baseline rows. Full results are saved in [results/yolo26_apples_to_apples_full_smoke.md](results/yolo26_apples_to_apples_full_smoke.md).

| Backend | FPS | Mean ms | Median ms | Notes |
|---|---:|---:|---:|---|
| CoreML/ANE | 448.4 | 2.23 | 2.25 | Swift/CoreML, `cpuAndNeuralEngine`, cache-busted |
| CoreML/GPU | 334.9 | 2.99 | 2.83 | Swift/CoreML, GPU-only |
| MLX/Metal | 155.2 | 6.44 | 6.25 | webAI script, converted `.npz` weights |
| PyTorch MPS | 99.7 | 10.03 | 10.04 | webAI script |
| CoreML CPU | 97.5 | 10.26 | 10.23 | Swift/CoreML, CPU-only |
| PyTorch CPU | 42.5 | 23.51 | 23.18 | webAI script |

Focused MLX-vs-ANE run: CoreML/ANE reached 513.8 FPS versus webAI MLX at 153.4 FPS, saved in [results/yolo26_webai_plus_coreml_smoke.md](results/yolo26_webai_plus_coreml_smoke.md).

Detection parity: CoreML and Ultralytics matched 5/5 detections on `bus.jpg`, with `xyxy` boxes and 0.951 mean IoU. See [results/yolo26_coreml_parity_smoke.json](results/yolo26_coreml_parity_smoke.json).

The result artifacts are stored in `results/`.

## How to run

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 14.0+ with CoreML support for the exported model
- Swift 5.9+ (ships with Xcode)
- Python 3.12 (for one-time model conversion only)

### Quick start

```bash
# 1. Clone and build
git clone https://github.com/elsheppo/yolo26-ane.git
cd yolo26-ane
swift build -c release

# 2. Create a Python environment for one-time conversion
python3.12 -m venv .venv
source .venv/bin/activate
pip install ultralytics coremltools torch==2.7.0

# 3. Put yolo26n.pt in the repo root, then export to CoreML
# This repo does not redistribute model weights.
python3 convert.py --model yolo26n --imgsz 640 --output models

# 4. Run live webcam detection
.build/release/yolo26-coreml webcam models/yolo26n.mlpackage --compute-units ane

# 5. Run on the included benchmark image
.build/release/yolo26-coreml predict models/yolo26n.mlpackage images/bus.jpg --compute-units ane --preprocess letterbox

# 6. Benchmark
.build/release/yolo26-coreml bench models/yolo26n.mlpackage images/bus.jpg --runs 300 --warmup 20 --compute-units ane --preprocess letterbox --cache-bust
```

For a clean reproduction checklist, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

### webAI benchmark shape

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py prepare-assets
python3 benchmarks/webai_apples_to_apples/benchmark.py throughput --models n --runs 10 --warmup 3
python3 benchmarks/webai_apples_to_apples/benchmark.py parity --model-size n --box-format auto
```

See [benchmarks/webai_apples_to_apples/README.md](benchmarks/webai_apples_to_apples/README.md) for the full MLX/CoreML/COCO validation flow.

### Rust wrapper

```bash
cd rust
cargo build --release
YOLO26_COREML_BIN=../.build/release/yolo26-coreml \
  ./target/release/yolo26-ane predict ../models/yolo26n.mlpackage ../images/bus.jpg
```

## Hardware used

- Apple M4 Pro, 48 GB unified memory
- macOS Tahoe 26.0.1 (25A362)

## Model variant

- Primary: **yolo26n** (2.4M params, 5.4 GFLOPs)
- Benchmark in this repo: yolo26n

## Architecture

```
Camera (AVFoundation, 720p)
  -> Resize (CIContext, GPU-accelerated)
    -> CoreML inference (ANE, 2ms)
      -> Post-processing (CPU, top-k selection)
        -> Overlay rendering (AppKit/CoreGraphics)
          -> Display (NSWindow)
```

The model is configured with `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`, which excludes the Metal GPU from CoreML inference scheduling. The GPU may still be used by the surrounding application for resize, display, rendering, or other workloads.

## Track

**Enterprise**: Local camera perception on Apple Silicon, with a separate execution path for the detector and the rest of the application.

## What makes this different

1. **Second Apple Silicon execution path.** webAI YOLO26-MLX covers the MLX/Metal path; this repo adds a Swift/CoreML Neural Engine path.
2. **2.9-3.4x faster than webAI MLX/Metal** in the included `bus.jpg` benchmark runs.
3. **GPU remains available** for the rest of the app while CoreML schedules compatible inference work on CPU plus Neural Engine.
4. **Zero Python at runtime.** The prediction, benchmark, and webcam paths run through a native Swift binary.
5. **YOLO26's NMS-free architecture enables this.** The exported model can use a static detector graph with simple tensor outputs.

## Files

```
yolo26-ane/
  Sources/main.swift     # Swift CoreML CLI (predict, bench, webcam)
  Package.swift          # Swift package manifest
  rust/                  # Rust subprocess wrapper
    Cargo.toml
    src/main.rs
  convert.py             # One-time model conversion script
  README.md              # This file
```

## License

AGPL-3.0 (same as upstream YOLO26 MLX).

This repository does not include YOLO26 `.pt`, `.mlpackage`, or `.mlmodelc` model weights. Reproduction docs describe how to generate local model artifacts from upstream sources. Third-party components and license context are summarized in `THIRD_PARTY_NOTICES.md`.
