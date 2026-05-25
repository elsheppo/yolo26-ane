# YOLO26-ANE Reproducibility Protocol

This document is the minimal protocol for reproducing the YOLO26-ANE result from a clean checkout on Apple Silicon.

## Notes

These commands reproduce the benchmark path used for the published results. Keep in mind:

1. The exact FPS will vary by chip, OS version, thermal state, camera/display path, and whether the benchmark uses a precompiled `.mlmodelc` or first-run `.mlpackage` compilation.
2. The repo does not ship YOLO26 weights or exported CoreML packages. Reproducers need to fetch/export the model weights according to the upstream model license.
3. CoreML does not expose a simple public per-op Neural Engine placement trace. This repo uses CoreML compute-unit selection plus CPU/GPU/ANE timing comparisons to validate the execution path.

Expect the repeatable pattern to matter more than one exact FPS number:

- `cpuAndNeuralEngine` should be much faster than CPU-only.
- `cpuAndNeuralEngine` should beat or match `all` for YOLO26n.
- `cpuAndNeuralEngine` should show lower p99 jitter than MLX/Metal.
- GPU-heavy workloads should remain usable while YOLO26 runs on `cpuAndNeuralEngine`.

## Hardware Used For Headline Result

- Machine: Apple Silicon Mac
- Tested headline machine: Apple M4 Pro, 48 GB unified memory
- OS: macOS Tahoe 26.0.1 (25A362)
- Swift: Swift 5.9+ / Xcode command line tools
- Python: Python 3.12 for one-time export

## Runtime Path Under Test

```text
YOLO26 .pt
  -> Ultralytics CoreML export
  -> .mlpackage
  -> Swift MLModel.compileModel
  -> MLModelConfiguration.computeUnits = .cpuAndNeuralEngine
  -> model.prediction(from:)
```

This path uses public CoreML APIs. It does not call private Apple Neural Engine APIs.

## Clean Build

From the repo root:

```bash
swift build -c release
```

Expected result:

```text
Build complete!
```

Optional Rust wrapper:

```bash
cd rust
cargo build --release
```

## Model Export

Use Python 3.12. Known-good pins from the original experiment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install ultralytics coremltools torch==2.7.0
```

Place `yolo26n.pt` in the repo root. This repository does not redistribute model weights; obtain them from an upstream source whose license permits your use.

Then export:

```bash
python3 convert.py --model yolo26n --imgsz 640 --output models
```

Expected output:

```text
models/yolo26n.mlpackage
```

The conversion script uses:

```python
model.export(format="coreml", nms=False, half=True, imgsz=640)
```

`nms=False` is important because it keeps the exported detector graph simple enough for the CoreML/ANE path.

## Benchmark Image

Use any local image that CoreML can load through `NSImage`. For easiest comparison, use the same image across every backend and compute-unit setting.

Suggested local path:

```bash
export IMAGE=/path/to/benchmark.jpg
export MODEL=models/yolo26n.mlpackage
```

The benchmark command now supports `--cache-bust`, which mutates one pixel in the input buffer between runs before timing begins. This reduces the chance that identical-input behavior is mistaken for real throughput.

## Swift/CoreML Benchmarks

Run the main ANE benchmark:

```bash
.build/release/yolo26-coreml bench "$MODEL" "$IMAGE" --runs 300 --warmup 20 --compute-units ane --cache-bust
```

Run comparison modes:

```bash
.build/release/yolo26-coreml bench "$MODEL" "$IMAGE" --runs 300 --warmup 20 --compute-units all --cache-bust
.build/release/yolo26-coreml bench "$MODEL" "$IMAGE" --runs 300 --warmup 20 --compute-units gpu --cache-bust
.build/release/yolo26-coreml bench "$MODEL" "$IMAGE" --runs 300 --warmup 20 --compute-units cpu --cache-bust
```

Expected JSON shape:

```json
{
  "fps": 505.0,
  "mean_ms": 1.98,
  "std_ms": 0.12,
  "median_ms": 1.95,
  "p50_ms": 1.95,
  "p99_ms": 2.34,
  "min_ms": 1.80,
  "max_ms": 2.70,
  "warmup": 20,
  "runs": 300,
  "compute_units": "ane",
  "cache_bust": true,
  "preprocess": "stretch"
}
```

Exact numbers will vary. The important comparison is relative throughput and jitter across compute-unit settings.

## Live Webcam Demo

```bash
.build/release/yolo26-coreml webcam "$MODEL" --compute-units ane --conf 0.25
```

Expected behavior:

- A camera window appears.
- Bounding boxes and labels render over the live feed.
- Terminal stderr prints live FPS and detection counts.
- Display and resize overhead can dominate; this is a system demo, not the cleanest inference benchmark.

## Rust Wrapper Benchmark

```bash
cd rust
YOLO26_COREML_BIN=../.build/release/yolo26-coreml \
  cargo run --release -- bench "../$MODEL" "$IMAGE" --runs 300 --warmup 20 --compute-units ane --cache-bust
```

The Rust wrapper is for integration shape. The cleanest headline benchmark is the Swift binary directly, because it avoids wrapper/subprocess concerns.

## Benchmark Notes

When sharing benchmark results, include:

- Machine model and chip
- RAM
- macOS version
- Swift version
- Python version and dependency pins used for export
- YOLO26 variant and input resolution
- Whether the model was `.mlpackage` or precompiled `.mlmodelc`
- Whether `--cache-bust` was used
- Runs and warmup count
- FPS, mean, p50, p99, min, max

Notes:

- First-run model compilation time is separate from steady-state inference latency.
- Live webcam display FPS includes camera, resize, overlay, and display overhead; use `bench` for model timing.
- `ALL` is not the same thing as `ANE`; `ALL` allows CoreML to schedule work across CPU, GPU, and Neural Engine.

## Expected Reproduction Pattern

From a clean checkout on Apple Silicon:

1. Build the Swift CLI.
2. Export YOLO26n to CoreML with `nms=False`.
3. Run `bench` with `--compute-units ane --cache-bust`.
4. Run the same benchmark with `all`, `gpu`, and `cpu`.
5. Compare relative throughput across CoreML compute-unit modes.

## webAI Apples-To-Apples Harness

For direct comparison against webAI's public benchmark shape, use the harness in `benchmarks/webai_apples_to_apples/`.

Prepare the shared test image:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py prepare-assets
```

Run CoreML compute-unit comparisons with webAI defaults: `bus.jpg`, `640x640`, 3 warmups, 10 timed runs, and webAI-compatible JSON stats.

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py throughput \
  --models n \
  --backends coreml_ane coreml_all coreml_gpu coreml_cpu \
  --runs 10 \
  --warmup 3
```

If first-run CoreML compilation or local binary launch is slow, raise the CoreML timeout:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py throughput \
  --models n \
  --backends coreml_ane \
  --runs 10 \
  --warmup 3 \
  --coreml-timeout 900
```

To include webAI's unchanged MLX/MPS/CPU script, pass the public repo URL or a local checkout:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py throughput \
  --models n s m \
  --backends mlx pytorch_mps pytorch_cpu coreml_ane coreml_all coreml_gpu coreml_cpu \
  --webai-repo https://github.com/thewebAI/yolo-mlx
```

`--webai-repo` accepts either a local path or the GitHub URL. URL mode clones/caches the repo at `/private/tmp/yolo-mlx-webai` unless `YOLO26_WEBAI_REPO_CACHE` is set.
The harness uses `python3.12`, `python3.11`, or `python3.10` for webAI when available; override with `--webai-python` or `YOLO26_WEBAI_PYTHON`.
When `uv` is available, the harness runs webAI through `uv run` from the cached checkout. It also copies the selected benchmark image into webAI's expected `images/bus.jpg` path. If webAI setup fails or times out, the webAI rows are marked skipped instead of blocking CoreML results; use `--webai-timeout 900` for slow first-time setup.
CoreML rows use the same fail-closed behavior through `--coreml-timeout`.

## Output Parity

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py parity \
  --model-size n \
  --box-format auto
```

The current `bus.jpg` parity check matched 5/5 detections against the Ultralytics reference with `xyxy` boxes and 0.951 mean IoU.

For a larger accuracy check, run COCO val2017 after the parity check identifies the correct coordinate interpretation:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py coco \
  --model-size n \
  --data datasets/coco \
  --subset 100 \
  --box-format xyxy
```

Remove `--subset` to evaluate all 5,000 COCO val2017 images with `pycocotools`.
