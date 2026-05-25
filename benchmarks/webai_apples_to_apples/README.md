# YOLO26 Apples-to-Apples Validation Harness

This harness compares YOLO26-ANE against the public webAI YOLO26-MLX benchmark shape instead of comparing unrelated numbers.

It covers three validation surfaces:

1. **Throughput parity**: same `bus.jpg`, image size, warmup count, timed run count, and stats shape as webAI's `benchmark_yolo26_inference.py`.
2. **Detection parity**: compares CoreML/ANE detections against Ultralytics on the same image and auto-tests `xywh` versus `xyxy` output interpretation.
3. **COCO validation**: exports CoreML/ANE detections to COCO JSON and evaluates with `pycocotools`.

## Quick Start

```bash
swift build -c release
python3 convert.py --model yolo26n --imgsz 640 --output models
python3 benchmarks/webai_apples_to_apples/benchmark.py prepare-assets
python3 benchmarks/webai_apples_to_apples/benchmark.py throughput --models n --runs 10
```

This writes:

```text
results/yolo26_apples_to_apples.json
results/yolo26_apples_to_apples.md
```

The CoreML rows require `models/yolo26n.mlpackage`. This repo does not include model weights or exported CoreML packages.

## Run webAI Baselines Unchanged

Pass the GitHub URL directly, or point at a local checkout:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py throughput \
  --models n s m \
  --backends mlx pytorch_mps pytorch_cpu coreml_ane coreml_all coreml_gpu coreml_cpu \
  --webai-repo https://github.com/thewebAI/yolo-mlx
```

The harness clones/caches the repo at `/private/tmp/yolo-mlx-webai` by default. Set `YOLO26_WEBAI_REPO_CACHE=/path/to/cache` to override it.

The harness delegates MLX/MPS/CPU to webAI's own script and only adds the CoreML rows. It chooses `python3.12`, `python3.11`, or `python3.10` for webAI if available; override with `--webai-python /path/to/python` or `YOLO26_WEBAI_PYTHON=/path/to/python`.

The harness runs webAI through `uv run` when `uv` is available and copies the selected benchmark image to `images/bus.jpg` inside the webAI checkout so both sides use the same image. You still need the model weights/assets expected by webAI's benchmark script for those rows to produce numbers. If webAI setup fails or times out, only the webAI rows are marked skipped and CoreML rows can still run. Use `--webai-timeout 900` for slow first-time setup.

CoreML rows also fail closed instead of hanging forever. If a local CoreML runner stalls during first-run compilation, Gatekeeper/dyld launch, or model loading, the row is marked skipped with the timeout reason. Use `--coreml-timeout 900` for slow first-run `.mlpackage` compilation, then rerun with the generated `.mlmodelc` when possible.

If the webAI MLX row is empty, install converted weights in the webAI checkout:

```bash
cd /private/tmp/yolo-mlx-webai
scripts/download_yolo26_models.sh --model n
uv run --extra convert yolo-mlx converters convert models/yolo26n.pt -o models/yolo26n.npz --verify
```

Then rerun the harness from this repo:

```bash
TMPDIR=/private/tmp python3 benchmarks/webai_apples_to_apples/benchmark.py throughput \
  --models n \
  --backends mlx pytorch_mps pytorch_cpu coreml_ane coreml_all coreml_gpu coreml_cpu \
  --webai-repo /private/tmp/yolo-mlx-webai \
  --output results/yolo26_apples_to_apples_full_smoke.json \
  --markdown results/yolo26_apples_to_apples_full_smoke.md
```

## Parity Smoke

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py parity \
  --model-size n \
  --box-format auto
```

Use the reported `best_box_format` for COCO validation. The current CoreML export reports `xyxy` on the `bus.jpg` parity check.

## COCO Validation

Download COCO val2017 using webAI's layout:

```text
datasets/coco/
  annotations/instances_val2017.json
  images/val2017/*.jpg
```

Then run a subset first:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py coco \
  --model-size n \
  --data datasets/coco \
  --subset 100 \
  --box-format xyxy
```

Full validation:

```bash
python3 benchmarks/webai_apples_to_apples/benchmark.py coco \
  --model-size n \
  --data datasets/coco \
  --box-format xyxy
```

## Validation Boundary

The included results cover `bus.jpg` throughput and detection parity. Run COCO validation for dataset-level accuracy reporting.
