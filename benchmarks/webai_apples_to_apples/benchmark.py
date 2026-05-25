#!/usr/bin/env python3
"""Apples-to-apples YOLO26 benchmark and validation harness.

This mirrors webAI's public benchmark shape while adding CoreML compute-unit
backends driven by the local Swift runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised by CLI users
    Image = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT / "images"
MODELS_DIR = ROOT / "models"
DEFAULT_IMAGE = IMAGES_DIR / "bus.jpg"
DEFAULT_SWIFT_BIN = ROOT / ".build" / "release" / "yolo26-coreml"
DEFAULT_WEBAI_REPO_CACHE = Path("/private/tmp/yolo-mlx-webai")
IMAGE_SIZE = 640
MODEL_SIZES = ["n", "s", "m", "l", "x"]
COREML_BACKENDS = {
    "coreml_ane": "ane",
    "coreml_all": "all",
    "coreml_gpu": "gpu",
    "coreml_cpu": "cpu",
}
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


@dataclass(frozen=True)
class Letterbox:
    ratio: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def device_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    for key, command in {
        "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "mac_model": ["sysctl", "-n", "hw.model"],
    }.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                info[key] = result.stdout.strip()
        except OSError:
            pass
    return info


def calculate_stats(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {}
    return {
        "mean_ms": float(statistics.fmean(times_ms)),
        "std_ms": float(statistics.pstdev(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "fps": float(1000.0 / statistics.fmean(times_ms)),
    }


def ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def model_path(model_size: str) -> Path | None:
    stem = f"yolo26{model_size}"
    for suffix in [".mlmodelc", ".mlpackage"]:
        candidate = MODELS_DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def download_bus_image(path: Path = DEFAULT_IMAGE) -> Path:
    ensure_results_dirs()
    if path.exists():
        return path
    url = "https://ultralytics.com/images/bus.jpg"
    urllib.request.urlretrieve(url, path)
    return path


def run_json(command: list[str], cwd: Path = ROOT, timeout_s: int | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout_s}s: {' '.join(command)}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(command)}\n{result.stdout}") from exc


def run_coreml_benchmark(
    swift_bin: Path,
    model: Path,
    image: Path,
    compute_units: str,
    warmup: int,
    runs: int,
    cache_bust: bool,
    preprocess: str,
    timeout_s: int | None,
) -> dict[str, Any]:
    command = [
        str(swift_bin),
        "bench",
        str(model),
        str(image),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--compute-units",
        compute_units,
        "--preprocess",
        preprocess,
    ]
    if cache_bust:
        command.append("--cache-bust")
    raw = run_json(command, timeout_s=timeout_s)
    return {
        "mean_ms": raw["mean_ms"],
        "std_ms": raw.get("std_ms", 0.0),
        "min_ms": raw["min_ms"],
        "max_ms": raw["max_ms"],
        "median_ms": raw.get("median_ms", raw.get("p50_ms")),
        "fps": raw["fps"],
        "p50_ms": raw.get("p50_ms"),
        "p99_ms": raw.get("p99_ms"),
        "warmup": raw.get("warmup", warmup),
        "runs": raw.get("runs", runs),
        "compute_units": raw.get("compute_units", compute_units),
        "cache_bust": raw.get("cache_bust", cache_bust),
        "preprocess": raw.get("preprocess", preprocess),
        "model_path": str(model),
    }


def run_webai_script(
    webai_repo: Path,
    models: list[str],
    backends: list[str],
    image: Path,
    warmup: int,
    runs: int,
    output_path: Path,
    python_bin: str | None = None,
    timeout_s: int = 300,
) -> dict[str, Any]:
    script = webai_repo / "scripts" / "benchmark_yolo26_inference.py"
    if not script.exists():
        raise FileNotFoundError(f"webAI benchmark script not found: {script}")
    webai_images = webai_repo / "images"
    webai_images.mkdir(parents=True, exist_ok=True)
    if image.exists():
        shutil.copyfile(image, webai_images / "bus.jpg")
    python = resolve_webai_python(python_bin)
    command = webai_command_prefix(python)
    command += [
        str(script),
        "--models",
        *models,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--output",
        str(output_path),
    ]
    if "mlx" not in backends:
        command.append("--skip-mlx")
    if "pytorch_mps" not in backends:
        command.append("--skip-mps")
    if "pytorch_cpu" not in backends:
        command.append("--skip-cpu")
    try:
        result = subprocess.run(
            command,
            cwd=webai_repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"webAI benchmark timed out after {timeout_s}s. "
            "Install/download required webAI assets first or increase --webai-timeout."
        ) from exc
    if result.returncode != 0:
        output = "\n".join([result.stdout[-2000:], result.stderr[-2000:]]).strip()
        raise RuntimeError(
            f"webAI benchmark failed with code {result.returncode}\n"
            f"{textwrap.indent(output, '  ')}"
        )
    return json.loads(output_path.read_text())


def resolve_webai_python(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env_python = os.environ.get("YOLO26_WEBAI_PYTHON")
    if env_python:
        return env_python
    for candidate in ["python3.12", "python3.11", "python3.10"]:
        found = shutil.which(candidate)
        if found:
            return found
        for prefix in ["/opt/homebrew/bin", "/usr/local/bin"]:
            path = Path(prefix) / candidate
            if path.exists():
                return str(path)
    if sys.version_info >= (3, 10):
        return sys.executable
    raise RuntimeError(
        "webAI's benchmark script requires Python 3.10+ syntax. "
        "Install python3.10+ or pass --webai-python /path/to/python."
    )


def webai_command_prefix(python: str) -> list[str]:
    uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
    if Path(uv).exists():
        return [uv, "run", "--python", python, "python"]
    return [python]


def resolve_webai_repo(webai_repo: str | None) -> Path | None:
    if not webai_repo:
        return None

    if webai_repo.startswith(("https://", "http://", "git@")):
        target = Path(os.environ.get("YOLO26_WEBAI_REPO_CACHE", str(DEFAULT_WEBAI_REPO_CACHE)))
        if (target / ".git").exists():
            subprocess.run(["git", "-C", str(target), "fetch", "--all", "--prune"], check=False)
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(["git", "clone", webai_repo, str(target)], check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone webAI repo from {webai_repo}")
        return target

    path = Path(webai_repo).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"webAI repo path does not exist: {path}")
    return path


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    results = report["results"]
    lines = [
        "# YOLO26 Apples-to-Apples Benchmark",
        "",
        f"Generated: `{report['timestamp']}`",
        "",
        "## Throughput",
        "",
        "| Model | Backend | FPS | Mean ms | Median ms | Std ms | Min ms | Max ms | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for backend, by_model in results.items():
        for size, stats in by_model.items():
            notes = []
            if stats.get("skipped"):
                notes.append(stats.get("reason", "skipped"))
                lines.append(f"| yolo26{size} | {backend} | - | - | - | - | - | - | {'; '.join(notes)} |")
                continue
            if stats.get("cache_bust"):
                notes.append("cache-bust")
            if stats.get("preprocess"):
                notes.append(str(stats["preprocess"]))
            lines.append(
                f"| yolo26{size} | {backend} | {stats['fps']:.1f} | "
                f"{stats['mean_ms']:.2f} | {stats.get('median_ms', 0):.2f} | "
                f"{stats.get('std_ms', 0):.2f} | {stats['min_ms']:.2f} | "
                f"{stats['max_ms']:.2f} | {', '.join(notes)} |"
            )
    output_path.write_text("\n".join(lines) + "\n")


def throughput(args: argparse.Namespace) -> int:
    ensure_results_dirs()
    image = Path(args.image)
    if args.download_bus:
        image = download_bus_image(image)
    if not image.exists():
        raise FileNotFoundError(f"Benchmark image missing: {image}")
    swift_bin = Path(args.swift_bin)
    if not swift_bin.exists():
        raise FileNotFoundError(f"Swift binary missing: {swift_bin}. Run: swift build -c release")

    results: dict[str, dict[str, Any]] = {backend: {} for backend in args.backends}
    for size in args.models:
        model = model_path(size)
        for backend in args.backends:
            if backend in COREML_BACKENDS:
                if model is None:
                    results[backend][size] = {
                        "skipped": True,
                        "reason": f"models/yolo26{size}.mlpackage or .mlmodelc not found",
                    }
                    continue
                try:
                    results[backend][size] = run_coreml_benchmark(
                        swift_bin=swift_bin,
                        model=model,
                        image=image,
                        compute_units=COREML_BACKENDS[backend],
                        warmup=args.warmup,
                        runs=args.runs,
                        cache_bust=args.cache_bust,
                        preprocess=args.preprocess,
                        timeout_s=args.coreml_timeout,
                    )
                except Exception as exc:
                    results[backend][size] = {
                        "skipped": True,
                        "reason": str(exc),
                    }
            else:
                results[backend][size] = {
                    "skipped": True,
                    "reason": "Use --webai-repo to run webAI MLX/MPS/CPU baseline unchanged",
                }

    webai_results = None
    webai_error = None
    webai_repo = resolve_webai_repo(args.webai_repo)
    if webai_repo:
        webai_output = RESULTS_DIR / "webai_yolo26_inference_three_way.json"
        try:
            webai_results = run_webai_script(
                webai_repo,
                args.models,
                args.backends,
                image,
                args.warmup,
                args.runs,
                webai_output,
                args.webai_python,
                args.webai_timeout,
            )
            for backend in ["mlx", "pytorch_mps", "pytorch_cpu"]:
                if backend in args.backends:
                    backend_results = webai_results.get("results", {}).get(backend, {})
                    if backend_results:
                        results[backend] = backend_results
                    else:
                        results[backend] = {
                            size: {
                                "skipped": True,
                                "reason": (
                                    "webAI benchmark emitted no rows for this backend; "
                                    "check that required weights/assets are installed in the webAI checkout"
                                ),
                            }
                            for size in args.models
                        }
        except Exception as exc:
            webai_error = str(exc)
            for backend in ["mlx", "pytorch_mps", "pytorch_cpu"]:
                if backend in args.backends:
                    results[backend] = {
                        size: {"skipped": True, "reason": webai_error}
                        for size in args.models
                    }

    output = {
        "benchmark": "YOLO26 apples-to-apples inference",
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info(),
        "configuration": {
            "image": str(image),
            "image_size": IMAGE_SIZE,
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "models": args.models,
            "backends": args.backends,
            "preprocess": args.preprocess,
            "cache_bust": args.cache_bust,
            "webai_repo": str(webai_repo) if webai_repo else None,
            "webai_python": resolve_webai_python(args.webai_python) if webai_repo else None,
            "coreml_timeout_s": args.coreml_timeout,
        },
        "results": results,
        "webai_results": webai_results,
        "webai_error": webai_error,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    write_markdown_report(output, Path(args.markdown))
    print(f"Wrote {output_path}")
    print(f"Wrote {args.markdown}")
    return 0


def letterbox_for_image(image_path: Path, size: int = IMAGE_SIZE) -> Letterbox:
    if Image is None:
        raise ImportError("Pillow is required for parity/COCO validation")
    with Image.open(image_path) as image:
        orig_w, orig_h = image.size
    ratio = min(size / orig_h, size / orig_w)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    return Letterbox(
        ratio=ratio,
        pad_x=(size - new_w) / 2.0,
        pad_y=(size - new_h) / 2.0,
        orig_w=orig_w,
        orig_h=orig_h,
    )


def detection_to_xyxy(det: dict[str, Any], box_format: str) -> tuple[float, float, float, float]:
    x = float(det["x"])
    y = float(det["y"])
    w = float(det["w"])
    h = float(det["h"])
    if box_format == "xyxy":
        return x, y, w, h
    if box_format == "xywh":
        return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0
    raise ValueError(f"Unknown box format: {box_format}")


def undo_letterbox(box: tuple[float, float, float, float], info: Letterbox) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(info.orig_w, (x1 - info.pad_x) / info.ratio))
    y1 = max(0.0, min(info.orig_h, (y1 - info.pad_y) / info.ratio))
    x2 = max(0.0, min(info.orig_w, (x2 - info.pad_x) / info.ratio))
    y2 = max(0.0, min(info.orig_h, (y2 - info.pad_y) / info.ratio))
    return x1, y1, x2, y2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_coreml_predict(
    swift_bin: Path,
    model: Path,
    image: Path,
    compute_units: str,
    conf: float,
    preprocess: str,
) -> list[dict[str, Any]]:
    return run_json([
        str(swift_bin),
        "predict",
        str(model),
        str(image),
        "--compute-units",
        compute_units,
        "--conf",
        str(conf),
        "--preprocess",
        preprocess,
    ])


def run_ultralytics_predict(model_name: str, image: Path, conf: float) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics to run parity smoke") from exc
    model = YOLO(f"{model_name}.pt")
    result = model.predict(str(image), imgsz=IMAGE_SIZE, conf=conf, verbose=False)[0]
    boxes = result.boxes
    out = []
    if boxes is None:
        return out
    for xyxy, score, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
        out.append({
            "box": [float(v) for v in xyxy],
            "conf": float(score),
            "class": int(cls),
        })
    return out


def compare_detections(
    reference: list[dict[str, Any]],
    coreml: list[dict[str, Any]],
    image: Path,
    box_format: str,
) -> dict[str, Any]:
    info = letterbox_for_image(image)
    converted = []
    for det in coreml:
        model_box = detection_to_xyxy(det, box_format)
        converted.append({
            "box": list(undo_letterbox(model_box, info)),
            "conf": float(det["conf"]),
            "class": int(det["class"]),
        })

    matched = []
    used = set()
    for ref in sorted(reference, key=lambda d: d["conf"], reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, cand in enumerate(converted):
            if idx in used or cand["class"] != ref["class"]:
                continue
            score = iou(tuple(ref["box"]), tuple(cand["box"]))
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            matched.append({
                "class": ref["class"],
                "reference_conf": ref["conf"],
                "coreml_conf": converted[best_idx]["conf"],
                "iou": best_iou,
            })
    return {
        "box_format": box_format,
        "reference_count": len(reference),
        "coreml_count": len(coreml),
        "matched_count": len(matched),
        "mean_iou": statistics.fmean([m["iou"] for m in matched]) if matched else 0.0,
        "matches": matched[:20],
    }


def parity(args: argparse.Namespace) -> int:
    image = Path(args.image)
    model = model_path(args.model_size)
    if model is None:
        raise FileNotFoundError(f"CoreML model missing for yolo26{args.model_size}")
    swift_bin = Path(args.swift_bin)
    reference = run_ultralytics_predict(f"yolo26{args.model_size}", image, args.conf)
    coreml = run_coreml_predict(swift_bin, model, image, "ane", args.conf, args.preprocess)
    formats = ["xywh", "xyxy"] if args.box_format == "auto" else [args.box_format]
    comparisons = [compare_detections(reference, coreml, image, fmt) for fmt in formats]
    best = max(comparisons, key=lambda item: item["mean_iou"])
    output = {
        "benchmark": "YOLO26 CoreML parity smoke",
        "timestamp": datetime.now().isoformat(),
        "image": str(image),
        "model": f"yolo26{args.model_size}",
        "conf": args.conf,
        "preprocess": args.preprocess,
        "best_box_format": best["box_format"],
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


def coreml_predict_list(
    swift_bin: Path,
    model: Path,
    image_paths: list[Path],
    compute_units: str,
    conf: float,
    preprocess: str,
) -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        list_path = Path(handle.name)
        for path in image_paths:
            handle.write(str(path) + "\n")
    command = [
        str(swift_bin),
        "predict-list",
        str(model),
        str(list_path),
        "--compute-units",
        compute_units,
        "--conf",
        str(conf),
        "--preprocess",
        preprocess,
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"predict-list failed: {result.stderr}")
    frames = []
    for line in result.stdout.splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


def load_coco_images(data: Path, subset: int | None) -> tuple[list[dict[str, Any]], Path]:
    ann_file = data / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file missing: {ann_file}")
    annotations = json.loads(ann_file.read_text())
    images = annotations["images"]
    if subset:
        images = images[:subset]
    return images, ann_file


def coco_eval(ann_file: Path, pred_file: Path, img_ids: list[int]) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("Install pycocotools to run COCO validation") from exc
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(pred_file))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = img_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "mAP50-95": float(evaluator.stats[0]),
        "mAP50": float(evaluator.stats[1]),
        "mAP75": float(evaluator.stats[2]),
        "mAP_small": float(evaluator.stats[3]),
        "mAP_medium": float(evaluator.stats[4]),
        "mAP_large": float(evaluator.stats[5]),
    }


def coco(args: argparse.Namespace) -> int:
    data = Path(args.data)
    images, ann_file = load_coco_images(data, args.subset)
    model = model_path(args.model_size)
    if model is None:
        raise FileNotFoundError(f"CoreML model missing for yolo26{args.model_size}")
    image_paths = [data / "images" / "val2017" / img["file_name"] for img in images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing COCO images, first missing: {missing[0]}")

    t0 = time.perf_counter()
    frames = coreml_predict_list(
        swift_bin=Path(args.swift_bin),
        model=model,
        image_paths=image_paths,
        compute_units=args.compute_units,
        conf=args.conf,
        preprocess=args.preprocess,
    )
    elapsed_total = time.perf_counter() - t0
    image_by_path = {str(path): img for path, img in zip(image_paths, images)}
    predictions = []
    inference_times = []
    for frame in frames:
        path = Path(frame["image"])
        img = image_by_path[str(path)]
        info = letterbox_for_image(path)
        inference_times.append(float(frame["elapsed_ms"]))
        for det in frame["detections"]:
            box = undo_letterbox(detection_to_xyxy(det, args.box_format), info)
            x1, y1, x2, y2 = box
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            class_id = int(det["class"])
            if class_id < 0 or class_id >= len(COCO_IDS):
                continue
            predictions.append({
                "image_id": int(img["id"]),
                "category_id": int(COCO_IDS[class_id]),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(det["conf"]),
            })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / f"yolo26{args.model_size}_coreml_{args.compute_units}_coco_predictions.json"
    pred_file.write_text(json.dumps(predictions, indent=2) + "\n")
    metrics = coco_eval(ann_file, pred_file, [int(img["id"]) for img in images]) if predictions else {}
    speed = calculate_stats(inference_times)
    result = {
        "model": f"yolo26{args.model_size}",
        "framework": "coreml",
        "compute_units": args.compute_units,
        "dataset": "coco_val2017",
        "num_images": len(images),
        "imgsz": IMAGE_SIZE,
        "conf_thresh": args.conf,
        "box_format": args.box_format,
        "preprocess": args.preprocess,
        "metrics": metrics,
        "speed": {
            **speed,
            "total_elapsed_s": elapsed_total,
            "end_to_end_fps": len(images) / elapsed_total if elapsed_total > 0 else 0.0,
        },
        "prediction_file": str(pred_file),
        "timestamp": datetime.now().isoformat(),
    }
    result_file = output_dir / f"yolo26{args.model_size}_coreml_{args.compute_units}_coco_results.json"
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26 apples-to-apples validation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-assets")
    prep.add_argument("--image", default=str(DEFAULT_IMAGE))
    prep.set_defaults(func=lambda args: print(download_bus_image(Path(args.image))) or 0)

    th = sub.add_parser("throughput")
    th.add_argument("--models", nargs="+", default=["n"], choices=MODEL_SIZES)
    th.add_argument("--backends", nargs="+", default=["coreml_ane", "coreml_all", "coreml_gpu", "coreml_cpu"])
    th.add_argument("--warmup", type=int, default=3)
    th.add_argument("--runs", type=int, default=10)
    th.add_argument("--image", default=str(DEFAULT_IMAGE))
    th.add_argument("--download-bus", action="store_true")
    th.add_argument("--swift-bin", default=str(DEFAULT_SWIFT_BIN))
    th.add_argument("--preprocess", choices=["stretch", "letterbox"], default="letterbox")
    th.add_argument("--cache-bust", action=argparse.BooleanOptionalAction, default=True)
    th.add_argument("--webai-repo", default=None)
    th.add_argument("--webai-python", default=None)
    th.add_argument("--webai-timeout", type=int, default=300)
    th.add_argument("--coreml-timeout", type=int, default=300)
    th.add_argument("--output", default=str(RESULTS_DIR / "yolo26_apples_to_apples.json"))
    th.add_argument("--markdown", default=str(RESULTS_DIR / "yolo26_apples_to_apples.md"))
    th.set_defaults(func=throughput)

    pa = sub.add_parser("parity")
    pa.add_argument("--model-size", default="n", choices=MODEL_SIZES)
    pa.add_argument("--image", default=str(DEFAULT_IMAGE))
    pa.add_argument("--swift-bin", default=str(DEFAULT_SWIFT_BIN))
    pa.add_argument("--conf", type=float, default=0.25)
    pa.add_argument("--preprocess", choices=["stretch", "letterbox"], default="letterbox")
    pa.add_argument("--box-format", choices=["auto", "xywh", "xyxy"], default="auto")
    pa.add_argument("--output", default=str(RESULTS_DIR / "yolo26_coreml_parity_smoke.json"))
    pa.set_defaults(func=parity)

    co = sub.add_parser("coco")
    co.add_argument("--model-size", default="n", choices=MODEL_SIZES)
    co.add_argument("--data", default=str(ROOT / "datasets" / "coco"))
    co.add_argument("--subset", type=int, default=None)
    co.add_argument("--swift-bin", default=str(DEFAULT_SWIFT_BIN))
    co.add_argument("--compute-units", choices=["ane", "all", "gpu", "cpu"], default="ane")
    co.add_argument("--conf", type=float, default=0.001)
    co.add_argument("--preprocess", choices=["stretch", "letterbox"], default="letterbox")
    co.add_argument("--box-format", choices=["xywh", "xyxy"], default="xyxy")
    co.add_argument("--output-dir", default=str(RESULTS_DIR))
    co.set_defaults(func=coco)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
