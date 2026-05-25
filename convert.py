#!/usr/bin/env python3
"""One-time conversion of YOLO26 .pt weights to CoreML .mlpackage.

Requirements:
    pip install ultralytics coremltools torch==2.7.0

Usage:
    python3 convert.py                    # converts yolo26n (default)
    python3 convert.py --model yolo26s    # converts yolo26s
    python3 convert.py --model yolo26m    # converts yolo26m
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO26 to CoreML for ANE")
    parser.add_argument("--model", default="yolo26n", help="Model name (yolo26n/s/m/l/x)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--output", default="models", help="Output directory")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Loading {args.model}.pt...")
    model = YOLO(f"{args.model}.pt")

    print(f"Exporting to CoreML (imgsz={args.imgsz}, half=True, nms=False)...")
    export_path = model.export(format="coreml", nms=False, half=True, imgsz=args.imgsz)

    # Move to output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    dest = output_dir / f"{args.model}.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(export_path, str(dest))

    print(f"Done: {dest}")
    print(f"Size: {sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
    print()
    print("Now run:")
    print(f"  .build/release/yolo26-coreml webcam {dest} --compute-units ane")


if __name__ == "__main__":
    main()
