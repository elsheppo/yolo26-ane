#!/usr/bin/env python3
"""Render YOLO26 CoreML detections onto bus.jpg for demo recording."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
}


def extract_json_array(text: str) -> list[dict]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("No JSON detection array found in prediction output")
    return json.loads(text[start : end + 1])


def undo_letterbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    orig_w: int,
    orig_h: int,
    size: int = 640,
) -> tuple[float, float, float, float]:
    ratio = min(size / orig_w, size / orig_h)
    new_w = orig_w * ratio
    new_h = orig_h * ratio
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    return (
        max(0.0, min(orig_w, (x1 - pad_x) / ratio)),
        max(0.0, min(orig_h, (y1 - pad_y) / ratio)),
        max(0.0, min(orig_w, (x2 - pad_x) / ratio)),
        max(0.0, min(orig_h, (y2 - pad_y) / ratio)),
    )


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: render_bus_overlay.py <image> <prediction-output> <output>", file=sys.stderr)
        return 2

    image_path = Path(sys.argv[1])
    pred_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    detections = extract_json_array(pred_path.read_text())
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    colors = {
        0: (0, 220, 255),
        5: (255, 190, 0),
    }

    for det in detections:
        cls = int(det["class"])
        conf = float(det["conf"])
        x1, y1, x2, y2 = undo_letterbox(
            float(det["x"]),
            float(det["y"]),
            float(det["w"]),
            float(det["h"]),
            image.width,
            image.height,
        )
        color = colors.get(cls, (80, 255, 120))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        label = f"{COCO_NAMES.get(cls, str(cls))} {conf:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_h = bbox[3] - bbox[1] + 10
        label_w = bbox[2] - bbox[0] + 14
        y_label = max(0, y1 - label_h)
        draw.rectangle((x1, y_label, x1 + label_w, y_label + label_h), fill=color)
        draw.text((x1 + 7, y_label + 4), label, fill=(0, 0, 0), font=font)

    banner = "YOLO26-ANE · CoreML/Apple Neural Engine · 5 detections"
    draw.rectangle((0, 0, image.width, 48), fill=(0, 0, 0))
    draw.text((16, 12), banner, fill=(255, 255, 255), font=small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
