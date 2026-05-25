# Third-Party Notices

This repository is a hackathon submission for the webAI YOLO26 MLX Build Challenge.

## webAI YOLO26-MLX

- Project: `thewebAI/yolo-mlx`
- URL: https://github.com/thewebAI/yolo-mlx
- License: GNU Affero General Public License v3.0
- Use in this submission: baseline methodology, benchmark harness compatibility, YOLO26 MLX comparison path, and challenge model context.

## Ultralytics YOLO26

- Project: Ultralytics YOLO models
- URL: https://docs.ultralytics.com/models/yolo26/
- Use in this submission: YOLO26 model export path and detection parity reference.

## Apple CoreML

- Project: Apple CoreML framework
- URL: https://developer.apple.com/documentation/coreml
- Use in this submission: public Swift/CoreML inference runtime targeting Apple Neural Engine through `MLModelConfiguration.computeUnits`.

## Python Packages Used For Validation

The runtime path is Swift/CoreML. Python is used for conversion, benchmark orchestration, and validation.

- `ultralytics`
- `coremltools`
- `torch`
- `pycocotools`
- `Pillow`

Each dependency remains subject to its own upstream license.

## Model Weights

This repository does not include YOLO26 `.pt`, `.mlpackage`, or `.mlmodelc` model weights. The reproduction docs describe how to generate local artifacts from properly licensed upstream sources.
