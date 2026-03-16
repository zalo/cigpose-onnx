<div align="center">
<h1>CIGPose ONNX Runtime</h1>
<p>Run state-of-the-art whole-body pose estimation with just <code>pip install onnxruntime opencv-python numpy</code>.</p>
<p>No PyTorch. No MMPose. No CUDA toolkit. Just download a model and go.</p>
<br>
<img src="demo.gif" width="640" />
</div>

---

This repo provides pre-exported ONNX models and a single-file inference script for [CIGPose](https://github.com/53mins/CIGPose), a causal-intervention-based pose estimator that achieves **67.5 Whole AP** on COCO-WholeBody — the current state of the art for whole-body pose estimation.

> **Original paper & code**: CIGPose was developed by the [53mins](https://github.com/53mins/CIGPose) team. All model weights originate from their training pipeline built on [MMPose](https://github.com/open-mmlab/mmpose). This repository only handles the ONNX conversion and lightweight inference wrapper.

## Why CIGPose?

Pose estimators tend to latch onto spurious visual cues — a hand near a coffee cup, a shoulder occluded by another person. CIGPose treats this as a **causal inference** problem:

1. **Structural Causal Model** — visual context is modeled as a confounder that creates a non-causal backdoor path between image features and pose predictions. CIGPose targets the interventional distribution P(Y|do(F)) instead of the observational P(Y|F).

2. **Causal Intervention Module (CIM)** — identifies which keypoint embeddings are "confused" by measuring predictive uncertainty, then swaps them out for learned context-invariant canonical embeddings.

3. **Hierarchical Graph Neural Network** — enforces anatomical plausibility through local (intra-part) and global (inter-part) message passing over the skeleton graph.

The net effect: fewer anatomically impossible predictions, especially under occlusion and clutter.

### Results

<div align="center">
  <img src="resources/result_contrast.png" width="100%" />
</div>

**(a)** CIGPose on COCO-WholeBody val — strong accuracy while being data-efficient. **(b)** Side-by-side with RTMPose-x: the baseline hallucinates limbs into background clutter; CIGPose doesn't.

<div align="center">
  <img src="resources/contrast.png" width="100%" />
</div>

Left to right: input, RTMPose-x, CIGPose-x. More examples below.

<div align="center">
  <img src="resources/more_contrast.png" width="100%" />
</div>

---

## Quick Start

```bash
pip install onnxruntime opencv-python numpy
# or: pip install onnxruntime-gpu opencv-python numpy
```

Download the model pack from the [Releases](../../releases) page:

```bash
# grab the latest release zip
wget https://github.com/YOUR_USER/cigpose_onnxruntime/releases/latest/download/cigpose_models.zip

# extract into the models/ directory
unzip cigpose_models.zip -d models/
```

You should end up with:

```
cigpose_onnxruntime/
  models/
    yolox_nano.onnx
    cigpose-m_coco-wholebody_256x192.onnx
    ...
  run_onnx.py
```

Run it:

```bash
# image
python run_onnx.py --model models/cigpose-m_coco-wholebody_256x192.onnx \
                   --detector models/yolox_nano.onnx --image photo.jpg

# video
python run_onnx.py --model models/cigpose-x_coco-ubody_384x288.onnx \
                   --detector models/yolox_nano.onnx --video clip.mp4

# webcam (q = quit, m = toggle keypoint mode)
python run_onnx.py --model models/cigpose-m_coco-wholebody_256x192.onnx \
                   --detector models/yolox_nano.onnx --webcam
```

Without `--detector`, the full frame is treated as one person (useful for pre-cropped inputs).

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *required* | CIGPose ONNX model |
| `--detector` | none | YOLOX ONNX detector |
| `--image / --video / --webcam` | — | Input source |
| `--output, -o` | auto | Output path |
| `--threshold` | 0.6 | Min keypoint confidence to draw |
| `--vis-mode` | yolo29 | `yolo29` (29 kpts) or `all` (full set) |
| `--det-threshold` | 0.5 | Person detection confidence |
| `--det-nms` | 0.45 | Detection NMS IoU |
| `--device` | cpu | `cpu` or `cuda` |
| `--cam-id` | 0 | Webcam index |

---

## Model Zoo

Every ONNX file is self-contained — input size, normalization constants, and split ratio are embedded as metadata. No sidecar configs needed.

### COCO-WholeBody v1.0 val (133 keypoints)

| Model | Input Size | GFLOPs | Body AP | Foot AP | Face AP | Hand AP | Whole AP | Size |
|-------|-----------|--------|---------|---------|---------|---------|----------|------|
| CIGPose-m | 256x192 | 2.3 | 69.0 | 64.3 | 82.1 | 49.7 | 59.9 | 71 MB |
| CIGPose-l | 256x192 | 4.6 | 71.2 | 69.0 | 83.3 | 54.0 | 62.6 | 131 MB |
| CIGPose-l | 384x288 | 10.7 | 73.0 | 72.0 | 88.3 | 59.8 | 66.3 | 142 MB |
| CIGPose-x | 384x288 | 18.7 | 73.5 | 72.3 | 88.1 | 60.2 | 67.0 | 230 MB |
| CIGPose-l +UBody | 256x192 | 4.6 | 71.3 | 66.2 | 83.4 | 55.5 | 63.1 | 131 MB |
| CIGPose-l +UBody | 384x288 | 10.7 | 73.1 | 72.3 | 88.0 | 61.2 | 66.9 | 142 MB |
| CIGPose-x +UBody | 384x288 | 18.7 | 73.5 | 70.3 | 88.4 | 62.6 | **67.5** | 230 MB |

### COCO val2017 (17 body keypoints)

| Model | Input Size | GFLOPs | Params | AP | AR | Size |
|-------|-----------|--------|--------|-----|-----|------|
| CIGPose-m | 256x192 | 1.9 | 14M | 76.6 | 79.3 | 54 MB |
| CIGPose-l | 256x192 | 4.2 | 28M | 77.6 | 80.3 | 108 MB |
| CIGPose-l | 384x288 | 9.4 | 29M | 78.5 | 81.1 | 109 MB |

### CrowdPose test (14 keypoints)

| Model | Input Size | Params | AP | AP easy | AP med | AP hard | Size |
|-------|-----------|--------|-----|---------|--------|---------|------|
| CIGPose-m | 256x192 | 14.4M | 71.4 | 81.0 | 72.7 | 58.9 | 54 MB |
| CIGPose-l | 256x192 | 28.4M | 73.7 | 82.8 | 75.1 | 61.2 | 108 MB |
| CIGPose-l | 384x288 | 28.8M | 74.2 | 82.9 | 75.6 | 62.5 | 109 MB |
| CIGPose-x | 384x288 | 50.4M | 75.8 | 84.2 | 77.3 | 63.6 | 191 MB |

### Person Detector

| Model | License | Input | Size |
|-------|---------|-------|------|
| [YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX) | Apache 2.0 | 416x416 | 3.5 MB |

### Picking a model

| Use case | Recommended | Why |
|----------|-------------|-----|
| Best accuracy | `cigpose-x_coco-ubody_384x288` | 67.5 Whole AP, trained on extra UBody data |
| Balanced | `cigpose-l_coco-wholebody_384x288` | 66.3 AP, ~40% smaller than x |
| Lightweight / real-time | `cigpose-m_coco-wholebody_256x192` | 71 MB, fastest |
| Body keypoints only | `cigpose-l_coco_384x288` | 78.5 AP, 17 standard COCO keypoints |
| Crowded scenes | `cigpose-x_crowdpose_384x288` | Trained on CrowdPose, handles overlap |

---

## Swapping the Detector

YOLOX-Nano ships as the default detector, but you can plug in anything that gives you person bounding boxes.

### Drop-in YOLOX upgrade

Any YOLOX variant (Tiny/S/M/L/X) from the [YOLOX repo](https://github.com/Megvii-BaseDetection/YOLOX) works with the existing `YOLOXDetector` class — just point `--detector` at the larger ONNX file.

### Custom detector

Implement a class with a `detect(frame) -> list[[x1,y1,x2,y2]]` method:

```python
class MyDetector:
    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(model_path, providers=providers or ['CPUExecutionProvider'])

    def detect(self, frame):
        # frame: BGR numpy (H, W, 3)
        # return: list of [x1, y1, x2, y2] in pixel coords
        ...
```

Then wire it up in `main()` or pass it programmatically.

### Pre-computed boxes

If you already have bounding boxes from a tracker or annotation file, skip the detector entirely — omit `--detector` and feed pre-cropped single-person images.

### License note

If you care about keeping your project permissively licensed, stick to detectors under Apache 2.0 (YOLOX, RT-DETR) or MIT (NanoDet). Ultralytics YOLO is AGPL-3.0, which is viral.

---

## How It Works

Standard top-down pipeline:

1. **Detect** — YOLOX finds person bounding boxes
2. **Crop** — each person is cropped with 1.25x padding, aspect-ratio-corrected, resized to model input
3. **Infer** — CIGPose predicts SimCC coordinate classifications (one distribution per keypoint per axis)
4. **Decode** — argmax gives the coordinate, raw logit peak gives confidence
5. **Remap** — coordinates are mapped back to the original frame

Model metadata (input dimensions, normalization constants, split ratio) is baked into each ONNX file, so there are no config files to keep in sync.

---

## Acknowledgements

This project would not exist without the work of the original CIGPose authors:

- **[CIGPose](https://github.com/53mins/CIGPose)** — Causal Intervention Graph Neural Network for Whole-Body Pose Estimation. The model architectures, training pipeline, and all checkpoint weights are their work.
- **[MMPose](https://github.com/open-mmlab/mmpose)** (OpenMMLab) — the pose estimation framework CIGPose is built on.
- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii) — the Apache 2.0 licensed object detector used here for person detection.

## Author

**Namas Bhandari** — [namas.brd@gmail.com](mailto:namas.brd@gmail.com)

ONNX conversion, runtime wrapper, and this repository.

## License

[Apache License 2.0](LICENSE)
