#!/usr/bin/env python3
# Copyright (c) 2025 Namas Bhandari <namas.brd@gmail.com>
# ONNX runtime wrapper for CIGPose pose estimation models.
# Original CIGPose by 53mins - https://github.com/53mins/CIGPose
# Licensed under Apache 2.0

"""
Top-down pose estimation using CIGPose ONNX models.

Supports COCO-WholeBody (133 kpts), COCO body (17 kpts), and CrowdPose (14 kpts).
Person detection via YOLOX (Apache 2.0, Megvii). Only needs onnxruntime, opencv, numpy.

Examples:
    python run_onnx.py --model models/cigpose-m_coco-wholebody_256x192.onnx \\
                       --detector models/yolox_nano.onnx --image test.jpg
    python run_onnx.py --model models/cigpose-x_coco-ubody_384x288.onnx \\
                       --detector models/yolox_nano.onnx --webcam
"""

import argparse
import json
import time

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Keypoint definitions & skeleton connectivity
# ---------------------------------------------------------------------------

# 29-keypoint subset used by YOLO-Pose, mapped from COCO-WholeBody 133
WHOLEBODY_TO_YOLO29 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  # body
    96, 100, 108, 117, 121, 129,                                   # hand tips
    17, 18, 19, 20, 21, 22,                                        # feet
]

YOLO29_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (9, 17), (9, 18), (9, 19), (10, 20), (10, 21), (10, 22),
    (15, 23), (15, 24), (15, 25), (16, 26), (16, 27), (16, 28),
]

COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

CROWDPOSE14_SKELETON = [
    (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 1), (0, 6), (1, 7),
    (6, 7), (6, 8), (8, 10), (7, 9), (9, 11),
    (12, 13),
]

# fmt: off
COCO133_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22),
    (9, 91), (10, 112),
    # left hand
    (91, 92), (92, 93), (93, 94), (94, 95),
    (91, 96), (96, 97), (97, 98), (98, 99),
    (91, 100), (100, 101), (101, 102), (102, 103),
    (91, 104), (104, 105), (105, 106), (106, 107),
    (91, 108), (108, 109), (109, 110), (110, 111),
    # right hand
    (112, 113), (113, 114), (114, 115), (115, 116),
    (112, 117), (117, 118), (118, 119), (119, 120),
    (112, 121), (121, 122), (122, 123), (123, 124),
    (112, 125), (125, 126), (126, 127), (127, 128),
    (112, 129), (129, 130), (130, 131), (131, 132),
    # face
    *[(23+i, 24+i) for i in range(16)],
    *[(40+i, 41+i) for i in range(4)],
    *[(45+i, 46+i) for i in range(4)],
    *[(50+i, 51+i) for i in range(3)], (50, 54), (54, 55), (55, 56), (56, 57), (57, 58),
    *[(59+i, 60+i) for i in range(5)], (59, 64),
    *[(65+i, 66+i) for i in range(5)], (65, 70),
    *[(71+i, 72+i) for i in range(11)], (71, 82),
    *[(83+i, 84+i) for i in range(7)], (83, 90),
]
# fmt: on

# ImageNet normalization (same as MMPose / torchvision defaults)
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Maps keypoint count -> skeleton lookup
_SKELETON_MAP = {
    133: {'yolo29': (WHOLEBODY_TO_YOLO29, YOLO29_SKELETON), 'all': (None, COCO133_SKELETON)},
    17:  {'all': (None, COCO17_SKELETON)},
    14:  {'all': (None, CROWDPOSE14_SKELETON)},
}


# ---------------------------------------------------------------------------
# Person detection (YOLOX)
# ---------------------------------------------------------------------------

class YOLOXDetector:
    """YOLOX person detector via ONNX Runtime.

    Handles the grid-based YOLOX output format: raw predictions need to be
    decoded using grid offsets and stride multipliers before they become
    pixel-space bounding boxes.

    Output: [1, N, 85] where 85 = cxcywh(4) + objectness(1) + classes(80).
    Person is class index 0.
    """

    def __init__(self, model_path, input_size=416, conf_thresh=0.35,
                 nms_thresh=0.45, providers=None):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.session = ort.InferenceSession(
            model_path, providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self._grids, self._strides = self._build_grids()

    def _build_grids(self):
        """Pre-compute grid coordinates and strides for decoding."""
        grids, strides = [], []
        for s in [8, 16, 32]:
            g = self.input_size // s
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing='ij')
            grids.append(np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32))
            strides.append(np.full((g * g, 1), s, dtype=np.float32))
        return np.concatenate(grids), np.concatenate(strides)

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        ratio = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        return blob[np.newaxis], ratio

    def _decode(self, raw):
        """Decode raw YOLOX grid predictions to pixel-space cxcywh."""
        decoded = raw.copy()
        decoded[:, :2] = (raw[:, :2] + self._grids) * self._strides
        decoded[:, 2:4] = np.exp(raw[:, 2:4]) * self._strides
        return decoded

    def detect(self, frame):
        """Return list of [x1, y1, x2, y2] person bboxes in original pixel coords."""
        blob, ratio = self._letterbox(frame)
        raw = self.session.run(None, {self.input_name: blob})[0][0]
        preds = self._decode(raw)

        scores = preds[:, 4] * preds[:, 5]  # objectness * person_class
        keep = scores >= self.conf_thresh
        if not np.any(keep):
            return []

        boxes, scores = preds[keep, :4], scores[keep]

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), scores.tolist(),
            self.conf_thresh, self.nms_thresh,
        )

        if len(indices) == 0:
            return []
        indices = indices.flatten()
        return [[x1[i] / ratio, y1[i] / ratio,
                 x2[i] / ratio, y2[i] / ratio] for i in indices]


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def preprocess_person(frame, bbox, input_w, input_h):
    """Crop a person bbox from the frame, resize, and normalize for the pose model."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1

    aspect = input_w / input_h
    if bw / max(bh, 1) > aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    bw *= 1.25
    bh *= 1.25

    sx1 = int(max(0, cx - bw / 2))
    sy1 = int(max(0, cy - bh / 2))
    sx2 = int(min(frame.shape[1], cx + bw / 2))
    sy2 = int(min(frame.shape[0], cy + bh / 2))

    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        crop = frame

    resized = cv2.resize(crop, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[np.newaxis]
    return tensor, (sx1, sy1, sx2, sy2)


def decode_simcc(simcc_x, simcc_y, input_w, input_h, split_ratio):
    """Decode SimCC logits to (K, 2) coords + (K,) confidence scores.

    Confidence is min(max_logit_x, max_logit_y) per keypoint, matching
    the MMPose get_simcc_maximum convention.
    """
    x_locs = np.argmax(simcc_x[0], axis=-1).astype(np.float32)
    y_locs = np.argmax(simcc_y[0], axis=-1).astype(np.float32)

    max_val_x = np.max(simcc_x[0], axis=-1)
    max_val_y = np.max(simcc_y[0], axis=-1)
    scores = np.minimum(max_val_x, max_val_y)

    kpts = np.stack([x_locs / split_ratio, y_locs / split_ratio], axis=-1)
    return kpts, scores


def remap_to_frame(kpts, crop_region, input_w, input_h):
    """Map keypoints from model-input coords back to original frame coords."""
    sx1, sy1, sx2, sy2 = crop_region
    scale_x = (sx2 - sx1) / input_w
    scale_y = (sy2 - sy1) / input_h
    out = kpts.copy()
    out[:, 0] = kpts[:, 0] * scale_x + sx1
    out[:, 1] = kpts[:, 1] * scale_y + sy1
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_pose(frame, keypoints, scores, threshold=0.6, mode='yolo29'):
    """Overlay skeleton and keypoints. Picks the right skeleton based on K."""
    n_kpts = len(keypoints)

    spec = _SKELETON_MAP.get(n_kpts, {})
    idx_map, skeleton = spec.get(mode, spec.get('all', (None, [])))

    if idx_map is not None:
        kpts = keypoints[idx_map]
        kscores = scores[idx_map]
    else:
        kpts, kscores = keypoints, scores

    K = len(kpts)
    for i, j in skeleton:
        if i >= K or j >= K:
            continue
        if kscores[i] < threshold or kscores[j] < threshold:
            continue
        cv2.line(frame,
                 (int(kpts[i, 0]), int(kpts[i, 1])),
                 (int(kpts[j, 0]), int(kpts[j, 1])),
                 (0, 255, 128), 2, cv2.LINE_AA)

    for k in range(K):
        if kscores[k] < threshold:
            continue
        x, y = int(kpts[k, 0]), int(kpts[k, 1])
        color = (0, 255, 0) if kscores[k] >= 0.5 else (0, 200, 255)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def _draw_bboxes(frame, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)


# ---------------------------------------------------------------------------
# Inference loops
# ---------------------------------------------------------------------------

def _infer_persons(session, frame, bboxes, input_w, input_h, split_ratio,
                   threshold, vis_mode):
    """Run pose estimation on each detected person and draw results."""
    vis = frame.copy()
    for bbox in bboxes:
        tensor, crop_region = preprocess_person(frame, bbox, input_w, input_h)
        simcc_x, simcc_y = session.run(None, {'input': tensor})
        kpts, scores = decode_simcc(simcc_x, simcc_y, input_w, input_h, split_ratio)
        kpts = remap_to_frame(kpts, crop_region, input_w, input_h)
        draw_pose(vis, kpts, scores, threshold, vis_mode)
    return vis


def run_on_image(session, image_path, input_w, input_h, split_ratio,
                 output_path, threshold, vis_mode, detector=None):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot read {image_path}")
        return

    bboxes = detector.detect(frame) if detector else [[0, 0, frame.shape[1], frame.shape[0]]]
    print(f"Detected {len(bboxes)} person(s)")

    vis = _infer_persons(session, frame, bboxes, input_w, input_h,
                         split_ratio, threshold, vis_mode)
    if detector:
        _draw_bboxes(vis, bboxes)

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Saved: {output_path}")
    else:
        cv2.imshow('CIGPose', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(session, video_path, input_w, input_h, split_ratio,
                 output_path, threshold, vis_mode, detector=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        bboxes = detector.detect(frame) if detector else [[0, 0, w, h]]
        vis = _infer_persons(session, frame, bboxes, input_w, input_h,
                             split_ratio, threshold, vis_mode)
        if detector:
            _draw_bboxes(vis, bboxes)
        dt = time.time() - t0

        idx += 1
        info = f'{1 / max(dt, 1e-6):.1f} FPS | {len(bboxes)} person(s)'
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if total > 0:
            print(f"\rFrame {idx}/{total} ({info})", end='', flush=True)

        if writer:
            writer.write(vis)
        else:
            cv2.imshow('CIGPose', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print()
    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {output_path}")
    cv2.destroyAllWindows()


def run_on_webcam(session, input_w, input_h, split_ratio, threshold, vis_mode,
                  detector=None, cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Cannot open webcam {cam_id}")
        return

    print("Controls: 'q' quit, 'm' toggle vis mode")
    cur_mode = vis_mode

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        bboxes = detector.detect(frame) if detector else [[0, 0, frame.shape[1], frame.shape[0]]]
        vis = _infer_persons(session, frame, bboxes, input_w, input_h,
                             split_ratio, threshold, cur_mode)
        if detector:
            _draw_bboxes(vis, bboxes)
        dt = time.time() - t0

        info = f'{1 / max(dt, 1e-6):.1f} FPS | {len(bboxes)} person(s) | {cur_mode}'
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('CIGPose', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            cur_mode = 'all' if cur_mode == 'yolo29' else 'yolo29'

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='CIGPose ONNX inference')
    p.add_argument('--model', required=True, help='CIGPose ONNX model path')
    p.add_argument('--detector', default=None, help='YOLOX ONNX detector (omit for single-person full-frame)')
    p.add_argument('--image', help='Input image')
    p.add_argument('--video', help='Input video')
    p.add_argument('--webcam', action='store_true', help='Use webcam')
    p.add_argument('--cam-id', type=int, default=0)
    p.add_argument('--output', '-o', default=None, help='Output path')
    p.add_argument('--threshold', type=float, default=0.6, help='Keypoint confidence threshold')
    p.add_argument('--vis-mode', choices=['yolo29', 'all'], default='yolo29',
                   help='yolo29: 29 kpts, all: full keypoint set')
    p.add_argument('--det-threshold', type=float, default=0.5)
    p.add_argument('--det-nms', type=float, default=0.45)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()

    if not any([args.image, args.video, args.webcam]):
        p.error('Specify --image, --video, or --webcam')

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.device == 'cuda' else ['CPUExecutionProvider'])

    print(f"Loading pose model: {args.model}")
    sess = ort.InferenceSession(args.model, providers=providers)

    # Model metadata is embedded in the ONNX file during export
    meta_str = sess.get_modelmeta().custom_metadata_map.get('cigpose_meta')
    if meta_str:
        meta = json.loads(meta_str)
        input_w, input_h = meta['input_w'], meta['input_h']
        split_ratio = meta['split_ratio']
        print(f"Model config: {input_w}x{input_h}, split_ratio={split_ratio}")
    else:
        input_w, input_h, split_ratio = 192, 256, 2.0
        print(f"No embedded metadata, falling back to {input_w}x{input_h}")

    detector = None
    if args.detector:
        print(f"Loading detector: {args.detector}")
        detector = YOLOXDetector(args.detector, conf_thresh=args.det_threshold,
                                 nms_thresh=args.det_nms, providers=providers)

    if args.image:
        out = args.output or args.image.rsplit('.', 1)[0] + '_pose.' + args.image.rsplit('.', 1)[-1]
        run_on_image(sess, args.image, input_w, input_h, split_ratio,
                     out, args.threshold, args.vis_mode, detector)
    elif args.video:
        out = args.output or args.video.rsplit('.', 1)[0] + '_pose.mp4'
        run_on_video(sess, args.video, input_w, input_h, split_ratio,
                     out, args.threshold, args.vis_mode, detector)
    elif args.webcam:
        run_on_webcam(sess, input_w, input_h, split_ratio,
                      args.threshold, args.vis_mode, detector, args.cam_id)


if __name__ == '__main__':
    main()
