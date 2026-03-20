"""Stereo CIGPose inference with real-time 3-D triangulation.

Reads a side-by-side stereo webcam, estimates 2-D poses on both halves,
matches persons across views, triangulates 3-D skeleton coordinates, and
streams them to a Three.js viewer over WebSocket.

Usage:
    python stereo_infer.py \\
        --model   models/cigpose-m_coco-wholebody_256x192.onnx \\
        --detector models/yolox_nano.onnx \\
        --calib   calibration.json

The viewer is served automatically and opened in your default browser.

Controls:
    q — quit
"""

import argparse
import asyncio
import http.server
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Re-use detection / pose helpers from run_onnx.py
sys.path.insert(0, str(Path(__file__).parent))
from run_onnx import (
    YOLOXDetector,
    _SKELETON_MAP,
    decode_simcc,
    draw_pose,
    preprocess_person,
    remap_to_frame,
)

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration(path):
    with open(path) as f:
        c = json.load(f)

    K1 = np.array(c['K1'], dtype=np.float64)
    D1 = np.array(c['D1'], dtype=np.float64)
    K2 = np.array(c['K2'], dtype=np.float64)
    D2 = np.array(c['D2'], dtype=np.float64)
    R1 = np.array(c['R1'], dtype=np.float64)
    R2 = np.array(c['R2'], dtype=np.float64)
    P1 = np.array(c['P1'], dtype=np.float64)
    P2 = np.array(c['P2'], dtype=np.float64)
    image_size = tuple(c['image_size'])   # (W, H) of each half

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    return dict(
        K1=K1, D1=D1, K2=K2, D2=D2,
        R1=R1, R2=R2, P1=P1, P2=P2,
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
    )


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def rectify_kpts(kpts, K, D, R, P):
    """Undistort + rectify (K,2) → (K,2) in rectified image coords."""
    pts = kpts.reshape(-1, 1, 2).astype(np.float32)
    return cv2.undistortPoints(pts, K, D, R=R, P=P).reshape(-1, 2)


def triangulate_kpts(P1, P2, kpts1, kpts2):
    """Triangulate (K,2) rectified pairs → (K,3) in camera-space metres."""
    pts4d = cv2.triangulatePoints(P1, P2, kpts1.T.astype(np.float64),
                                           kpts2.T.astype(np.float64))
    w = pts4d[3:4]
    w[np.abs(w) < 1e-9] = 1e-9
    return (pts4d[:3] / w).T   # (K, 3)


# ---------------------------------------------------------------------------
# Batch inference over a list of bboxes from one camera half
# ---------------------------------------------------------------------------

def run_half(pose_session, frame_half, bboxes, input_w, input_h, split_ratio):
    """Return list of dicts {bbox, kpts, scores} for each detected person."""
    if not bboxes:
        return []

    tensors, regions = [], []
    for bbox in bboxes:
        t, r = preprocess_person(frame_half, bbox, input_w, input_h)
        tensors.append(t)
        regions.append(r)

    batch = np.concatenate(tensors, axis=0)          # [N, 3, H, W]
    sx_batch, sy_batch = pose_session.run(None, {'input': batch})

    results = []
    for i, (bbox, region) in enumerate(zip(bboxes, regions)):
        kpts, scores = decode_simcc(
            sx_batch[i:i+1], sy_batch[i:i+1], input_w, input_h, split_ratio)
        kpts = remap_to_frame(kpts, region, input_w, input_h)
        results.append({'bbox': bbox, 'kpts': kpts, 'scores': scores})
    return results


# ---------------------------------------------------------------------------
# Person matching across views (epipolar: same y after rectification)
# ---------------------------------------------------------------------------

def match_persons(left_persons, right_persons, calib, max_y_diff=40):
    """Match left↔right persons by rectified-image y-centroid proximity.

    Returns list of (left_idx, right_idx) pairs.
    """
    if not left_persons or not right_persons:
        return []

    K1, D1, R1, P1 = calib['K1'], calib['D1'], calib['R1'], calib['P1']
    K2, D2, R2, P2 = calib['K2'], calib['D2'], calib['R2'], calib['P2']

    def y_centroid(person, K, D, R, P):
        kpts_rect = rectify_kpts(person['kpts'], K, D, R, P)
        return float(np.median(kpts_rect[:, 1]))

    left_y  = [y_centroid(p, K1, D1, R1, P1) for p in left_persons]
    right_y = [y_centroid(p, K2, D2, R2, P2) for p in right_persons]

    matches, used_right = [], set()
    for li, ly in enumerate(left_y):
        best_diff, best_ri = min(
            ((abs(ly - ry), ri) for ri, ry in enumerate(right_y)
             if ri not in used_right),
            default=(float('inf'), -1),
        )
        if best_diff < max_y_diff:
            matches.append((li, best_ri))
            used_right.add(best_ri)
    return matches


# ---------------------------------------------------------------------------
# WebSocket streamer
# ---------------------------------------------------------------------------

class PoseStreamer:
    """Thread-safe WebSocket broadcaster.  Call .send(data_dict) from any thread."""

    def __init__(self, host='localhost', port=8765):
        if not _HAS_WS:
            return
        self._clients = set()
        self._loop    = asyncio.new_event_loop()
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.host, self.port = host, port

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"WebSocket: ws://{self.host}:{self.port}  →  open viewer/index.html")
            await asyncio.Future()

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    def send(self, data: dict):
        if not _HAS_WS or not self._clients:
            return
        msg = json.dumps(data)
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    async def _broadcast(self, msg):
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead


# ---------------------------------------------------------------------------
# HTTP viewer server
# ---------------------------------------------------------------------------

def serve_viewer(viewer_dir: Path, http_port: int, ws_port: int, open_browser: bool):
    """Serve viewer/index.html in a daemon thread and optionally open a browser tab.

    The viewer HTML is patched on-the-fly so the embedded WS URL reflects the
    actual ws_port, avoiding hard-coded values in the static file.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = viewer_dir / 'index.html'
            html = path.read_bytes().replace(
                b'ws://localhost:8765',
                f'ws://localhost:{ws_port}'.encode(),
            )
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *_):   # silence request logs
            pass

    server = http.server.HTTPServer(('localhost', http_port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://localhost:{http_port}/'
    print(f"Viewer:    {url}")

    if open_browser:
        # Small delay so the server is ready before the browser hits it
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Stereo CIGPose 3-D inference')
    ap.add_argument('--model',      required=True)
    ap.add_argument('--detector',   default=None)
    ap.add_argument('--calib',      required=True, help='calibration.json')
    ap.add_argument('--camera',     type=int,   default=0)
    ap.add_argument('--threshold',  type=float, default=0.5)
    ap.add_argument('--det-threshold', type=float, default=0.45)
    ap.add_argument('--det-nms',    type=float, default=0.45)
    ap.add_argument('--device',     choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--ws-port',    type=int,   default=8765)
    ap.add_argument('--http-port',  type=int,   default=8766, help='Port for the HTML viewer (default: 8766)')
    ap.add_argument('--no-browser', action='store_true', help='Serve viewer but do not auto-open browser')
    ap.add_argument('--no-viewer',  action='store_true', help='Disable WebSocket server and viewer entirely')
    args = ap.parse_args()

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.device == 'cuda' else ['CPUExecutionProvider'])

    # --- Load calibration ---
    print(f"Loading calibration: {args.calib}")
    calib = load_calibration(args.calib)
    calib_img_w, calib_img_h = calib['P1'][0, 2] * 2, calib['P1'][1, 2] * 2  # approx

    # --- Load models ---
    print(f"Loading pose model: {args.model}")
    pose_sess = ort.InferenceSession(args.model, providers=providers)
    meta_str  = pose_sess.get_modelmeta().custom_metadata_map.get('cigpose_meta')
    if meta_str:
        meta  = json.loads(meta_str)
        input_w, input_h = meta['input_w'], meta['input_h']
        split_ratio       = meta['split_ratio']
    else:
        input_w, input_h, split_ratio = 192, 256, 2.0
    print(f"  Pose model: {input_w}x{input_h}, split_ratio={split_ratio}")

    detector = None
    if args.detector:
        print(f"Loading detector: {args.detector}")
        detector = YOLOXDetector(
            args.detector,
            conf_thresh=args.det_threshold,
            nms_thresh=args.det_nms,
            providers=providers,
        )

    # --- WebSocket streamer + HTTP viewer ---
    streamer = None
    if not args.no_viewer and _HAS_WS:
        streamer = PoseStreamer(port=args.ws_port)
        viewer_dir = Path(__file__).parent / 'viewer'
        if viewer_dir.is_dir():
            serve_viewer(viewer_dir, args.http_port, args.ws_port,
                         open_browser=not args.no_browser)
        else:
            print(f"viewer/ directory not found at {viewer_dir} — skipping HTTP server")
    elif not _HAS_WS:
        print("websockets not installed — 3-D viewer disabled")

    # --- Open camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W  = fw // 2
    print(f"Stereo frame: {fw}x{fh}  |  each half: {W}x{fh}")
    print("Press 'q' to quit")

    skeleton = _SKELETON_MAP.get(17, _SKELETON_MAP.get(14, []))  # prefer COCO-17

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0    = time.time()
        left  = frame[:, :W]
        right = frame[:, W:]

        # --- Detection ---
        if detector:
            left_bboxes  = detector.detect(left)
            right_bboxes = detector.detect(right)
        else:
            left_bboxes  = [[0, 0, W, fh]]
            right_bboxes = [[0, 0, W, fh]]

        # --- Pose estimation (batched per half) ---
        left_persons  = run_half(pose_sess, left,  left_bboxes,  input_w, input_h, split_ratio)
        right_persons = run_half(pose_sess, right, right_bboxes, input_w, input_h, split_ratio)

        # --- Match and triangulate ---
        persons_3d = []
        matches = match_persons(left_persons, right_persons, calib)

        for li, ri in matches:
            lp = left_persons[li]
            rp = right_persons[ri]

            # Undistort + rectify keypoints from each view
            kpts1_rect = rectify_kpts(lp['kpts'], calib['K1'], calib['D1'],
                                      calib['R1'], calib['P1'])
            kpts2_rect = rectify_kpts(rp['kpts'], calib['K2'], calib['D2'],
                                      calib['R2'], calib['P2'])

            kpts3d = triangulate_kpts(calib['P1'], calib['P2'],
                                      kpts1_rect, kpts2_rect)

            # Average confidence across views
            scores = (lp['scores'] + rp['scores']) / 2.0

            persons_3d.append({
                'kpts3d': kpts3d.tolist(),
                'scores': scores.tolist(),
            })

        # --- Stream to viewer ---
        if streamer:
            K = len(persons_3d[0]['kpts3d']) if persons_3d else 17
            streamer.send({
                'type':     'pose',
                'persons':  persons_3d,
                'skeleton': _SKELETON_MAP.get(K, skeleton),
            })

        # --- 2-D visualisation ---
        vis = frame.copy()
        for p in left_persons:
            draw_pose(vis[:, :W], p['kpts'], p['scores'], args.threshold)
        for p in right_persons:
            draw_pose(vis[:, W:], p['kpts'], p['scores'], args.threshold)

        # Highlight matched persons with a coloured bbox tint
        for li, ri in matches:
            lp, rp = left_persons[li], right_persons[ri]
            x1, y1, x2, y2 = [int(v) for v in lp['bbox']]
            cv2.rectangle(vis[:, :W], (x1, y1), (x2, y2), (0, 255, 180), 2)
            x1, y1, x2, y2 = [int(v) for v in rp['bbox']]
            cv2.rectangle(vis[:, W:], (x1, y1), (x2, y2), (0, 255, 180), 2)

        dt   = time.time() - t0
        info = (f"{1/max(dt,1e-6):.1f} FPS | "
                f"L:{len(left_persons)} R:{len(right_persons)} "
                f"matched:{len(matches)}")
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow('Stereo CIGPose', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
