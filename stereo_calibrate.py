"""Stereo camera calibration using a ChArUco board.

Displays the board on-screen, captures from a side-by-side stereo UVC camera
once per second, accumulates valid stereo pairs, then runs stereoCalibrate and
saves the result to JSON.

Usage:
    python stereo_calibrate.py --camera 0 --output calibration.json

Controls (camera feed window):
    q  — quit (calibrates with pairs collected so far if >= 4)
    s  — force a snapshot right now (instead of waiting for the timer)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board config
# ---------------------------------------------------------------------------

BOARD_COLS = 5       # squares horizontally  (reduced for low-res cameras)
BOARD_ROWS = 3       # squares vertically
SQUARE_LEN = 0.04    # metres  (physical size doesn't matter for relative 3-D,
MARKER_LEN = 0.03    # metres   but it sets the scale of the output T vector)
DICT_ID    = cv2.aruco.DICT_5X5_100

MIN_PAIRS       = 35
CAPTURE_INTERVAL = 1.0   # seconds between automatic captures


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def create_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_LEN, MARKER_LEN, dictionary)
    return board, dictionary


def render_board(board, size=(1200, 900)):
    img = board.generateImage(size, marginSize=30, borderBits=1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_detector(board):
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    def detect(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        return corners, ids

    return detect


# ---------------------------------------------------------------------------
# Point helpers
# ---------------------------------------------------------------------------

def find_common_pts(board, left_corners, left_ids, right_corners, right_ids):
    """Return (obj_pts, left_pts, right_pts) for IDs seen in both views."""
    l_map = {int(i): c for i, c in zip(left_ids.flatten(),  left_corners)}
    r_map = {int(i): c for i, c in zip(right_ids.flatten(), right_corners)}
    common = sorted(set(l_map) & set(r_map))
    if len(common) < 4:
        return None, None, None

    board_corners = board.getChessboardCorners()     # (N_total, 3)
    obj_pts  = np.array([board_corners[i] for i in common], dtype=np.float32)
    left_pts = np.array([l_map[i]         for i in common], dtype=np.float32)
    right_pts= np.array([r_map[i]         for i in common], dtype=np.float32)
    return obj_pts, left_pts, right_pts


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _init_K(image_size):
    """Return a reasonable initial camera matrix for the given image size.

    Using CALIB_USE_INTRINSIC_GUESS with this seed bypasses initIntrinsicParams2D,
    which fails via a homography assertion when corner coverage is sparse or
    the image is low-resolution.  f = max(W, H) is a safe prior for cameras
    with a field-of-view up to ~50°.
    """
    w, h = image_size
    f = float(max(w, h))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0,     1.0]], dtype=np.float64)


def run_stereo_calibration(pairs, image_size):
    obj_list, l_list, r_list = zip(*pairs)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 200, 1e-6)
    flags_mono = cv2.CALIB_USE_INTRINSIC_GUESS   # skip fragile homography bootstrap

    # Step 1: calibrate each camera individually.
    print("  Calibrating left camera…")
    _, K1, D1, _, _ = cv2.calibrateCamera(
        obj_list, l_list, image_size,
        _init_K(image_size), np.zeros(5, dtype=np.float64),
        flags=flags_mono, criteria=criteria)

    print("  Calibrating right camera…")
    _, K2, D2, _, _ = cv2.calibrateCamera(
        obj_list, r_list, image_size,
        _init_K(image_size), np.zeros(5, dtype=np.float64),
        flags=flags_mono, criteria=criteria)

    # Step 2: stereo calibration — fix individual intrinsics, solve only for R and T.
    print("  Running stereoCalibrate…")
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_list, l_list, r_list,
        K1, D1, K2, D2, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0)

    return {
        'image_size': list(image_size),
        'rms':  float(rms),
        'K1':   K1.tolist(),   'D1': D1.tolist(),
        'K2':   K2.tolist(),   'D2': D2.tolist(),
        'R':    R.tolist(),    'T':  T.tolist(),
        'E':    E.tolist(),    'F':  F.tolist(),
        'R1':   R1.tolist(),   'R2': R2.tolist(),
        'P1':   P1.tolist(),   'P2': P2.tolist(),
        'Q':    Q.tolist(),
    }


# ---------------------------------------------------------------------------
# Pair loading from disk
# ---------------------------------------------------------------------------

def load_existing_pairs(save_dir, detect, board):
    pairs = []
    max_idx = 0
    for left_path in sorted(save_dir.glob('pair_*_left.png')):
        num = int(left_path.stem.split('_')[1])
        right_path = save_dir / f'pair_{num:04d}_right.png'
        if not right_path.exists():
            continue
        left_img  = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        lc, li = detect(left_img)
        rc, ri = detect(right_img)
        if lc is None or rc is None:
            continue
        obj_pts, lpts, rpts = find_common_pts(board, lc, li, rc, ri)
        if obj_pts is not None:
            pairs.append((obj_pts, lpts, rpts))
            max_idx = max(max_idx, num + 1)
    return pairs, max_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Stereo ChArUco calibration')
    ap.add_argument('--camera',    type=int,   default=1)
    ap.add_argument('--output',    default='calibration.json')
    ap.add_argument('--save-dir',  default='calibration_pairs')
    ap.add_argument('--min-pairs', type=int,   default=MIN_PAIRS)
    ap.add_argument('--interval',  type=float, default=CAPTURE_INTERVAL)
    ap.add_argument('--board-size',default='1200x900')
    args = ap.parse_args()

    board_w, board_h = map(int, args.board_size.split('x'))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    board, _ = create_board()
    detect    = make_detector(board)

    # Load any pairs already saved to disk before touching the camera.
    pairs, pair_idx = load_existing_pairs(save_dir, detect, board)
    if pairs:
        print(f"Re-loaded {len(pairs)} existing pairs from '{save_dir}/'")

    # Derive image_size from disk pairs (stored images) if we already have enough,
    # so we can calibrate without ever opening the camera.
    image_size = None
    if pairs:
        sample = cv2.imread(str(next(save_dir.glob('pair_*_left.png'))))
        if sample is not None:
            image_size = (sample.shape[1], sample.shape[0])

    if len(pairs) >= args.min_pairs and image_size is not None:
        print(f"Already have {len(pairs)} valid pairs — skipping capture, going straight to calibration.")
    else:
        # Need more pairs: open the camera and start collecting.
        board_img = render_board(board, (board_w, board_h))
        cv2.namedWindow('ChArUco Board — point camera here', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ChArUco Board — point camera here', board_w, board_h)
        cv2.imshow('ChArUco Board — point camera here', board_img)

        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Cannot open camera {args.camera}")
            return

        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W   = fw // 2
        image_size = (W, fh)

        print(f"Stereo frame: {fw}x{fh}  |  each half: {W}x{fh}")
        print(f"Collecting {args.min_pairs} valid stereo pairs — press 'q' to stop early, 's' to force snapshot")

        last_capture = time.time() - args.interval   # fire immediately on first frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            left  = frame[:, :W]
            right = frame[:, W:]
            vis   = frame.copy()
            now   = time.time()

            force_snap = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                force_snap = True

            do_capture = force_snap or (now - last_capture >= args.interval)
            captured   = False

            if do_capture:
                last_capture = now
                lc, li = detect(left)
                rc, ri = detect(right)

                if lc is not None and li is not None and rc is not None and ri is not None:
                    obj_pts, lpts, rpts = find_common_pts(board, lc, li, rc, ri)
                    if obj_pts is not None:
                        pairs.append((obj_pts, lpts, rpts))
                        # Always append; never overwrite existing files.
                        cv2.imwrite(str(save_dir / f'pair_{pair_idx:04d}_left.png'),  left)
                        cv2.imwrite(str(save_dir / f'pair_{pair_idx:04d}_right.png'), right)
                        pair_idx += 1
                        captured  = True
                        print(f"\r  Valid pairs: {len(pairs)}/{args.min_pairs}", end='', flush=True)

                if lc is not None:
                    cv2.aruco.drawDetectedCornersCharuco(vis[:, :W], lc, li)
                if rc is not None:
                    cv2.aruco.drawDetectedCornersCharuco(vis[:, W:], rc, ri)

            n, goal = len(pairs), args.min_pairs
            bar_w   = int(W * min(n / goal, 1.0))
            cv2.rectangle(vis, (0, fh - 12), (bar_w, fh), (0, 220, 0), -1)
            color = (0, 255, 80) if captured else (0, 200, 255)
            cv2.putText(vis, f"Pairs: {n}/{goal}  (q=quit  s=snap)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('Stereo Calibration', vis)

            if len(pairs) >= args.min_pairs:
                print(f"\nReached {len(pairs)} pairs — calibrating…")
                break

        cap.release()
        cv2.destroyAllWindows()

    if len(pairs) < 4:
        print(f"Only {len(pairs)} pairs — need at least 4. Exiting.")
        return

    print(f"\nRunning stereoCalibrate with {len(pairs)} pairs…")
    calib = run_stereo_calibration(pairs, image_size)

    with open(args.output, 'w') as f:
        json.dump(calib, f, indent=2)

    print(f"Saved: {args.output}")
    print(f"RMS reprojection error: {calib['rms']:.4f} px")
    if calib['rms'] > 1.5:
        print("  ⚠  RMS > 1.5 px — try collecting more varied poses or recapturing")


if __name__ == '__main__':
    main()
