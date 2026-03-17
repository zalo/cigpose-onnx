# Author: Namas Bhandari <namas.brd@gmail.com>
# Original CIGPose by 53mins - https://github.com/53mins/CIGPose
# Licensed under Apache 2.0

"""CLI entry point for CIGPose ONNX inference."""

import argparse
import os
import sys

from cigpose import __version__
from cigpose.inference import (
    YOLOXDetector,
    load_pose_model,
    run_on_image,
    run_on_video,
    run_on_webcam,
)

_MODELS_URL = 'https://github.com/namas191297/cigpose-onnx/releases/latest/download/cigpose_models.zip'


def main():
    p = argparse.ArgumentParser(
        prog='cigpose',
        description=(
            'CIGPose ONNX Runtime inference.\n\n'
            'Run whole-body (133 kpts), body (17 kpts), or CrowdPose (14 kpts)\n'
            'pose estimation on images, videos, or webcam. The model auto-detects\n'
            'input size, normalization, and skeleton from embedded ONNX metadata.\n\n'
            'Models: https://github.com/namas191297/cigpose-onnx/releases'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'examples:\n'
            '  cigpose --model cigpose-x_coco-ubody_384x288.onnx \\\n'
            '          --detector yolox_nano.onnx --image photo.jpg\n\n'
            '  cigpose --model cigpose-m_coco-wholebody_256x192.onnx \\\n'
            '          --detector yolox_nano.onnx --video clip.mp4 -o out.mp4\n\n'
            '  cigpose --model cigpose-m_coco-wholebody_256x192.onnx \\\n'
            '          --detector yolox_nano.onnx --webcam\n\n'
            '  cigpose --model cigpose-l_coco_384x288.onnx --image cropped.jpg\n'
            '          (no --detector = full frame treated as one person)\n\n'
            'Original CIGPose by 53mins: https://github.com/53mins/CIGPose'
        ),
    )

    p.add_argument('--version', action='version', version=f'cigpose {__version__}')

    # model args
    model_group = p.add_argument_group('model')
    model_group.add_argument(
        '--model', required=True,
        help='path to CIGPose ONNX model (e.g. cigpose-x_coco-ubody_384x288.onnx)')
    model_group.add_argument(
        '--detector', default=None,
        help='path to YOLOX ONNX detector (e.g. yolox_nano.onnx). '
             'omit to treat the full frame as a single person')

    # input args
    input_group = p.add_argument_group('input')
    input_group.add_argument('--image', help='path to input image')
    input_group.add_argument('--video', help='path to input video')
    input_group.add_argument('--webcam', action='store_true', help='use webcam as input')
    input_group.add_argument('--cam-id', type=int, default=0, help='webcam device index (default: 0)')

    # output args
    output_group = p.add_argument_group('output')
    output_group.add_argument(
        '--output', '-o', default=None,
        help='output path. auto-generated if not set (e.g. photo_pose.jpg). '
             'for webcam, pass a .mp4 path to record')
    output_group.add_argument(
        '--no-display', action='store_true',
        help='skip cv2.imshow windows (for headless servers). requires --output')

    # threshold args
    thresh_group = p.add_argument_group('thresholds')
    thresh_group.add_argument(
        '--threshold', type=float, default=0.6,
        help='keypoint confidence threshold - keypoints below this are hidden (default: 0.6)')
    thresh_group.add_argument(
        '--det-threshold', type=float, default=0.5,
        help='person detection confidence threshold (default: 0.5)')
    thresh_group.add_argument(
        '--det-nms', type=float, default=0.45,
        help='detection NMS IoU threshold (default: 0.45)')

    # runtime args
    runtime_group = p.add_argument_group('runtime')
    runtime_group.add_argument(
        '--device', choices=['cpu', 'cuda'], default='cpu',
        help='inference device (default: cpu). cuda requires onnxruntime-gpu')

    args = p.parse_args()

    if not any([args.image, args.video, args.webcam]):
        p.error('specify one of --image, --video, or --webcam')

    if args.no_display and not args.output:
        p.error('--no-display requires --output')

    if not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}\n")
        print("You need to download the ONNX models before running inference.")
        print(f"Grab them from: {_MODELS_URL}")
        print("\nExtract the zip into a models/ folder, then run something like:")
        print("  cigpose --model models/cigpose-x_coco-ubody_384x288.onnx "
              "--detector models/yolox_nano.onnx --image photo.jpg")
        print(f"\nSee all available models at: https://github.com/namas191297/cigpose-onnx/releases")
        sys.exit(1)

    if args.detector and not os.path.isfile(args.detector):
        print(f"Error: detector file not found: {args.detector}\n")
        print("The detector (yolox_nano.onnx) is included in the models download.")
        print(f"Grab the models zip from: {_MODELS_URL}")
        sys.exit(1)

    show = not args.no_display

    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.device == 'cuda' else ['CPUExecutionProvider'])

    print(f"Loading pose model: {args.model}")
    sess, input_w, input_h, split_ratio = load_pose_model(args.model, providers)
    print(f"Model config: {input_w}x{input_h}, split_ratio={split_ratio}")

    detector = None
    if args.detector:
        print(f"Loading detector: {args.detector}")
        detector = YOLOXDetector(args.detector, conf_thresh=args.det_threshold,
                                 nms_thresh=args.det_nms, providers=providers)

    if args.image:
        out = args.output or args.image.rsplit('.', 1)[0] + '_pose.' + args.image.rsplit('.', 1)[-1]
        run_on_image(sess, args.image, input_w, input_h, split_ratio,
                     out, args.threshold, detector, show=show)
    elif args.video:
        out = args.output or args.video.rsplit('.', 1)[0] + '_pose.mp4'
        run_on_video(sess, args.video, input_w, input_h, split_ratio,
                     out, args.threshold, detector, show=show)
    elif args.webcam:
        run_on_webcam(sess, input_w, input_h, split_ratio,
                      args.threshold, detector, args.cam_id,
                      output_path=args.output, show=show)


if __name__ == '__main__':
    main()
