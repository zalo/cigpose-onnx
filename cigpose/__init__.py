# Author: Namas Bhandari <namas.brd@gmail.com>
# Original CIGPose by 53mins - https://github.com/53mins/CIGPose
# Licensed under Apache 2.0

__version__ = "1.0.1"

from cigpose.inference import (
    YOLOXDetector,
    load_pose_model,
    preprocess_person,
    decode_simcc,
    remap_to_frame,
    draw_pose,
    draw_bboxes,
    infer_persons,
    run_on_image,
    run_on_video,
    run_on_webcam,
    COCO17_SKELETON,
    COCO133_SKELETON,
    CROWDPOSE14_SKELETON,
)
