from __future__ import annotations

import numpy as np

from .models import FramePose, Landmark3D
from .realsense import RealSenseCamera, RealSenseFrame

try:
    import mediapipe as mp
except Exception:
    mp = None


LANDMARK_INDEX = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}


class PoseEstimator:
    def __init__(self) -> None:
        self._pose = None
        if mp is not None:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    @staticmethod
    def available() -> bool:
        return mp is not None

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()

    def _sample_depth_m(self, depth_image: np.ndarray, pixel_x: int, pixel_y: int, depth_scale: float) -> float | None:
        height, width = depth_image.shape[:2]
        if not (0 <= pixel_x < width and 0 <= pixel_y < height):
            return None
        y0 = max(0, pixel_y - 2)
        y1 = min(height, pixel_y + 3)
        x0 = max(0, pixel_x - 2)
        x1 = min(width, pixel_x + 3)
        patch = depth_image[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid) * depth_scale)

    def estimate(self, frame: RealSenseFrame) -> FramePose | None:
        if self._pose is None:
            return None

        result = self._pose.process(frame.color_image)
        if result.pose_landmarks is None:
            return None

        image_height, image_width = frame.color_image.shape[:2]
        landmarks: dict[str, Landmark3D] = {}
        qualities: list[float] = []

        for name, index in LANDMARK_INDEX.items():
            landmark = result.pose_landmarks.landmark[index]
            pixel_x = int(np.clip(landmark.x * image_width, 0, image_width - 1))
            pixel_y = int(np.clip(landmark.y * image_height, 0, image_height - 1))
            depth_m = self._sample_depth_m(frame.depth_image, pixel_x, pixel_y, frame.depth_scale)
            if depth_m is None:
                continue
            x_value, y_value, z_value = RealSenseCamera.deproject_pixel(
                frame.intrinsics,
                pixel_x,
                pixel_y,
                depth_m,
            )
            qualities.append(float(landmark.visibility))
            landmarks[name] = Landmark3D(
                name=name,
                x=x_value,
                y=y_value,
                z=z_value,
                visibility=float(landmark.visibility),
                pixel_x=pixel_x,
                pixel_y=pixel_y,
            )

        if not landmarks:
            return None

        quality = float(np.mean(qualities)) if qualities else 0.0
        return FramePose(
            timestamp_ms=frame.timestamp_ms,
            frame_index=frame.frame_index,
            quality=quality,
            landmarks=landmarks,
        )
