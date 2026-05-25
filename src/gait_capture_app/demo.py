from __future__ import annotations

from math import pi, sin

from .models import FramePose, Landmark3D


def generate_demo_poses(duration_s: float = 8.0, fps: int = 30) -> list[FramePose]:
    poses: list[FramePose] = []
    frame_count = int(duration_s * fps)
    for frame_index in range(frame_count):
        time_s = frame_index / fps
        pelvis_z = 0.45 * time_s
        pelvis_y = 1.0
        stride = 0.22 * sin(2 * pi * 1.2 * time_s)
        knee_cycle_left = 35 + 20 * max(0.0, sin(2 * pi * 1.2 * time_s))
        knee_cycle_right = 35 + 20 * max(0.0, sin(2 * pi * 1.2 * time_s + pi))

        left_ankle_z = pelvis_z + stride
        right_ankle_z = pelvis_z - stride
        left_knee_y = 0.58 + 0.04 * sin(2 * pi * 1.2 * time_s)
        right_knee_y = 0.58 + 0.04 * sin(2 * pi * 1.2 * time_s + pi)

        landmarks = {
            "LEFT_HIP": Landmark3D("LEFT_HIP", -0.10, pelvis_y, pelvis_z, 0.99, 0, 0),
            "RIGHT_HIP": Landmark3D("RIGHT_HIP", 0.10, pelvis_y, pelvis_z, 0.99, 0, 0),
            "LEFT_SHOULDER": Landmark3D("LEFT_SHOULDER", -0.14, 1.45, pelvis_z, 0.99, 0, 0),
            "RIGHT_SHOULDER": Landmark3D("RIGHT_SHOULDER", 0.14, 1.45, pelvis_z, 0.99, 0, 0),
            "LEFT_KNEE": Landmark3D("LEFT_KNEE", -0.10, left_knee_y, pelvis_z + stride * 0.45, 0.98, 0, 0),
            "RIGHT_KNEE": Landmark3D("RIGHT_KNEE", 0.10, right_knee_y, pelvis_z - stride * 0.45, 0.98, 0, 0),
            "LEFT_ANKLE": Landmark3D("LEFT_ANKLE", -0.10, 0.14, left_ankle_z, 0.98, 0, 0),
            "RIGHT_ANKLE": Landmark3D("RIGHT_ANKLE", 0.10, 0.14, right_ankle_z, 0.98, 0, 0),
            "LEFT_HEEL": Landmark3D("LEFT_HEEL", -0.10, 0.05, left_ankle_z - 0.03, 0.98, 0, 0),
            "RIGHT_HEEL": Landmark3D("RIGHT_HEEL", 0.10, 0.05, right_ankle_z - 0.03, 0.98, 0, 0),
            "LEFT_FOOT_INDEX": Landmark3D("LEFT_FOOT_INDEX", -0.08, 0.03, left_ankle_z + 0.07, 0.98, 0, 0),
            "RIGHT_FOOT_INDEX": Landmark3D("RIGHT_FOOT_INDEX", 0.08, 0.03, right_ankle_z + 0.07, 0.98, 0, 0),
        }
        landmarks["LEFT_KNEE"].visibility = min(1.0, knee_cycle_left / 55.0)
        landmarks["RIGHT_KNEE"].visibility = min(1.0, knee_cycle_right / 55.0)

        poses.append(
            FramePose(
                timestamp_ms=time_s * 1000.0,
                frame_index=frame_index,
                quality=0.98,
                landmarks=landmarks,
            )
        )
    return poses
