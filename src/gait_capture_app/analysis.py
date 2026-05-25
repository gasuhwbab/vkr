from __future__ import annotations

from math import acos, degrees
from statistics import median

import numpy as np

from .models import FramePose


def _landmark_vector(pose: FramePose, name: str) -> np.ndarray | None:
    landmark = pose.landmarks.get(name)
    if landmark is None:
        return None
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def _dominant_progression_axis(poses: list[FramePose]) -> tuple[int, float]:
    pelvis_points: list[np.ndarray] = []
    for pose in poses:
        left = _landmark_vector(pose, "LEFT_HIP")
        right = _landmark_vector(pose, "RIGHT_HIP")
        if left is None or right is None:
            continue
        pelvis_points.append((left + right) / 2.0)
    if len(pelvis_points) < 2:
        return 2, 1.0

    pelvis = np.vstack(pelvis_points)
    spans = np.ptp(pelvis[:, [0, 2]], axis=0)
    axis = 0 if spans[0] >= spans[1] else 2
    direction = np.sign(pelvis[-1, axis] - pelvis[0, axis]) or 1.0
    return axis, float(direction)


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_step_events(poses: list[FramePose]) -> list[dict[str, float | str]]:
    if len(poses) < 10:
        return []

    axis, direction = _dominant_progression_axis(poses)
    samples: list[tuple[float, float, float]] = []
    for pose in poses:
        left = _landmark_vector(pose, "LEFT_HEEL")
        if left is None:
            left = _landmark_vector(pose, "LEFT_ANKLE")
        right = _landmark_vector(pose, "RIGHT_HEEL")
        if right is None:
            right = _landmark_vector(pose, "RIGHT_ANKLE")
        if left is None or right is None:
            continue
        diff = direction * (left[axis] - right[axis])
        samples.append((pose.timestamp_ms, diff, abs(diff)))

    if len(samples) < 10:
        return []

    timestamps = np.array([item[0] for item in samples], dtype=float)
    diff = _smooth(np.array([item[1] for item in samples], dtype=float), window=7)
    amplitude = float(np.percentile(np.abs(diff), 60)) if len(diff) else 0.0
    threshold = max(0.05, amplitude * 0.35)
    events: list[dict[str, float | str]] = []

    for index in range(1, len(diff) - 1):
        prev_value = diff[index - 1]
        value = diff[index]
        next_value = diff[index + 1]
        if value >= threshold and value >= prev_value and value > next_value:
            events.append(
                {
                    "timestamp_ms": float(timestamps[index]),
                    "side": "LEFT",
                    "step_length_m": float(value),
                }
            )
        elif value <= -threshold and value <= prev_value and value < next_value:
            events.append(
                {
                    "timestamp_ms": float(timestamps[index]),
                    "side": "RIGHT",
                    "step_length_m": float(abs(value)),
                }
            )

    filtered: list[dict[str, float | str]] = []
    min_gap_ms = 250.0
    for event in events:
        if not filtered:
            filtered.append(event)
            continue
        last = filtered[-1]
        if float(event["timestamp_ms"]) - float(last["timestamp_ms"]) < min_gap_ms:
            continue
        filtered.append(event)
    return filtered


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return degrees(acos(cosine))


def joint_angle_series(poses: list[FramePose]) -> dict[str, list[tuple[float, float]]]:
    output = {
        "left_knee": [],
        "right_knee": [],
        "left_hip": [],
        "right_hip": [],
    }
    for pose in poses:
        left_hip = _landmark_vector(pose, "LEFT_HIP")
        right_hip = _landmark_vector(pose, "RIGHT_HIP")
        left_knee = _landmark_vector(pose, "LEFT_KNEE")
        right_knee = _landmark_vector(pose, "RIGHT_KNEE")
        left_ankle = _landmark_vector(pose, "LEFT_ANKLE")
        right_ankle = _landmark_vector(pose, "RIGHT_ANKLE")
        left_shoulder = _landmark_vector(pose, "LEFT_SHOULDER")
        right_shoulder = _landmark_vector(pose, "RIGHT_SHOULDER")

        left_knee_angle = None
        right_knee_angle = None
        left_hip_angle = None
        right_hip_angle = None

        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_knee_angle = _angle(left_hip, left_knee, left_ankle)
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_knee_angle = _angle(right_hip, right_knee, right_ankle)
        if left_shoulder is not None and left_hip is not None and left_knee is not None:
            left_hip_angle = _angle(left_shoulder, left_hip, left_knee)
        if right_shoulder is not None and right_hip is not None and right_knee is not None:
            right_hip_angle = _angle(right_shoulder, right_hip, right_knee)

        if left_knee_angle is not None:
            output["left_knee"].append((pose.timestamp_ms, left_knee_angle))
        if right_knee_angle is not None:
            output["right_knee"].append((pose.timestamp_ms, right_knee_angle))
        if left_hip_angle is not None:
            output["left_hip"].append((pose.timestamp_ms, left_hip_angle))
        if right_hip_angle is not None:
            output["right_hip"].append((pose.timestamp_ms, right_hip_angle))
    return output


def pelvis_path(poses: list[FramePose]) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for pose in poses:
        left = _landmark_vector(pose, "LEFT_HIP")
        right = _landmark_vector(pose, "RIGHT_HIP")
        if left is None or right is None:
            continue
        center = (left + right) / 2.0
        points.append((pose.timestamp_ms, float(center[0]), float(center[2])))
    return points


def compute_session_metrics(poses: list[FramePose]) -> dict[str, float | int | str]:
    if len(poses) < 2:
        return {}

    duration_s = max((poses[-1].timestamp_ms - poses[0].timestamp_ms) / 1000.0, 1e-6)
    events = detect_step_events(poses)
    pelvis = pelvis_path(poses)

    path_length = 0.0
    if len(pelvis) >= 2:
        last = np.array([pelvis[0][1], pelvis[0][2]], dtype=float)
        for _, x_value, z_value in pelvis[1:]:
            current = np.array([x_value, z_value], dtype=float)
            path_length += float(np.linalg.norm(current - last))
            last = current

    step_lengths = [float(event["step_length_m"]) for event in events]
    step_times = [
        (float(current["timestamp_ms"]) - float(previous["timestamp_ms"])) / 1000.0
        for previous, current in zip(events, events[1:])
    ]
    same_side_intervals = []
    for index in range(2, len(events)):
        if events[index]["side"] == events[index - 2]["side"]:
            same_side_intervals.append(
                (float(events[index]["timestamp_ms"]) - float(events[index - 2]["timestamp_ms"])) / 1000.0
            )

    cadence = 0.0 if duration_s <= 0 else (len(events) / duration_s) * 60.0
    speed = 0.0 if duration_s <= 0 else path_length / duration_s

    qualities = [pose.quality for pose in poses]
    return {
        "duration_s": round(duration_s, 2),
        "step_count": len(events),
        "cadence_steps_min": round(cadence, 2),
        "mean_step_length_m": round(float(np.mean(step_lengths)), 3) if step_lengths else 0.0,
        "median_step_length_m": round(float(median(step_lengths)), 3) if step_lengths else 0.0,
        "mean_step_time_s": round(float(np.mean(step_times)), 3) if step_times else 0.0,
        "mean_stride_time_s": round(float(np.mean(same_side_intervals)), 3) if same_side_intervals else 0.0,
        "speed_m_s": round(speed, 3),
        "path_length_m": round(path_length, 3),
        "mean_pose_quality": round(float(np.mean(qualities)), 3) if qualities else 0.0,
    }
