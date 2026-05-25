from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


class RealSenseRuntimeError(RuntimeError):
    pass


@dataclass(slots=True)
class RealSenseFrame:
    timestamp_ms: float
    frame_index: int
    color_image: np.ndarray
    depth_image: np.ndarray
    intrinsics: Any
    depth_scale: float


class RealSenseCamera:
    def __init__(self) -> None:
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.playback = None

    @staticmethod
    def available() -> bool:
        return rs is not None

    def start(
        self,
        record_path: Path | None = None,
        playback_path: Path | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        repeat_playback: bool = False,
    ) -> None:
        if rs is None:
            raise RealSenseRuntimeError("pyrealsense2 is not installed")
        if record_path is not None and playback_path is not None:
            raise RealSenseRuntimeError("record_path and playback_path cannot be used together")

        self.pipeline = rs.pipeline()
        config = rs.config()
        if playback_path is not None:
            config.enable_device_from_file(str(playback_path), repeat_playback=repeat_playback)
        else:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            if record_path is not None:
                config.enable_record_to_file(str(record_path))

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.playback = None
        if playback_path is not None:
            self.playback = profile.get_device().as_playback()
            self.playback.set_real_time(False)

    def read(self, timeout_ms: int = 1000) -> RealSenseFrame | None:
        if self.pipeline is None or self.align is None:
            raise RealSenseRuntimeError("camera is not started")

        success, frames = self.pipeline.try_wait_for_frames(timeout_ms)
        if not success:
            return None
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        return RealSenseFrame(
            timestamp_ms=float(color_frame.get_timestamp()),
            frame_index=int(color_frame.get_frame_number()),
            color_image=color_image,
            depth_image=depth_image,
            intrinsics=intrinsics,
            depth_scale=float(self.depth_scale),
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None
        self.playback = None

    def is_playback_finished(self) -> bool:
        if self.playback is None:
            return False
        return self.playback.current_status() == rs.playback_status.stopped

    @staticmethod
    def deproject_pixel(intrinsics: Any, pixel_x: int, pixel_y: int, depth_m: float) -> tuple[float, float, float]:
        if rs is None:
            raise RealSenseRuntimeError("pyrealsense2 is not installed")
        x_value, y_value, z_value = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(pixel_x), float(pixel_y)],
            float(depth_m),
        )
        return float(x_value), float(y_value), float(z_value)
