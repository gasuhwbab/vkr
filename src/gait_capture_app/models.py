from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Landmark3D:
    name: str
    x: float
    y: float
    z: float
    visibility: float
    pixel_x: int
    pixel_y: int


@dataclass(slots=True)
class FramePose:
    timestamp_ms: float
    frame_index: int
    quality: float
    landmarks: dict[str, Landmark3D] = field(default_factory=dict)


@dataclass(slots=True)
class SessionMetadata:
    session_id: str
    participant_id: str
    operator_name: str
    protocol_name: str
    diagnosis: str
    notes: str
    camera_model: str
    created_at: str


@dataclass(slots=True)
class PreviewUpdate:
    timestamp_ms: float
    frame_index: int
    color_image: Any | None
    depth_image: Any | None
    pose: FramePose | None
    metrics: dict[str, float | int | str]
    status_text: str


@dataclass(slots=True)
class SessionResult:
    session_dir: str
    metrics: dict[str, float | int | str]


@dataclass(slots=True)
class AppEvent:
    kind: str
    payload: dict[str, Any]
