from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import SESSIONS_DIR, TRACKED_LANDMARKS
from .models import FramePose, SessionMetadata


class SessionWriter:
    def __init__(self, session_dir: Path, metadata: SessionMetadata) -> None:
        self.session_dir = session_dir
        self.metadata = metadata
        self._landmarks_path = session_dir / "landmarks.csv"
        self._metrics_path = session_dir / "metrics.json"
        self._metadata_path = session_dir / "metadata.json"
        self._csv_file = self._landmarks_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._build_header())
        self._writer.writeheader()
        self._metadata_path.write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_header(self) -> list[str]:
        columns = ["frame_index", "timestamp_ms", "quality"]
        for landmark_name in TRACKED_LANDMARKS:
            lower = landmark_name.lower()
            columns.extend(
                [
                    f"{lower}_x",
                    f"{lower}_y",
                    f"{lower}_z",
                    f"{lower}_visibility",
                    f"{lower}_pixel_x",
                    f"{lower}_pixel_y",
                ]
            )
        return columns

    def append_pose(self, pose: FramePose) -> None:
        row: dict[str, float | int] = {
            "frame_index": pose.frame_index,
            "timestamp_ms": pose.timestamp_ms,
            "quality": pose.quality,
        }
        for landmark_name in TRACKED_LANDMARKS:
            landmark = pose.landmarks.get(landmark_name)
            lower = landmark_name.lower()
            if landmark is None:
                row[f"{lower}_x"] = ""
                row[f"{lower}_y"] = ""
                row[f"{lower}_z"] = ""
                row[f"{lower}_visibility"] = ""
                row[f"{lower}_pixel_x"] = ""
                row[f"{lower}_pixel_y"] = ""
                continue
            row[f"{lower}_x"] = landmark.x
            row[f"{lower}_y"] = landmark.y
            row[f"{lower}_z"] = landmark.z
            row[f"{lower}_visibility"] = landmark.visibility
            row[f"{lower}_pixel_x"] = landmark.pixel_x
            row[f"{lower}_pixel_y"] = landmark.pixel_y
        self._writer.writerow(row)
        self._csv_file.flush()

    def finalize(self, metrics: dict[str, float | int | str]) -> None:
        self._metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._csv_file.close()


class SessionStore:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or SESSIONS_DIR
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, metadata: SessionMetadata) -> SessionWriter:
        session_dir = self.root_dir / metadata.session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        return SessionWriter(session_dir, metadata)

    def list_sessions(self) -> list[dict[str, str]]:
        sessions: list[dict[str, str]] = []
        for session_dir in sorted(self.root_dir.glob("*"), reverse=True):
            metadata_path = session_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "session_id": metadata["session_id"],
                    "participant_id": metadata["participant_id"],
                    "created_at": metadata["created_at"],
                    "path": str(session_dir),
                }
            )
        return sessions

    def load_session(self, session_dir: Path) -> tuple[dict, list[dict], dict]:
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        metrics = {}
        metrics_path = session_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        rows: list[dict] = []
        with (session_dir / "landmarks.csv").open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                rows.append(row)
        return metadata, rows, metrics

    def export_summary_csv(self, output_path: Path | None = None) -> Path:
        output = output_path or (self.root_dir / "summary.csv")
        sessions = self.list_sessions()
        rows: list[dict[str, str | float | int]] = []
        for session in sessions:
            session_dir = Path(session["path"])
            metadata, _, metrics = self.load_session(session_dir)
            row: dict[str, str | float | int] = {
                "session_id": metadata["session_id"],
                "participant_id": metadata["participant_id"],
                "created_at": metadata["created_at"],
                "protocol_name": metadata["protocol_name"],
                "diagnosis": metadata["diagnosis"],
            }
            row.update(metrics)
            rows.append(row)

        fieldnames = sorted({key for row in rows for key in row}) or [
            "session_id",
            "participant_id",
            "created_at",
        ]
        with output.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output


def build_session_metadata(
    participant_id: str,
    operator_name: str,
    protocol_name: str,
    diagnosis: str,
    notes: str,
    camera_model: str,
) -> SessionMetadata:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{participant_id or 'anon'}"
    return SessionMetadata(
        session_id=session_id,
        participant_id=participant_id or "anon",
        operator_name=operator_name,
        protocol_name=protocol_name,
        diagnosis=diagnosis,
        notes=notes,
        camera_model=camera_model,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
