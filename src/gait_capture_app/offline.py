from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .analysis import compute_session_metrics
from .models import SessionMetadata, SessionResult
from .pose import PoseEstimator
from .realsense import RealSenseCamera, RealSenseRuntimeError
from .storage import SessionStore


@dataclass(slots=True)
class BagProcessingReport:
    bag_path: Path
    session_result: SessionResult
    frames_processed: int
    poses_saved: int


def _iter_bag_files(paths: list[Path]) -> list[Path]:
    bag_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            bag_files.extend(sorted(path.glob("*.bag")))
            continue
        if path.suffix.lower() == ".bag":
            bag_files.append(path)
    return bag_files


def _unique_session_id(store: SessionStore, base_id: str) -> str:
    candidate = base_id
    index = 1
    while (store.root_dir / candidate).exists():
        candidate = f"{base_id}_{index}"
        index += 1
    return candidate


def _build_offline_metadata(store: SessionStore, bag_path: Path) -> SessionMetadata:
    created_at = datetime.now().isoformat(timespec="seconds")
    session_id = _unique_session_id(store, f"archive_{bag_path.stem}")
    return SessionMetadata(
        session_id=session_id,
        participant_id=bag_path.stem,
        operator_name="offline-import",
        protocol_name="bag-playback",
        diagnosis="",
        notes=f"Source bag: {bag_path.resolve()}",
        camera_model="Intel RealSense playback",
        created_at=created_at,
    )


def process_bag_file(bag_path: Path, store: SessionStore | None = None) -> BagProcessingReport:
    if not bag_path.exists():
        raise FileNotFoundError(f"bag file not found: {bag_path}")
    if bag_path.suffix.lower() != ".bag":
        raise ValueError(f"unsupported file type: {bag_path}")
    if not RealSenseCamera.available():
        raise RealSenseRuntimeError("pyrealsense2 is not installed; run pip install -e '.[camera]'")
    if not PoseEstimator.available():
        raise RuntimeError("mediapipe is not installed; run pip install -e '.[camera]'")

    session_store = store or SessionStore()
    metadata = _build_offline_metadata(session_store, bag_path)
    writer = None
    camera = RealSenseCamera()
    estimator = PoseEstimator()
    poses = []
    frames_processed = 0

    try:
        camera.start(playback_path=bag_path)
        writer = session_store.create_session(metadata)

        while True:
            frame = camera.read(timeout_ms=1000)
            if frame is None:
                if camera.is_playback_finished():
                    break
                continue

            frames_processed += 1
            pose = estimator.estimate(frame)
            if pose is None:
                continue

            poses.append(pose)
            writer.append_pose(pose)

        metrics = compute_session_metrics(poses) if poses else {}
        writer.finalize(metrics)
        return BagProcessingReport(
            bag_path=bag_path,
            session_result=SessionResult(session_dir=str(writer.session_dir), metrics=metrics),
            frames_processed=frames_processed,
            poses_saved=len(poses),
        )
    finally:
        estimator.close()
        camera.stop()
        if writer is not None and not (Path(writer.session_dir) / "metrics.json").exists():
            writer.finalize(compute_session_metrics(poses) if poses else {})


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline processing of RealSense .bag files into gait session outputs",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("archive")],
        help="One or more .bag files or directories with .bag files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional directory for generated session outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    bag_files = _iter_bag_files(args.paths)
    if not bag_files:
        parser.error("no .bag files found")

    store = SessionStore(root_dir=args.output_root) if args.output_root else SessionStore()
    reports: list[BagProcessingReport] = []

    for bag_path in bag_files:
        report = process_bag_file(bag_path=bag_path, store=store)
        reports.append(report)
        print(f"[OK] {bag_path}")
        print(f"  session: {report.session_result.session_dir}")
        print(f"  frames: {report.frames_processed}")
        print(f"  poses: {report.poses_saved}")
        print(f"  metrics: {report.session_result.metrics}")

    print(f"Processed {len(reports)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
