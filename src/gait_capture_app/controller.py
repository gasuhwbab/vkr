from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from .analysis import compute_session_metrics
from .demo import generate_demo_poses
from .models import AppEvent, PreviewUpdate, SessionResult
from .pose import PoseEstimator
from .realsense import RealSenseCamera, RealSenseRuntimeError
from .storage import SessionStore


class CaptureService:
    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.events: queue.Queue[AppEvent] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def start_live_session(self, metadata, record_bag: bool = True) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_live_session,
            args=(metadata, record_bag),
            daemon=True,
        )
        self._thread.start()

    def start_demo_session(self, metadata) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_demo_session,
            args=(metadata,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def drain_events(self) -> list[AppEvent]:
        events: list[AppEvent] = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        return events

    def _emit(self, kind: str, **payload) -> None:
        self.events.put(AppEvent(kind=kind, payload=payload))

    def _run_live_session(self, metadata, record_bag: bool) -> None:
        self._running = True
        writer = None
        camera = RealSenseCamera()
        estimator = PoseEstimator()
        poses = []
        try:
            writer = self.store.create_session(metadata)
            session_dir = Path(writer.session_dir)
            bag_path = session_dir / "recording.bag" if record_bag else None
            self._emit("status", message="Инициализация RealSense и модулей обработки")
            camera.start(record_path=bag_path)
            self._emit(
                "status",
                message=(
                    "Запись начата"
                    if estimator.available()
                    else "Запись начата, но mediapipe недоступен — сохраняется сырой поток"
                ),
            )

            while not self._stop_event.is_set():
                frame = camera.read(timeout_ms=1000)
                if frame is None:
                    continue

                pose = estimator.estimate(frame)
                if pose is not None:
                    poses.append(pose)
                    writer.append_pose(pose)

                if frame.frame_index % 3 == 0:
                    metrics = compute_session_metrics(poses) if poses else {}
                    self._emit(
                        "preview",
                        update=PreviewUpdate(
                            timestamp_ms=frame.timestamp_ms,
                            frame_index=frame.frame_index,
                            color_image=frame.color_image,
                            depth_image=frame.depth_image,
                            pose=pose,
                            metrics=metrics,
                            status_text="Идёт захват RealSense",
                        ),
                    )

            metrics = compute_session_metrics(poses) if poses else {}
            writer.finalize(metrics)
            self._emit(
                "session_saved",
                result=SessionResult(session_dir=str(writer.session_dir), metrics=metrics),
            )
        except FileExistsError:
            self._emit("error", message="Сеанс с таким идентификатором уже существует")
        except RealSenseRuntimeError as error:
            self._emit("error", message=str(error))
        except Exception as error:
            self._emit("error", message=f"Ошибка захвата: {error}")
        finally:
            estimator.close()
            camera.stop()
            if writer is not None and not (Path(writer.session_dir) / "metrics.json").exists():
                writer.finalize(compute_session_metrics(poses) if poses else {})
            self._running = False
            self._emit("status", message="Запись остановлена")

    def _run_demo_session(self, metadata) -> None:
        self._running = True
        writer = None
        poses = []
        try:
            writer = self.store.create_session(metadata)
            self._emit("status", message="Запущен demo-режим")
            for pose in generate_demo_poses():
                if self._stop_event.is_set():
                    break
                poses.append(pose)
                writer.append_pose(pose)
                if pose.frame_index % 3 == 0:
                    self._emit(
                        "preview",
                        update=PreviewUpdate(
                            timestamp_ms=pose.timestamp_ms,
                            frame_index=pose.frame_index,
                            color_image=None,
                            depth_image=None,
                            pose=pose,
                            metrics=compute_session_metrics(poses),
                            status_text="Demo-режим без камеры",
                        ),
                    )
                time.sleep(1 / 30)

            metrics = compute_session_metrics(poses)
            writer.finalize(metrics)
            self._emit(
                "session_saved",
                result=SessionResult(session_dir=str(writer.session_dir), metrics=metrics),
            )
        except FileExistsError:
            self._emit("error", message="Сеанс с таким идентификатором уже существует")
        except Exception as error:
            self._emit("error", message=f"Ошибка demo-сеанса: {error}")
        finally:
            if writer is not None and not (Path(writer.session_dir) / "metrics.json").exists():
                writer.finalize(compute_session_metrics(poses))
            self._running = False
            self._emit("status", message="Demo-сеанс завершён")
