from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

APP_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
APP_CACHE_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(APP_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import DEFAULT_CAMERA_MODEL
from .controller import CaptureService
from .storage import SessionStore, build_session_metadata


class GaitCaptureApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RGBD Gait Capture")
        self.geometry("1380x880")
        self.minsize(1200, 760)

        self.store = SessionStore()
        self.capture_service = CaptureService(self.store)
        self.current_session_dir: Path | None = None
        self.preview_metrics: dict[str, float | int | str] = {}

        self._build_layout()
        self._refresh_sessions()
        self.after(120, self._poll_events)

    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.capture_tab = ttk.Frame(notebook, padding=10)
        self.results_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.capture_tab, text="Регистрация")
        notebook.add(self.results_tab, text="Результаты")

        self._build_capture_tab()
        self._build_results_tab()

    def _build_capture_tab(self) -> None:
        self.capture_tab.columnconfigure(0, weight=3)
        self.capture_tab.columnconfigure(1, weight=2)
        self.capture_tab.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(self.capture_tab, text="Предпросмотр")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_figure = Figure(figsize=(8, 5), dpi=100)
        self.preview_ax_color = self.preview_figure.add_subplot(1, 2, 1)
        self.preview_ax_depth = self.preview_figure.add_subplot(1, 2, 2)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_frame)
        self.preview_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._reset_preview_axes()

        controls = ttk.LabelFrame(self.capture_tab, text="Сеанс")
        controls.grid(row=0, column=1, sticky="nsew")
        for row_index in range(11):
            controls.rowconfigure(row_index, weight=0)
        controls.columnconfigure(1, weight=1)

        self.participant_var = tk.StringVar(value="subject01")
        self.operator_var = tk.StringVar(value="operator")
        self.protocol_var = tk.StringVar(value="10m_walk_test")
        self.diagnosis_var = tk.StringVar(value="")
        self.camera_var = tk.StringVar(value=DEFAULT_CAMERA_MODEL)
        self.status_var = tk.StringVar(value="Ожидание")

        ttk.Label(controls, text="Испытуемый").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.participant_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Оператор").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.operator_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Протокол").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.protocol_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Диагноз").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.diagnosis_var).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Камера").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.camera_var).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="Заметки").grid(row=5, column=0, sticky="nw", pady=4)
        self.notes_text = tk.Text(controls, height=8, width=32)
        self.notes_text.grid(row=5, column=1, sticky="ew", pady=4)

        buttons = ttk.Frame(controls)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for column_index in range(3):
            buttons.columnconfigure(column_index, weight=1)
        ttk.Button(buttons, text="Старт записи", command=self._start_live_capture).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(buttons, text="Demo", command=self._start_demo_capture).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(buttons, text="Стоп", command=self._stop_capture).grid(row=0, column=2, sticky="ew", padx=2)

        ttk.Label(controls, text="Статус").grid(row=7, column=0, sticky="w", pady=(8, 4))
        ttk.Label(controls, textvariable=self.status_var, wraplength=320).grid(row=7, column=1, sticky="w", pady=(8, 4))

        ttk.Label(controls, text="Текущие метрики").grid(row=8, column=0, sticky="nw", pady=4)
        self.metrics_text = tk.Text(controls, height=12, width=32)
        self.metrics_text.grid(row=8, column=1, sticky="ew", pady=4)

    def _build_results_tab(self) -> None:
        self.results_tab.columnconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=1)

        top = ttk.Frame(self.results_tab)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Сеанс").grid(row=0, column=0, padx=(0, 8))
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(top, textvariable=self.session_var, state="readonly")
        self.session_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Обновить", command=self._refresh_sessions).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Загрузить", command=self._load_selected_session).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Экспорт CSV", command=self._export_summary).grid(row=0, column=4, padx=4)

        content = ttk.Frame(self.results_tab)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        summary_frame = ttk.LabelFrame(content, text="Сводка")
        summary_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_frame, width=40)
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        charts = ttk.LabelFrame(content, text="Графики")
        charts.grid(row=0, column=1, sticky="nsew")
        charts.rowconfigure(0, weight=1)
        charts.columnconfigure(0, weight=1)

        self.results_figure = Figure(figsize=(8, 6), dpi=100)
        self.results_axes = [
            self.results_figure.add_subplot(2, 2, 1),
            self.results_figure.add_subplot(2, 2, 2),
            self.results_figure.add_subplot(2, 2, 3),
            self.results_figure.add_subplot(2, 2, 4),
        ]
        self.results_canvas = FigureCanvasTkAgg(self.results_figure, master=charts)
        self.results_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._reset_results_axes()

    def _reset_preview_axes(self) -> None:
        self.preview_ax_color.clear()
        self.preview_ax_depth.clear()
        self.preview_ax_color.set_title("RGB")
        self.preview_ax_depth.set_title("Depth")
        self.preview_ax_color.text(0.5, 0.5, "Нет сигнала", ha="center", va="center")
        self.preview_ax_depth.text(0.5, 0.5, "Нет сигнала", ha="center", va="center")
        self.preview_ax_color.set_xticks([])
        self.preview_ax_color.set_yticks([])
        self.preview_ax_depth.set_xticks([])
        self.preview_ax_depth.set_yticks([])
        self.preview_canvas.draw_idle()

    def _reset_results_axes(self) -> None:
        titles = ["Траектория таза", "Сепарация стоп", "Угол колена", "Угол тазобедренного сустава"]
        for axis, title in zip(self.results_axes, titles):
            axis.clear()
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
        self.results_canvas.draw_idle()

    def _collect_metadata(self):
        return build_session_metadata(
            participant_id=self.participant_var.get().strip(),
            operator_name=self.operator_var.get().strip(),
            protocol_name=self.protocol_var.get().strip(),
            diagnosis=self.diagnosis_var.get().strip(),
            notes=self.notes_text.get("1.0", tk.END).strip(),
            camera_model=self.camera_var.get().strip() or DEFAULT_CAMERA_MODEL,
        )

    def _start_live_capture(self) -> None:
        if self.capture_service.is_running():
            return
        metadata = self._collect_metadata()
        self.status_var.set("Запуск записи RealSense")
        self.capture_service.start_live_session(metadata, record_bag=True)

    def _start_demo_capture(self) -> None:
        if self.capture_service.is_running():
            return
        metadata = self._collect_metadata()
        self.status_var.set("Запуск demo-сеанса")
        self.capture_service.start_demo_session(metadata)

    def _stop_capture(self) -> None:
        self.capture_service.stop()
        self.status_var.set("Останавливается")

    def _poll_events(self) -> None:
        for event in self.capture_service.drain_events():
            if event.kind == "status":
                self.status_var.set(event.payload["message"])
            elif event.kind == "preview":
                self._update_preview(event.payload["update"])
            elif event.kind == "session_saved":
                result = event.payload["result"]
                self.current_session_dir = Path(result.session_dir)
                self.preview_metrics = result.metrics
                self._refresh_sessions(select_session=self.current_session_dir.name)
                self.status_var.set(f"Сеанс сохранён: {self.current_session_dir.name}")
            elif event.kind == "error":
                self.status_var.set(event.payload["message"])
                messagebox.showerror("Ошибка", event.payload["message"])
        self.after(120, self._poll_events)

    def _update_preview(self, update) -> None:
        self.preview_ax_color.clear()
        self.preview_ax_depth.clear()
        self.preview_ax_color.set_title("RGB")
        self.preview_ax_depth.set_title("Depth")
        self.preview_ax_color.set_xticks([])
        self.preview_ax_color.set_yticks([])
        self.preview_ax_depth.set_xticks([])
        self.preview_ax_depth.set_yticks([])

        if update.color_image is not None:
            self.preview_ax_color.imshow(update.color_image)
        else:
            self.preview_ax_color.text(0.5, 0.5, "Demo 3D pose", ha="center", va="center")

        if update.pose is not None and update.color_image is not None:
            xs = [landmark.pixel_x for landmark in update.pose.landmarks.values()]
            ys = [landmark.pixel_y for landmark in update.pose.landmarks.values()]
            self.preview_ax_color.scatter(xs, ys, s=15, c="#00ff99")

        if update.depth_image is not None:
            depth = update.depth_image.astype(float)
            positive = depth[depth > 0]
            vmax = np.percentile(positive, 98) if positive.size else 1
            self.preview_ax_depth.imshow(depth, cmap="viridis", vmin=0, vmax=vmax)
        else:
            self.preview_ax_depth.text(0.5, 0.5, "Нет depth", ha="center", va="center")

        self.preview_canvas.draw_idle()
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, self._format_metrics(update.metrics))

    def _format_metrics(self, metrics: dict[str, float | int | str]) -> str:
        if not metrics:
            return "Недостаточно данных для расчёта."
        lines = []
        for key, value in metrics.items():
            label = key.replace("_", " ")
            lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def _refresh_sessions(self, select_session: str | None = None) -> None:
        sessions = self.store.list_sessions()
        labels = [
            f"{item['session_id']} | {item['participant_id']} | {item['created_at']}"
            for item in sessions
        ]
        self._session_lookup = {label: item["path"] for label, item in zip(labels, sessions)}
        self.session_combo["values"] = labels

        if select_session is not None:
            for label in labels:
                if select_session in label:
                    self.session_var.set(label)
                    return
        if labels and not self.session_var.get():
            self.session_var.set(labels[0])

    def _load_selected_session(self) -> None:
        selected = self.session_var.get()
        if not selected:
            return
        session_path = Path(self._session_lookup[selected])
        metadata, rows, metrics = self.store.load_session(session_path)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(
            tk.END,
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n\n" + json.dumps(metrics, ensure_ascii=False, indent=2),
        )
        self._plot_session(rows)

    def _plot_session(self, rows: list[dict]) -> None:
        self._reset_results_axes()
        timestamps = []
        left_sep = []
        pelvis_x = []
        pelvis_z = []
        left_knee = []
        right_knee = []
        left_hip = []
        right_hip = []

        for row in rows:
            try:
                timestamp = float(row["timestamp_ms"]) / 1000.0
            except (TypeError, ValueError):
                continue
            timestamps.append(timestamp)

            try:
                left_heel_z = float(row["left_heel_z"])
                right_heel_z = float(row["right_heel_z"])
                left_sep.append(left_heel_z - right_heel_z)
            except (TypeError, ValueError):
                pass

            try:
                lx = float(row["left_hip_x"])
                rx = float(row["right_hip_x"])
                lz = float(row["left_hip_z"])
                rz = float(row["right_hip_z"])
                pelvis_x.append((lx + rx) / 2.0)
                pelvis_z.append((lz + rz) / 2.0)
            except (TypeError, ValueError):
                pass

            left_knee_angle = self._row_angle(row, "left_hip", "left_knee", "left_ankle")
            right_knee_angle = self._row_angle(row, "right_hip", "right_knee", "right_ankle")
            left_hip_angle = self._row_angle(row, "left_shoulder", "left_hip", "left_knee")
            right_hip_angle = self._row_angle(row, "right_shoulder", "right_hip", "right_knee")
            if left_knee_angle is not None:
                left_knee.append((timestamp, left_knee_angle))
            if right_knee_angle is not None:
                right_knee.append((timestamp, right_knee_angle))
            if left_hip_angle is not None:
                left_hip.append((timestamp, left_hip_angle))
            if right_hip_angle is not None:
                right_hip.append((timestamp, right_hip_angle))

        if pelvis_x and pelvis_z:
            self.results_axes[0].plot(pelvis_z, pelvis_x, color="#0b7285")
            self.results_axes[0].set_xlabel("Z, м")
            self.results_axes[0].set_ylabel("X, м")
        if timestamps and left_sep:
            self.results_axes[1].plot(timestamps[: len(left_sep)], left_sep, color="#d9480f")
            self.results_axes[1].set_xlabel("t, c")
            self.results_axes[1].set_ylabel("Lheel-Rheel, м")
        if left_knee:
            self.results_axes[2].plot([item[0] for item in left_knee], [item[1] for item in left_knee], label="Left", color="#2b8a3e")
        if right_knee:
            self.results_axes[2].plot([item[0] for item in right_knee], [item[1] for item in right_knee], label="Right", color="#c92a2a")
        if left_hip:
            self.results_axes[3].plot([item[0] for item in left_hip], [item[1] for item in left_hip], label="Left", color="#2b8a3e")
        if right_hip:
            self.results_axes[3].plot([item[0] for item in right_hip], [item[1] for item in right_hip], label="Right", color="#c92a2a")
        self.results_axes[2].legend(loc="best")
        self.results_axes[3].legend(loc="best")
        self.results_canvas.draw_idle()

    def _row_angle(self, row: dict, a_prefix: str, b_prefix: str, c_prefix: str) -> float | None:
        try:
            a = np.array([float(row[f"{a_prefix}_x"]), float(row[f"{a_prefix}_y"]), float(row[f"{a_prefix}_z"])])
            b = np.array([float(row[f"{b_prefix}_x"]), float(row[f"{b_prefix}_y"]), float(row[f"{b_prefix}_z"])])
            c = np.array([float(row[f"{c_prefix}_x"]), float(row[f"{c_prefix}_y"]), float(row[f"{c_prefix}_z"])])
        except (KeyError, TypeError, ValueError):
            return None
        ba = a - b
        bc = c - b
        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denominator == 0:
            return None
        cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def _export_summary(self) -> None:
        path = self.store.export_summary_csv()
        self.status_var.set(f"Экспортировано: {path.name}")
        messagebox.showinfo("Экспорт", f"Сводка сохранена в {path}")


def main() -> None:
    app = GaitCaptureApp()
    app.mainloop()
