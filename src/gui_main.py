#!/usr/bin/env python3
"""
Ultimate Modern GUI for the Real-Time 3D Ball Tracker (V4 Offline Simulator Edition).
Features Playback Mode, Picture-in-Picture AI Debug, Image Filtering (CLAHE), DVR, and Theme Engine.
"""

import sys
import os
import time
import csv
import math
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import psutil

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QFont, QTextCursor, QAction, QVector3D
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QSlider, QGroupBox, QFormLayout, QPushButton, QComboBox,
        QPlainTextEdit, QMessageBox, QSizePolicy, QDockWidget, QTabWidget,
        QLineEdit, QToolBar, QMenu, QFileDialog, QCheckBox
    )
    import qdarktheme
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError:
    print("Please install requirements: pip3 install PyQt6 pyqtdarktheme pyqtgraph PyOpenGL PyOpenGL_accelerate psutil")
    sys.exit(1)

from pc_tracker import load_config, _build_cameras
from common.network import UDPSender
from detection.ball_detector import BallDetector, DetectionResult
from stereo.triangulation import StereoTriangulator
from detection.camera import MockCamera

# ---------------------------------------------------------------------------
# Qt Log Handler
# ---------------------------------------------------------------------------

class QtLogSignals(QObject):
    log_msg = pyqtSignal(str)

class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.signals = QtLogSignals()

    def emit(self, record):
        msg = self.format(record)
        self.signals.log_msg.emit(msg)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
qt_handler = QtLogHandler()
qt_handler.setFormatter(formatter)
logger.addHandler(qt_handler)


# ---------------------------------------------------------------------------
# Background Tracker Thread (with V4 Playback & Filters)
# ---------------------------------------------------------------------------

class TrackerThread(QThread):
    frames_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    connection_error = pyqtSignal(str)
    recording_status = pyqtSignal(bool, str)

    def __init__(self, config: Dict[str, Any], playback_paths: Optional[Tuple[str, str]] = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.playback_paths = playback_paths
        self._running = True
        
        self.detector = None
        self.cam_left = None
        self.cam_right = None
        self.sender = None
        
        # DVR State
        self.is_recording = False
        self.video_out_l = None
        self.video_out_r = None
        self.csv_file = None
        self.csv_writer = None
        self.rec_start_time = 0.0
        
        # V4 Commands / State
        self._cmd_exposure = -1
        self._cmd_gain = -1.0
        self._cmd_hsv_lower = None
        self._cmd_hsv_upper = None
        self._cmd_method = None
        self._cmd_reset = False
        self._cmd_network = None
        self._cmd_toggle_rec = False
        self.playback_delay_ms = 0
        self.use_clahe = False
        self.use_blur = False
        self.show_ai_debug = True

    def run(self):
        net_cfg    = self.config["network"]
        cam_cfg    = self.config["camera"]
        stereo_cfg = self.config["stereo"]
        det_cfg    = self.config["detection"]

        self.sender = UDPSender(ip=net_cfg["rpi_ip"], port=net_cfg["port"])
        
        # V4: Use MockCamera for Video Playback
        if self.playback_paths:
            w, h = cam_cfg["resolution"]["width"], cam_cfg["resolution"]["height"]
            self.cam_left = MockCamera(width=w, height=h, fps=30, is_left=True, video_path=self.playback_paths[0])
            self.cam_right = MockCamera(width=w, height=h, fps=30, is_left=False, video_path=self.playback_paths[1])
            logging.info(f"Started Playback Mode with files: {self.playback_paths}")
        else:
            self.cam_left, self.cam_right = _build_cameras(cam_cfg, stereo_cfg)

        kalman_cfg = det_cfg.get("kalman", {})
        if not kalman_cfg.get("enabled", True):
            kalman_cfg = {"max_coast_frames": 0}

        self.detector = BallDetector(
            method=det_cfg.get("method", "hybrid"),
            yolo_model_path=det_cfg.get("yolo_model_path", "yolov8n.pt"),
            hsv_bounds=det_cfg.get("hsv_bounds"),
            hough_cfg=det_cfg.get("hough"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.4),
            kalman_cfg=kalman_cfg,
            roi_cfg=det_cfg.get("roi"),
        )

        triangulator = StereoTriangulator(
            baseline_mm=stereo_cfg["baseline_mm"],
            focal_length_px=stereo_cfg["focal_length_px"],
            cx=stereo_cfg.get("principal_point_x", cam_cfg["resolution"]["width"] / 2.0),
            cy=stereo_cfg.get("principal_point_y", cam_cfg["resolution"]["height"] / 2.0),
        )

        if not self.cam_left.open() or not self.cam_right.open():
            self.connection_error.emit("Failed to open cameras.")
            self.sender.close()
            return

        while self._running:
            t_start = time.perf_counter()

            # Apply commands
            self._process_commands()

            # Capture
            ret_l, frame_l = self.cam_left.read()
            ret_r, frame_r = self.cam_right.read()

            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                time.sleep(0.01)
                continue
                
            # V4: Playback Slowdown
            if self.playback_paths and self.playback_delay_ms > 0:
                time.sleep(self.playback_delay_ms / 1000.0)

            timestamp = self.cam_left.get_timestamp()
            
            # V4: Image Filtering
            if self.use_clahe or self.use_blur:
                frame_l = self._apply_filters(frame_l)
                frame_r = self._apply_filters(frame_r)

            # Detect & Triangulate
            result_l, result_r = self.detector.detect_stereo(frame_l, frame_r)
            x_3d = y_3d = z_3d = 0.0
            tracking_success = False

            if result_l.success and result_r.success:
                tracking_success, x_3d, y_3d, z_3d = triangulator.triangulate(
                    (result_l.x, result_l.y),
                    (result_r.x, result_r.y),
                )
                if tracking_success:
                    self._draw_overlay(frame_l, result_l)
                    self._draw_overlay(frame_r, result_r)
                    
            # V4: AI Debug Picture-in-Picture (HSV Mask)
            if self.show_ai_debug:
                self._apply_ai_debug(frame_l)
                self._apply_ai_debug(frame_r)

            # Send UDP
            self.sender.send_target_position(x_3d, y_3d, z_3d, tracking_success, timestamp)

            # Record Data (DVR & CSV)
            if self.is_recording:
                if self.video_out_l and self.video_out_r:
                    self.video_out_l.write(frame_l)
                    self.video_out_r.write(frame_r)
                if self.csv_writer:
                    rel_t = time.time() - self.rec_start_time
                    self.csv_writer.writerow([f"{rel_t:.3f}", 1 if tracking_success else 0, f"{x_3d:.1f}", f"{y_3d:.1f}", f"{z_3d:.1f}"])

            # Emit
            fps = 1.0 / max(time.perf_counter() - t_start, 1e-9)
            stats = {
                "fps": fps,
                "tracked": tracking_success,
                "x": x_3d, "y": y_3d, "z": z_3d,
                "res_l": result_l,
                "res_r": result_r
            }
            self.frames_ready.emit(frame_l.copy(), frame_r.copy(), stats)

        # Cleanup
        self._stop_recording()
        self.cam_left.close()
        self.cam_right.close()
        self.sender.close()

    def _apply_filters(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        if self.use_blur:
            out = cv2.GaussianBlur(out, (5, 5), 0)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out

    def _apply_ai_debug(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.detector.lower_hsv, self.detector.upper_hsv)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 1/4 size pip
        h, w = frame.shape[:2]
        pip_w, pip_h = w // 4, h // 4
        pip = cv2.resize(mask_bgr, (pip_w, pip_h))
        
        # Draw red border around pip
        cv2.rectangle(pip, (0,0), (pip_w-1, pip_h-1), (0,0,255), 2)
        cv2.putText(pip, "AI DEBUG", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Place in bottom right corner
        frame[h-pip_h:h, w-pip_w:w] = pip

    def _process_commands(self):
        if self._cmd_exposure > 0:
            self.cam_left.set_exposure(self._cmd_exposure)
            self.cam_right.set_exposure(self._cmd_exposure)
            self._cmd_exposure = -1
        if self._cmd_gain >= 0:
            self.cam_left.set_gain(self._cmd_gain)
            self.cam_right.set_gain(self._cmd_gain)
            self._cmd_gain = -1.0
        if self._cmd_hsv_lower is not None and self._cmd_hsv_upper is not None:
            self.detector.lower_hsv = self._cmd_hsv_lower
            self.detector.upper_hsv = self._cmd_hsv_upper
            self._cmd_hsv_lower = None
            self._cmd_hsv_upper = None
        if self._cmd_method is not None:
            self.detector.method = self._cmd_method
            self._cmd_method = None
        if self._cmd_reset:
            self.detector._kalman_left.reset()
            self.detector._kalman_right.reset()
            self._cmd_reset = False
            logging.info("Kalman filters reset by user.")
        if self._cmd_network is not None:
            ip, port = self._cmd_network
            self.sender.close()
            self.sender = UDPSender(ip=ip, port=port)
            self._cmd_network = None
            logging.info(f"UDP Target updated to {ip}:{port}")
        if self._cmd_toggle_rec:
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()
            self._cmd_toggle_rec = False

    def _start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"recordings/session_{timestamp}"
        os.makedirs("recordings", exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w, h = self.config["camera"]["resolution"]["width"], self.config["camera"]["resolution"]["height"]
        
        self.video_out_l = cv2.VideoWriter(f"{prefix}_camL.mp4", fourcc, 30.0, (w, h))
        self.video_out_r = cv2.VideoWriter(f"{prefix}_camR.mp4", fourcc, 30.0, (w, h))
        
        self.csv_file = open(f"{prefix}_telemetry.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time_s", "tracked", "x_mm", "y_mm", "z_mm"])
        
        self.rec_start_time = time.time()
        self.is_recording = True
        self.recording_status.emit(True, prefix)
        logging.info(f"Started DVR Recording to: {prefix}")

    def _stop_recording(self):
        self.is_recording = False
        if self.video_out_l: self.video_out_l.release()
        if self.video_out_r: self.video_out_r.release()
        if self.csv_file: self.csv_file.close()
        self.video_out_l = self.video_out_r = self.csv_writer = self.csv_file = None
        self.recording_status.emit(False, "")

    def stop(self):
        self._running = False
        self.wait()

    def set_exposure(self, us: int): self._cmd_exposure = us
    def set_gain(self, gain: float): self._cmd_gain = gain
    def set_hsv(self, lower: np.ndarray, upper: np.ndarray):
        self._cmd_hsv_lower = lower
        self._cmd_hsv_upper = upper
    def set_method(self, method: str): self._cmd_method = method
    def reset_tracker(self): self._cmd_reset = True
    def set_network(self, ip: str, port: int): self._cmd_network = (ip, port)
    def toggle_recording(self): self._cmd_toggle_rec = True

    def _draw_overlay(self, frame: np.ndarray, result: DetectionResult):
        color = (0, 255, 0)
        if result.method == "hsv": color = (255, 255, 0)
        elif result.method == "kalman": color = (0, 220, 255)
        cx, cy, r = result.x, result.y, max(result.radius, 8)
        if result.is_predicted:
            cv2.circle(frame, (cx, cy), r, color, 1)
            cv2.circle(frame, (cx, cy), r + 3, color, 1)
        else:
            cv2.circle(frame, (cx, cy), r, color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 15, 2)


# ---------------------------------------------------------------------------
# Main Window (V4 Offline Simulator Edition)
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.setWindowTitle("Robot Goalkeeper 3D - Ultimate Control Center V4")
        self.resize(1600, 900)
        self.config = config
        
        self.current_theme = "dark"
        self.Z_GOAL = 600.0  
        self.playback_paths = None
        
        self.history_len = 300
        self.data_t = deque(maxlen=self.history_len)
        self.data_x = deque(maxlen=self.history_len)
        self.data_y = deque(maxlen=self.history_len)
        self.data_z = deque(maxlen=self.history_len)
        self.t_start = time.time()
        
        self.grp_boxes = []
        self._build_ui()
        self.apply_theme(self.current_theme)
        
        qt_handler.signals.log_msg.connect(self.on_log_message)
        logging.info("Ultimate GUI V4 Initialized.")

        self._start_thread()
        
        self.sys_timer = QTimer(self)
        self.sys_timer.timeout.connect(self._update_system_stats)
        self.sys_timer.start(1000)

    def _start_thread(self):
        if hasattr(self, 'tracker_thread') and self.tracker_thread.isRunning():
            self.tracker_thread.stop()
        
        self.tracker_thread = TrackerThread(self.config, self.playback_paths)
        self.tracker_thread.frames_ready.connect(self.on_frames_ready)
        self.tracker_thread.connection_error.connect(self.on_error)
        self.tracker_thread.recording_status.connect(self.on_recording_status)
        
        # Apply current toggles
        self.tracker_thread.show_ai_debug = self.chk_ai_debug.isChecked()
        self.tracker_thread.use_clahe = self.chk_clahe.isChecked()
        self.tracker_thread.use_blur = self.chk_blur.isChecked()
        self.tracker_thread.playback_delay_ms = self.sl_slowmo.value()
        
        self.tracker_thread.start()

    def _build_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        act_playback = QAction("Load Playback Video...", self)
        act_playback.triggered.connect(self.on_load_playback)
        file_menu.addAction(act_playback)
        act_live = QAction("Switch to Live Cameras", self)
        act_live.triggered.connect(self.on_live_cameras)
        file_menu.addAction(act_live)
        
        file_menu.addSeparator()
        save_act = QAction("Save Configuration", self)
        save_act.triggered.connect(self.on_save_config)
        file_menu.addAction(save_act)
        
        prof_menu = file_menu.addMenu("Load Lighting Profile")
        act_sunny = QAction("Outdoor (Sunny)", self)
        act_sunny.triggered.connect(lambda: self.load_profile("sunny"))
        act_cloudy = QAction("Outdoor (Cloudy)", self)
        act_cloudy.triggered.connect(lambda: self.load_profile("cloudy"))
        act_indoor = QAction("Indoor (Fluorescent)", self)
        act_indoor.triggered.connect(lambda: self.load_profile("indoor"))
        prof_menu.addAction(act_sunny)
        prof_menu.addAction(act_cloudy)
        prof_menu.addAction(act_indoor)
        
        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)
        
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        self.act_record = QAction("🔴 Start DVR Recording", self)
        self.act_record.triggered.connect(self.on_toggle_record)
        toolbar.addAction(self.act_record)
        toolbar.addSeparator()
        act_theme = QAction("🌗 Toggle Theme", self)
        act_theme.triggered.connect(self.on_toggle_theme)
        toolbar.addAction(act_theme)
        
        self.statusBar().showMessage("System Ready.")
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tab_cams = QWidget()
        self._build_tab_cameras(self.tab_cams)
        self.tabs.addTab(self.tab_cams, "📷 Dual Cameras")
        
        self.tab_plot = QWidget()
        self._build_tab_telemetry(self.tab_plot)
        self.tabs.addTab(self.tab_plot, "📈 2D Telemetry")
        
        self.tab_3d = QWidget()
        self._build_tab_3d_arena(self.tab_3d)
        self.tabs.addTab(self.tab_3d, "🧊 3D Robot Simulator")
        
        self.dock_settings = QDockWidget("Hardware Settings", self)
        self.dock_settings.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        widget_settings = QWidget()
        self._build_dock_settings(widget_settings)
        self.dock_settings.setWidget(widget_settings)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_settings)
        
        self.dock_stats = QDockWidget("Tracker Analytics", self)
        widget_stats = QWidget()
        self._build_dock_stats(widget_stats)
        self.dock_stats.setWidget(widget_stats)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_stats)
        
        self.dock_logs = QDockWidget("System Console", self)
        widget_logs = QWidget()
        self._build_dock_logs(widget_logs)
        self.dock_logs.setWidget(widget_logs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_logs)
        
        view_menu = menubar.addMenu("View")
        view_menu.addAction(self.dock_settings.toggleViewAction())
        view_menu.addAction(self.dock_stats.toggleViewAction())
        view_menu.addAction(self.dock_logs.toggleViewAction())
        view_menu.addSeparator()
        reset_layout_act = QAction("Reset Window Layout", self)
        reset_layout_act.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_act)
        
        self._default_state = self.saveState()

    def reset_layout(self):
        if hasattr(self, '_default_state'):
            self.restoreState(self._default_state)
            self.dock_settings.setVisible(True)
            self.dock_stats.setVisible(True)
            self.dock_logs.setVisible(True)

    # -----------------------------------------------------------------------
    # V4: Playback Logic
    # -----------------------------------------------------------------------
    def on_load_playback(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Left Camera Video", "recordings", "Videos (*.mp4 *.avi)")
        if file_path:
            # Try to guess the right camera file
            if "_camL" in file_path:
                right_path = file_path.replace("_camL", "_camR")
            else:
                right_path = file_path # Fallback to same video for both
                
            if not os.path.exists(right_path):
                QMessageBox.warning(self, "Warning", f"Could not find matching right camera video:\n{right_path}\nWill use left video for both.")
                right_path = file_path

            self.playback_paths = (file_path, right_path)
            self.statusBar().showMessage(f"Playback Mode: {os.path.basename(file_path)}", 5000)
            self.setWindowTitle("Robot Goalkeeper 3D - Playback Mode")
            self._start_thread()

    def on_live_cameras(self):
        self.playback_paths = None
        self.statusBar().showMessage("Switched to Live Cameras", 5000)
        self.setWindowTitle("Robot Goalkeeper 3D - Ultimate Control Center V4")
        self._start_thread()

    # -----------------------------------------------------------------------
    # Theme Engine
    # -----------------------------------------------------------------------
    def on_toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme(self.current_theme)

    def apply_theme(self, theme: str):
        app = QApplication.instance()
        if app:
            app.setStyleSheet(qdarktheme.load_stylesheet(theme))
            
        if theme == "dark":
            grp_border = "#444"
            vid_bg = "#050505"
            vid_border = "#222"
            plot_bg = "#151515"
            plot_fg = "#BBBBBB"
            cons_bg = "#0F0F0F"
            cons_fg = "#CCCCCC"
        else:
            grp_border = "#CCC"
            vid_bg = "#F0F0F0"
            vid_border = "#AAA"
            plot_bg = "#FFFFFF"
            plot_fg = "#000000"
            cons_bg = "#FAFAFA"
            cons_fg = "#222222"

        grp_style = f"""
        QGroupBox {{ font-weight: bold; border: 1px solid {grp_border}; border-radius: 6px; margin-top: 12px; padding-top: 10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: #00A650; }}
        """
        for grp in self.grp_boxes:
            grp.setStyleSheet(grp_style)

        self.lbl_vid_l.setStyleSheet(f"background-color: {vid_bg}; border: 2px solid {vid_border}; border-radius: 8px;")
        self.lbl_vid_r.setStyleSheet(f"background-color: {vid_bg}; border: 2px solid {vid_border}; border-radius: 8px;")
        self.console.setStyleSheet(f"background-color: {cons_bg}; color: {cons_fg}; font-family: 'Consolas', 'Monospace'; font-size: 11px;")

        self.plot_widget.setBackground(plot_bg)
        self.plot_widget.getAxis('bottom').setPen(plot_fg)
        self.plot_widget.getAxis('bottom').setTextPen(plot_fg)
        self.plot_widget.getAxis('left').setPen(plot_fg)
        self.plot_widget.getAxis('left').setTextPen(plot_fg)

    def load_profile(self, mode: str):
        if mode == "sunny":
            self.sl_exp.setValue(200)
            self.sl_gain.setValue(0)
        elif mode == "cloudy":
            self.sl_exp.setValue(1000)
            self.sl_gain.setValue(0)
        elif mode == "indoor":
            self.sl_exp.setValue(8000)
            self.sl_gain.setValue(50) 
        self.statusBar().showMessage(f"Loaded {mode.capitalize()} lighting profile.", 3000)

    def on_toggle_record(self):
        self.tracker_thread.toggle_recording()

    @pyqtSlot(bool, str)
    def on_recording_status(self, is_recording: bool, prefix: str):
        if is_recording:
            self.act_record.setText("⬛ Stop DVR Recording")
            self.statusBar().showMessage(f"Recording active: {prefix}", 5000)
        else:
            self.act_record.setText("🔴 Start DVR Recording")
            self.statusBar().showMessage("Recording stopped.", 5000)

    def _build_tab_cameras(self, parent: QWidget):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        self.lbl_vid_l = QLabel("Left Camera")
        self.lbl_vid_r = QLabel("Right Camera")
        self.lbl_vid_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_vid_r.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_vid_l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.lbl_vid_r.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        layout.addWidget(self.lbl_vid_l, 1)
        layout.addWidget(self.lbl_vid_r, 1)

    def _build_tab_telemetry(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget(title="Real-time 3D Trajectory")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setLabel('left', 'Position (mm)')
        self.plot_widget.addLegend(offset=(10, 10))
        
        self.curve_x = self.plot_widget.plot(pen=pg.mkPen('#FF5252', width=3), name="X (Horizontal)")
        self.curve_y = self.plot_widget.plot(pen=pg.mkPen('#4CAF50', width=3), name="Y (Vertical)")
        self.curve_z = self.plot_widget.plot(pen=pg.mkPen('#448AFF', width=3), name="Z (Depth)")
        
        layout.addWidget(self.plot_widget)

    def _build_tab_3d_arena(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.gl_view = gl.GLViewWidget()
        self.gl_view.opts['distance'] = 4000
        self.gl_view.opts['elevation'] = 20
        self.gl_view.opts['azimuth'] = 45
        
        grid = gl.GLGridItem()
        grid.setSize(x=5000, y=5000, z=0)
        grid.setSpacing(x=500, y=500, z=0)
        self.gl_view.addItem(grid)
        
        goal_box = gl.GLBoxItem(size=QVector3D(1200, 200, 800), color=(0.2, 0.8, 0.2, 0.3))
        goal_box.translate(-600, -100, self.Z_GOAL)
        self.gl_view.addItem(goal_box)
        
        self.gl_ball = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), color=(1, 0.6, 0.1, 1), size=40, pxMode=True)
        self.gl_view.addItem(self.gl_ball)
        
        self.gl_pred = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,0]]), color=(0, 1, 1, 0.8), width=3, antialias=True)
        self.gl_view.addItem(self.gl_pred)
        
        self.gl_robot_arm = gl.GLLinePlotItem(pos=np.array([[0,0,self.Z_GOAL], [0, 800, self.Z_GOAL]]), color=(1, 0, 0.5, 1), width=6, antialias=True)
        self.gl_view.addItem(self.gl_robot_arm)
        
        layout.addWidget(self.gl_view)

    def _build_dock_settings(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        
        # V4: Software Filters & Playback Controls
        self.grp_sim = QGroupBox("Simulation & Filters")
        sim_lay = QFormLayout(self.grp_sim)
        
        self.chk_ai_debug = QCheckBox("Show AI Debug Mask (PiP)")
        self.chk_ai_debug.setChecked(True)
        self.chk_ai_debug.stateChanged.connect(lambda state: setattr(self.tracker_thread, 'show_ai_debug', bool(state)))
        
        self.chk_clahe = QCheckBox("Enable CLAHE Filter")
        self.chk_clahe.stateChanged.connect(lambda state: setattr(self.tracker_thread, 'use_clahe', bool(state)))
        
        self.chk_blur = QCheckBox("Enable Gaussian Blur")
        self.chk_blur.stateChanged.connect(lambda state: setattr(self.tracker_thread, 'use_blur', bool(state)))
        
        self.sl_slowmo = QSlider(Qt.Orientation.Horizontal)
        self.sl_slowmo.setRange(0, 500)
        self.sl_slowmo.setValue(0)
        self.sl_slowmo.valueChanged.connect(lambda v: setattr(self.tracker_thread, 'playback_delay_ms', v))
        self.lbl_slowmo = QLabel("0 ms")
        self.sl_slowmo.valueChanged.connect(lambda v: self.lbl_slowmo.setText(f"{v} ms"))
        
        sim_lay.addRow(self.chk_ai_debug)
        sim_lay.addRow(self.chk_clahe)
        sim_lay.addRow(self.chk_blur)
        sim_lay.addRow("Playback Delay:", self.sl_slowmo)
        sim_lay.addRow("", self.lbl_slowmo)
        layout.addWidget(self.grp_sim)
        self.grp_boxes.append(self.grp_sim)
        
        # Hardware
        self.grp_cam = QGroupBox("Camera Exposure & Gain")
        cam_lay = QFormLayout(self.grp_cam)
        self.sl_exp = QSlider(Qt.Orientation.Horizontal)
        self.sl_exp.setRange(100, 20000)
        self.sl_exp.setValue(self.config["camera"].get("exposure_time_us", 800))
        self.sl_exp.valueChanged.connect(lambda v: self.tracker_thread.set_exposure(v))
        self.lbl_exp_val = QLabel(f"{self.sl_exp.value()} µs")
        self.sl_exp.valueChanged.connect(lambda v: self.lbl_exp_val.setText(f"{v} µs"))
        self.sl_gain = QSlider(Qt.Orientation.Horizontal)
        self.sl_gain.setRange(0, 240)
        self.sl_gain.setValue(int(self.config["camera"].get("gain", 0.0) * 10))
        self.sl_gain.valueChanged.connect(lambda v: self.tracker_thread.set_gain(v / 10.0))
        self.lbl_gain_val = QLabel(f"{self.sl_gain.value()/10.0} dB")
        self.sl_gain.valueChanged.connect(lambda v: self.lbl_gain_val.setText(f"{v/10.0} dB"))
        cam_lay.addRow("Exposure:", self.sl_exp)
        cam_lay.addRow("", self.lbl_exp_val)
        cam_lay.addRow("Gain:", self.sl_gain)
        cam_lay.addRow("", self.lbl_gain_val)
        layout.addWidget(self.grp_cam)
        self.grp_boxes.append(self.grp_cam)
        
        self.grp_hsv = QGroupBox("HSV Fallback")
        hsv_lay = QFormLayout(self.grp_hsv)
        bounds = self.config["detection"].get("hsv_bounds", {})
        self.sliders_hsv = {}
        for name, v_min, v_max, default in [
            ("lower_h", 0, 179, bounds.get("lower_h", 0)),
            ("upper_h", 0, 179, bounds.get("upper_h", 180)),
            ("lower_s", 0, 255, bounds.get("lower_s", 0)),
            ("upper_s", 0, 255, bounds.get("upper_s", 50)),
            ("lower_v", 0, 255, bounds.get("lower_v", 200)),
            ("upper_v", 0, 255, bounds.get("upper_v", 255)),
        ]:
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(v_min, v_max)
            sl.setValue(default)
            sl.valueChanged.connect(self._on_hsv_changed)
            self.sliders_hsv[name] = sl
            hsv_lay.addRow(name, sl)
        layout.addWidget(self.grp_hsv)
        self.grp_boxes.append(self.grp_hsv)
        
        self.grp_net = QGroupBox("Network Target")
        net_lay = QFormLayout(self.grp_net)
        self.le_ip = QLineEdit(self.config["network"]["rpi_ip"])
        self.le_port = QLineEdit(str(self.config["network"]["port"]))
        btn_apply_net = QPushButton("Apply")
        btn_apply_net.clicked.connect(self._on_network_apply)
        net_lay.addRow("IP:", self.le_ip)
        net_lay.addRow("Port:", self.le_port)
        net_lay.addRow("", btn_apply_net)
        layout.addWidget(self.grp_net)
        self.grp_boxes.append(self.grp_net)
        
        layout.addStretch()

    def _build_dock_stats(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        
        self.grp_algo = QGroupBox("Algorithm Control")
        algo_lay = QVBoxLayout(self.grp_algo)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["hybrid", "yolo", "hsv"])
        self.cb_mode.setCurrentText(self.config["detection"].get("method", "hybrid"))
        self.cb_mode.currentTextChanged.connect(lambda t: self.tracker_thread.set_method(t))
        algo_lay.addWidget(QLabel("Primary Method:"))
        algo_lay.addWidget(self.cb_mode)
        
        btn_reset = QPushButton("↺ Reset Kalman Filter")
        btn_reset.setStyleSheet("background-color: #A03030; color: white; padding: 6px;")
        btn_reset.clicked.connect(lambda: self.tracker_thread.reset_tracker())
        algo_lay.addWidget(btn_reset)
        layout.addWidget(self.grp_algo)
        self.grp_boxes.append(self.grp_algo)
        
        self.grp_stats = QGroupBox("Live Output")
        stat_lay = QFormLayout(self.grp_stats)
        
        self.lbl_fps = QLabel("0.0")
        self.lbl_status = QLabel("N/A")
        self.lbl_3d = QLabel("[0.0, 0.0, 0.0]")
        self.lbl_angle = QLabel("0.0°")
        self.lbl_conf_l = QLabel("--")
        self.lbl_conf_r = QLabel("--")
        
        self.lbl_fps.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 16px;")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.lbl_3d.setStyleSheet("color: #00E676; font-family: monospace; font-size: 14px;")
        self.lbl_angle.setStyleSheet("color: #FF1744; font-weight: bold; font-size: 16px;")
        
        stat_lay.addRow("System FPS:", self.lbl_fps)
        stat_lay.addRow("Tracking State:", self.lbl_status)
        stat_lay.addRow("Target (mm):", self.lbl_3d)
        stat_lay.addRow("Req. Tilt Angle:", self.lbl_angle)
        stat_lay.addRow("Camera L:", self.lbl_conf_l)
        stat_lay.addRow("Camera R:", self.lbl_conf_r)
        layout.addWidget(self.grp_stats)
        self.grp_boxes.append(self.grp_stats)
        
        self.grp_hw = QGroupBox("Hardware Monitor")
        hw_lay = QFormLayout(self.grp_hw)
        self.lbl_cpu = QLabel("0 %")
        self.lbl_ram = QLabel("0 %")
        hw_lay.addRow("CPU Usage:", self.lbl_cpu)
        hw_lay.addRow("RAM Usage:", self.lbl_ram)
        layout.addWidget(self.grp_hw)
        self.grp_boxes.append(self.grp_hw)
        
        layout.addStretch()

    def _build_dock_logs(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

    def _update_system_stats(self):
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            self.lbl_cpu.setText(f"{cpu:.1f} %")
            self.lbl_ram.setText(f"{ram:.1f} %")
            self.lbl_cpu.setStyleSheet("color: #FF5252;" if cpu > 85 else "color: inherit;")
            self.lbl_ram.setStyleSheet("color: #FF5252;" if ram > 85 else "color: inherit;")
        except Exception:
            pass

    def _on_hsv_changed(self):
        lower = np.array([self.sliders_hsv["lower_h"].value(), self.sliders_hsv["lower_s"].value(), self.sliders_hsv["lower_v"].value()], dtype=np.uint8)
        upper = np.array([self.sliders_hsv["upper_h"].value(), self.sliders_hsv["upper_s"].value(), self.sliders_hsv["upper_v"].value()], dtype=np.uint8)
        self.tracker_thread.set_hsv(lower, upper)

    def _on_network_apply(self):
        ip = self.le_ip.text()
        try:
            port = int(self.le_port.text())
            self.tracker_thread.set_network(ip, port)
            self.statusBar().showMessage(f"Network updated to {ip}:{port}", 3000)
        except ValueError:
            QMessageBox.warning(self, "Invalid Port", "Port must be a number.")

    def on_save_config(self):
        try:
            import yaml
            self.config["network"]["rpi_ip"] = self.le_ip.text()
            self.config["network"]["port"] = int(self.le_port.text())
            self.config["camera"]["exposure_time_us"] = self.sl_exp.value()
            self.config["camera"]["gain"] = self.sl_gain.value() / 10.0
            
            self.config["detection"]["hsv_bounds"] = {
                "lower_h": self.sliders_hsv["lower_h"].value(),
                "lower_s": self.sliders_hsv["lower_s"].value(),
                "lower_v": self.sliders_hsv["lower_v"].value(),
                "upper_h": self.sliders_hsv["upper_h"].value(),
                "upper_s": self.sliders_hsv["upper_s"].value(),
                "upper_v": self.sliders_hsv["upper_v"].value(),
            }
            self.config["detection"]["method"] = self.cb_mode.currentText()
            
            with open("config/system_config.yaml", "w") as fh:
                yaml.safe_dump(self.config, fh, default_flow_style=False)
            logging.info("Configuration saved successfully to YAML.")
            self.statusBar().showMessage("Configuration saved to config/system_config.yaml", 3000)
        except Exception as exc:
            logging.error(f"Failed to save config: {exc}")
            QMessageBox.critical(self, "Error", f"Failed to save:\n{exc}")

    @pyqtSlot(str)
    def on_log_message(self, msg):
        self.console.appendPlainText(msg)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    @pyqtSlot(np.ndarray, np.ndarray, dict)
    def on_frames_ready(self, frame_l, frame_r, stats):
        if self.tabs.currentIndex() == 0:
            self.lbl_vid_l.setPixmap(self._cv_to_pixmap(frame_l, self.lbl_vid_l.size()))
            self.lbl_vid_r.setPixmap(self._cv_to_pixmap(frame_r, self.lbl_vid_r.size()))
        
        self.lbl_fps.setText(f"{stats['fps']:.1f}")
        current_time = time.time() - self.t_start
        self.data_t.append(current_time)
        
        if stats['tracked']:
            self.lbl_status.setText("TRACKING")
            self.lbl_status.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 14px;")
            self.lbl_3d.setText(f"[{stats['x']:.0f}, {stats['y']:.0f}, {stats['z']:.0f}]")
            
            self.data_x.append(stats['x'])
            self.data_y.append(stats['y'])
            self.data_z.append(stats['z'])
            
            self.gl_ball.setData(pos=np.array([[stats['x'], stats['y'], stats['z']]]))
            
            if len(self.data_x) > 5 and not np.isnan(self.data_x[-5]):
                dt = self.data_t[-1] - self.data_t[-5]
                if dt > 0:
                    vx = (self.data_x[-1] - self.data_x[-5]) / dt
                    vy = (self.data_y[-1] - self.data_y[-5]) / dt
                    vz = (self.data_z[-1] - self.data_z[-5]) / dt
                    
                    pred_pts = []
                    for t_fut in np.linspace(0, 1.5, 20):
                        px = stats['x'] + vx * t_fut
                        py = stats['y'] + vy * t_fut 
                        pz = stats['z'] + vz * t_fut
                        pred_pts.append([px, py, pz])
                    self.gl_pred.setData(pos=np.array(pred_pts))
                    
                    if abs(vz) > 0.1:
                        t_hit = (self.Z_GOAL - stats['z']) / vz
                        if 0 < t_hit < 3.0: 
                            x_hit = stats['x'] + vx * t_hit
                            y_hit = stats['y'] + vy * t_hit
                            
                            arm_length = 800.0
                            clamped_x = max(min(x_hit, arm_length), -arm_length) 
                            angle_rad = math.asin(clamped_x / arm_length)
                            angle_deg = math.degrees(angle_rad)
                            
                            self.lbl_angle.setText(f"{angle_deg:.1f}°")
                            
                            end_x = math.sin(angle_rad) * arm_length
                            end_y = math.cos(angle_rad) * arm_length
                            self.gl_robot_arm.setData(pos=np.array([[0, 0, self.Z_GOAL], [end_x, end_y, self.Z_GOAL]]))
        else:
            self.lbl_status.setText("LOST")
            self.lbl_status.setStyleSheet("color: #FF0000; font-weight: bold; font-size: 14px;")
            self.lbl_3d.setText("[---, ---, ---]")
            
            self.data_x.append(np.nan)
            self.data_y.append(np.nan)
            self.data_z.append(np.nan)
            self.gl_pred.setData(pos=np.array([[0,0,0], [0,0,0]])) 
            
            self.gl_robot_arm.setData(pos=np.array([[0,0,self.Z_GOAL], [0, 800, self.Z_GOAL]]))
            self.lbl_angle.setText("0.0°")
            
        def fmt_conf(res: DetectionResult) -> str:
            if not res.success: return "--"
            mark = "~" if res.is_predicted else ""
            return f"{res.method}{mark} {res.confidence:.2f}"
            
        self.lbl_conf_l.setText(fmt_conf(stats['res_l']))
        self.lbl_conf_r.setText(fmt_conf(stats['res_r']))
        
        if self.tabs.currentIndex() == 1:
            self.curve_x.setData(list(self.data_t), list(self.data_x))
            self.curve_y.setData(list(self.data_t), list(self.data_y))
            self.curve_z.setData(list(self.data_t), list(self.data_z))

    @pyqtSlot(str)
    def on_error(self, msg):
        self.lbl_status.setText("ERROR")
        logging.error(f"Tracker Error: {msg}")

    def _cv_to_pixmap(self, cv_img: np.ndarray, size) -> QPixmap:
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        return pm.scaled(size.width(), size.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def closeEvent(self, event):
        if hasattr(self, 'tracker_thread') and self.tracker_thread.isRunning():
            self.tracker_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    config = load_config("config/system_config.yaml")
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
