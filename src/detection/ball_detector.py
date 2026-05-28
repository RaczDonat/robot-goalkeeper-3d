import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from detection.ball_tracker import BallKalmanTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Unified result from any detection layer.

    Fields
    ------
    success      : True if a valid ball position is available (even if predicted).
    x, y         : Ball centre in pixels.
    radius       : Estimated ball radius in pixels.
    confidence   : 0.0–1.0 score.  1.0 = high-confidence YOLO hit, lower for
                   HSV/Hough, lower still for Kalman-only coasting.
    method       : Which layer produced this result: "yolo" | "hsv" | "kalman" | "none"
    is_predicted : True when the position is a Kalman-only extrapolation (no raw detection).
    """
    success: bool = False
    x: int = 0
    y: int = 0
    radius: int = 0
    confidence: float = 0.0
    method: str = "none"
    is_predicted: bool = False

    def as_tuple(self) -> Optional[Tuple[int, int, int]]:
        """Returns (x, y, radius) or None if not successful."""
        return (self.x, self.y, self.radius) if self.success else None


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class BallDetector:
    """
    Three-layer hybrid ball detector for a soccer ball in a stereo camera setup.

    Detection pipeline (per frame)
    ──────────────────────────────
    Layer 1 – YOLOv8 (primary, GPU-accelerated)
        Uses COCO class 32 ("sports ball").  A fine-tuned model path can be
        supplied via ``yolo_model_path`` to improve accuracy over the generic
        COCO checkpoint.  Both left and right frames are inferred in a single
        batched call.

    Layer 2 – HSV colour threshold + Hough Circle Transform (fallback)
        Applied only when YOLO fails or returns low confidence.
        Designed for outdoor, bright-sunlight conditions with a white ball.
        Hough circles give more robust radius estimation than the previous
        min-enclosing-circle heuristic.

    Layer 3 – Kalman filter temporal smoothing / coasting
        After every successful detection (from any layer) the per-camera Kalman
        filter is *updated*.  When all raw detection layers fail, the filter is
        *predicted* (coasting) for up to ``max_coast_frames`` frames before the
        tracker is declared lost.

    Public API
    ──────────
    detect(frame)               → DetectionResult
    detect_stereo(left, right)  → (DetectionResult_L, DetectionResult_R)
    """

    # COCO class index for "sports ball"
    _COCO_BALL_CLASS = 32

    def __init__(
        self,
        method: str = "hybrid",
        yolo_model_path: str = "yolov8n.pt",
        hsv_bounds: Optional[Dict[str, int]] = None,
        hough_cfg: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.4,
        kalman_cfg: Optional[Dict[str, Any]] = None,
        roi_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param method:               "hybrid" | "yolo" | "hsv"
        :param yolo_model_path:      Path or name of the YOLO model file.
                                     Use a fine-tuned soccer ball model for best results.
        :param hsv_bounds:           Dict with keys lower_h/s/v, upper_h/s/v.
                                     Defaults to outdoor white-ball bounds.
        :param hough_cfg:            Dict with Hough circle parameters
                                     (min_dist, param1, param2, min_radius, max_radius).
        :param confidence_threshold: Minimum YOLO confidence accepted (0–1).
        :param kalman_cfg:           Dict with process_noise, measurement_noise,
                                     max_coast_frames.
        :param roi_cfg:              Dict with enabled (bool), padding_factor (float).
        """
        self.method = method.lower()
        self.confidence_threshold = confidence_threshold

        # ── HSV bounds ────────────────────────────────────────────────────────
        b = hsv_bounds or {}
        self.lower_hsv = np.array([
            b.get("lower_h", 0),
            b.get("lower_s", 0),
            b.get("lower_v", 200),   # outdoor sunlight: high brightness
        ], dtype=np.uint8)
        self.upper_hsv = np.array([
            b.get("upper_h", 180),
            b.get("upper_s", 50),    # low saturation → white ball
            b.get("upper_v", 255),
        ], dtype=np.uint8)

        # ── Hough config ──────────────────────────────────────────────────────
        hc = hough_cfg or {}
        self._hough_min_dist   = hc.get("min_dist",   30)
        self._hough_param1     = hc.get("param1",    100)
        self._hough_param2     = hc.get("param2",     30)
        self._hough_min_radius = hc.get("min_radius",  8)
        self._hough_max_radius = hc.get("max_radius", 120)

        # ── ROI config ────────────────────────────────────────────────────────
        rc = roi_cfg or {}
        self._roi_enabled        = rc.get("enabled", True)
        self._roi_padding_factor = rc.get("padding_factor", 2.5)

        # Per-camera ROI state (updated each frame)
        self._roi_left:  Optional[Tuple[int, int, int, int]] = None   # x1,y1,x2,y2
        self._roi_right: Optional[Tuple[int, int, int, int]] = None

        # ── Kalman trackers (one per camera) ──────────────────────────────────
        kc = kalman_cfg or {}
        kalman_kwargs = {
            "process_noise":     kc.get("process_noise",    1e-2),
            "measurement_noise": kc.get("measurement_noise", 1e-1),
            "max_coast_frames":  kc.get("max_coast_frames",  10),
        }
        self._kalman_left  = BallKalmanTracker(**kalman_kwargs)
        self._kalman_right = BallKalmanTracker(**kalman_kwargs)

        # ── YOLO ──────────────────────────────────────────────────────────────
        self.yolo_model = None
        if self.method in ("yolo", "hybrid"):
            self._init_yolo(yolo_model_path)

        logger.info(
            "BallDetector ready | method=%s | yolo=%s",
            self.method,
            "loaded" if self.yolo_model else "unavailable",
        )

    # ------------------------------------------------------------------
    # YOLO initialisation
    # ------------------------------------------------------------------

    def _init_yolo(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import]
            self.yolo_model = YOLO(model_path)
            logger.info("YOLOv8 model loaded: %s", model_path)
        except ImportError:
            logger.warning(
                "ultralytics not installed – YOLO layer disabled. "
                "Install with: pip install ultralytics"
            )
            if self.method == "yolo":
                self.method = "hsv"
        except Exception as exc:
            logger.warning("Failed to load YOLO model '%s': %s", model_path, exc)
            if self.method == "yolo":
                self.method = "hsv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect the ball in a single frame.

        :param frame: BGR image
        :return: DetectionResult
        """
        if frame is None:
            return DetectionResult()

        result, _ = self._detect_frame(frame, self._kalman_left, roi=self._roi_left)
        self._roi_left = self._make_roi(result, frame.shape)
        return result

    def detect_stereo(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
    ) -> Tuple["DetectionResult", "DetectionResult"]:
        """
        Detect the ball in both stereo frames.

        In "yolo" / "hybrid" mode the two frames are passed as a batch so that
        YOLO runs a single GPU inference call for both cameras.

        :return: (result_left, result_right)
        """
        if frame_left is None or frame_right is None:
            return DetectionResult(), DetectionResult()

        yolo_left: Optional[DetectionResult] = None
        yolo_right: Optional[DetectionResult] = None

        # ── YOLO batch (primary layer) ────────────────────────────────────────
        if self.method in ("yolo", "hybrid") and self.yolo_model is not None:
            yolo_left, yolo_right = self._yolo_batch(frame_left, frame_right)

        # ── Per-camera fallback + Kalman ──────────────────────────────────────
        result_left,  self._roi_left  = self._finalise(
            frame_left,  yolo_left,  self._kalman_left,  self._roi_left
        )
        result_right, self._roi_right = self._finalise(
            frame_right, yolo_right, self._kalman_right, self._roi_right
        )

        return result_left, result_right

    # ------------------------------------------------------------------
    # Internal pipeline helpers
    # ------------------------------------------------------------------

    def _finalise(
        self,
        frame: np.ndarray,
        yolo_result: Optional["DetectionResult"],
        kalman: BallKalmanTracker,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> Tuple["DetectionResult", Optional[Tuple[int, int, int, int]]]:
        """
        Combine the YOLO result (may be None) with HSV/Hough fallback and Kalman.
        Returns the final DetectionResult and the updated ROI for the next frame.
        """
        # Layer 1: use YOLO if confident enough
        if yolo_result is not None and yolo_result.success:
            sx, sy = kalman.update(yolo_result.x, yolo_result.y)
            yolo_result.x, yolo_result.y = sx, sy
            new_roi = self._make_roi(yolo_result, frame.shape)
            return yolo_result, new_roi

        # Layer 2: HSV + Hough fallback
        if self.method in ("hsv", "hybrid"):
            hsv_result = self._detect_hsv_hough(frame, roi)
            if hsv_result.success:
                sx, sy = kalman.update(hsv_result.x, hsv_result.y)
                hsv_result.x, hsv_result.y = sx, sy
                new_roi = self._make_roi(hsv_result, frame.shape)
                return hsv_result, new_roi

        # Layer 3: Kalman coasting
        predicted = kalman.predict()
        if predicted is not None:
            return DetectionResult(
                success=True,
                x=predicted[0],
                y=predicted[1],
                radius=self._last_radius(kalman),
                confidence=kalman.confidence,
                method="kalman",
                is_predicted=True,
            ), roi   # keep old ROI while coasting

        # Total failure
        return DetectionResult(), None

    def _detect_frame(
        self,
        frame: np.ndarray,
        kalman: BallKalmanTracker,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> Tuple["DetectionResult", Optional[Tuple[int, int, int, int]]]:
        """Single-frame detect path (used by public detect() method)."""
        yolo_result: Optional[DetectionResult] = None

        if self.method in ("yolo", "hybrid") and self.yolo_model is not None:
            results = self.yolo_model.predict(frame, verbose=False, conf=self.confidence_threshold)
            yolo_result = self._parse_yolo_result(results[0])

        return self._finalise(frame, yolo_result, kalman, roi)

    # ------------------------------------------------------------------
    # YOLO helpers
    # ------------------------------------------------------------------

    def _yolo_batch(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
    ) -> Tuple[Optional["DetectionResult"], Optional["DetectionResult"]]:
        """Run batched YOLO inference on both stereo frames in one GPU call."""
        try:
            results = self.yolo_model.predict(
                [frame_left, frame_right],
                verbose=False,
                conf=self.confidence_threshold,
            )
            return (
                self._parse_yolo_result(results[0]),
                self._parse_yolo_result(results[1]),
            )
        except Exception as exc:
            logger.warning("YOLO batch inference failed: %s", exc)
            return None, None

    def _parse_yolo_result(self, result: Any) -> Optional["DetectionResult"]:
        """
        Extract the highest-confidence sports-ball (class 32) detection.

        Also validates the bounding box aspect ratio – a soccer ball should be
        roughly square (width / height between 0.5 and 2.0).
        """
        best_conf = 0.0
        best_box  = None

        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != self._COCO_BALL_CLASS:
                continue
            if conf < self.confidence_threshold:
                continue

            # Aspect-ratio sanity check
            xyxy   = box.xyxy[0].cpu().numpy()
            width  = float(xyxy[2] - xyxy[0])
            height = float(xyxy[3] - xyxy[1])
            if height < 1:
                continue
            aspect = width / height
            if not (0.4 < aspect < 2.5):
                continue   # very elongated → not a ball

            if conf > best_conf:
                best_conf = conf
                best_box  = xyxy

        if best_box is None:
            return None

        x_center = int((best_box[0] + best_box[2]) / 2)
        y_center = int((best_box[1] + best_box[3]) / 2)
        radius   = int((best_box[2] - best_box[0] + best_box[3] - best_box[1]) / 4)

        return DetectionResult(
            success=True,
            x=x_center,
            y=y_center,
            radius=max(radius, 1),
            confidence=best_conf,
            method="yolo",
            is_predicted=False,
        )

    # ------------------------------------------------------------------
    # HSV + Hough Circle fallback
    # ------------------------------------------------------------------

    def _detect_hsv_hough(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> "DetectionResult":
        """
        HSV colour thresholding followed by Hough Circle Transform.

        Steps
        -----
        1. (Optional) Crop to ROI from previous detection for speed.
        2. Convert to HSV and threshold for white/light-coloured balls.
        3. Morphological close+open to fill ball interior and remove shadow noise.
        4. cv2.HoughCircles to find the most circular blob.
        5. Map result back to full-frame coordinates.
        """
        h_frame, w_frame = frame.shape[:2]
        x_offset = y_offset = 0

        # Apply ROI crop if available and enabled
        work = frame
        if self._roi_enabled and roi is not None:
            x1, y1, x2, y2 = roi
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_frame, x2); y2 = min(h_frame, y2)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                work     = frame[y1:y2, x1:x2]
                x_offset = x1
                y_offset = y1

        # ── HSV threshold ──────────────────────────────────────────────────
        hsv  = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Morphological clean-up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)  # fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)   # remove small noise

        # Blur for Hough stability
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)

        # ── Hough Circle Transform ─────────────────────────────────────────
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self._hough_min_dist,
            param1=self._hough_param1,
            param2=self._hough_param2,
            minRadius=self._hough_min_radius,
            maxRadius=self._hough_max_radius,
        )

        if circles is None:
            return DetectionResult()

        # Pick the circle with the largest accumulator vote (most prominent)
        circles = np.round(circles[0]).astype(int)
        # circles shape: (N, 3) → [cx, cy, r]
        best = circles[0]   # HoughCircles returns them sorted by vote count

        cx = int(best[0]) + x_offset
        cy = int(best[1]) + y_offset
        r  = int(best[2])

        # Basic circularity validation using the mask
        if r < 2:
            return DetectionResult()

        # Compute pixel coverage inside the detected circle on the original mask
        if self._roi_enabled and roi is not None:
            full_mask = cv2.inRange(
                cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                self.lower_hsv, self.upper_hsv,
            )
        else:
            full_mask = mask   # already full-frame when no ROI

        coverage = self._circle_mask_coverage(full_mask, cx, cy, r)
        if coverage < 0.25:
            # Less than 25% of the circle area is white → probably noise
            return DetectionResult()

        # Confidence: scale coverage → 0.5–0.9 range (HSV is less certain than YOLO)
        conf = 0.5 + 0.4 * min(coverage, 1.0)

        return DetectionResult(
            success=True,
            x=cx,
            y=cy,
            radius=r,
            confidence=conf,
            method="hsv",
            is_predicted=False,
        )

    @staticmethod
    def _circle_mask_coverage(mask: np.ndarray, cx: int, cy: int, r: int) -> float:
        """Fraction of pixels inside the circle that are white (255) in the mask."""
        h, w = mask.shape[:2]
        # Create a single-channel circle mask
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (cx, cy), max(r, 1), 255, -1)
        circle_area = np.count_nonzero(circle_mask)
        if circle_area == 0:
            return 0.0
        overlap = np.count_nonzero(cv2.bitwise_and(mask, circle_mask))
        return overlap / circle_area

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    def _make_roi(
        self,
        result: "DetectionResult",
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute the next-frame ROI from a detection result."""
        if not result.success or result.radius < 1:
            return None
        pad = int(result.radius * self._roi_padding_factor)
        h, w = frame_shape[:2]
        x1 = max(0,      result.x - pad)
        y1 = max(0,      result.y - pad)
        x2 = min(w - 1,  result.x + pad)
        y2 = min(h - 1,  result.y + pad)
        return (x1, y1, x2, y2)

    @staticmethod
    def _last_radius(kalman: BallKalmanTracker) -> int:
        """Return a safe default radius for Kalman-only coasting frames."""
        # We don't track radius in Kalman; return a reasonable placeholder.
        return 15
