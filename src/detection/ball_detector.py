import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class BallDetector:
    """
    Detects a soccer ball in 2D camera frames.
    Supports high-speed HSV color thresholding and robust YOLOv8 deep learning inference.
    Includes a stereo-batch optimization to process left and right frames in parallel.
    """
    def __init__(self, 
                 method: str = "hsv", 
                 hsv_bounds: Optional[Dict[str, int]] = None,
                 confidence_threshold: float = 0.5) -> None:
        self.method: str = method.lower()
        self.confidence_threshold: float = confidence_threshold
        
        # Configure HSV bounds
        if hsv_bounds:
            self.lower_hsv = np.array([
                hsv_bounds.get("lower_h", 0),
                hsv_bounds.get("lower_s", 0),
                hsv_bounds.get("lower_v", 180)
            ])
            self.upper_hsv = np.array([
                hsv_bounds.get("upper_h", 180),
                hsv_bounds.get("upper_s", 60),
                hsv_bounds.get("upper_v", 255)
            ])
        else:
            # Default: White ball (High brightness, low saturation, any hue)
            self.lower_hsv = np.array([0, 0, 180])
            self.upper_hsv = np.array([180, 60, 255])

        # YOLOv8 initialization
        self.yolo_model = None
        if self.method == "yolo":
            try:
                from ultralytics import YOLO
                # Load a small nano model (automatically downloads the official COCO model)
                self.yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 model loaded successfully for stereo detection.")
            except ImportError:
                logger.warning("ultralytics package not found. Falling back to HSV method.")
                self.method = "hsv"

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """
        Processes a single BGR frame and returns the ball's center and radius.
        
        :param frame: BGR input image
        :return: (success, (x, y, radius))
        """
        if frame is None:
            return False, None

        if self.method == "yolo":
            return self._detect_yolo_single(frame)
        else:
            return self._detect_hsv_single(frame)

    def detect_stereo(self, 
                      frame_left: np.ndarray, 
                      frame_right: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]], bool, Optional[Tuple[int, int, int]]]:
        """
        Processes a stereo pair of frames (left and right).
        Uses batching in YOLOv8 mode to process both frames in a single inference pass,
        resulting in a significant speedup on both GPU and CPU.
        
        :return: (success_left, ball_left, success_right, ball_right)
                 where ball_data is (x, y, radius)
        """
        if frame_left is None or frame_right is None:
            return False, None, False, None

        if self.method == "yolo" and self.yolo_model is not None:
            return self._detect_yolo_stereo_batch(frame_left, frame_right)
        else:
            # HSV runs sequentially (very lightweight, no batch benefit)
            ret_l, ball_l = self._detect_hsv_single(frame_left)
            ret_r, ball_r = self._detect_hsv_single(frame_right)
            return ret_l, ball_l, ret_r, ball_r

    def _detect_hsv_single(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Color thresholding (HSV) detection - optimized for white/colored spherical shapes."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Perform thresholding
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Morphological operations to clean up noise (erode and dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None
            
        best_circle_data = None
        max_circularity = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Ignore tiny noise artifacts
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Circularity metric: 4 * pi * Area / Perimeter^2 (1.0 is a perfect circle)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Find minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            area_ratio = area / circle_area if circle_area > 0 else 0
            
            # Filter for spherical shapes (circularity > 0.4 and area ratio > 0.4)
            if circularity > 0.4 and area_ratio > 0.4:
                if circularity > max_circularity:
                    max_circularity = circularity
                    best_circle_data = (int(x), int(y), int(radius))
        
        if best_circle_data is not None:
            return True, best_circle_data
            
        # Fallback: if no highly circular shape is found, take the largest contour exceeding area threshold
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            return True, (int(x), int(y), int(radius))

        return False, None

    def _detect_yolo_single(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Detects a ball in a single frame using YOLOv8."""
        if self.yolo_model is None:
            return False, None
        
        results = self.yolo_model.predict(frame, verbose=False, conf=self.confidence_threshold)
        return self._parse_yolo_result(results[0])

    def _detect_yolo_stereo_batch(self, 
                                  frame_left: np.ndarray, 
                                  frame_right: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]], bool, Optional[Tuple[int, int, int]]]:
        """Runs parallel batch inference on both frames, saving processing time."""
        # Pass a list of frames to the model (YOLOv8 runs them as a batch)
        results = self.yolo_model.predict([frame_left, frame_right], verbose=False, conf=self.confidence_threshold)
        
        ret_l, ball_l = self._parse_yolo_result(results[0])
        ret_r, ball_r = self._parse_yolo_result(results[1])
        
        return ret_l, ball_l, ret_r, ball_r

    def _parse_yolo_result(self, result: Any) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Parses a YOLOv8 result object and extracts class 32 (sports ball) bounding boxes."""
        best_box = None
        max_conf = 0.0
        
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # COCO Class 32 is "sports ball" (which fits soccer balls, tennis balls, etc.)
            if class_id == 32:
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
                    
        if best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            
            # Radius estimation from average dimension
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            radius = int((width + height) / 4)
            
            return True, (x_center, y_center, radius)
            
        return False, None
