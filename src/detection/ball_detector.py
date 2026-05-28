import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class BallDetector:
    """
    Detects a soccer ball in a 2D camera frame.
    Supports high-speed HSV color thresholding and robust YOLOv8 deep learning inference.
    """
    def __init__(self, 
                 method: str = "hsv", 
                 hsv_bounds: Optional[Dict[str, int]] = None,
                 confidence_threshold: float = 0.5) -> None:
        self.method: str = method.lower()
        self.confidence_threshold: float = confidence_threshold
        
        # Load HSV bounds
        if hsv_bounds:
            self.lower_hsv = np.array([
                hsv_bounds.get("lower_h", 5),
                hsv_bounds.get("lower_s", 100),
                hsv_bounds.get("lower_v", 100)
            ])
            self.upper_hsv = np.array([
                hsv_bounds.get("upper_h", 25),
                hsv_bounds.get("upper_s", 255),
                hsv_bounds.get("upper_v", 255)
            ])
        else:
            # Default: Orange ball
            self.lower_hsv = np.array([5, 100, 100])
            self.upper_hsv = np.array([25, 255, 255])

        # YOLOv8 initialization
        self.yolo_model = None
        if self.method == "yolo":
            try:
                from ultralytics import YOLO
                # Load a small nano model (will automatically download if not present)
                self.yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLOv8 model loaded successfully.")
            except ImportError:
                logger.warning("ultralytics package not found. Falling back to HSV method.")
                self.method = "hsv"

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """
        Processes a single BGR frame and returns the ball's center and radius.
        
        :param frame: BGR input image
        :return: (success_boolean, (x, y, radius))
        """
        if frame is None:
            return False, None

        if self.method == "yolo":
            return self._detect_yolo(frame)
        else:
            return self._detect_hsv(frame)

    def _detect_hsv(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Color thresholding (HSV) detection - low latency, high FPS."""
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
            
        # Find the largest circular contour
        best_contour = None
        max_circularity = 0.0
        best_circle_data = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Ignore very small noise contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Circularity metric: 4 * pi * Area / Perimeter^2 (1.0 is a perfect circle)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # Find minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            
            # Area ratio: contour area vs enclosing circle area (should be close to 1 for circles)
            area_ratio = area / circle_area if circle_area > 0 else 0
            
            # Filter for circular shape and reasonable size
            if circularity > 0.4 and area_ratio > 0.4:
                if circularity > max_circularity:
                    max_circularity = circularity
                    best_contour = contour
                    best_circle_data = (int(x), int(y), int(radius))
        
        if best_circle_data is not None:
            return True, best_circle_data
            
        # Fallback: if no circular contour found, take the largest contour exceeding area threshold
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            return True, (int(x), int(y), int(radius))

        return False, None

    def _detect_yolo(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
        """Deep learning YOLOv8 detection - high accuracy, higher latency."""
        if self.yolo_model is None:
            return False, None

        # Run inference (specifically looking for sports ball, class index 32 in COCO)
        results = self.yolo_model.predict(frame, verbose=False, conf=self.confidence_threshold)
        
        best_box = None
        max_conf = 0.0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Class 32 is "sports ball" in COCO dataset
                if class_id == 32:
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box
                        
        if best_box is not None:
            # Get bounding box coordinates [x_min, y_min, x_max, y_max]
            xyxy = best_box.xyxy[0].cpu().numpy()
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            
            # Estimate radius as half the average width/height
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            radius = int((width + height) / 4)
            
            return True, (x_center, y_center, radius)
            
        return False, None
