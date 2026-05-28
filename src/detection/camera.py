import time
import math
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class CameraInterface(ABC):
    """
    Abstract base class defining the standard interface for cameras.
    Ensures that MindVision, OpenCV, and Mock cameras can be swapped transparently.
    """
    @abstractmethod
    def open(self) -> bool:
        """Opens the camera stream. Returns True if successful."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame from the camera. Returns (success, frame)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the camera stream and releases resources."""
        pass

    @abstractmethod
    def get_timestamp(self) -> float:
        """Returns the hardware/software timestamp of the last captured frame."""
        pass


class MockCamera(CameraInterface):
    """
    Simulates a camera stream for development and testing on PC/Laptop.
    Can read from a local video file, or generate a synthetic bouncing 3D soccer ball 
    projected onto the left/right 2D camera planes to validate stereo triangulation.
    """
    def __init__(self, 
                 width: int = 1280, 
                 height: int = 720, 
                 fps: int = 60,
                 is_left: bool = True, 
                 video_path: Optional[str] = None) -> None:
        self.width: int = width
        self.height: int = height
        self.fps: int = fps
        self.is_left: bool = is_left
        self.video_path: Optional[str] = video_path
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_timestamp: float = 0.0
        self.frame_count: int = 0
        
        # Stereo parameters for simulation projection
        self.baseline_mm: float = 300.0
        self.focal_length_px: float = 1200.0
        self.cx: float = width / 2.0
        self.cy: float = height / 2.0

    def open(self) -> bool:
        if self.video_path:
            logger.info(f"MockCamera: Opening video file {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            return self.cap.isOpened()
        else:
            logger.info(f"MockCamera: Running in synthetic 3D projection mode (is_left={self.is_left})")
            self.last_timestamp = time.time()
            return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Simulate frame-rate timing delay
        sleep_time = 1.0 / self.fps
        time.sleep(sleep_time)
        self.last_timestamp = time.time()
        self.frame_count += 1

        if self.video_path:
            if self.cap is None:
                return False, None
            ret, frame = self.cap.read()
            if not ret:
                # Loop video for continuous testing
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (self.width, self.height))
            return ret, frame
        else:
            # Generate synthetic frame with a 3D bouncing ball projected onto 2D
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw playing field grid lines for visual reference
            for y_line in range(0, self.height, 100):
                cv2.line(frame, (0, y_line), (self.width, y_line), (40, 40, 40), 1)
            for x_line in range(0, self.width, 100):
                cv2.line(frame, (x_line, 0), (x_line, self.height), (40, 40, 40), 1)
            
            # Calculate ball's simulated 3D position (X, Y, Z) in mm over time
            t = self.frame_count / self.fps
            
            # Z: Ball starts 3 meters away (3000mm) and flies towards the goal (Z -> 0mm) at 4 m/s
            z_3d = 3000.0 - (4000.0 * (t % 1.5))
            
            # If ball passes the goal, reset simulation loop
            if z_3d < 100.0:
                z_3d = 3000.0
                self.frame_count = 0
            
            # X: Bounces left and right
            x_3d = 400.0 * math.sin(2 * math.pi * 0.5 * t)
            
            # Y: Bounces up and down (parabolic bounce)
            y_3d = 200.0 + 300.0 * abs(math.cos(2 * math.pi * 0.8 * t)) - 100.0
            
            # Project 3D coordinate to 2D image plane using pinhole camera equations
            # Left camera shifts X by +baseline/2, Right camera shifts X by -baseline/2
            offset_x = (self.baseline_mm / 2.0) if self.is_left else -(self.baseline_mm / 2.0)
            
            x_proj = self.cx + (self.focal_length_px * (x_3d + offset_x)) / z_3d
            y_proj = self.cy - (self.focal_length_px * y_3d) / z_3d
            
            # Ball radius in pixels decreases as Z increases (perspective effect)
            ball_radius_px = int((35.0 * self.focal_length_px) / z_3d)  # 35mm physical radius (soccer ball is ~110mm, let's use 110mm)
            ball_radius_px = max(5, min(ball_radius_px, 150))
            
            # Draw synthetic orange soccer ball
            center_pt = (int(x_proj), int(y_proj))
            if 0 <= center_pt[0] < self.width and 0 <= center_pt[1] < self.height:
                # Orange fill
                cv2.circle(frame, center_pt, ball_radius_px, (30, 120, 240), -1)
                # Black outline
                cv2.circle(frame, center_pt, ball_radius_px, (0, 0, 0), 2)
                # Draw pentagon-like patterns to mimic a soccer ball
                for angle in range(0, 360, 72):
                    rad = math.radians(angle + t * 45)
                    pattern_pt = (
                        int(center_pt[0] + (ball_radius_px * 0.5) * math.cos(rad)),
                        int(center_pt[1] + (ball_radius_px * 0.5) * math.sin(rad))
                    )
                    cv2.circle(frame, pattern_pt, max(2, int(ball_radius_px * 0.15)), (0, 0, 0), -1)
                    cv2.line(frame, center_pt, pattern_pt, (0, 0, 0), 1)

            # Draw HUD
            cam_label = "LEFT CAMERA (MOCK)" if self.is_left else "RIGHT CAMERA (MOCK)"
            cv2.putText(frame, cam_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sim 3D Pos: [{x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f}] mm", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return True, frame

    def close(self) -> None:
        if self.cap:
            self.cap.release()
        logger.info("MockCamera closed.")

    def get_timestamp(self) -> float:
        return self.last_timestamp


class MindVisionCamera(CameraInterface):
    """
    Interfaces directly with the MindVision MC023CG-SY-UB camera using the native Python SDK (mvsdk).
    Utilizes Direct Memory Access (DMA) and hardware callbacks for minimal CPU usage.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.camera_index: int = camera_index
        self.width: int = width
        self.height: int = height
        self.h_camera: Any = None
        self.sdk_loaded: bool = False
        self.last_timestamp: float = 0.0

        # Try importing the MindVision SDK python module
        try:
            global mvsdk
            import mvsdk
            self.sdk_loaded = True
            logger.info("MindVision SDK (mvsdk) successfully loaded.")
        except ImportError:
            logger.warning("MindVision SDK (mvsdk) not found. Falling back to OpenCV wrapper.")
            self.sdk_loaded = False

    def open(self) -> bool:
        if not self.sdk_loaded:
            logger.warning(f"MindVision SDK not available. Attempting OpenCV fallback on index {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return True
            return False

        try:
            # Enumerate cameras
            device_list = mvsdk.CameraEnumerateDevice()
            num_devices = len(device_list)
            if num_devices == 0:
                logger.error("No MindVision cameras found.")
                return False

            if self.camera_index >= num_devices:
                logger.error(f"Target camera index {self.camera_index} out of bounds (found {num_devices} devices).")
                return False

            device = device_list[self.camera_index]
            self.h_camera = mvsdk.CameraInit(device, -1, -1)
            
            # Print camera model details
            cap_desc = mvsdk.CameraGetFriendlyName(self.h_camera)
            logger.info(f"Connected to MindVision camera: {cap_desc}")

            # Configure trigger mode (hardware sync sync cable triggers)
            # 0: Continuous mode, 1: Software trigger, 2: Hardware trigger
            mvsdk.CameraSetTriggerMode(self.h_camera, 0) # Default to continuous for now

            # Set resolution
            # mvsdk.CameraSetImageResolution(self.h_camera, resolution_struct)
            
            # Start camera play
            mvsdk.CameraPlay(self.h_camera)
            
            # Allocate frame buffer
            cap_info = mvsdk.CameraGetCapability(self.h_camera)
            self.buffer_size = cap_info.sResolutionRange.iWidthMax * cap_info.sResolutionRange.iHeightMax * 3
            self.p_frame_buffer = mvsdk.CameraAlignMalloc(self.buffer_size, 16)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing MindVision Camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self.last_timestamp = time.time()
        
        if not self.sdk_loaded:
            ret, frame = self.cap.read()
            return ret, frame

        try:
            # Read frame with a 500ms timeout
            p_raw_data, frame_head = mvsdk.CameraGetImageBuffer(self.h_camera, 500)
            
            # Image conversion (Demosaic RAW to RGB) on the SDK side (C++ optimization)
            mvsdk.CameraImageProcess(self.h_camera, p_raw_data, self.p_frame_buffer, frame_head)
            
            # Release buffer back to camera queue
            mvsdk.CameraReleaseImageBuffer(self.h_camera, p_raw_data)
            
            # Create numpy array view over the C buffer
            # MindVision outputs RGB, so we reshape and can convert to BGR for OpenCV
            frame_data = (mvsdk.c_char * frame_head.uBytes).from_address(self.p_frame_buffer)
            frame_np = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame_np.reshape((frame_head.iHeight, frame_head.iWidth, 3))
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return True, frame_bgr
        except Exception as e:
            logger.error(f"Error reading frame from MindVision Camera: {e}")
            return False, None

    def close(self) -> None:
        if not self.sdk_loaded:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            return

        if self.h_camera:
            mvsdk.CameraUnInit(self.h_camera)
            mvsdk.CameraAlignFree(self.p_frame_buffer)
            self.h_camera = None
            logger.info("MindVision Camera closed.")

    def get_timestamp(self) -> float:
        return self.last_timestamp
