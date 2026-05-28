import time
import logging
import argparse
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

# Import local modules
from common.network import UDPSender
from detection.camera import MockCamera, MindVisionCamera
from detection.ball_detector import BallDetector
from stereo.triangulation import StereoTriangulator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration, falling back to defaults if YAML loader is not available or file is missing."""
    defaults = {
        "network": {"rpi_ip": "127.0.0.1", "port": 5005},
        "camera": {"type": "mock", "resolution": {"width": 1280, "height": 720}, "fps": 60},
        "stereo": {"baseline_mm": 300.0, "focal_length_px": 1200.0, "cx": 640.0, "cy": 360.0, "left_camera_index": 0, "right_camera_index": 1},
        "detection": {"method": "hsv", "confidence_threshold": 0.5, "hsv_bounds": {"lower_h": 0, "lower_s": 0, "lower_v": 180, "upper_h": 180, "upper_s": 60, "upper_v": 255}}
    }
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info("Successfully loaded system configuration from YAML.")
            return config
    except ImportError:
        logger.warning("PyYAML not installed. Using default configuration settings.")
        return defaults
    except Exception as e:
        logger.warning(f"Could not load config file ({e}). Using default settings.")
        return defaults

def save_hsv_config(config_path: str, config: Dict[str, Any], lh: int, uh: int, ls: int, us: int, lv: int, uv: int) -> None:
    """Saves the calibrated HSV bounds back to the system config file."""
    try:
        import yaml
        config["detection"]["hsv_bounds"] = {
            "lower_h": lh,
            "upper_h": uh,
            "lower_s": ls,
            "upper_s": us,
            "lower_v": lv,
            "upper_v": uv
        }
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info(f"CALIBRATION SAVED: Updated HSV bounds in {config_path}")
    except Exception as e:
        logger.error(f"Failed to save HSV calibration: {e}")

def nothing(x: int) -> None:
    """Placeholder callback for OpenCV trackbars."""
    pass

def setup_calibration_window(hsv_bounds: Dict[str, int]) -> None:
    """Creates trackbars for HSV calibration."""
    cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Calibration", 400, 350)
    
    cv2.createTrackbar("Lower H", "HSV Calibration", hsv_bounds.get("lower_h", 0), 179, nothing)
    cv2.createTrackbar("Upper H", "HSV Calibration", hsv_bounds.get("upper_h", 180), 179, nothing)
    cv2.createTrackbar("Lower S", "HSV Calibration", hsv_bounds.get("lower_s", 0), 255, nothing)
    cv2.createTrackbar("Upper S", "HSV Calibration", hsv_bounds.get("upper_s", 60), 255, nothing)
    cv2.createTrackbar("Lower V", "HSV Calibration", hsv_bounds.get("lower_v", 180), 255, nothing)
    cv2.createTrackbar("Upper V", "HSV Calibration", hsv_bounds.get("upper_v", 255), 255, nothing)

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time 3D Ball Tracker (PC Side)")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Path to config file")
    parser.add_argument("--calibrate", action="store_true", help="Launch in HSV calibration mode with sliders")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    net_cfg = config["network"]
    cam_cfg = config["camera"]
    stereo_cfg = config["stereo"]
    det_cfg = config["detection"]

    # Initialize UDP Sender
    sender = UDPSender(ip=net_cfg["rpi_ip"], port=net_cfg["port"])

    # Initialize Cameras (Left and Right)
    width = cam_cfg["resolution"]["width"]
    height = cam_cfg["resolution"]["height"]
    fps = cam_cfg["fps"]
    cam_type = cam_cfg.get("type", "mock").lower()

    if cam_type == "mindvision":
        logger.info("Initializing MindVision cameras...")
        cam_left = MindVisionCamera(camera_index=stereo_cfg["left_camera_index"], width=width, height=height)
        cam_right = MindVisionCamera(camera_index=stereo_cfg["right_camera_index"], width=width, height=height)
    elif cam_type == "opencv":
        logger.info("Initializing standard OpenCV USB cameras...")
        cam_left = MindVisionCamera(camera_index=stereo_cfg["left_camera_index"], width=width, height=height)
        cam_left.sdk_loaded = False  # Bypasses SDK, forces standard V4L2VideoCapture
        cam_right = MindVisionCamera(camera_index=stereo_cfg["right_camera_index"], width=width, height=height)
        cam_right.sdk_loaded = False
    else:
        logger.info("Initializing mock cameras with simulated 3D projection...")
        cam_left = MockCamera(width=width, height=height, fps=fps, is_left=True)
        cam_right = MockCamera(width=width, height=height, fps=fps, is_left=False)

    # Initialize a single Ball Detector for stereo tracking (supports parallel batching)
    detector = BallDetector(
        method=det_cfg["method"], 
        hsv_bounds=det_cfg.get("hsv_bounds"), 
        confidence_threshold=det_cfg["confidence_threshold"]
    )

    # Initialize Stereo Triangulator
    triangulator = StereoTriangulator(
        baseline_mm=stereo_cfg["baseline_mm"],
        focal_length_px=stereo_cfg["focal_length_px"],
        cx=stereo_cfg.get("cx", stereo_cfg.get("principal_point_x", 640.0)),
        cy=stereo_cfg.get("cy", stereo_cfg.get("principal_point_y", 360.0))
    )

    # Open camera streams
    if not cam_left.open() or not cam_right.open():
        logger.error("Failed to open one or both camera streams. Exiting.")
        return

    logger.info("Camera streams successfully opened. Starting tracking loop. Press 'q' to quit.")

    # Setup calibration window if requested
    if args.calibrate and det_cfg["method"] == "hsv":
        logger.info("Starting HSV Calibration mode. Adjust sliders. Press 's' to save bounds, 'q' to exit.")
        setup_calibration_window(det_cfg.get("hsv_bounds", {}))

    window_name = "Real-Time 3D Ball Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            t_start = time.time()

            # 1. Read calibration trackbars if in calibration mode
            if args.calibrate and det_cfg["method"] == "hsv":
                lh = cv2.getTrackbarPos("Lower H", "HSV Calibration")
                uh = cv2.getTrackbarPos("Upper H", "HSV Calibration")
                ls = cv2.getTrackbarPos("Lower S", "HSV Calibration")
                us = cv2.getTrackbarPos("Upper S", "HSV Calibration")
                lv = cv2.getTrackbarPos("Lower V", "HSV Calibration")
                uv = cv2.getTrackbarPos("Upper V", "HSV Calibration")
                
                # Dynamic update of HSV bounds
                new_lower = np.array([lh, ls, lv])
                new_upper = np.array([uh, us, uv])
                detector.lower_hsv = new_lower
                detector.upper_hsv = new_upper

            # 2. Capture frames
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                logger.warning("Empty frame received. Skipping frame.")
                continue

            timestamp = cam_left.get_timestamp()

            # 3. Run Optimized 2D Stereo Ball Detection (Parallel batching for YOLO)
            success_l, ball_l, success_r, ball_r = detector.detect_stereo(frame_l, frame_r)

            x_3d, y_3d, z_3d = 0.0, 0.0, 0.0
            tracking_success = False

            # 4. If ball detected in both images, calculate 3D position
            if success_l and success_r and ball_l and ball_r:
                pt_l = (ball_l[0], ball_l[1])
                pt_r = (ball_r[0], ball_r[1])
                
                # Triangulate
                tracking_success, x_3d, y_3d, z_3d = triangulator.triangulate(pt_l, pt_r)
                
                if tracking_success:
                    # Draw tracking markers on left frame
                    cv2.circle(frame_l, pt_l, ball_l[2], (0, 255, 0), 2)
                    cv2.drawMarker(frame_l, pt_l, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                    
                    # Draw tracking markers on right frame
                    cv2.circle(frame_r, pt_r, ball_r[2], (0, 255, 0), 2)
                    cv2.drawMarker(frame_r, pt_r, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
            
            # 5. Stream coordinates to Raspberry Pi via UDP
            sender.send_target_position(x_3d, y_3d, z_3d, tracking_success, timestamp)

            # 6. Visualizations & GUI
            # Stack raw streams horizontally
            raw_stacked = np.hstack((frame_l, frame_r))

            if args.calibrate and det_cfg["method"] == "hsv":
                # In calibration mode, stack raw frames on top and binary masks on the bottom
                hsv_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2HSV)
                hsv_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2HSV)
                mask_l = cv2.inRange(hsv_l, detector.lower_hsv, detector.upper_hsv)
                mask_r = cv2.inRange(hsv_r, detector.lower_hsv, detector.upper_hsv)
                
                # Convert masks to 3-channel BGR so we can stack them with BGR frames
                mask_l_bgr = cv2.cvtColor(mask_l, cv2.COLOR_GRAY2BGR)
                mask_r_bgr = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
                masks_stacked = np.hstack((mask_l_bgr, mask_r_bgr))
                
                # Stack raw on top, masks below
                display_frame = np.vstack((raw_stacked, masks_stacked))
                hud_y = height * 2 - 30
                pos_y = height * 2 - 60
            else:
                display_frame = raw_stacked
                hud_y = height - 30
                pos_y = height - 60
            
            # Draw HUD status info
            fps_actual = 1.0 / (time.time() - t_start)
            status_text = f"FPS: {fps_actual:.1f} | Tracked: {tracking_success}"
            cv2.putText(display_frame, status_text, (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if args.calibrate:
                cv2.putText(display_frame, "CALIBRATION MODE - Press 's' to Save current values, 'q' to Quit", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if tracking_success:
                pos_text = f"3D Pos: [{x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f}] mm"
                cv2.putText(display_frame, pos_text, (20, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "BALL OUT OF SIGHT", (20, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display composite window
            cv2.imshow(window_name, display_frame)

            # Handle keystrokes
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.calibrate:
                # Save calibration values to system configuration
                save_hsv_config(args.config, config, lh, uh, ls, us, lv, uv)

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Closing tracker.")
    finally:
        # Clean up
        cam_left.close()
        cam_right.close()
        sender.close()
        cv2.destroyAllWindows()
        logger.info("Tracker shutdown complete.")

if __name__ == "__main__":
    main()
