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
        "camera": {"resolution": {"width": 1280, "height": 720}, "fps": 60},
        "stereo": {"baseline_mm": 300.0, "focal_length_px": 1200.0, "cx": 640.0, "cy": 360.0},
        "detection": {"method": "hsv", "confidence_threshold": 0.5, "hsv_bounds": {"lower_h": 5, "lower_s": 100, "lower_v": 100, "upper_h": 25, "upper_s": 255, "upper_v": 255}}
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time 3D Ball Tracker (PC Side)")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Path to config file")
    parser.add_argument("--use-real-cameras", action="store_true", help="Set to use real MindVision cameras instead of mocks")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    net_cfg = config["network"]
    cam_cfg = config["camera"]
    stereo_cfg = config["stereo"]
    det_cfg = config["detection"]

    # Initialize UDP Sender
    # If testing locally on the same PC, rpi_ip can be "127.0.0.1" (localhost)
    sender = UDPSender(ip=net_cfg["rpi_ip"], port=net_cfg["port"])

    # Initialize Cameras (Left and Right)
    width = cam_cfg["resolution"]["width"]
    height = cam_cfg["resolution"]["height"]
    fps = cam_cfg["fps"]

    if args.use_real_cameras:
        logger.info("Initializing physical MindVision cameras...")
        cam_left = MindVisionCamera(camera_index=stereo_cfg["left_camera_index"], width=width, height=height)
        cam_right = MindVisionCamera(camera_index=stereo_cfg["right_camera_index"], width=width, height=height)
    else:
        logger.info("Initializing mock cameras with simulated 3D projection...")
        cam_left = MockCamera(width=width, height=height, fps=fps, is_left=True)
        cam_right = MockCamera(width=width, height=height, fps=fps, is_left=False)

    # Initialize Ball Detectors
    detector_left = BallDetector(
        method=det_cfg["method"], 
        hsv_bounds=det_cfg.get("hsv_bounds"), 
        confidence_threshold=det_cfg["confidence_threshold"]
    )
    detector_right = BallDetector(
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

    # Create visualization window
    window_name = "Real-Time 3D Ball Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            t_start = time.time()

            # 1. Capture frames
            ret_l, frame_l = cam_left.read()
            ret_r, frame_r = cam_right.read()

            if not ret_l or not ret_r or frame_l is None or frame_r is None:
                logger.warning("Empty frame received. Skipping frame.")
                continue

            # Capture timestamp
            timestamp = cam_left.get_timestamp()

            # 2. Run 2D Ball Detection
            success_l, ball_l = detector_left.detect(frame_l)
            success_r, ball_r = detector_right.detect(frame_r)

            x_3d, y_3d, z_3d = 0.0, 0.0, 0.0
            tracking_success = False

            # 3. If ball detected in both images, calculate 3D position
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
            
            # 4. Stream coordinates to Raspberry Pi via UDP
            sender.send_target_position(x_3d, y_3d, z_3d, tracking_success, timestamp)

            # 5. Visualizations & GUI
            # Stack images horizontally for side-by-side view
            combined_view = np.hstack((frame_l, frame_r))
            
            # Print tracking status overlay
            fps_actual = 1.0 / (time.time() - t_start)
            status_text = f"FPS: {fps_actual:.1f} | Tracked: {tracking_success}"
            cv2.putText(combined_view, status_text, (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if tracking_success:
                pos_text = f"3D Pos (X, Y, Z): [{x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f}] mm"
                cv2.putText(combined_view, pos_text, (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(combined_view, "BALL OUT OF SIGHT", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display window
            cv2.imshow(window_name, combined_view)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
