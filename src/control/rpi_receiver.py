import time
import sys
import os
import logging
from typing import Dict, Any

# Adjust path to import common networking module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.network import UDPReceiver

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ball_data(packet: Dict[str, Any]) -> None:
    """
    Callback function that processes incoming ball data packets from the laptop.
    
    :param packet: Data dictionary containing:
                   - "x": float (X coordinate in mm)
                   - "y": float (Y coordinate in mm)
                   - "z": float (Z coordinate in mm)
                   - "det": int (1 if ball detected, 0 if tracking lost)
                   - "t": float (capture timestamp on PC)
    """
    x = packet.get("x", 0.0)
    y = packet.get("y", 0.0)
    z = packet.get("z", 0.0)
    detected = bool(packet.get("det", 0))
    pc_timestamp = packet.get("t", 0.0)
    
    # Calculate latency (current time - capture time on PC)
    # Note: Requires system time synchronization between PC and Pi (e.g. NTP or local network sync)
    current_time = time.time()
    latency_ms = (current_time - pc_timestamp) * 1000.0

    if detected:
        print(f"[TRACKING ACTIVE] X: {x:8.2f} mm | Y: {y:8.2f} mm | Z: {z:8.2f} mm | Latency: {latency_ms:6.2f} ms", end='\r')
    else:
        print("[TRACKING INACTIVE] Ball out of sight                                                      ", end='\r')
    
    # Flush stdout to ensure real-time print updates in the console
    sys.stdout.flush()

def main() -> None:
    # Default IP: Listen on all interfaces ("0.0.0.0") on port 5005
    # This allows it to receive packets from any laptop on the same network
    ip = "0.0.0.0"
    port = 5005

    # Try loading config if running in the workspace directory
    try:
        import yaml
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/system_config.yaml'))
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                port = config["network"]["port"]
                logger.info(f"Loaded receiver port {port} from configuration file.")
    except Exception as e:
        logger.warning(f"Could not load config file ({e}). Using default port {port}.")

    logger.info("Initializing Raspberry Pi 5 Goalkeeper Receiver...")
    receiver = UDPReceiver(ip=ip, port=port)

    logger.info("Starting UDP listener loop. Press Ctrl+C to terminate...")
    
    try:
        # Start listening in a background thread and handle callback
        receiver.start_listening(process_ball_data)
        
        # Keep main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("\nTermination signal received.")
    finally:
        logger.info("Stopping UDP listener and cleaning up...")
        receiver.stop_listening()
        logger.info("Receiver shut down successfully.")

if __name__ == "__main__":
    main()
