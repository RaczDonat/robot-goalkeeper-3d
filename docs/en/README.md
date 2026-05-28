# Thesis Documentation Outline & Guidelines
## Real-Time 3D Ball Detection and Robot Goalkeeper Control

This document contains the structure and chapter outlines of the Computer Engineering BSc thesis project at the University of Debrecen, Faculty of Informatics, in English. This outline is designed to serve as the foundation for the minimum 40-page Word document.

---

### Academic Formatting Guidelines (UD FI)
* **Font:** Times New Roman 12pt (body text)
* **Line Spacing:** 1.5 lines, justified alignment
* **Margins:** Left: 3.5 cm (for binding), Right: 2.5 cm, Top/Bottom: 2.5 cm
* **Heading Numbering:** Decimal system (e.g., 1., 1.1., 1.1.1.)
* **References:** IEEE or Harvard style, cited in text (e.g., `[1]`)
* **Figures & Tables:** Every figure and table must have a unique number and caption, and must be referenced in the body text.

---

# Proposed Thesis Chapter Structure

## 1. Introduction
* **Objective:** Introduce the project topic, relevance, and the significance of robot goalkeeper projects in industry and education (e.g., mechatronics, computer vision, real-time systems).
* **Project Goal:** Build a physical testbed capable of intercepting a rolled/thrown ball along a goal line.
* **Division of Labor (Morvai Roland & Rácz Donát):**
  * *Morvai Roland:* Image processing, camera handling (MindVision SDK), stereo calibration, 3D reconstruction, and AI-based detection using the Hailo-8L NPU.
  * *Rácz Donát:* Trajectory estimation, microcontroller-based control, physical mechatronics design, motor driving, and serial communication.

## 2. Literature Review and Theoretical Background
* **Computer Vision:** From 2D imaging to 3D reconstruction. Working principles of stereo camera systems (Epipolar geometry, triangulation).
* **Object Detection:** Traditional color thresholding (OpenCV HSV-based segmentation) vs modern deep learning methods (YOLO architectures).
* **Real-Time Systems:** Latency sources (camera exposure, USB transfer time, software demosaicing, processing time, communication jitter).
* **Control Theory:** Trajectory estimation (physical filters like Kalman filter or ballistics models) and control of actuators (PID control, stepper/servo motors).

## 3. System Specification and Hardware Architecture
* **Control Box and Power Supply:**
  * Sizing the 5V DC industrial power supply (requiring at least 5A to drive the Raspberry Pi 5 and peripherals under heavy load).
  * Safety and professional wiring standards (grounding, noise suppression, active cooling).
* **Computing Unit:** Raspberry Pi 5 (8GB RAM) + AI Hat (Hailo-8L, 13 TOPS).
* **Camera System:** 
  * 2x MindVision MC023CG-SY-UB cameras (2.3 Megapixel, Global Shutter, USB3.0).
  * Importance of Global Shutter: eliminating image distortion (jello effect) caused by rolling shutter during fast ball movement.
  * Lens focal length selection to optimize the Field of View (FOV).
  * Data transmission: 2x EP-USB3HybridcableU-20 active optical cables (lossless high-speed transmission up to 20 meters).
  * Synchronization: 2x CBL-702-8P-SYNC-5M0 cables for hardware triggering of both cameras (critical in stereo vision to ensure frames are captured at the exact same millisecond).

## 4. Software Architecture and Optimization
* **Analysis of the Raspberry Pi 5 Performance Bottleneck:**
  * Why did the test code slow down? (Slow generic V4L2 backend in OpenCV, software demosaicing/Bayer conversion on Pi's CPU, single-threaded I/O and processing).
* **Optimization Solutions:**
  * *MindVision SDK Integration:* Utilizing Direct Memory Access (DMA) and hardware-level adjustments (exposure, gain, pixel format).
  * *Multithreading/Multiprocessing:* Dedicated threads/processes for receiving camera frames (Frame Reader Threads) and a separate thread for processing and visualization. Double buffering technique.
  * *NPU Acceleration:* Exporting YOLOv8-nano model to Hailo HEF format. Utilizing the AI Hat (Hailo-8L) for ball detection, reducing CPU usage to a minimum.
  * *Resolution and ROI (Region of Interest) optimization:* Reading and processing only the relevant playing field area to reduce pixel data size.

## 5. 3D Ball Detection and Stereo Vision
* **Camera Calibration:** Chessboard-pattern calibration, determination of intrinsic and extrinsic camera parameters.
* **Rectification:** Removing lens distortion and aligning stereo frame pairs.
* **2D Detection:** Segmenting the ball using the Hailo NPU or optimized color-based segmentation.
* **Triangulation:** Calculating 3D world coordinates (X, Y, Z) from 2D pixel coordinates (x1, y1) and (x2, y2) based on the camera baseline distance.

## 6. Trajectory Estimation and Robot Control
* **Trajectory Modeling:** Accounting for gravity and air resistance. Equations of motion for the ball in 3D space.
* **Goal Line Intersection Prediction:** Estimating where (X, Y) and when the ball will cross the goal line based on Z-axis movement.
* **Communication:** Serial communication protocol (UART / USB CDC) between the Raspberry Pi 5 and the robot control board (e.g., STM32, Arduino, or ESP32).
* **Actuation Unit:** Driving stepper/servo motors, designing acceleration and deceleration profiles (S-curve), positioning with minimized overshoot.

## 7. Experimental Results and Evaluation
* **FPS and Latency Analysis:** Measuring frame rate and latency across different resolutions and optimization levels.
* **Detection Accuracy:** Tracking error rates under various ball speeds.
* **Goalkeeper Efficiency:** Success rate of saving incoming shots.

## 8. Summary and Future Work
* Summary of achievements.
* Future improvements (e.g., predicting curved trajectories, smarter defense strategies).

---

# Current Development Focus
We are currently in **Phase 1**:
1. **Interfacing the cameras under Linux using the MindVision SDK.**
2. **Developing an optimized multithreaded I/O pipeline in Python / C++.**
3. **Running benchmark tests on the Pi, documenting FPS and CPU usage.**
