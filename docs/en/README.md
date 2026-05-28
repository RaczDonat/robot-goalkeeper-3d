# Thesis Theoretical and Hardware Chapters Outline
## Real-Time 3D Ball Detection and Robot Goalkeeper Control

This document contains the structure and detailed outlines of the thesis chapters that you can write **right now, before discussing the programming and implementation details**. This forms the backbone of the first 15-20 pages of your thesis.

---

# Proposed Chapter Structure (Theory and Hardware Phase)

## 1. Introduction and System Concept
*This chapter introduces the project background, motivation, and objectives.*
* **1.1 Foreword and Motivation:**
  - The significance of real-time systems and mechatronics in modern industry.
  - Why the robot goalkeeper topic is exciting (combining high-speed vision, sensor fusion, and fast actuator control).
* **1.2 Topic Selection and Relevance:**
  - The importance of high-speed image processing (autonomous driving, industrial quality inspection connection).
* **1.3 Objectives:**
  - Building a physical, real-time goalkeeper testbed capable of blocking a ball along the goal line.
  - Making the system modular and low-latency.
* **1.4 History of Robotics:**
  - A brief historical overview from industrial robotic arms (e.g., Unimate) to modern collaborative robots (cobots) and autonomous systems.
* **1.5 Concept and Literature Review of Robot Goalkeepers:**
  - Existing solutions (e.g., the RoboKeeper concept, university research projects).
  - Differences between passive and active defense mechanisms.
* **1.6 Division of Labor:**
  - *Morvai Roland:* Image processing, stereo calibration, 3D reconstruction, and AI-based detection.
  - *Rácz Donát:* Trajectory estimation, microcontroller-based control, physical mechatronics, and motor driving.

## 2. Theoretical Foundations (Computer Vision and Mathematics)
*This chapter clarifies the mathematical and theoretical background of 3D reconstruction and image processing.*
* **2.1 Principles of Computer Vision:**
  - Digital imaging process (pixel grids, color spaces: RGB vs. HSV color space theory, and why HSV is preferred for color thresholding).
* **2.2 Stereo Vision Theory:**
  - How human depth perception works and how it can be replicated using two cameras.
  - Epipolar geometry theory (essential and fundamental matrices).
* **2.3 Triangulation and Depth Estimation:**
  - Mathematical definition of stereo disparity.
  - Calculating 3D coordinates from 2D pixel coordinates (similar triangles principle, baseline and focal length relations).
* **2.4 Theory of Object Detection Methods:**
  - *Traditional methods:* Thresholding, contour detection, circle fitting (Hough Transform).
  - *Modern methods:* Convolutional Neural Networks (CNN) principles, YOLO (You Only Look Once) architecture evolution and operations (Bounding boxes, confidence score, class predictions).

## 3. Hardware Architecture and Component Overview
*This chapter presents the physical components, detailing technical specifications and design choices.*
* **3.1 Computing Unit and AI Hat:**
  - **Raspberry Pi 5 (8GB RAM):** Processor architecture (Broadcom BCM2712), performance updates compared to Pi 4, GPIO interfaces.
  - **Raspberry Pi AI Hat (Hailo-8L NPU):** What is an NPU? 13 TOPS computing capacity and its role in accelerating deep learning models.
* **3.2 Industrial Cameras and Optical Accessories:**
  - **MindVision MC023CG-SY-UB:** Image sensor properties (Sony IMX392), Global Shutter theory (why it is critical to prevent rolling shutter distortion/jello effect in fast ball tracking).
  - **Lenses:** Focal length, aperture, Field of View (FOV) planning.
  - **EP-USB3HybridcableU-20 Active Optical USB 3.0 Cable:** Why active cables are required over 20 meters (signal attenuation and noise shielding).
  - **CBL-702-8P-SYNC-5M0 Sync Cable:** Hardware trigger theory in stereo vision.
* **3.3 Control Box and Power Management:**
  - Engineering calculations for sizing the 5V DC power supply (incorporating Pi 5 power demands, AI Hat load, and USB camera power draws - requiring at least 5A / 25W).
  - Control box layout, active cooling, and safety (grounding) aspects.
* **3.4 Goalkeeper Mechanics and Actuators:**
  - Description of Donát's physical mechanical rails, timing belts, and linear guides.
  - Stepper/servo motors and drive boards (e.g., TB6600 or TMC drivers).
