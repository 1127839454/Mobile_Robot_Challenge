# Raspberry Pi Autonomous Car Challenge

## Project Overview
This project implements an autonomous car using a Raspberry Pi and various sensors (camera, infrared line sensors, ultrasonic distance sensor, etc.), coupled with servos and LED pixel strips. Core capabilities include background-difference object detection, gap localization and navigation, and homography-based mapping from image coordinates to real-world ground coordinates.

## Features
- **Video Capture & Streaming**  
  Leverages `picamera2` and `libcamera` for video recording and real-time streaming (`camera.py`).

- **Background-Difference Detection**  
  In `detect_300400_backgrounddiff.py`, compares live frames against a pre-captured background image to locate the target gap and compute its center.

- **Autonomous Driving Control**  
  The `Car` class (`car.py`) integrates DC motor control (`motor.py`), servo steering (`servo.py`), LED strip management (`rpi_ledpixel.py` / `spi_ledpixel.py`), infrared line following (`infrared.py`), and ultrasonic obstacle avoidance (`ultrasonic.py`).

- **Homography Calibration & Mapping**  
  In `calculate_homography/mapping_computation.py`, manually select four corner points in the camera view, compute the homography matrix, and map pixel coordinates to ground coordinates (cm) for accurate navigation.

- **Configurable Parameters**  
  `parameter.py` and `params.json` automatically adapt settings for different PCB revisions and Raspberry Pi models.

## Hardware Requirements
- Raspberry Pi (4 or 5 recommended)  
- Pi Camera Module  
- Infrared line-tracking sensors  
- HC-SR04 ultrasonic sensor  
- DC motor driver board  
- 3× servo motors  
- WS281X or SPI-driven LED pixel strip  
- Power supply and wiring

## Software Dependencies
See [requirements.txt](./requirements.txt) for full details.

## Installation & Usage

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd challenge

2.	Install Python dependencies
    pip3 install -r requirements.txt

3.	Configure parameters
    python3 parameter.py

4.	calculate homography data 
cd calculate_homography
python3 mapping_computation.py

5.	Run background-difference detection & navigation
   python3 detect_300400_backgrounddiff.py




Project Structure
challenge/
├── camera.py                       # PiCamera2 streaming & recording examples
├── car.py                          # Car class integrating sensors and drivers
├── detect_300400_backgrounddiff.py # Main background-difference & navigation script
├── infrared.py                     # Infrared line-tracking sensor module
├── motor.py                        # DC motor driver interface
├── parameter.py                    # Parameter file generator/updater
├── params.json                     # Hardware version configuration
├── rpi_ledpixel.py                 # WS281X LED strip driver
├── servo.py                        # Multi‐servo control via pigpio
├── spi_ledpixel.py                 # SPI LED strip driver
├── ultrasonic.py                   # Ultrasonic distance sensor module
├── homography_data.npz             # Precomputed homography matrix
├── calculate_homography/
│   ├── mapping_computation.py      # Homography calibration & mapping
│   └── video.h264                  # Sample test video
└── result/                         # Example outputs
    ├── challenge_view.png
    ├── dbg_*.png
    └── navigation_map.png