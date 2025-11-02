Real-Time Object Detection Project

This project uses Python, OpenCV, and the YOLOv8 model to perform real-time object detection from a webcam or a video file.

Setup

Install Python: Make sure you have Python 3.10, 3.11, or 3.12 installed.

Install Dependencies: Open your terminal in this folder and run the following commands:

pip install ultralytics
pip install opencv-python


How to Run

You have two scripts to choose from:

1. Webcam Detection

This script will turn on your computer's webcam and start detecting objects.

To Run:

python detect.py


To Quit: Press the 'q' key on your keyboard while the video window is active.

2. Video File Detection

This script will read a video file (like traffic.mp4), detect objects, and show you the result.

To Run:

python detect_video.py


To Quit: Press the 'q' key on your keyboard.

(Note: This script looks for a file named traffic.mp4. You can change the filename in detect_video.py to use a different video.)