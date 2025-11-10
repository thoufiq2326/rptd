Real-Time Object Detection Project

This project uses Python, OpenCV, and the YOLOv8 model to perform real-time object detection. It includes scripts to run a pre-trained model and scripts to train your own custom model.

Setup

Install Python: Make sure you have Python 3.10, 3.11, or 3.12 installed.

Install Dependencies: Open your terminal in this folder and run the following commands:

pip install ultralytics
pip install opencv-python


How to Run (Using the Pre-Trained Model)

These scripts use the generic yolov8n.pt model that can detect 80 different objects (people, cars, ties, etc.).

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

How to Train Your Own Model (ML Project)

These scripts allow you to train a new model for a specific task (like detecting hard hats) and then use it.

Step 1: Train the Custom Model

This script downloads a hard hat detection dataset and trains a new model called best.pt. This is the "machine learning" part.

To Run:

python train.py


This will take a few minutes. When it's done, your new model will be saved in runs/detect/train/weights/best.pt.

Step 2: Run Your Custom Model

This script is just like detect_video.py, but it loads your new best.pt model instead of the generic one.

To Run:

python detect_video_custom.py

To Quit: Press the 'q' key on your keyboard.