Real-Time Object Detection Project (YOLOv8)

This is a simple Python project for real-time object detection using YOLOv8 and OpenCV.

It is broken into two parts:

Part 1: Run a generic, pre-trained model to detect 80 different objects.

Part 2: Train your own custom model on a small dataset and then use it.

How to Run

Create a Virtual Environment (a clean "project box"):

python -m venv venv


Activate the Environment:

You may first need to allow scripts to run (do this once per terminal):

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


Then, activate it:

.\venv\Scripts\Activate.ps1


Your terminal prompt should now start with (venv).

Install Dependencies:

With your (venv) active, install the required libraries:

pip install ultralytics opencv-contrib-python


Run a Script:

Follow the steps for Part 1 or Part 2 below.

Part 1: Run the Generic Pre-Trained Model

This script uses the official yolov8n.pt model, which can detect 80 classes (person, car, dog, etc.).

File: detect_video.py

To Run:

Edit detect_video.py:

Open the file and make sure Line 7 points to the video you want to use (e.g., traffic.mp4 or pedestrians.mp4).

# Line 7:
cap = cv2.VideoCapture('traffic.mp4') 


Run the script:

python detect_video.py


A window will pop up showing the video with all detected objects.

To quit, press the 'q' key on your keyboard.

Part 2: Train and Run Your Own Custom Model

This part shows you the full Machine Learning workflow. You will train a new "brain" on a tiny, built-in dataset called coco8 (which has 8 images).

Step 1: Train the Model

File: train_coco8.py

What it does: This script loads the generic yolov8n.pt model and re-trains it on the coco8 dataset for 50 epochs (passes). This teaches you the training process.

To Run:

python train_coco8.py


This will take a minute or two. When it's finished, your new, custom-trained "brain" will be saved at:
runs/detect/coco8_run/weights/best.pt

Step 2: Run Your Custom Model

File: detect_coco8.py

What it does: This script is already set up to load the best.pt model you just trained in Step 1.

To Run:

python detect_coco8.py


A window will pop up showing the video, but this time it's using your custom-trained AI to find the objects!
