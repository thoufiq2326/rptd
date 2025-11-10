import cv2
from ultralytics import YOLO  # <-- This line is now correct

# 1. Load the "brain"
model = YOLO('yolov8n.pt') 

# 2. Set up the "eyes" to read from "traffic.mp4"
# Make sure you downloaded "traffic.mp4" into this folder
cap = cv2.VideoCapture('pedestrians.mp4') 


# 3. This is a loop that will run until the video ends
while True:
    # Read one "picture" (frame) from the video
    ret, frame = cap.read()
    
    # If 'ret' is false, it means the video is over or the file is bad
    if not ret:
        print("Video finished or file error. Exiting.")
        break

    # 4. Tell the "brain" to find objects in the current frame
    results = model(frame)

    # 5. Get the frame with the boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # 6. Show the final image in a window
    cv2.imshow("Video Object Detection (Press 'q' to quit)", annotated_frame)

    # 7. Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up
cap.release()
cv2.destroyAllWindows() 