import cv2
from ultralytics import YOLO

# 1. Load the "brain"
# We are using 'yolov8n.pt'. 'n' means 'nano', it's the smallest and fastest version.
model = YOLO('yolov8n.pt') 

# 2. Set up the "eyes" (your webcam)
# 0 means the default webcam. If you have multiple, you can try 1, 2, etc.
cap = cv2.VideoCapture(0)

# 3. This is a loop that will run forever (until you press 'q')
while True:
    # Read one "picture" (frame) from the webcam
    ret, frame = cap.read()
    
    # If it couldn't read a frame (e.g., webcam disconnected), break the loop
    if not ret:
        break

    # 4. Tell the "brain" to find objects in the current frame
    # This one line does all the hard work!
    results = model(frame)

    # 5. Get the frame with the boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # 6. Show the final image in a window
    cv2.imshow("Real-time Object Detection (Press 'q' to quit)", annotated_frame)

    # 7. Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up
cap.release()
cv2.destroyAllWindows()