import cv2
from ultralytics import YOLO

# 1. Load your CUSTOM-TRAINED "brain"
# This path points to the model you will create with 'train_coco8.py'
model_path = 'runs/detect/coco8_run/weights/best.pt'
print(f"Loading custom-trained model from: {model_path}")
model = YOLO(model_path) 
print("Model loaded successfully.")

# 2. Set up the "eyes" to read from "traffic.mp4"
cap = cv2.VideoCapture('pedestrian.mp4') 

# 3. This is a loop that will run until the video ends
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Video finished or file error. Exiting.")
        break

    # 4. Tell your new "brain" to find objects
    results = model(frame)

    # 5. Get the frame with the boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # 6. Show the final image in a window
    cv2.imshow("My Custom-Trained Model (Press 'q' to quit)", annotated_frame)

    # 7. Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up
cap.release()
cv2.destroyAllWindows()