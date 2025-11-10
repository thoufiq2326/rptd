from ultralytics import YOLO
import platform
import torch

print(f"--- System Info ---")
print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("-------------------")

try:
    # 1. Load a pre-trained model
    print("Loading pre-trained model 'yolov8n.pt'...")
    model = YOLO('yolov8n.pt') 
    print("Model loaded successfully.")

    # 2. Define the dataset
    # This is a built-in dataset from Ultralytics. No download needed!
    dataset_yaml = 'coco8.yaml'
    print(f"Using built-in dataset: {dataset_yaml}")

    print("\nStarting training... This will be very fast.")

    # 3. Train the model
    # We train for 50 epochs (passes) since the dataset is tiny.
    # We force it to use the CPU to avoid errors.
    # We give it a custom name so we can find the results easily.
    results = model.train(
        data=dataset_yaml, 
        epochs=50, 
        imgsz=640, 
        device='cpu', 
        name='coco8_run'  # <-- This saves results to 'runs/detect/coco8_run'
    )

    print("\n--- Training Finished! ---")
    print("Your new model is saved in 'runs/detect/coco8_run/weights/best.pt'")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED! ---")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    print("\n--- End of Error ---")