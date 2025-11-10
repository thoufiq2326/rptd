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
    # This is the NEW, LOCAL path.
    # It assumes you unzipped the folder into your project.
    # Make sure the folder name "Personal-Protective-Equipment-Combined-Model-8" is correct!
    dataset_yaml = 'Personal-Protective-Equipment-Combined-Model-8/data.yaml'
    print(f"Using local dataset: {dataset_yaml}")

    print("\nStarting training... This might take a few minutes.")

    # 3. Train the model
    # We add device='cpu' to force it to use your CPU,
    # which avoids any of the graphics card (DLL) errors we saw before.
    results = model.train(data=dataset_yaml, epochs=10, imgsz=640, device='cpu')

    print("\n--- Training Finished! ---")
    print("Your new model is saved in the 'runs/detect/train/weights/' folder.")
    print("The best model is called 'best.pt'.")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED! ---")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    print("\n--- End of Error ---")