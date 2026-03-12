from ultralytics import YOLO
import os

def train_custom_model():
    # 1. Load the base YOLOv8 nano model
    # We start with nano (yolov8n) because it's fast and perfect for a B.Tech project
    model = YOLO('yolov8n.pt')

    # 2. Start the training
    # data: points to your data.yaml which has our 8 classes
    # epochs: 50 iterations through your data
    # imgsz: 640 is the standard training resolution
    # device: 0 if you have a GPU (CUDA), otherwise 'cpu'
    results = model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16,
        device=0,  # <--- THIS IS THE MAGIC KEY
        name='football_custom_training'
    )
    
    print("✅ Training complete!")
    print("Your custom 'best.pt' is located in: runs/detect/football_custom_training/weights/best.pt")

if __name__ == "__main__":
    train_custom_model()