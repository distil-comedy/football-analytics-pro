from ultralytics import YOLO

def main():
    # 1. Load the pre-trained YOLOv8 Nano model (Best for real-time video FPS)
    model = YOLO("yolov8n.pt")
    
    # 2. Start the GPU Training Process
    print("🚀 Igniting GTX 1080 Ti for End-to-End Model Training...")
    results = model.train(
        data=r"D:\football_analytics_pro\data.yaml",
        epochs=100,         # 100 epochs is a solid baseline. YOLO auto-stops if it peaks early.
        imgsz=640,          # HD resolution scaling
        batch=16,           # Optimal batch size for an 11GB GTX 1080 Ti
        device=0,           # 0 targets your dedicated NVIDIA GPU
        project="runs/detect", 
        name="pro_football_model",
        cache=True          # Speeds up training by loading images into RAM
    )
    print("✅ Training Complete! Your new brain is ready.")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    main()