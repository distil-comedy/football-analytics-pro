import os
import cv2
import easyocr
from ultralytics import YOLO

# 🚨 Configuration for OCR storage
MODEL_STORAGE_PATH = r"D:\football_analytics_pro\ocr_models"
os.environ["EASYOCR_MODULE_PATH"] = MODEL_STORAGE_PATH

def test_model():
    model_path = r"D:\football_analytics_pro\runs\detect\runs\detect\pro_football_model2\weights\best.pt"
    model = YOLO(model_path)
    
    reader = easyocr.Reader(['en'], gpu=False)
    
    video_path = r"D:\football_analytics_pro\data\raw_video\sample_match.mp4" 
    
    # --- NEW: Setup OpenCV Video Capture & Writer ---
    cap = cv2.VideoCapture(video_path)
    # Get video details for the writer (width, height, frames per second)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    output_path = r"D:\football_analytics_pro\final_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # ------------------------------------------------
    
    print(f"🎥 Running Elite Custom Inference on: {video_path}")
    print(f"💾 Saving final video to: {output_path}")
    
    results = model.predict(
        source=video_path,
        show=False,       
        save=False,       
        conf=0.4,        
        line_width=2,
        stream=True      
    )
    
    for r in results:
        annotated_frame = r.plot()
        orig_img = r.orig_img
        
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if "player" in class_name or "goalkeeper" in class_name:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                player_crop = orig_img[max(0, y1):y2, max(0, x1):x2]
                
                # --- NEW: Image Enhancement for Better OCR ---
                # 1. Grayscale
                gray_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
                # 2. Resize to make it 2x bigger (helps OCR see small pixels)
                zoomed_crop = cv2.resize(gray_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                # -------------------------------------------
                
                # Pass the ENHANCED image to OCR
                ocr_res = reader.readtext(zoomed_crop, allowlist='0123456789')
                
                if ocr_res:
                    num_text = f"#{ocr_res[0][1]}"
                    
                    font = cv2.FONT_HERSHEY_PLAIN
                    font_scale = 0.65  
                    thickness = 1
                    
                    (tw, th), baseline = cv2.getTextSize(num_text, font, font_scale, thickness)
                    cv2.rectangle(annotated_frame, (x1, y2 - th - 4), (x1 + tw + 4, y2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, num_text, (x1 + 2, y2 - 2), 
                                font, font_scale, (255, 255, 255), thickness)
        
        # --- NEW: Save the frame to the video file ---
        out.write(annotated_frame)
        
        cv2.imshow("Custom Player Analytics", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- NEW: Release the writer to finalize the file ---
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Inference Complete! Video saved at: {output_path}")

if __name__ == '__main__':
    test_model()