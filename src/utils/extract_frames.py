import cv2
import os

def extract_all_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    print("Starting frame extraction...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # CVAT usually names frames as 000000.jpg, 000001.jpg etc.
        frame_name = f"frame_{count:06d}.jpg" 
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        count += 1
        if count % 100 == 0:
            print(f"Extracted {count} frames...")
            
    cap.release()
    print(f"Done! Extracted {count} frames to {output_dir}")

if __name__ == "__main__":
    # Update these paths to match your folders
    extract_all_frames("data/raw_video/sample_match.mp4", "data/dataset/images")