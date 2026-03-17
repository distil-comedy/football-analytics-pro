import xml.etree.ElementTree as ET
import cv2
import os
import random

# --- CONFIGURATION (UPDATED PATHS) ---
XML_FILE = r"D:\football_analytics_pro\data\dataset\annotations.xml"          

# Assuming CVAT put your images in an 'images' subfolder. 
# If they are sitting directly next to the XML, change this to just r"D:\football_analytics_pro\data\dataset"
IMAGES_DIR = r"D:\football_analytics_pro\data\dataset\images"                 

OUTPUT_DIR = r"D:\football_analytics_pro\research_screenshots_cvat"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors for specific classes (BGR format for OpenCV)
COLORS = {
    "PLAYER": (0, 0, 255),      # Red
    "REFEREE": (0, 255, 0),     # Green
    "GOALKEEPER": (255, 255, 0),# Cyan
    "BALL": (255, 0, 255)       # Magenta
}

def draw_real_annotations(num_samples=10):
    print(f"Parsing CVAT XML: {XML_FILE}...")
    
    if not os.path.exists(XML_FILE):
        print(f"❌ ERROR: Cannot find XML file at {XML_FILE}")
        return
        
    tree = ET.parse(XML_FILE)
    root = tree.getroot()
    
    # Get all images from the XML
    images = root.findall('image')
    
    # Filter to only images that actually exist in your folder
    valid_images = [img for img in images if os.path.exists(os.path.join(IMAGES_DIR, img.get('name')))]
    
    if not valid_images:
        print(f"❌ ERROR: Found the XML, but couldn't find any matching images in {IMAGES_DIR}. Check your image path!")
        return
        
    # Pick random samples
    selected_images = random.sample(valid_images, min(num_samples, len(valid_images)))
    
    for img_elem in selected_images:
        img_name = img_elem.get('name')
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"Processing: {img_name}...")
        
        # Loop through every bounding box in this image
        for box in img_elem.findall('box'):
            label = box.get('label').upper()
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            
            color = COLORS.get(label, (255, 165, 0)) # Default Orange
            
            # 1. Draw the Bounding Box
            cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color, 2)
            
            # 2. Gather Real Attributes from XML
            attributes = [label] # First line is the class name
            for attr in box.findall('attribute'):
                attr_name = attr.get('name')
                attr_value = attr.text
                if attr_value: # Only add if it's not empty
                    attributes.append(f"{attr_name}: {attr_value}")
            
            # 3. Draw Clean Multi-line Text Background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_height = 18
            
            # Calculate background box size based on longest text
            if attributes:
                max_text_width = max([cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text in attributes])
                bg_height = len(attributes) * line_height + 5
                
                # Draw semi-transparent background so it's readable
                overlay = img.copy()
                cv2.rectangle(overlay, (xtl, ytl - bg_height), (xtl + max_text_width + 10, ytl), color, -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                
                # 4. Write the Real Text
                y_text = ytl - bg_height + 15
                for text in attributes:
                    # Class name in white, attributes in a slight off-white for contrast
                    text_color = (255, 255, 255) if text == label else (220, 220, 220)
                    cv2.putText(img, text, (xtl + 5, y_text), font, font_scale, text_color, thickness)
                    y_text += line_height

        # Save the final image
        save_path = os.path.join(OUTPUT_DIR, f"real_cvat_{img_name}")
        cv2.imwrite(save_path, img)
        print(f"✅ Saved: {save_path}")

    print(f"\n🎉 Done! Check the '{OUTPUT_DIR}' folder for your research images.")

# Run the script
draw_real_annotations(10)