import xml.etree.ElementTree as ET
import os
import yaml

def generate_full_yolo_dataset(xml_path, output_dir, yaml_output_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("🔍 Scanning XML for unique annotation combinations...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- PASS 1: AUTO-DISCOVER ALL UNIQUE CLASSES ---
    unique_classes = set()

    for box in root.findall('.//box'):
        label = box.get('label').lower()
        
        # Grab all attributes and make them lowercase & snake_case
        attr_dict = {attr.get('name').lower(): attr.text.lower().replace(' ', '_') for attr in box.findall('attribute')}
        
        class_parts = [label] # Start with base class (e.g., 'player', 'ball', 'goal_post')
        
        # Build dynamic names based on your CVAT screenshots
        if label in ['player', 'goalkeeper']:
            if 'team' in attr_dict: class_parts.append(attr_dict['team'])
            if 'action' in attr_dict: class_parts.append(attr_dict['action'])
            
        elif label == 'referee':
            if 'action' in attr_dict: class_parts.append(attr_dict['action'])
            
        elif label == 'ball':
            if 'state' in attr_dict: class_parts.append(attr_dict['state'])
            if 'speed' in attr_dict: class_parts.append(attr_dict['speed'])
            
        # Combine them (e.g., "goalkeeper_team_a_diving" or "goal_post")
        final_class_name = "_".join(class_parts)
        unique_classes.add(final_class_name)

    # Sort classes alphabetically and assign YOLO IDs (0, 1, 2, 3...)
    class_list = sorted(list(unique_classes))
    class_mapping = {name: i for i, name in enumerate(class_list)}
    
    print(f"✅ Discovered {len(class_mapping)} unique classes!")
    for name, class_id in class_mapping.items():
        print(f"  {class_id}: {name}")

    # --- PASS 2: GENERATE YOLO .TXT LABELS ---
    print("\n✍️ Generating YOLO .txt files...")
    for image in root.findall('image'):
        img_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))
        
        txt_filename = os.path.splitext(os.path.basename(img_name))[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        with open(txt_filepath, 'w') as f:
            for box in image.findall('box'):
                label = box.get('label').lower()
                attr_dict = {attr.get('name').lower(): attr.text.lower().replace(' ', '_') for attr in box.findall('attribute')}
                
                class_parts = [label]
                if label in ['player', 'goalkeeper']:
                    if 'team' in attr_dict: class_parts.append(attr_dict['team'])
                    if 'action' in attr_dict: class_parts.append(attr_dict['action'])
                elif label == 'referee':
                    if 'action' in attr_dict: class_parts.append(attr_dict['action'])
                elif label == 'ball':
                    if 'state' in attr_dict: class_parts.append(attr_dict['state'])
                    if 'speed' in attr_dict: class_parts.append(attr_dict['speed'])
                    
                final_class_name = "_".join(class_parts)
                class_id = class_mapping[final_class_name]
                
                # YOLO Bounding Box Math
                xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
                xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
                x_center = ((xtl + xbr) / 2) / img_width
                y_center = ((ytl + ybr) / 2) / img_height
                box_width = (xbr - xtl) / img_width
                box_height = (ybr - ytl) / img_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    # --- PASS 3: AUTO-CREATE DATA.YAML ---
    print("\n📄 Creating dynamic data.yaml...")
    yaml_content = {
        'path': 'D:/football_analytics_pro/data/dataset',
        'train': 'images',
        'val': 'images',
        'names': {class_id: name for name, class_id in class_mapping.items()}
    }
    
    with open(yaml_output_path, 'w') as yf:
        yaml.dump(yaml_content, yf, default_flow_style=False, sort_keys=False)
        
    print(f"✅ Setup Complete! Your dataset is fully prepared for training.")

if __name__ == "__main__":
    XML_FILE = r"D:\football_analytics_pro\data\dataset\annotations.xml"
    LABELS_OUT = r"D:\football_analytics_pro\data\dataset\labels"
    YAML_OUT = r"D:\football_analytics_pro\data.yaml"
    
    generate_full_yolo_dataset(XML_FILE, LABELS_OUT, YAML_OUT)