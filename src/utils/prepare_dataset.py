import os
import shutil

# 1. Define your folder paths
label_dir = "data/dataset/labels"
all_images_dir = "data/dataset/images"
final_images_dir = "data/dataset/train_images"

print("--- 🔍 Starting Dataset Preparation ---")

# 2. Check if the 'images' folder actually exists where it should be
if not os.path.exists(all_images_dir):
    print(f"❌ Error: The folder '{all_images_dir}' was not found.")
    print("Please make sure you moved your extracted images into this folder.")
    exit()

# 3. Create the new clean folder for matched images
if not os.path.exists(final_images_dir):
    os.makedirs(final_images_dir)
    print(f"📁 Created folder: {final_images_dir}")

# 4. Find all the .txt files (your manual CVAT annotations)
if not os.path.exists(label_dir):
    print(f"❌ Error: The folder '{label_dir}' was not found.")
    exit()

annotated_frames = [f.replace('.txt', '') for f in os.listdir(label_dir) if f.endswith('.txt')]
print(f"✅ Found {len(annotated_frames)} label (.txt) files. Starting match...\n")

if len(annotated_frames) == 0:
    print("❌ Error: No .txt files found in the labels folder!")
    exit()

# 5. Match and copy the images
count = 0
missing_count = 0

for frame in annotated_frames:
    # Most extracted frames are .jpg. If yours are .png, change the extension below.
    img_name = f"{frame}.jpg"
    src_path = os.path.join(all_images_dir, img_name)
    dst_path = os.path.join(final_images_dir, img_name)
    
    # If the image exists in your big folder, copy it to the clean folder
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        count += 1
    else:
        missing_count += 1
        # Print warning for the first few missing images to avoid spamming your terminal
        if missing_count <= 5:
            print(f"⚠️ Warning: Could not find matching image '{img_name}' in '{all_images_dir}'")

if missing_count > 5:
    print(f"⚠️ ... and {missing_count - 5} more missing images.")

print(f"\n🎉 DONE! Successfully matched and copied {count} images to {final_images_dir}")