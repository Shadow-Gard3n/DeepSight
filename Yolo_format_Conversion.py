import os
import json
import cv2
import glob

def convert_json_to_yolo(image_dir, json_dir, yolo_out_dir):
    os.makedirs(yolo_out_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    for json_path in json_files:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Find the image (supporting both .jpg and .png)
        img_path = os.path.join(image_dir, f"{base_name}.jpg") 
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{base_name}.png")
            
        if not os.path.exists(img_path):
            print(f"Image not found for {base_name}, skipping.")
            continue
            
        # Read image to get its width and height
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # Open and load the JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        yolo_lines = []
        for obj in data:
            class_id = 0
            x_tl = obj['x']  # Top-left x
            y_tl = obj['y']  # Top-left y
            box_w = obj['width']
            box_h = obj['height']
            
            # Calculate Center X and Center Y
            x_center = x_tl + (box_w / 2.0)
            y_center = y_tl + (box_h / 2.0)
            
            # Normalize to values between 0.0 and 1.0
            x_norm = x_center / img_w
            y_norm = y_center / img_h
            w_norm = box_w / img_w
            h_norm = box_h / img_h
            
            # Append formatted string
            yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        # Save to a .txt file
        txt_path = os.path.join(yolo_out_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            
    print(f"Conversion complete! YOLO txt files saved in: {yolo_out_dir}")

# --- Setup Paths ---
image_folder = "train_set/images"
json_folder = "train_set/labels"
yolo_folder = "train_set/labels_yolo"

# Run the function
convert_json_to_yolo(image_folder, json_folder, yolo_folder)