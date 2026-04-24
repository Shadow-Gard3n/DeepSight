import os
import cv2
import glob

def test_yolo_labels(image_dir, yolo_label_dir, test_out_dir):
    os.makedirs(test_out_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(yolo_label_dir, '*.txt'))
    
    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        
        # Find the image
        img_path = os.path.join(image_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{base_name}.png")
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_norm, y_norm, w_norm, h_norm = map(float, parts[1:])
            
            # Convert YOLO normalized coordinates back to absolute pixel values
            box_w = int(w_norm * img_w)
            box_h = int(h_norm * img_h)
            x_center = int(x_norm * img_w)
            y_center = int(y_norm * img_h)
            
            # Find the top-left corner
            x_tl = int(x_center - box_w / 2)
            y_tl = int(y_center - box_h / 2)
            
            # Draw the bounding box and label
            cv2.rectangle(img, (x_tl, y_tl), (x_tl + box_w, y_tl + box_h), (0, 255, 0), 2)
            cv2.putText(img, f"ID:{class_id}", (x_tl, y_tl - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Save the drawn image to the test folder
        out_path = os.path.join(test_out_dir, f"{base_name}_test.jpg")
        cv2.imwrite(out_path, img)
        
    print(f"Testing complete! Open the '{test_out_dir}' folder to verify the boxes.")

# --- Setup Paths ---
image_folder = "train_set/images"
yolo_folder = "train_set/labels_yolo"
test_folder = "train_set/test_visualizations"

# Run the function
test_yolo_labels(image_folder, yolo_folder, test_folder)