import os
import glob

def rename_dataset(image_dir, label_dir):
    # Grab all images in the directory
    images = sorted(glob.glob(os.path.join(image_dir, '*.*')))
    
    for index, img_path in enumerate(images):
        # Extract the file extension and base name
        ext = os.path.splitext(img_path)[1]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Locate the corresponding JSON file
        json_path = os.path.join(label_dir, f"{base_name}.json")
        
        if os.path.exists(json_path):
            # Create a 6-digit padded sequential name
            new_base = f"{index:06d}"
            new_img_path = os.path.join(image_dir, f"{new_base}{ext}")
            new_json_path = os.path.join(label_dir, f"{new_base}.json")
            
            # Rename both files
            os.rename(img_path, new_img_path)
            os.rename(json_path, new_json_path)
            print(f"Renamed {base_name} -> {new_base}")
        else:
            print(f"Warning: JSON not found for {img_path}")

# --- Setup Paths ---
image_folder = "train_set/images"
json_folder = "train_set/labels"

# Run the function
rename_dataset(image_folder, json_folder)