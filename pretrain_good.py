import os
import json
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
images_dir = "images"
output_dir = "output_full_vehicles"
labels_dir = os.path.join(output_dir, "labels")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

model = YOLO("yolov8n.pt")  # or "yolov8s.pt" for better accuracy

VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

# ----------------------------------------

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    image = cv2.imread(img_path)

    results = model.predict(
        source=img_path,
        conf=0.5,       # 🔥 reduce noise
        imgsz=640,
        device="cpu"
    )

    boxes_data = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])

            # ✅ keep only vehicles
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            # 🔥 remove tiny false detections
            if width < 50 or height < 50:
                continue

            conf = float(box.conf[0])

            boxes_data.append({
                "class_id": cls,
                "confidence": conf,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

            # Draw box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Save image only if vehicles detected
    if boxes_data:
        cv2.imwrite(os.path.join(output_dir, img_file), image)

    # Save JSON (even if empty → useful for analysis)
    with open(os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".json"), "w") as f:
        json.dump(boxes_data, f, indent=4)

print("✅ Full dataset processed (vehicle-only detection)")