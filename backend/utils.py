import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import uuid

# =========================
# Load YOLO model
# =========================
MODEL_PATH = "weights/best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model weights not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# =========================
# EasyOCR Init
# =========================
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=False if needed

# =========================
# Debug folder
# =========================
DEBUG_DIR = "debug_crops"
os.makedirs(DEBUG_DIR, exist_ok=True)


# =========================
# OCR FUNCTION (EasyOCR)
# =========================
def get_ocr_text(crop, crop_id):
    try:
        if crop is None or crop.size == 0:
            return None, 0.0

        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"{DEBUG_DIR}/{crop_id}.jpg", crop)

        results = reader.readtext(crop_rgb)

        if not results:
            return None, 0.0

        texts = []
        confidences = []

        for (bbox, text, conf) in results:
            clean_text = "".join([c for c in text if c.isalnum()])
            
            if len(clean_text) >= 2:   # allow small parts like "L8"
                texts.append(clean_text)
                confidences.append(conf)

        if not texts:
            return None, 0.0

        # 🔥 SORT by vertical position (top → bottom)
        results_sorted = sorted(results, key=lambda x: min([p[1] for p in x[0]]))

        final_text = ""
        for (_, text, _) in results_sorted:
            clean = "".join([c for c in text if c.isalnum()])
            if len(clean) >= 2:
                final_text += clean

        avg_conf = sum(confidences) / len(confidences)

        print(f"✅ OCR [{crop_id}] -> {final_text}")

        return final_text, float(avg_conf)

    except Exception as e:
        print(f"OCR Error: {e}")
        return None, 0.0

# =========================
# IMAGE PROCESSING
# =========================
def process_image(image):
    print(f"\n[STEP 1] Running YOLO detection...")
    results = model(image, imgsz=320, verbose=False)

    plates = []

    found_count = len(results[0].boxes)
    print(f"[STEP 2] YOLO found {found_count} bounding boxes.")

    for r in results:
        if r.boxes:
            for box in r.boxes.xyxy.cpu().numpy():
                crop_id = f"plate_{uuid.uuid4().hex[:6]}"

                x1, y1, x2, y2 = map(int, box)

                # 🔥 IMPORTANT FIX: bigger padding
                h, w, _ = image.shape
                pad = 15

                crop = image[
                    max(0, y1 - pad):min(h, y2 + pad),
                    max(0, x1 - pad):min(w, x2 + pad)
                ]

                print(f"[STEP 3] Processing {crop_id}...")
                text, conf = get_ocr_text(crop, crop_id)

                if text:
                    plates.append({
                        "text": text,
                        "confidence": conf,
                        "debug_id": crop_id
                    })
                else:
                    print(f"   ⚠️ OCR failed for {crop_id}")

    return plates


# =========================
# VIDEO PROCESSING
# =========================
def process_video_stream(video_path):
    print(f"\n🎥 Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    tracked_plates = {}
    final_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, imgsz=320, verbose=False)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box, tid in zip(boxes, ids):
                if tid not in tracked_plates:
                    x1, y1, x2, y2 = map(int, box)

                    crop_id = f"track_{tid}"

                    crop = frame[y1:y2, x1:x2]

                    text, conf = get_ocr_text(crop, crop_id)

                    if text and len(text) >= 5:
                        print(f"✅ Detected Plate: {text}")
                        tracked_plates[tid] = text
                        final_results.append(text)

    cap.release()
    return list(set(final_results))