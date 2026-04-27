# 🚗 DeepSight ANPR (Automatic Number Plate Recognition)

Welcome to **DeepSight**, an efficient and lightweight Automatic Number Plate Recognition (ANPR) system. DeepSight leverages a fine-tuned YOLOv11 Nano model to achieve real-time license plate detection on both images and video streams. 

The project features a decoupled architecture with a FastAPI backend hosted on Hugging Face Spaces and a responsive frontend deployed on Vercel.

---

## 🔗 Live Links

- **Frontend (UI):** [https://deep-sight-ashy.vercel.app/](https://deep-sight-ashy.vercel.app/)
- **Backend API:** [https://shadowgard3n-deepsight.hf.space/docs](https://shadowgard3n-deepsight.hf.space/docs)

---

## 📂 Repository Structure

The project is logically divided into training scripts, a backend API, and a frontend interface:

```text
DeepSight/
├── DeepSight.ipynb               # Jupyter notebook containing YOLOv11 training pipeline and evaluation
├── Testing_Bounding_Boxes.py     # Script to visualize and verify dataset bounding boxes
├── Yolo_format_Conversion.py     # Data processing script for YOLO format conversion
├── data.yaml                     # YOLO configuration file for dataset paths and classes
├── pretrain_good.py              # Script for preliminary training/testing
├── rename.py                     # Utility script for standardizing dataset filenames
├── backend/                      # FastAPI Backend Services
│   ├── main.py                   # Application entry point & API routes (/predict-image, /predict-video)
│   ├── utils.py                  # Core inference logic and media stream processing
│   ├── Dockerfile                # Containerization config for deployment (Hugging Face)
│   ├── requirements.txt          # Python dependencies for the backend
│   ├── weights/
│   │   └── best.pt               # Fine-tuned YOLOv11 Nano model weights
│   └── debug_crops/              # Sample cropped plate images for debugging
└── frontend/                     # Web User Interface
    └── index.html                # Responsive HTML/JS/CSS frontend
```

## ⚙️ Pipeline Flow

The DeepSight pipeline spans from data preprocessing to real-time web inference:

### Data Preprocessing
* Raw images are standardized using `rename.py`.
* Annotations are processed and structured into the required format using `Yolo_format_Conversion.py`.
* `Testing_Bounding_Boxes.py` is used to validate the accuracy of ground-truth labels before training.

### Model Training (via Colab)
* The model is trained using `DeepSight.ipynb`.
* We utilize the ultra-fast YOLOv11 Nano (`yolo11n.pt`) architecture to ensure low computational overhead (only 6.3 GFLOPs), making it ideal for real-time video processing.
* Robust data augmentations (Mosaic, Mixup, HSV adjustments, scaling, and Albumentations like Blur and CLAHE) are applied to improve the model's resilience to different lighting and weather conditions.

### Backend Inference (FastAPI)
* The backend (`main.py`) exposes two primary POST endpoints: `/predict-image` and `/predict-video`.
* When a payload is received, `utils.py` processes the media. For videos, it extracts frames, runs the YOLO model to detect bounding boxes, and tracks the plates.
* The best detections are returned as a JSON response.

### Frontend Client
* A lightweight vanilla HTML/JS client (`index.html`) provides a dark-themed, modern interface.
* Users can upload images or videos, which are sent via HTTP requests to the Hugging Face backend.
* The UI parses the API response and displays the detected text directly to the user.

---

## 🧠 Model Details & Accuracies

The core of DeepSight is a custom-trained **YOLOv11 Nano** model, specifically optimized for single-class object detection (`license_plate`).

### Training Configuration
* **Base Model:** YOLOv11n (Nano)
* **Epochs:** 20
* **Image Size:** 416x416
* **Batch Size:** 16
* **Hardware:** NVIDIA Tesla T4 GPU
* **Parameters:** 2,582,347

### Final Evaluation Metrics (Validation Set)
After 20 epochs, the model achieved exceptional accuracy, indicating highly reliable license plate detection:
* **mAP@50 (Mean Average Precision):** 0.983 (98.3%)
* **mAP@50-95:** 0.699 (69.9%)


### Inference Speed
* **YOLO:** ~70 ms per image
* **EasyOCR:** ~150 ms per image

---

## 🚀 API Endpoints

### 1. `POST /predict-image`
* **Description:** Analyzes a single image and extracts license plate text.
* **Payload:** `multipart/form-data` containing an `image/*` file.
* **Response:** JSON containing a status and an array of detected plates.

### 2. `POST /predict-video`
* **Description:** Analyzes a video file, tracking and extracting plates frame-by-frame.
* **Payload:** `multipart/form-data` containing a `video/*` file.
* **Response:** JSON containing unique detected plates across the video timeline.

---

## 🛠️ Local Development

To run the backend locally:

1. Navigate to the `backend/` directory.
2. Install dependencies:
```pip install -r requirements.txt```
3. Start the FastAPI server:
```uvicorn main:app --host 0.0.0.0 --port 7860 --reload```
4. Open the frontend/index.html file in your browser to interact with the local API (ensure API_URL in the frontend is pointed to http://localhost:7860 for local testing).