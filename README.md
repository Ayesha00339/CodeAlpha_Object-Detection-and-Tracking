# 🎯 Task 4: Real-Time Object Detection & Tracking

AI Internship by **CodeAlpha** — Computer Vision Module

---

# 📌 Introduction

Object detection and tracking is one of the most impactful applications of computer vision. It powers self-driving cars, surveillance systems, sports analytics, robotics, and crowd monitoring. The core idea is straightforward: detect what objects are present in each frame, then persistently follow each object across frames by assigning it a unique ID.
This project implements a complete real-time detection + tracking pipeline from scratch using:
MobileNet SSD — a fast, lightweight neural network for detecting 20 types of objects
SORT (Simple Online and Realtime Tracking) — a Kalman-filter-based multi-object tracker with Hungarian algorithm assignment
The system accepts either a live webcam feed or any pre-recorded video, processes every frame, and outputs annotated video with bounding boxes, class labels, confidence scores, track IDs, and motion trails.

---

# ✨ Features

| Feature | Detail |
|---|---|
| 🔍 Real-time detection | MobileNet SSD processes each frame via OpenCV DNN backend |
| 🏷️ 20 object classes | Person, car, bus, bicycle, dog, cat, bottle, chair, and more (PASCAL VOC) |
| 🔢 Unique Track IDs | Every detected object gets a persistent ID that survives brief occlusions |
| 📐 Kalman Filtering | Smooth bounding box prediction even when detector misses a frame |
| 🔗 Hungarian Algorithm | Optimal detection-to-track assignment per frame (scipy) |
| 🌈 Colour-coded tracks | Each track ID gets its own colour — visually distinct at a glance |
| 🐾 Motion trails | 30-frame centroid trail drawn per object to visualise movement path |
| 📊 Live HUD | FPS counter, active track count, and frame number overlaid on video |
| 💾 Video export | Output saved as .mp4 (downloadable via Streamlit or CLI) |
| 🌐 Streamlit Web App | Browser-based UI — upload a video and watch it process live |
| 🧪 Demo generator | Synthetic test video with moving objects — no webcam needed |
| ⚙️ Fully configurable | Confidence, IoU, max_age, min_hits all tunable via CLI flags or UI sliders |

---

# 🤔 Why These Choices?

## Why MobileNet SSD instead of YOLO or Faster R-CNN?

| Model | Speed | Accuracy | Dependency |
|---|---|---|---|
| MobileNet SSD | ✅ Very fast (CPU-friendly) | Good for 20 classes | OpenCV only — no PyTorch/TF |
| YOLOv8 | Fast (needs GPU ideally) | Excellent | Requires ultralytics + PyTorch |
| Faster R-CNN | Slow | Highest | Heavy PyTorch dependency |

MobileNet SSD runs comfortably at 20–30 FPS on CPU with no GPU required, making it ideal for an internship demo and real deployment on modest hardware. It uses the OpenCV DNN module directly — no additional deep learning framework needed.

## Why SORT instead of Deep SORT?

| Tracker | Complexity | Accuracy | Re-ID? |
|---|---|---|---|
| SORT | ✅ Simple, fast | Good | No (IoU only) |
| Deep SORT | Moderate | Better | Yes (appearance embedding) |
| ByteTrack | Complex | Best | Partial |

SORT is the foundation of almost all modern trackers. It is elegant, well-understood, and perfectly suitable for this task. Deep SORT adds a deep appearance descriptor for re-identification after long occlusions — overkill for a demonstration and adds a heavy model dependency.

## Why OpenCV DNN?

OpenCV's cv2.dnn module can load Caffe, TensorFlow, ONNX, and Darknet models and run them on CPU with zero additional dependencies. This keeps the project lightweight, portable, and easy to run on any machine without CUDA or cloud GPUs.

---

# 🏗️ System Architecture

```
  ┌──────────────────────────────────────────────────────┐
  │                    INPUT STAGE                       │
  │   Webcam  ──┐                                        │
  │             ├──▶  cv2.VideoCapture  ──▶  BGR Frame   │
  │   Video  ───┘                                        │
  └─────────────────────────┬────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────┐
  │                  DETECTION STAGE                     │
  │                                                      │
  │   Frame ──▶ blobFromImage (300×300, scale 0.007843)  │
  │         ──▶ MobileNet SSD (OpenCV DNN / Caffe)       │
  │         ──▶ [class_id, confidence, x1, y1, x2, y2]  │
  │         ──▶ Confidence filter (threshold e.g. 0.4)   │
  └─────────────────────────┬────────────────────────────┘
                            │  N detections per frame
                            ▼
  ┌──────────────────────────────────────────────────────┐
  │                  TRACKING STAGE  (SORT)              │
  │                                                      │
  │  ① Kalman Predict   → estimate current positions     │
  │     state: [cx, cy, s, r, vx, vy, vs]               │
  │                                                      │
  │  ② IoU Cost Matrix  → pairwise(detections, tracks)  │
  │                                                      │
  │  ③ Hungarian Assign → optimal det ↔ track pairing   │
  │                                                      │
  │  ④ Kalman Update    → correct estimates with meas.   │
  │                                                      │
  │  ⑤ Lifecycle Mgmt   → create / confirm / delete     │
  └─────────────────────────┬────────────────────────────┘
                            │  M tracks [x1,y1,x2,y2, ID]
                            ▼
  ┌──────────────────────────────────────────────────────┐
  │               VISUALISATION STAGE                    │
  │                                                      │
  │  • Per-ID coloured bounding box (2px border)         │
  │  • Dark label tag: "ID:N  label  0.87"               │
  │  • 30-point centroid motion trail                    │
  │  • HUD: FPS | Active Tracks | Frame counter          │
  │  • Watermark: model & tracker name                   │
  └─────────────────────────┬────────────────────────────┘
                            │
                            ▼
              Live Window  /  MP4 Output  /  Streamlit UI
```

---

# 📦 Project Structure

```
task4_detection_tracking/
│
├── detect_track.py          ← Main pipeline (CLI entry point)
├── sort_tracker.py          ← Full SORT implementation from scratch
├── generate_demo_video.py   ← Synthetic test video generator
├── app.py                   ← Streamlit web interface
├── README.md                ← This file
│
├── models/
│   ├── deploy.prototxt              ← MobileNet SSD network definition
│   └── MobileNetSSD_deploy.caffemodel  ← Pre-trained weights (23 MB)
│
├── demo_input.mp4           ← Auto-generated synthetic test video
└── output_tracked.mp4       ← Sample output (pre-run on demo video)

```
---

# 🚀 How to Run

* Step 1 — Install dependencies
```bash
pip install opencv-python numpy scipy streamlit
```
* Step 2 — Download the model (one-time)
```bash
mkdir models

wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt \
     -O models/deploy.prototxt

wget https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel \
     -O models/MobileNetSSD_deploy.caffemodel
```
* Step 3a — Run on webcam (live)
```bash
python detect_track.py
# Press 'q' to quit
```
* Step 3b — Run on a video file
```bash
python detect_track.py --input my_video.mp4 --output result.mp4
```
* Step 3c — Headless (no display window)
```bash
python detect_track.py --input my_video.mp4 --no-display
```
* Step 3d — Launch the Streamlit web app
```bash
streamlit run app.py
# Open http://localhost:8501 in your browser
# Upload a video or click "Generate & Use Demo Video"
```

---

# ⚙️ CLI Reference

| Flag | Default | Description |
|---|---|---|
| --input | 0 (webcam) | Video file path or camera index |
| --proto | models/deploy.prototxt | Path to network definition |
| --model | models/MobileNetSSD_deploy.caffemodel | Path to weights |
| --conf | 0.4 | Minimum detection confidence (0–1) |
| --max-age | 5 | SORT: frames to keep a lost track alive |
| --min-hits | 2 | SORT: detections before track is confirmed |
| --iou | 0.3 | SORT: IoU threshold for det↔track matching |
| --output | output_tracked.mp4 | Output video save path |
| --no-display | None | Disable live window (server/headless mode) |

---

# 🔬 Algorithm Details

MobileNet SSD
Backbone: MobileNet (depthwise-separable convolutions — very fast on CPU)
Head: SSD multi-scale anchor boxes for detection
Input size: 300 × 300 pixels
Training data: PASCAL VOC 2007 + 2012
Classes: 20 (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor)
SORT — Kalman Filter
Each track maintains a 7-dimensional state vector:
```
x = [cx, cy, s, r, dx, dy, ds]ᵀ
```
where `cx, cy` = bounding box centre, `s` = area (scale), `r` = aspect ratio (constant), and `dx, dy, ds` are their respective velocities. The constant-velocity motion model predicts the next position even when the detector temporarily misses an object.
SORT — Hungarian Algorithm
Each frame, an IoU cost matrix is built between the set of new detections and predicted track positions. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) finds the globally optimal one-to-one assignment in O(n³). Pairs below the IoU threshold are rejected — the detection spawns a new track and the unmatched existing track ages toward deletion.
Track Lifecycle
```
Detection → [tentative] → (min_hits confirmed) → [active] → (max_age missed) → [deleted]
```

---

# 📊 Results (Demo Video)

| Metric | Value |
|---|---|
| Video resolution | 854 × 480 |
| Total frames processed | 250 (10 seconds @ 25 fps) |
| Avg processing speed | ~20–30 FPS on CPU |
| Unique track IDs assigned | 3–8 per run |
| Classes detectable | 20 (PASCAL VOC set) |
| Model size | 23 MB (caffemodel) |
| External GPU required | ❌ No — CPU only |

---

Developed as part of AI Internship by **CodeAlpha** — Task 4: Object Detection & Tracking  
Stack: Python · OpenCV · NumPy · SciPy · Streamlit
