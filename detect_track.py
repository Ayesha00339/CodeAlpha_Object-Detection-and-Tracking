"""
Object Detection & Tracking  —  Task 4
=======================================
• Detector  : MobileNet SSD via OpenCV DNN
• Tracker   : SORT (Kalman Filter + Hungarian Algorithm)
• Input     : webcam (default) or any video file
• Output    : annotated video saved + optional live display

Usage
-----
  # webcam live
  python detect_track.py

  # process a video file
  python detect_track.py --input my_video.mp4

  # save output without display window (headless)
  python detect_track.py --input my_video.mp4 --no-display
"""

import argparse
import time
import cv2
import numpy as np

from sort_tracker import Sort

# ─────────────────────────────────────────────
# MobileNet SSD class labels  (COCO-like set)
# ─────────────────────────────────────────────
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Colour palette — one unique colour per class
np.random.seed(42)
PALETTE = {cls: tuple(int(c) for c in np.random.randint(60, 230, 3))
           for cls in CLASSES}

# Tracker ID colours (cycle through)
ID_COLOURS = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 255, 80), (80, 255, 255), (255, 80, 255),
    (255, 160, 80), (160, 255, 80), (80, 160, 255),
    (200, 80, 200), (80, 200, 200), (200, 200, 80),
]


def load_model(proto_path: str, model_path: str):
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect(net, frame, conf_threshold: float = 0.4):
    """
    Run MobileNet SSD on one frame.
    Returns list of dicts: {label, confidence, x1, y1, x2, y2}
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()   # shape (1,1,N,7)

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        class_id = int(detections[0, 0, i, 1])
        if class_id >= len(CLASSES):
            continue
        label = CLASSES[class_id]
        if label == "background":
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        results.append({
            "label": label,
            "confidence": confidence,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return results


def draw_detection(frame, det, color):
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    label = f"{det['label']} {det['confidence']:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_track(frame, x1, y1, x2, y2, track_id, label, conf, trail):
    color = ID_COLOURS[int(track_id) % len(ID_COLOURS)]

    # Draw trail
    for j in range(1, len(trail)):
        if trail[j - 1] is None or trail[j] is None:
            continue
        thickness = max(1, int(3 * (j / len(trail))))
        cv2.line(frame, trail[j - 1], trail[j], color, thickness)

    # Bounding box (thicker)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    header = f"ID:{int(track_id)}  {label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, header, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)


def draw_hud(frame, fps, n_objects, frame_idx, total_frames=None):
    """Heads-up display overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 90), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"FPS: {fps:.1f}",
        f"Objects tracked: {n_objects}",
        f"Frame: {frame_idx}" + (f" / {total_frames}" if total_frames else ""),
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1)

    # Corner watermark
    cv2.putText(frame, "SORT Tracker  |  MobileNet SSD",
                (w - 310, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run(
    input_source=0,
    proto_path="models/deploy.prototxt",
    model_path="models/MobileNetSSD_deploy.caffemodel",
    conf_threshold=0.4,
    max_age=5,
    min_hits=2,
    iou_threshold=0.3,
    output_path="output_tracked.mp4",
    display=True,
    max_frames=None,
):
    net = load_model(proto_path, model_path)
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {input_source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (W, H))

    # Track-ID → centroid trail  (last 30 points)
    trails: dict[int, list] = {}
    # Detect → track ID mapping cache (id assignment by IoU)
    id_label_map: dict[int, tuple] = {}  # track_id → (label, conf)

    frame_idx = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0.0

    print(f"[INFO] Source: {input_source}  |  Resolution: {W}×{H}  |  FPS: {fps_src:.1f}")
    print("[INFO] Processing… Press 'q' to quit (if display window is open).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # ── Detection ───────────────────────────
        dets_raw = detect(net, frame, conf_threshold)

        # Build array for SORT:  [x1,y1,x2,y2,conf]
        if dets_raw:
            det_arr = np.array([
                [d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]]
                for d in dets_raw
            ], dtype=float)
        else:
            det_arr = np.empty((0, 5))

        # ── Tracking ────────────────────────────
        tracks = tracker.update(det_arr)   # (M,5) → [x1,y1,x2,y2,id]

        # Map track centroids → nearest raw detection for label
        for trk in tracks:
            tx1, ty1, tx2, ty2, tid = trk
            tid = int(tid)
            tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

            # Find closest detection by centroid distance
            best_label, best_conf = "object", 0.0
            best_dist = 1e9
            for d in dets_raw:
                dcx = (d["x1"] + d["x2"]) / 2
                dcy = (d["y1"] + d["y2"]) / 2
                dist = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_label, best_conf = d["label"], d["confidence"]

            if best_dist < (W * H * 0.1):   # reasonable match
                id_label_map[tid] = (best_label, best_conf)

            # Update trail
            cx, cy = int(tcx), int(tcy)
            if tid not in trails:
                trails[tid] = []
            trails[tid].append((cx, cy))
            if len(trails[tid]) > 30:
                trails[tid].pop(0)

        # ── Drawing ─────────────────────────────
        for trk in tracks:
            tx1, ty1, tx2, ty2, tid = [int(v) for v in trk]
            tid_key = int(trk[4])
            label, conf = id_label_map.get(tid_key, ("object", 0.0))
            trail = trails.get(tid_key, [])
            draw_track(frame, tx1, ty1, tx2, ty2, tid_key, label, conf, trail)

        # ── FPS ──────────────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_hud(frame, fps_display, len(tracks), frame_idx, total_frames)

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow("Object Detection & Tracking — SORT + MobileNet SSD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Frames processed: {frame_idx}")
    if output_path:
        print(f"[INFO] Output saved → {output_path}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection & Tracking")
    parser.add_argument("--input", default=0,
                        help="Video file path or webcam index (default: 0)")
    parser.add_argument("--proto", default="models/deploy.prototxt")
    parser.add_argument("--model", default="models/MobileNetSSD_deploy.caffemodel")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Detection confidence threshold")
    parser.add_argument("--max-age", type=int, default=5,
                        help="SORT max frames before track is deleted")
    parser.add_argument("--min-hits", type=int, default=2,
                        help="SORT min hits before track is shown")
    parser.add_argument("--iou", type=float, default=0.3,
                        help="SORT IoU threshold for matching")
    parser.add_argument("--output", default="output_tracked.mp4",
                        help="Path to save output video (empty = no save)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live display window (headless mode)")
    args = parser.parse_args()

    input_src = args.input
    if isinstance(input_src, str) and input_src.isdigit():
        input_src = int(input_src)

    run(
        input_source=input_src,
        proto_path=args.proto,
        model_path=args.model,
        conf_threshold=args.conf,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou,
        output_path=args.output,
        display=not args.no_display,
    )
