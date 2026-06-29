"""
app.py  —  Streamlit Web Interface
====================================
Upload a video → see object detection + SORT tracking result.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path

from sort_tracker import Sort

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
  body { background: #0d0d0d; }
  .main { background: #111; color: #eee; }
  .block-container { padding-top: 1rem; }
  h1 { color: #7ecfff; font-family: 'Segoe UI', sans-serif; }
  h3 { color: #a8d8ff; }
  .metric-box {
    background: #1a2535;
    border: 1px solid #2a4060;
    border-radius: 10px;
    padding: 12px 18px;
    text-align: center;
    margin: 4px;
  }
  .metric-value { font-size: 2em; font-weight: bold; color: #7ecfff; }
  .metric-label { font-size: 0.8em; color: #88aacc; margin-top: 2px; }
  .stButton > button {
    background: linear-gradient(135deg, #1a6faa, #0a4a80);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-size: 1rem; cursor: pointer;
    transition: all 0.2s;
  }
  .stButton > button:hover { background: linear-gradient(135deg, #2a8fd0, #1a6faa); }
  .info-box {
    background: #0f2030; border-left: 4px solid #1a6faa;
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0;
    font-size: 0.9em; color: #aac8e8;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.markdown("# 🎯 Object Detection & Tracking")
st.markdown("**MobileNet SSD Detector  ·  SORT Kalman-Filter Tracker**")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    conf_thresh = st.slider("Detection Confidence", 0.2, 0.9, 0.4, 0.05,
                            help="Minimum confidence to accept a detection")
    max_age = st.slider("SORT Max Age", 1, 15, 5,
                        help="Frames a track survives without a detection")
    min_hits = st.slider("SORT Min Hits", 1, 5, 2,
                         help="Detections needed before track is shown")
    iou_thresh = st.slider("IoU Threshold", 0.1, 0.7, 0.3, 0.05,
                           help="IoU needed to match detection ↔ track")
    max_frames = st.slider("Max Frames to Process", 50, 500, 150, 25)

    st.markdown("---")
    st.markdown("### 🏷️ About")
    st.markdown("""
**Detector:** MobileNet SSD (Caffe)  
**Tracker:** SORT (Kalman + Hungarian)  
**Classes:** 20 PASCAL VOC objects  
**Framework:** OpenCV DNN  
    """)

# ── Model paths ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROTO = str(SCRIPT_DIR / "models" / "deploy.prototxt")
MODEL = str(SCRIPT_DIR / "models" / "MobileNetSSD_deploy.caffemodel")

CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus",
    "car","cat","chair","cow","diningtable","dog","horse","motorbike",
    "person","pottedplant","sheep","sofa","train","tvmonitor",
]
np.random.seed(42)
PALETTE = {c: tuple(int(x) for x in np.random.randint(60, 230, 3)) for c in CLASSES}
ID_COLS = [
    (255,80,80),(80,255,80),(80,80,255),(255,255,80),
    (80,255,255),(255,80,255),(255,160,80),(160,255,80),
]

@st.cache_resource
def load_net():
    return cv2.dnn.readNetFromCaffe(PROTO, MODEL)

def detect_frame(net, frame, thresh):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843,(300,300),127.5)
    net.setInput(blob)
    dets = net.forward()
    results = []
    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf < thresh: continue
        cid = int(dets[0,0,i,1])
        if cid >= len(CLASSES) or CLASSES[cid] == "background": continue
        box = dets[0,0,i,3:7] * np.array([w,h,w,h])
        x1,y1,x2,y2 = box.astype(int)
        results.append({"label":CLASSES[cid],"conf":conf,
                        "x1":max(0,x1),"y1":max(0,y1),
                        "x2":min(w-1,x2),"y2":min(h-1,y2)})
    return results

def annotate(frame, tracks, dets_raw, id_label_map, trails):
    for trk in tracks:
        tx1,ty1,tx2,ty2,tid = [int(v) for v in trk]
        tid_key = int(trk[4])
        col = ID_COLS[tid_key % len(ID_COLS)]

        # trail
        trail = trails.get(tid_key,[])
        for j in range(1,len(trail)):
            if trail[j-1] and trail[j]:
                cv2.line(frame, trail[j-1], trail[j], col,
                         max(1, int(3*(j/max(len(trail),1)))))

        cv2.rectangle(frame,(tx1,ty1),(tx2,ty2),col,2)
        label,conf = id_label_map.get(tid_key,("object",0.0))
        header = f"ID:{tid_key}  {label} {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(header,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
        cv2.rectangle(frame,(tx1,ty1-th-8),(tx1+tw+6,ty1),col,-1)
        cv2.putText(frame,header,(tx1+3,ty1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(10,10,10),2)
    return frame

# ── Main UI ──────────────────────────────────────────────
col_upload, col_demo = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        "📁 Upload a video file",
        type=["mp4","avi","mov","mkv"],
        help="Upload any video to run detection + tracking on it"
    )

with col_demo:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.button("🎬 Generate & Use Demo Video")

# ── Generate demo ─────────────────────────────────────────
demo_path = str(SCRIPT_DIR / "demo_input.mp4")
if use_demo:
    with st.spinner("Generating synthetic demo video…"):
        import subprocess, sys
        subprocess.run([sys.executable,
                        str(SCRIPT_DIR / "generate_demo_video.py")], check=True)
    st.success("Demo video ready!")

# ── Process ──────────────────────────────────────────────
video_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush()
    video_path = tmp.name
elif use_demo and Path(demo_path).exists():
    video_path = demo_path
elif Path(demo_path).exists():
    st.markdown('<div class="info-box">💡 No video uploaded — using existing demo video. Click <b>Generate & Use Demo Video</b> or upload your own.</div>', unsafe_allow_html=True)
    video_path = demo_path

if video_path:
    net = load_net()
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    st.markdown(f"**Video:** {W}×{H}  ·  {src_fps:.0f} fps  ·  {total} frames")

    # ── Live preview columns ──────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📷 Detection Only")
        ph_det = st.empty()
    with c2:
        st.markdown("### 🔢 Detection + Tracking (SORT)")
        ph_trk = st.empty()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    ph_fps  = m1.empty()
    ph_obj  = m2.empty()
    ph_frm  = m3.empty()
    ph_uid  = m4.empty()

    prog = st.progress(0)
    status = st.empty()

    # ── Processing loop ───────────────────────────────────
    id_label_map = {}
    trails = {}
    seen_ids = set()
    frame_idx = 0
    t0 = time.time()
    out_frames = []

    while frame_idx < min(max_frames, total):
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        dets_raw = detect_frame(net, frame, conf_thresh)

        det_frame = frame.copy()
        for d in dets_raw:
            col = PALETTE[d["label"]]
            cv2.rectangle(det_frame,(d["x1"],d["y1"]),(d["x2"],d["y2"]),col,2)
            lbl = f"{d['label']} {d['conf']:.2f}"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(det_frame,(d["x1"],d["y1"]-th-6),(d["x1"]+tw+4,d["y1"]),col,-1)
            cv2.putText(det_frame,lbl,(d["x1"]+2,d["y1"]-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        det_arr = (np.array([[d["x1"],d["y1"],d["x2"],d["y2"],d["conf"]]
                             for d in dets_raw], dtype=float)
                   if dets_raw else np.empty((0,5)))
        tracks = tracker.update(det_arr)

        for trk in tracks:
            tx1,ty1,tx2,ty2,tid = trk
            tid_key = int(tid)
            seen_ids.add(tid_key)
            tcx,tcy = (tx1+tx2)/2,(ty1+ty2)/2
            best,bd = ("object",0.0), 1e9
            for d in dets_raw:
                dist = ((tcx-(d["x1"]+d["x2"])/2)**2 +
                        (tcy-(d["y1"]+d["y2"])/2)**2)
                if dist < bd:
                    bd = dist
                    best = (d["label"],d["conf"])
            if bd < W*H*0.1:
                id_label_map[tid_key] = best
            trails.setdefault(tid_key,[]).append((int(tcx),int(tcy)))
            if len(trails[tid_key]) > 30:
                trails[tid_key].pop(0)

        trk_frame = frame.copy()
        annotate(trk_frame, tracks, dets_raw, id_label_map, trails)

        elapsed = max(time.time()-t0, 0.001)
        fps_now = frame_idx / elapsed

        # Update every 5 frames to avoid Streamlit flood
        if frame_idx % 5 == 0 or frame_idx == 1:
            ph_det.image(cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB),
                         channels="RGB", use_container_width=True)
            ph_trk.image(cv2.cvtColor(trk_frame, cv2.COLOR_BGR2RGB),
                         channels="RGB", use_container_width=True)
            ph_fps.markdown(f'<div class="metric-box"><div class="metric-value">{fps_now:.1f}</div><div class="metric-label">Proc FPS</div></div>', unsafe_allow_html=True)
            ph_obj.markdown(f'<div class="metric-box"><div class="metric-value">{len(tracks)}</div><div class="metric-label">Active Tracks</div></div>', unsafe_allow_html=True)
            ph_frm.markdown(f'<div class="metric-box"><div class="metric-value">{frame_idx}</div><div class="metric-label">Frames Done</div></div>', unsafe_allow_html=True)
            ph_uid.markdown(f'<div class="metric-box"><div class="metric-value">{len(seen_ids)}</div><div class="metric-label">Unique IDs</div></div>', unsafe_allow_html=True)
            prog.progress(frame_idx / min(max_frames, total))
            status.markdown(f"Processing frame **{frame_idx}** / {min(max_frames,total)} …")

        out_frames.append(trk_frame)

    cap.release()
    prog.progress(1.0)
    status.success(f"✅ Done! Processed {frame_idx} frames in {time.time()-t0:.1f}s  |  Unique track IDs: {len(seen_ids)}")

    # ── Save output video ─────────────────────────────────
    if out_frames:
        out_path = str(SCRIPT_DIR / "output_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wrt = cv2.VideoWriter(out_path, fourcc, src_fps, (W, H))
        for f in out_frames:
            wrt.write(f)
        wrt.release()

        with open(out_path, "rb") as vf:
            st.download_button("⬇️ Download Tracked Video",
                               data=vf.read(),
                               file_name="object_tracking_output.mp4",
                               mime="video/mp4")

else:
    st.markdown("""
    <div class="info-box">
    👆 Upload a video file above, or click <b>Generate &amp; Use Demo Video</b> to create a synthetic test video automatically.
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram placeholder
    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
| Stage | Component | Detail |
|---|---|---|
| **Input** | OpenCV VideoCapture | Webcam (index 0) or any video file |
| **Detection** | MobileNet SSD (Caffe) | 20 PASCAL VOC classes, 300×300 input |
| **Pre-processing** | `cv2.dnn.blobFromImage` | Scale 0.007843, mean 127.5 |
| **Post-processing** | Confidence filter | Threshold configurable (default 0.4) |
| **Tracking** | SORT Algorithm | Kalman Filter + Hungarian Algorithm |
| **State vector** | `[cx, cy, s, r, vx, vy, vs]` | 7-dimensional motion model |
| **Association** | IoU + `scipy.linear_sum_assignment` | Greedy optimal matching |
| **Visualisation** | OpenCV drawing | Per-ID colour, centroid trail, HUD |
| **Output** | MP4 via VideoWriter | Annotated + downloadable |
    """)
