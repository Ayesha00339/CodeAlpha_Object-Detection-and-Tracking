"""
SORT (Simple Online and Realtime Tracking) — Lightweight Implementation
Based on: Bewley et al., 2016 — https://arxiv.org/abs/1602.00763
"""

import numpy as np


# ─────────────────────────────────────────────
# Kalman Filter for bounding-box motion model
# ─────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single bounding box using a constant-velocity Kalman filter.
    State vector: [x, y, s, r, dx, dy, ds]
      x, y  = centre coordinates
      s     = scale (area)
      r     = aspect ratio (fixed)
      dx,dy,ds = velocities
    """

    count = 0  # global ID counter

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.history = []

        # State transition matrix  (7×7)
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix  (4×7)
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # Measurement noise
        self.R = np.eye(4, dtype=float)
        self.R[2:, 2:] *= 10.0

        # Process noise
        self.Q = np.eye(7, dtype=float)
        self.Q[4:, 4:] *= 0.01

        # Covariance
        self.P = np.eye(7, dtype=float)
        self.P[4:, 4:] *= 1000.0   # high uncertainty on velocity at start
        self.P *= 10.0

        # Initial state
        self.x = np.zeros((7, 1), dtype=float)
        m = self._bbox_to_z(bbox)
        self.x[:4] = m

    # ── helpers ──────────────────────────────

    @staticmethod
    def _bbox_to_z(bbox):
        """[x1,y1,x2,y2] → [cx, cy, s, r]ᵀ"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[cx], [cy], [s], [r]], dtype=float)

    @staticmethod
    def _z_to_bbox(z, score=None):
        """[cx, cy, s, r] → [x1, y1, x2, y2, (score)]"""
        w = np.sqrt(abs(z[2] * z[3]))
        h = z[2] / w if w > 0 else 0
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        if score is None:
            return np.array([x1, y1, x2, y2]).flatten()
        return np.array([x1, y1, x2, y2, score]).flatten()

    # ── Kalman predict / update ───────────────

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.x[:4]))
        return self.history[-1]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        return self._z_to_bbox(self.x[:4])


# ─────────────────────────────────────────────
# IoU utility
# ─────────────────────────────────────────────

def iou_batch(bb_test, bb_gt):
    """
    Vectorised IoU between two sets of bboxes.
    bb_test: (N,4), bb_gt: (M,4)  → (N,M)
    """
    bb_gt = np.expand_dims(bb_gt, 0)   # (1,M,4)
    bb_test = np.expand_dims(bb_test, 1)  # (N,1,4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_t = ((bb_test[..., 2] - bb_test[..., 0]) *
              (bb_test[..., 3] - bb_test[..., 1]))
    area_g = ((bb_gt[..., 2] - bb_gt[..., 0]) *
              (bb_gt[..., 3] - bb_gt[..., 1]))

    union = area_t + area_g - inter
    return inter / np.where(union == 0, 1e-6, union)


def linear_assignment(cost_matrix):
    """Hungarian algorithm via scipy."""
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(row_ind, col_ind)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        matched_idx = linear_assignment(-iou_matrix)
        matched = matched_idx[iou_matrix[matched_idx[:, 0], matched_idx[:, 1]] >= iou_threshold]
    else:
        matched = np.empty((0, 2), dtype=int)

    unmatched_det = [d for d in range(len(detections))
                     if d not in matched[:, 0]] if len(matched) else list(range(len(detections)))
    unmatched_trk = [t for t in range(len(trackers))
                     if t not in matched[:, 1]] if len(matched) else list(range(len(trackers)))

    return matched, np.array(unmatched_det), np.array(unmatched_trk)


# ─────────────────────────────────────────────
# SORT main class
# ─────────────────────────────────────────────

class Sort:
    """
    SORT tracker.
    Parameters
    ----------
    max_age       : frames to keep a lost track alive
    min_hits      : min detections before a track is confirmed
    iou_threshold : IoU to match detection ↔ track
    """

    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0  # reset IDs

    def update(self, dets=np.empty((0, 5))):
        """
        dets: np.array of shape (N, 5) → [x1, y1, x2, y2, confidence]
        Returns: np.array (M, 5) → [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict positions of existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # Collect active tracks
        results = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update <= self.max_age and
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                results.append(np.concatenate([d, [trk.id]]).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(results, axis=0) if results else np.empty((0, 5))
