"""
generate_demo_video.py
======================
Creates a synthetic 10-second demo video with moving coloured blobs
that simulate real objects, so the tracker can be tested without a webcam.
"""

import cv2
import numpy as np
import random
import math

W, H = 854, 480
FPS = 25
DURATION = 10  # seconds

OBJECT_CLASSES = ["person", "car", "bicycle", "dog", "bus", "bottle"]
COLOURS = [
    (60,  180, 255),   # orange-ish
    (80,  255, 120),   # green
    (255, 100, 100),   # blue
    (200, 80,  255),   # violet
    (255, 220,  60),   # cyan-yellow
    (60,  220, 220),   # teal
]


class FakeObject:
    def __init__(self, oid):
        self.oid = oid
        self.label = random.choice(OBJECT_CLASSES)
        self.color = random.choice(COLOURS)
        self.w = random.randint(60, 160)
        self.h = random.randint(60, 120)
        self.x = random.randint(0, W - self.w)
        self.y = random.randint(0, H - self.h)
        speed = random.uniform(1.5, 4.0)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        # gentle sine wobble
        self.wobble_amp = random.uniform(0, 1.5)
        self.wobble_freq = random.uniform(0.05, 0.15)
        self.t = 0

    def step(self):
        self.t += 1
        wobble = self.wobble_amp * math.sin(self.wobble_freq * self.t)
        self.x += self.vx + wobble
        self.y += self.vy + wobble

        # bounce off walls
        if self.x < 0 or self.x + self.w > W:
            self.vx = -self.vx
            self.x = max(0, min(W - self.w, self.x))
        if self.y < 0 or self.y + self.h > H:
            self.vy = -self.vy
            self.y = max(0, min(H - self.h, self.y))

    def draw(self, frame):
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.w, y1 + self.h

        # filled rounded rectangle body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        # darker border
        dark = tuple(max(0, c - 60) for c in self.color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), dark, 2)

        # label text inside
        cv2.putText(frame, self.label,
                    (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (10, 10, 10), 1)


def generate_demo_video(out_path="demo_input.mp4", n_objects=6):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    objects = [FakeObject(i) for i in range(n_objects)]
    total_frames = FPS * DURATION

    # Background — gradient
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for row in range(H):
        t = row / H
        bg[row] = (int(30 + 20 * t), int(30 + 15 * t), int(50 + 25 * t))

    print(f"[INFO] Generating demo video: {W}×{H} @ {FPS}fps × {DURATION}s → {out_path}")

    for f in range(total_frames):
        frame = bg.copy()

        # Subtle grid
        for gx in range(0, W, 80):
            cv2.line(frame, (gx, 0), (gx, H), (45, 45, 65), 1)
        for gy in range(0, H, 60):
            cv2.line(frame, (0, gy), (W, gy), (45, 45, 65), 1)

        for obj in objects:
            obj.step()
            obj.draw(frame)

        # Frame counter watermark
        cv2.putText(frame, f"DEMO  Frame {f + 1}/{total_frames}",
                    (W - 240, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        writer.write(frame)

    writer.release()
    print(f"[INFO] Demo video saved → {out_path}")


if __name__ == "__main__":
    generate_demo_video()
