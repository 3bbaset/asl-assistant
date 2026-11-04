import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"   # 24 static letters (no J, Z)

def make_detector(min_conf=0.5):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)

def detect(detector, rgb):
    """Return the 21 landmarks of the first hand, or None if no hand found."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    return result.hand_landmarks[0] if result.hand_landmarks else None

def normalize(landmarks):
    """21 landmarks -> 21x3 array, centered on the wrist and scaled by hand size."""
    pts = np.array([[p.x, p.y, p.z] for p in landmarks])
    pts -= pts[0]                                        # wrist becomes (0,0,0)
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))   # biggest distance from wrist
    return pts / scale

import cv2

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20), (0,17),
]

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
from itertools import combinations

CHAINS = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
TIPS = [4, 8, 12, 16, 20]

def renorm(P):
    """P is (N, 21, 3). Center on the wrist and scale by hand size again."""
    P = P - P[:, :1, :]
    scale = np.linalg.norm(P[:, :, :2], axis=2).max(axis=1)[:, None, None]
    return P / (scale + 1e-6)

def _cos(u, v):
    return (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-6)

def flat_to_features(X):
    """N x 63 raw numbers -> N x 101 features (coordinates, finger angles, distances)."""
    P = np.asarray(X, dtype=np.float64).reshape(-1, 21, 3)
    feats = [P.reshape(len(P), -1)]
    for chain in CHAINS:                                   # how bent is each finger joint
        v = [P[:, chain[i + 1]] - P[:, chain[i]] for i in range(4)]
        feats += [_cos(v[i], v[i + 1])[:, None] for i in range(3)]
    for a, b in combinations(TIPS, 2):                     # fingertip to fingertip
        feats.append(np.linalg.norm(P[:, a] - P[:, b], axis=1)[:, None])
    for t in (5, 9, 13, 17, 6, 10, 14, 18):                # thumb tip to the other fingers
        feats.append(np.linalg.norm(P[:, 4] - P[:, t], axis=1)[:, None])
    for t in TIPS:                                         # fingertip to wrist
        feats.append(np.linalg.norm(P[:, t] - P[:, 0], axis=1)[:, None])
    return np.hstack(feats)
