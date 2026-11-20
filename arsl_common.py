"""arsl_common.py - Shared 1-hand/2-hand geometric feature extraction for Arabic/Egyptian Sign Language."""
from itertools import combinations
import numpy as np
from common import CHAINS, TIPS
from motion_common import fill_gaps
from phrase_common import frame_features


def _cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))


def _hand_geom_single(P):
    """P: (21, 3) wrist-centered normalized landmarks -> 33 geometric scalars."""
    if not P.any():
        return np.zeros(33, dtype=np.float32)
    feats = []
    for chain in CHAINS:
        v = [P[chain[i + 1]] - P[chain[i]] for i in range(4)]
        for i in range(3):
            feats.append(_cos(v[i], v[i + 1]))
    for a, b in combinations(TIPS, 2):
        feats.append(float(np.linalg.norm(P[a] - P[b])))
    for t in (5, 9, 13, 17):
        feats.append(float(np.linalg.norm(P[4] - P[t])))
    for t in TIPS[:4]:
        feats.append(float(np.linalg.norm(P[t] - P[0])))
    return np.asarray(feats, dtype=np.float32)


def arsl_row_features(row130):
    """Convert a 130-dim phrase_common.frame_features vector -> 205-dim ArSL word vector.

    Slots:
      - Hand 0: 63 coords + 33 geom + 2 wrist = 98
      - Hand 1: 63 coords + 33 geom + 2 wrist = 98
      - Global 2-hand interaction = 9
    Total = 205 floats.
    """
    r = np.asarray(row130, dtype=np.float32).ravel()
    h0 = r[:65]
    h1 = r[65:130]
    P0 = h0[:63].reshape(21, 3)
    P1 = h1[:63].reshape(21, 3)
    w0 = h0[63:65]
    w1 = h1[63:65]

    present0 = bool(P0.any())
    present1 = bool(P1.any())
    # Re-scale per-hand P by max 2D extent for extra robustness while keeping wrist coords
    if present0:
        s0 = float(np.max(np.linalg.norm(P0[:, :2], axis=1))) + 1e-6
        P0n = P0 / s0
    else:
        P0n = P0
    if present1:
        s1 = float(np.max(np.linalg.norm(P1[:, :2], axis=1))) + 1e-6
        P1n = P1 / s1
    else:
        P1n = P1

    f0 = np.concatenate([P0n.ravel(), _hand_geom_single(P0n), w0 if present0 else np.zeros(2, dtype=np.float32)])
    f1 = np.concatenate([P1n.ravel(), _hand_geom_single(P1n), w1 if present1 else np.zeros(2, dtype=np.float32)])

    n_hands = float(int(present0) + int(present1))
    both = float(present0 and present1)
    if present0 and present1:
        dx = float(w1[0] - w0[0])
        dy = float(w1[1] - w0[1])
        dist = float(np.hypot(dx, dy))
        # Approximate relative tip distance in image space (wrist + scaled hand)
        idx_d = float(np.linalg.norm((w1 + 0.12 * P1n[8, :2]) - (w0 + 0.12 * P0n[8, :2])))
        thm_d = float(np.linalg.norm((w1 + 0.12 * P1n[4, :2]) - (w0 + 0.12 * P0n[4, :2])))
        cx, cy = float((w0[0] + w1[0]) * 0.5), float((w0[1] + w1[1]) * 0.5)
    elif present0:
        dx = dy = dist = idx_d = thm_d = 0.0
        cx, cy = float(w0[0]), float(w0[1])
    elif present1:
        dx = dy = dist = idx_d = thm_d = 0.0
        cx, cy = float(w1[0]), float(w1[1])
    else:
        dx = dy = dist = idx_d = thm_d = cx = cy = 0.0

    glob = np.array([n_hands / 2.0, both, dx, dy, dist, idx_d, thm_d, cx, cy], dtype=np.float32)
    return np.concatenate([f0, f1, glob]).astype(np.float32)


def arsl_batch_features(rows130):
    """Vector/batch helper so FunctionTransformer or training loops share one path."""
    arr = np.asarray(rows130, dtype=np.float32)
    if arr.ndim == 1:
        return arsl_row_features(arr)[None, :]
    return np.vstack([arsl_row_features(r) for r in arr])


def mirror_row130(row130):
    """Mirror a 130-dim frame_features row horizontally and keep left-to-right hand slot order."""
    r = np.asarray(row130, dtype=np.float32).copy()
    slots = []
    for i in (0, 1):
        seg = r[i * 65:(i + 1) * 65].copy()
        if seg[:63].any():
            pts = seg[:63].reshape(21, 3)
            pts[:, 0] *= -1.0
            seg[:63] = pts.ravel()
            seg[63] = 1.0 - seg[63]
            slots.append((float(seg[63]), seg))
    slots.sort(key=lambda item: item[0])
    out = np.zeros(130, dtype=np.float32)
    for i, (_, seg) in enumerate(slots):
        out[i * 65:(i + 1) * 65] = seg
    return out


def augment_row130(row130, rng):
    """Small scale/rotation/jitter augmentation on a 130-dim frame vector."""
    r = np.asarray(row130, dtype=np.float32).copy()
    ang = np.radians(rng.uniform(-12.0, 12.0))
    c, s = float(np.cos(ang)), float(np.sin(ang))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    scale = float(rng.uniform(0.92, 1.08))
    dx, dy = float(rng.uniform(-0.04, 0.04)), float(rng.uniform(-0.04, 0.04))
    for i in (0, 1):
        seg = r[i * 65:(i + 1) * 65]
        if seg[:63].any():
            pts = seg[:63].reshape(21, 3)
            pts[:, :2] = (pts[:, :2] @ rot.T) * scale
            pts += rng.normal(0.0, 0.015, pts.shape).astype(np.float32)
            seg[:63] = pts.ravel()
            seg[63] = np.clip(seg[63] + dx, 0.0, 1.0)
            seg[64] = np.clip(seg[64] + dy, 0.0, 1.0)
    return r


def arsl_frame_features(hand_landmarks_result):
    """Extract 205-dim ArSL feature vector directly from MediaPipe HandLandmarkerResult."""
    return arsl_row_features(frame_features(hand_landmarks_result))


def classify_arsl_word(model, clip_30x130):
    """Classify a live WordSegmenter (30, 130) clip using an ArSL word classifier.

    Focuses on the apex/active middle window of the clip (frames 6..24 where hands
    are present) and averages predicted probabilities across up to 10 sampled frames.
    """
    raw_clip = fill_gaps(np.asarray(clip_30x130, dtype=np.float32).reshape(30, 130))
    active_idx = np.where(raw_clip[:, :63].any(axis=1) | raw_clip[:, 65:128].any(axis=1))[0]
    if len(active_idx) == 0:
        classes = [str(c) for c in model.classes_]
        return classes[0], 0.0, [(c, 0.0) for c in classes]

    mid = [i for i in active_idx if 5 <= i <= 24]
    use_idx = mid if len(mid) >= 3 else active_idx.tolist()
    if len(use_idx) > 10:
        sel = np.linspace(0, len(use_idx) - 1, 10).round().astype(int)
        use_idx = [use_idx[j] for j in sel]

    feats = arsl_batch_features(raw_clip[use_idx])
    probs = model.predict_proba(feats).mean(axis=0)
    order = probs.argsort()[::-1]
    ranked = [(str(model.classes_[i]), float(probs[i])) for i in order]
    return ranked[0][0], ranked[0][1], ranked
