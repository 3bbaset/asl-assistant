"""Shared by training and the live app, so features are IDENTICAL in both places."""
import numpy as np

N_FRAMES = 30
# per frame: 2 hand slots x (21 landmarks x 3 relative-to-wrist + 2 wrist position) = 130
FEAT = 2 * (21 * 3 + 2)


def frame_features(result):
    """result = MediaPipe HandLandmarkerResult for ONE frame -> vector of length 130."""
    hands = []
    for lm in result.hand_landmarks:
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        wrist = pts[0].copy()
        rel = pts - wrist
        scale = np.linalg.norm(rel[9]) + 1e-6          # wrist -> middle knuckle = hand size
        hands.append((wrist[0], np.concatenate([(rel / scale).ravel(), wrist[:2]])))
    hands.sort(key=lambda h: h[0])                      # slot 0 = left-most hand in the image
    out = np.zeros(FEAT, dtype=np.float32)
    for i, (_, f) in enumerate(hands[:2]):
        out[i * 65:(i + 1) * 65] = f
    return out


def resample(seq, n=N_FRAMES):
    """seq = (T, FEAT) -> (n, FEAT) by linear interpolation over time."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((n, FEAT), dtype=np.float32)
    if len(seq) == 1:
        return np.repeat(seq, n, axis=0)
    src = np.linspace(0, 1, len(seq))
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, seq[:, j]) for j in range(seq.shape[1])], axis=1)


def clip_vector(seq):
    return resample(seq).ravel()          # 30 * 130 = 3900 numbers
