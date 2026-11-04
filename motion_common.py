import numpy as np

T = 30
SEEN = 0.99      # on a really tracked frame, wrist -> middle knuckle has length 1

def _seen(clip, slot):
    o = 65 * slot
    return np.linalg.norm(clip[:, o + 27:o + 30], axis=1) > SEEN

def fill_gaps(clip):
    """clip is (30, 130). Frames where the hand was lost get filled by interpolating."""
    clip = clip.copy()
    t = np.arange(T)
    for slot in (0, 1):
        o = 65 * slot
        m = _seen(clip, slot)
        if m.sum() < 3:
            clip[:, o:o + 65] = 0                    # this hand is basically not there
            continue
        for j in range(o, o + 65):
            clip[:, j] = np.interp(t, t[m], clip[m, j])
    return clip

def mirror_clip(clip):
    """Left/right mirror: what a left-handed signer's version of the same sign looks like."""
    c = clip.copy()
    present = [bool(c[:, 65 * s:65 * s + 63].any()) for s in (0, 1)]
    for s in (0, 1):
        if present[s]:
            o = 65 * s
            c[:, o:o + 63:3] *= -1                   # x of all 21 relative landmarks
            c[:, o + 63] = 1 - c[:, o + 63]          # wrist x position
    if present[0] and present[1]:                    # slot 0 = the left-most hand, so swap
        c = np.concatenate([c[:, 65:], c[:, :65]], axis=1)
    return c

def clip_features(clip):
    """One clip (30, 130) -> a compact feature vector (about 137 numbers)."""
    s = clip[:, :65]
    rel = s[:, :63].reshape(T, 21, 3)
    wrist = s[:, 63:65]
    parts = []
    for tip in (8, 20):                              # index fingertip, pinky fingertip
        pos = wrist + 0.15 * rel[:, tip, :2]         # rough fingertip position in the picture
        parts.append((pos - pos.mean(axis=0)).ravel())            # the shape of the path
        parts.append([np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()])  # how far it moved
    parts.append(rel[:, [4, 8, 12, 16, 20], :].mean(axis=0).ravel())  # average handshape
    return np.concatenate([np.ravel(p) for p in parts])
def _main_slot(clip):
    """Which of the two hand slots holds the hand we should follow."""
    e = [np.abs(clip[:, 65 * s:65 * s + 63]).sum() for s in (0, 1)]
    return int(np.argmax(e))

def _tip_path(clip, slot, tip):
    o = 65 * slot
    rel = clip[:, o:o + 63].reshape(T, 21, 3)
    pos = clip[:, o + 63:o + 65] + 0.15 * rel[:, tip, :2]
    pad = np.pad(pos, ((1, 1), (0, 0)), mode="edge")        # light smoothing
    return np.stack([np.convolve(pad[:, j], np.ones(3) / 3, mode="valid") for j in (0, 1)], axis=1)

def _shape_feats(pos):
    seg = np.diff(pos, axis=0)
    length = np.linalg.norm(seg, axis=1)
    total = length.sum()
    ext = pos.max(axis=0) - pos.min(axis=0)
    norm_pos = (pos - pos.mean(axis=0)) / max(ext.max(), 1e-3)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    moving = length > 0.15 * (length.max() + 1e-9)
    if moving.sum() >= 2:
        turn = np.angle(np.exp(1j * np.diff(ang[moving])))   # turning between moves
    else:
        turn = np.zeros(1)
    hist = np.zeros(8)
    for b, l in zip(((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8, length):
        hist[b] += l
    hist /= total + 1e-9
    straight = np.linalg.norm(pos[-1] - pos[0]) / (total + 1e-9)
    return np.concatenate([
        norm_pos[::3].ravel(),                                # the shape, 10 points
        ext, [ext[0] / (ext[1] + 1e-3)],                      # width, height, aspect
        [total, straight],
        [np.abs(turn).sum(), np.abs(turn).max(), (np.abs(turn) > 1.2).sum()],   # sharp turns
        hist, pos[-1] - pos[0],
        [length.mean(), length.max(), moving.mean()],
    ])

def clip_features2(clip):
    """One clip (30, 130) after fill_gaps -> 112 numbers about shape, turns, speed, handshape."""
    slot = _main_slot(clip)
    o = 65 * slot
    parts = [_shape_feats(_tip_path(clip, slot, tip)) for tip in (8, 20)]   # index tip, pinky tip
    tips = clip[:, o:o + 63].reshape(T, 21, 3)[:, [4, 8, 12, 16, 20], :]
    parts += [tips.mean(axis=0).ravel(), tips.std(axis=0).ravel()]          # handshape, and its change
    return np.concatenate(parts)

def clips_to_features(C):
    return np.array([clip_features2(c) for c in np.asarray(C)])
