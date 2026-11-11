"""Pose landmarks -> per-frame feature vectors (46 values).

Pure NumPy so the same code runs on Kaggle (training) and on the low-RAM
client node (live). Input landmarks follow MediaPipe Pose (33 points, each
x, y, z, visibility; x and y normalised to image width and height).
"""
import numpy as np

FPS = 30.0
VIS_THRESHOLD = 0.5
MAX_GAP = 5          # frames; longer gaps split a clip
MIN_SEGMENT = 16     # frames; shorter segments are dropped

# MediaPipe indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
JOINTS = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]
JOINT_NAMES = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
               'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
GATE = [L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN]   # must be visible for a frame to count

# (a, b, c): angle at b between b->a and b->c
ANGLES = [
    ('l_elbow', L_SH, L_EL, L_WR), ('r_elbow', R_SH, R_EL, R_WR),
    ('l_shoulder', L_EL, L_SH, L_HIP), ('r_shoulder', R_EL, R_SH, R_HIP),
    ('l_hip', L_SH, L_HIP, L_KN), ('r_hip', R_SH, R_HIP, R_KN),
    ('l_knee', L_HIP, L_KN, L_AN), ('r_knee', R_HIP, R_KN, R_AN),
]
LIMBS = [('l_wrist', L_WR), ('r_wrist', R_WR), ('l_ankle', L_AN), ('r_ankle', R_AN)]

FEATURE_NAMES = (
    [f'{j}_{c}' for j in JOINT_NAMES for c in ('x', 'y')]
    + [f'ang_{a[0]}' for a in ANGLES]
    + [f'angvel_{a[0]}' for a in ANGLES]
    + [f'speed_{l[0]}' for l in LIMBS]
    + ['torso_incl', 'torso_incl_vel']
)
N_FEATURES = len(FEATURE_NAMES)  # 46
assert N_FEATURES == 46

# Index map for the paper's 9-D ablation: 3 left-side angles, velocities, accelerations
PAPER_ANGLES = ['l_elbow', 'l_shoulder', 'l_knee']

# Mirror permutation for MediaPipe's 33 landmarks (left <-> right)
_MP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
             (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MIRROR_PERM = np.arange(33)
for a, b in _MP_PAIRS:
    MIRROR_PERM[a], MIRROR_PERM[b] = b, a


def frame_valid(lm):
    """lm: (n, 33, 4). True where all gate joints are visible enough."""
    return np.all(lm[:, GATE, 3] >= VIS_THRESHOLD, axis=1)


def segments(lm):
    """Split a clip into usable segments. Gaps up to MAX_GAP frames are filled by
    linear interpolation; longer gaps split the clip. Returns list of (n_i, 33, 4)."""
    lm = np.asarray(lm, dtype=np.float32)
    ok = frame_valid(lm)
    out, start, n = [], None, len(lm)
    i = 0
    idx_ok = np.where(ok)[0]
    if len(idx_ok) == 0:
        return []
    # walk through valid frames, splitting where the gap is too long
    seg = [idx_ok[0]]
    for a, b in zip(idx_ok[:-1], idx_ok[1:]):
        if b - a - 1 > MAX_GAP:
            out.append(seg); seg = [b]
        else:
            seg.append(b)
    out.append(seg)
    result = []
    for s in out:
        lo, hi = s[0], s[-1]
        if hi - lo + 1 < MIN_SEGMENT:
            continue
        x = lm[lo:hi + 1].copy()
        valid = ok[lo:hi + 1]
        if not valid.all():
            t = np.arange(len(x)); good = t[valid]
            flat = x.reshape(len(x), -1)
            for c in range(flat.shape[1]):
                flat[~valid, c] = np.interp(t[~valid], good, flat[valid, c])
            x = flat.reshape(x.shape)
        result.append(x)
    return result


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(-1)
    return np.degrees(np.arctan2(np.abs(cross), dot))


def _vel(x, fps):
    """Backward difference; first frame copies the second."""
    v = np.zeros_like(x)
    if len(x) > 1:
        v[1:] = (x[1:] - x[:-1]) * fps
        v[0] = v[1]
    return v


def frame_features(lm, aspect=1.0, fps=FPS, prev=None):
    """lm: (n, 33, >=2) landmarks of one continuous segment.
    aspect: image width / height (MediaPipe x is relative to width).
    prev: optional (33, >=2) landmark of the frame before lm[0], used by the
          live stream so velocities at the start of a chunk are correct.
    Returns (n, 46) float32."""
    lm = np.asarray(lm, dtype=np.float32)[..., :2].copy()
    lm[..., 0] *= aspect                      # x into image-height units
    if prev is not None:
        p = np.asarray(prev, dtype=np.float32)[None, :, :2].copy(); p[..., 0] *= aspect
        lm = np.concatenate([p, lm], 0)
    hip = (lm[:, L_HIP] + lm[:, R_HIP]) / 2
    sh = (lm[:, L_SH] + lm[:, R_SH]) / 2
    torso = np.maximum(np.linalg.norm(sh - hip, axis=1), 1e-3)
    norm = (lm - hip[:, None, :]) / torso[:, None, None]
    pos = norm[:, JOINTS, :].reshape(len(lm), -1)                              # 24
    ang = np.stack([_angle(norm[:, a], norm[:, b], norm[:, c]) for _, a, b, c in ANGLES], 1)  # 8
    angv = _vel(ang, fps)                                                       # 8
    spd = np.stack([np.linalg.norm(_vel(norm[:, j], fps), axis=1) for _, j in LIMBS], 1)  # 4
    spine = sh - hip
    incl = np.degrees(np.arctan2(spine[:, 0], -spine[:, 1]))[:, None]           # 0 = upright
    inclv = _vel(incl, fps)
    f = np.concatenate([pos, ang, angv, spd, incl, inclv], 1).astype(np.float32)
    if prev is not None:
        f = f[1:]
    return f


def paper_features(feat46, fps=FPS):
    """The original paper's 9-D vector from the 46-D features: three left-side
    angles, their velocities and accelerations."""
    idx = [FEATURE_NAMES.index('ang_' + a) for a in PAPER_ANGLES]
    ang = feat46[:, idx]
    vel = _vel(ang, fps)
    acc = _vel(vel, fps)
    return np.concatenate([ang, vel, acc], 1).astype(np.float32)


def windows(feat, T=48, stride=4):
    """Cut (n, F) into (m, T, F) windows. Clips shorter than T are padded by
    repeating the last frame."""
    n = len(feat)
    if n < T:
        feat = np.concatenate([feat, np.repeat(feat[-1:], T - n, 0)], 0); n = T
    starts = list(range(0, n - T + 1, stride))
    return np.stack([feat[s:s + T] for s in starts]).astype(np.float32)


# augmentation
def augment_landmarks(lm, rng, mirror=None, rot_deg=10.0, speed=(0.8, 1.2), drop=0.05):
    """Augment raw landmarks (n, 33, 4) of one segment before feature extraction.
    Mirror swaps left/right and flips x; rotation is about the mid-hip; speed
    resamples time; frame drop removes ~5% of frames and re-interpolates."""
    x = np.asarray(lm, dtype=np.float32).copy()
    if mirror is None:
        mirror = rng.random() < 0.5
    if mirror:
        x = x[:, MIRROR_PERM]
        x[..., 0] = 1.0 - x[..., 0]
    th = np.radians(rng.uniform(-rot_deg, rot_deg))
    c, s = np.cos(th), np.sin(th)
    hip = (x[:, L_HIP, :2] + x[:, R_HIP, :2]) / 2
    xy = x[..., :2] - hip[:, None]
    x[..., 0] = c * xy[..., 0] - s * xy[..., 1] + hip[:, None, 0]
    x[..., 1] = s * xy[..., 0] + c * xy[..., 1] + hip[:, None, 1]
    n = len(x)
    k = rng.uniform(*speed)
    m = max(int(round(n / k)), 2)
    t_new = np.linspace(0, n - 1, m)
    keep = rng.random(m) >= drop
    keep[0] = keep[-1] = True
    t_src = np.arange(n)
    flat = x.reshape(n, -1)
    res = np.empty((m, flat.shape[1]), np.float32)
    for ci in range(flat.shape[1]):
        res[:, ci] = np.interp(t_new, t_src, flat[:, ci])
    if not keep.all():
        tt = np.arange(m)
        for ci in range(res.shape[1]):
            res[~keep, ci] = np.interp(tt[~keep], tt[keep], res[keep, ci])
    return res.reshape(m, 33, x.shape[2])


# live stream
class StreamFeaturizer:
    """Turns irregular (t, landmarks) samples from the browser into 30 fps
    feature frames, holding the last T frames for the classifier."""

    def __init__(self, T=48, fps=FPS):
        self.T, self.fps = T, fps
        self.reset()

    def reset(self):
        self.last_t = None; self.last_lm = None; self.next_t = None
        self.prev_grid = None; self.buf = []; self.missing = 0

    def push(self, t, lm, aspect=1.0):
        """t in seconds, lm (33, 4) or None. Returns number of new frames."""
        if lm is None or not np.all(np.asarray(lm)[GATE, 3] >= VIS_THRESHOLD):
            # skip the frame; the next good one is interpolated across the gap,
            # and a gap over 1 s starts a new clip (checked below)
            self.missing += 1
            return 0
        self.missing = 0
        lm = np.asarray(lm, dtype=np.float32)
        if self.last_t is None or t <= self.last_t or t - self.last_t > 1.0:
            if self.last_t is not None and t - self.last_t > 1.0:
                self.reset()
            self.last_t, self.last_lm, self.next_t = t, lm, t
        new = []
        while self.next_t <= t:
            a = 0.0 if t == self.last_t else (self.next_t - self.last_t) / (t - self.last_t)
            new.append(self.last_lm + a * (lm - self.last_lm))
            self.next_t += 1.0 / self.fps
        self.last_t, self.last_lm = t, lm
        if not new:
            return 0
        new = np.stack(new)
        f = frame_features(new, aspect=aspect, fps=self.fps, prev=self.prev_grid)
        self.prev_grid = new[-1]
        self.buf.extend(list(f))
        self.buf = self.buf[-self.T:]
        return len(new)

    def window(self, min_frames=None):
        """Last T frames as (1, T, F). With min_frames, a shorter buffer is padded
        by repeating its last frame, the same rule used for short training clips."""
        n = len(self.buf)
        if n < (min_frames or self.T) or n == 0:
            return None
        x = np.stack(self.buf)
        if n < self.T:
            x = np.concatenate([x, np.repeat(x[-1:], self.T - n, 0)], 0)
        return x[None]
