"""Stage 'extract': MediaPipe Pose Landmarker on every Penn Action frame.

Runs in its own Python process with MediaPipe installed in a separate folder,
so MediaPipe's NumPy/OpenCV versions cannot break the notebook's libraries.
Writes cache/lm/<video>.npy with shape (frames, 33, 4): x, y, z, visibility.
Frames with no detected person are all zeros. Safe to re-run: finished videos
are skipped.
"""
import json, os, sys, time
from multiprocessing import Pool

WORK = os.environ.get('BTP_WORK', '/kaggle/working')
CACHE = os.path.join(WORK, 'cache')
MODEL = os.environ.get('BTP_POSE_MODEL', os.path.join(CACHE, 'pose_landmarker_full.task'))


def run_video(m):
    import numpy as np
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    out = os.path.join(CACHE, 'lm', m['vid'] + '.npy')
    if os.path.exists(out):
        return m['vid'], 'cached'
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
    files = sorted(f for f in os.listdir(m['frames_dir']) if f.endswith('.jpg'))
    arr = np.zeros((len(files), 33, 4), np.float16)
    with vision.PoseLandmarker.create_from_options(opts) as lmk:
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(m['frames_dir'], f))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / 30))
            if res.pose_landmarks:
                arr[i] = [[l.x, l.y, l.z, (l.visibility if l.visibility is not None else 0.0)] for l in res.pose_landmarks[0]]
    np.save(out + '.tmp.npy', arr)
    os.replace(out + '.tmp.npy', out)
    return m['vid'], 'done'


if __name__ == '__main__':
    with open(os.path.join(CACHE, 'meta.json')) as f:
        meta = json.load(f)
    limit = int(os.environ.get('BTP_LIMIT', '0'))
    if limit:
        meta = meta[:limit]
    todo = [m for m in meta if not os.path.exists(os.path.join(CACHE, 'lm', m['vid'] + '.npy'))]
    print(f'{len(meta) - len(todo)} videos already done, {len(todo)} to go', flush=True)
    t0 = time.time()
    workers = max(1, (os.cpu_count() or 2))
    with Pool(workers) as pool:
        for k, (vid, st) in enumerate(pool.imap_unordered(run_video, todo, chunksize=4), 1):
            if k % 50 == 0 or k == len(todo):
                el = time.time() - t0
                print(f'{k}/{len(todo)} videos, {el / 60:.1f} min elapsed, about {el / k * (len(todo) - k) / 60:.0f} min left', flush=True)
    print('extraction finished', flush=True)
