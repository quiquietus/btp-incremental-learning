"""BTP training pipeline (runs on Kaggle).

Stages (each can be re-run; results are cached under WORK/cache):
  meta      read Penn Action labels -> cache/meta.json
  extract   MediaPipe Pose on every frame (separate process, see extract.py)
  data      landmarks -> features, splits, federated partitions
  central   centralised training: Bi-LSTM, Bi-RNN, ablations, baselines, 14 and 15 classes
  cl        incremental-learning experiments A (new client) and B (new classes)
  fl        federated simulation: FedAvg vs FedProx + FedAdam + Ditto, contributions
  export    model files, node data, figures, tables, params.json -> outputs.zip
"""
import json, os, sys, time, math, glob, copy, shutil, zipfile, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from btpcore import features as FT
from btpcore import model as NPM
from btpcore import fl_core as FL

# configuration
P = {
    'seed_list': [0, 1, 2],
    'data': {'fps': FT.FPS, 'T': 48, 'stride': 4, 'vis_threshold': FT.VIS_THRESHOLD, 'max_gap': FT.MAX_GAP,
             'min_segment': FT.MIN_SEGMENT, 'val_share': 0.15, 'aug_copies': 4,
             'augmentation': {'mirror_p': 0.5, 'rotation_deg': 10, 'speed': [0.8, 1.2], 'frame_drop': 0.05,
                              'note': 'scale augmentation is omitted because torso-length normalisation removes scale'},
             'normalisation': 'mid-hip origin, torso-length scale, then z-score with training statistics',
             'short_clip_padding': 'repeat last frame up to T'},
    'pose': {'model': 'pose_landmarker_full.task', 'mode': 'VIDEO', 'num_poses': 1, 'min_detection_conf': 0.5,
             'min_presence_conf': 0.5, 'min_tracking_conf': 0.5},
    'model': {'H': 64, 'A': 64, 'D': 64, 'dropout': 0.3, 'layers': 2},
    'train': {'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4, 'warmup_epochs': 5, 'min_lr': 1e-6,
              'max_epochs': 80, 'patience': 12, 'batch': 64, 'label_smoothing': 0.05, 'grad_clip': 1.0,
              'sampler': 'class-balanced', 'early_stop_metric': 'validation window macro-F1'},
    'incremental': {'buffer': 16, 'lr': 1e-4, 'optimizer': 'Adam', 'replay_ratio': 0.2, 'reservoir': 2000,
                    'lambda_ewc': 1000.0, 'lambda_ewc_sensitivity': [100.0, 1000.0, 10000.0],
                    'fisher_samples': 500, 'passes_per_chunk': 3, 'chunks_expA': 4, 'class_tasks_expB': [10, 2, 2],
                    'expB_passes': 10, 'expB_lr': 5e-4, 'expB_new_head_init': 'mean of seen-class output rows'},
    'federated': {'clients': 3, 'rounds': 30, 'eta_search_rounds': 10, 'local_epochs': 2, 'local_lr': 1e-3,
                  'local_batch': 64, 'participation': 'all', 'dirichlet_alpha': 0.5, 'client_test_share': 0.2,
                  'mu': 0.01, 'eta_grid': [1e-3, 1e-2, 3e-2, 1e-1], 'beta1': 0.9, 'beta2': 0.99, 'tau': 1e-3,
                  'ditto_lambda': 0.1, 'server_val_windows': 400, 'shapley': 'exact (2^3 subsets)'},
    'live': {'predict_every_frames': 8, 'smoothing_predictions': 5},
}
QUICK = os.environ.get('BTP_QUICK') == '1'      # tiny settings for a smoke test
if QUICK:
    P['seed_list'] = [0]; P['train'].update(max_epochs=3, patience=2); P['data']['aug_copies'] = 1
    P['federated'].update(rounds=3, eta_search_rounds=1, local_epochs=1)
    P['incremental'].update(passes_per_chunk=1, fisher_samples=50, lambda_ewc_sensitivity=[1000.0])

CLASSES15 = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowling', 'clean_and_jerk', 'golf_swing',
             'jump_rope', 'jumping_jacks', 'pull_ups', 'push_ups', 'sit_ups', 'squats', 'strumming_guitar',
             'tennis_forehand', 'tennis_serve']
CLASSES14 = [c for c in CLASSES15 if c != 'strumming_guitar']
# Penn Action's .mat files use short names for six actions
PENN_NAMES = {'bowl': 'bowling', 'pullup': 'pull_ups', 'pushup': 'push_ups', 'situp': 'sit_ups',
              'squat': 'squats', 'strum_guitar': 'strumming_guitar'}

WORK = os.environ.get('BTP_WORK', '/kaggle/working')
CACHE = os.path.join(WORK, 'cache')
OUT = os.path.join(WORK, 'outputs')
for d in [CACHE, OUT, os.path.join(CACHE, 'lm')]:
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


# stage: meta
def find_dataset():
    """Return the Penn Action root (folder with frames/ and labels/)."""
    cands = glob.glob('/kaggle/input/**/labels/0001.mat', recursive=True) + \
        glob.glob(os.path.join(WORK, '**/labels/0001.mat'), recursive=True) + \
        glob.glob('/tmp/penn/**/labels/0001.mat', recursive=True)
    for c in cands:
        root = os.path.dirname(os.path.dirname(c))
        if os.path.isdir(os.path.join(root, 'frames')):
            return root
    return None


def download_dataset():
    """Stream the archive straight into tar, so the 3 GB file is never stored twice."""
    url = 'https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz'
    os.makedirs('/tmp/penn', exist_ok=True)
    log('downloading and unpacking', url, '(about 3 GB, usually 5 to 15 minutes)')
    r = subprocess.run(f'wget -q --tries=3 -O - "{url}" | tar -xz -C /tmp/penn', shell=True)
    if r.returncode != 0 or find_dataset() is None:
        raise RuntimeError('Download failed. Click "Add Data", search "Penn Action", add a copy with frames/ and labels/ folders, and re-run this cell.')


def stage_meta():
    import scipy.io as sio
    root = find_dataset()
    if root is None:
        download_dataset(); root = find_dataset()
    if root is None:
        raise RuntimeError('Penn Action not found after download.')
    log('dataset root:', root)
    meta = []
    for f in sorted(glob.glob(os.path.join(root, 'labels', '*.mat'))):
        m = sio.loadmat(f)
        vid = os.path.splitext(os.path.basename(f))[0]
        dims = np.asarray(m['dimensions']).ravel()
        act = str(np.asarray(m['action']).ravel()[0]).strip()
        meta.append({'vid': vid, 'action': PENN_NAMES.get(act, act),
                     'train': int(np.asarray(m['train']).ravel()[0]) == 1,
                     'nframes': int(np.asarray(m['nframes']).ravel()[0]),
                     'H': int(dims[0]), 'W': int(dims[1]),
                     'x': np.asarray(m['x'], float).round(1).tolist(), 'y': np.asarray(m['y'], float).round(1).tolist(),
                     'vis': np.asarray(m['visibility']).astype(int).tolist(),
                     'bbox': np.asarray(m['bbox'], float).round(1).tolist(),
                     'frames_dir': os.path.join(root, 'frames', vid)})
    save_json(os.path.join(CACHE, 'meta.json'), meta)
    from collections import Counter
    log('videos:', len(meta), 'train:', sum(m['train'] for m in meta), 'classes:', dict(Counter(m['action'] for m in meta)))


# stage: data
def load_meta():
    meta = load_json(os.path.join(CACHE, 'meta.json'))
    for m in meta:
        m['action'] = PENN_NAMES.get(m['action'], m['action'])
    missing = sorted(set(CLASSES15) - {m['action'] for m in meta})
    if missing:
        raise RuntimeError(f'no videos for {missing}; action names in the labels: {sorted({m["action"] for m in meta})}')
    return meta


def pck(meta):
    """PCK@0.2 of MediaPipe against Penn Action annotations (threshold 0.2 x max bbox side).
    Penn joint order: head, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_kn, r_kn, l_an, r_an."""
    mp_idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
    hits = {'as_is': [0, 0], 'swapped': [0, 0]}
    per_joint = np.zeros((2, 13, 2))
    for m in meta:
        f = os.path.join(CACHE, 'lm', m['vid'] + '.npy')
        if not os.path.exists(f):
            continue
        lm = np.load(f).astype(np.float32)
        gx, gy, gv = np.asarray(m['x']), np.asarray(m['y']), np.asarray(m['vis'])
        bb = np.asarray(m['bbox'])
        n = min(len(lm), len(gx), len(bb))
        for t in range(n):
            if lm[t, :, 3].max() == 0:
                continue
            thr = 0.2 * max(bb[t, 2] - bb[t, 0], bb[t, 3] - bb[t, 1])
            px = lm[t, mp_idx, 0] * m['W']; py = lm[t, mp_idx, 1] * m['H']
            for mode, order in (('as_is', list(range(13))), ('swapped', swap)):
                d = np.hypot(px[order] - gx[t], py[order] - gy[t])
                ok = gv[t] > 0
                hits[mode][0] += int(((d < thr) & ok).sum()); hits[mode][1] += int(ok.sum())
                mi = 0 if mode == 'as_is' else 1
                per_joint[mi, :, 0] += (d < thr) & ok; per_joint[mi, :, 1] += ok
    res = {k: (v[0] / v[1] if v[1] else None) for k, v in hits.items()}
    best = max(res, key=lambda k: res[k] or 0)
    names = ['head', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    mi = 0 if best == 'as_is' else 1
    res['per_joint'] = {names[j]: float(per_joint[mi, j, 0] / max(per_joint[mi, j, 1], 1)) for j in range(13)}
    res['convention_used'] = best
    res['note'] = 'PCK@0.2 with threshold 0.2 x max(bbox width, height); MediaPipe nose used for the head joint.'
    return res


def stage_data():
    meta = load_meta()
    rng = np.random.default_rng(0)
    vids = []
    no_pose = 0
    for m in meta:
        f = os.path.join(CACHE, 'lm', m['vid'] + '.npy')
        if not os.path.exists(f):
            continue
        lm = np.load(f).astype(np.float32)
        segs = FT.segments(lm)
        if not segs:
            no_pose += 1
            continue
        vids.append({'vid': m['vid'], 'action': m['action'], 'train': m['train'], 'aspect': m['W'] / m['H'], 'segs': segs})
    log('videos with usable pose:', len(vids), 'dropped (no usable pose):', no_pose)
    # validation split: 15% of training videos, stratified by class
    by_cls = {}
    for v in vids:
        if v['train']:
            by_cls.setdefault(v['action'], []).append(v['vid'])
    val = set()
    for c, lst in by_cls.items():
        lst = sorted(lst); rng.shuffle(lst)
        val.update(lst[:max(1, int(round(P['data']['val_share'] * len(lst))))])
    for v in vids:
        v['split'] = 'test' if not v['train'] else ('val' if v['vid'] in val else 'train')
    # features (original + augmented copies for training videos)
    arng = np.random.default_rng(1)
    for v in vids:
        v['feat'] = [FT.frame_features(s, aspect=v['aspect']) for s in v['segs']]
        v['aug'] = []
        if v['split'] == 'train':
            for _ in range(P['data']['aug_copies']):
                v['aug'].append([FT.frame_features(FT.augment_landmarks(s, arng), aspect=v['aspect']) for s in v['segs']])
        del v['segs']
    # federated partitions (14-class training videos): Dirichlet label skew
    fed = P['federated']
    tr14 = [v for v in vids if v['split'] == 'train' and v['action'] in CLASSES14]
    for attempt in range(100):
        prng = np.random.default_rng(100 + attempt)
        assign = {}
        for c in CLASSES14:
            lst = sorted([v['vid'] for v in tr14 if v['action'] == c]); prng.shuffle(lst)
            q = prng.dirichlet([fed['dirichlet_alpha']] * fed['clients'])
            cuts = (np.cumsum(q)[:-1] * len(lst)).astype(int)
            for k, part in enumerate(np.split(np.array(lst), cuts)):
                for vid in part:
                    assign[vid] = k + 1
        sizes = [sum(1 for a in assign.values() if a == k + 1) for k in range(fed['clients'])]
        if min(sizes) >= max(40, len(tr14) // 8):
            break
    P['federated']['partition_seed'] = 100 + attempt
    for v in vids:
        v['client'] = assign.get(v['vid'])
    # client test hold-out
    crng = np.random.default_rng(7)
    for k in range(1, fed['clients'] + 1):
        lst = sorted([v['vid'] for v in vids if v.get('client') == k]); crng.shuffle(lst)
        held = set(lst[:int(round(fed['client_test_share'] * len(lst)))])
        for v in vids:
            if v['vid'] in held:
                v['client_split'] = 'test'
            elif v.get('client') == k:
                v['client_split'] = 'train'
    import pickle
    with open(os.path.join(CACHE, 'videos.pkl'), 'wb') as f:
        pickle.dump(vids, f)
    stats = {'videos': len(vids), 'by_split': {s: sum(v['split'] == s for v in vids) for s in ('train', 'val', 'test')},
             'client_sizes': {k: {'train': sum(1 for v in vids if v.get('client') == k and v.get('client_split') == 'train'),
                                  'test': sum(1 for v in vids if v.get('client') == k and v.get('client_split') == 'test')}
                              for k in range(1, fed['clients'] + 1)},
             'client_class_counts': {k: {c: sum(1 for v in vids if v.get('client') == k and v['action'] == c) for c in CLASSES14}
                                     for k in range(1, fed['clients'] + 1)},
             'dropped_no_pose': no_pose}
    stats['pck'] = pck(meta)
    save_json(os.path.join(OUT, 'metrics', 'data_stats.json'), stats)
    log('data stats:', json.dumps({k: stats[k] for k in ('by_split', 'client_sizes')}))
    log('PCK@0.2:', stats['pck'].get(stats['pck']['convention_used']))


def load_videos():
    import pickle
    with open(os.path.join(CACHE, 'videos.pkl'), 'rb') as f:
        return pickle.load(f)


# tensors
def make_windows(vids, classes, feat_fn=None, aug=False, stride=None):
    """Return X (n,T,F), y (n,), video ids (n,)."""
    T = P['data']['T']; stride = stride or P['data']['stride']
    cidx = {c: i for i, c in enumerate(classes)}
    X, y, g = [], [], []
    for v in vids:
        if v['action'] not in cidx:
            continue
        seq_sets = [v['feat']] + (v['aug'] if aug else [])
        for segs in seq_sets:
            for s in segs:
                f = feat_fn(s) if feat_fn else s
                w = FT.windows(f, T, stride)
                X.append(w); y += [cidx[v['action']]] * len(w); g += [v['vid']] * len(w)
    if not X:
        return np.zeros((0, T, 46), np.float32), np.zeros(0, int), np.zeros(0, object)
    return np.concatenate(X), np.array(y), np.array(g)


def norm_stats(X):
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(0); sd = flat.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


# torch model
def torch_setup():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch, dev


def build_net(F, K, cell='lstm', pool='attn'):
    import torch, torch.nn as nn
    H, A, D, dr = P['model']['H'], P['model']['A'], P['model']['D'], P['model']['dropout']

    class HARNet(nn.Module):
        def __init__(s):
            super().__init__()
            R = nn.LSTM if cell == 'lstm' else nn.RNN
            s.rnn1 = R(F, H, batch_first=True, bidirectional=True)
            s.rnn2 = R(2 * H, H, batch_first=True, bidirectional=True)
            s.drop = nn.Dropout(dr); s.pool = pool
            s.att_W = nn.Linear(2 * H, A); s.att_v = nn.Linear(A, 1, bias=False)
            s.fc1 = nn.Linear(2 * H, D); s.out = nn.Linear(D, K)

        def forward(s, x, mask=None):
            h, _ = s.rnn1(x); h = s.drop(h); h, _ = s.rnn2(h)
            if s.pool == 'attn':
                e = s.att_v(torch.tanh(s.att_W(h))).squeeze(-1)
                a = torch.softmax(e, 1); z = (a.unsqueeze(-1) * h).sum(1)
            else:
                z = h.max(1).values
            z = s.drop(z)
            o = s.out(s.drop(torch.relu(s.fc1(z))))
            if mask is not None:
                o = o.masked_fill(~mask, -1e9)
            return o
    return HARNet()


def predict(net, X, dev, mask=None, batch=1024):
    import torch
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], device=dev)
            out.append(torch.softmax(net(xb, mask), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def macro_f1(y, pred, labels=None):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average='macro', labels=labels, zero_division=0))


def train_net(Xtr, ytr, Xva, yva, K, seed, cell='lstm', pool='attn', mask=None, max_epochs=None, init=None, log_every=10):
    """Centralised training with AdamW, warm-up + cosine, label smoothing,
    class-balanced sampling, gradient clipping and early stopping on validation macro-F1."""
    torch, dev = torch_setup()
    import torch.nn.functional as Fn
    tp = P['train']
    max_epochs = max_epochs or tp['max_epochs']
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_net(Xtr.shape[2], K, cell, pool).to(dev)
    if init is not None:
        net.load_state_dict(init)
    opt = torch.optim.AdamW(net.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])
    def lr_at(ep):
        if ep < tp['warmup_epochs']:
            return (ep + 1) / tp['warmup_epochs']
        prog = (ep - tp['warmup_epochs']) / max(1, max_epochs - tp['warmup_epochs'])
        return (tp['min_lr'] / tp['lr']) + (1 - tp['min_lr'] / tp['lr']) * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    Xt = torch.as_tensor(Xtr, device=dev); yt = torch.as_tensor(ytr, device=dev)
    counts = np.bincount(ytr, minlength=K).astype(float)
    w = torch.as_tensor(np.where(counts[ytr] > 0, 1.0 / counts[ytr], 0), device=dev, dtype=torch.float)
    mtorch = None if mask is None else torch.as_tensor(mask, device=dev)
    hist = []; best = (-1, None, 0); bad = 0
    for ep in range(max_epochs):
        net.train(); t0 = time.time(); tl = 0; nb = 0
        idx = torch.multinomial(w, len(ytr), replacement=True)
        for i in range(0, len(idx), tp['batch']):
            b = idx[i:i + tp['batch']]
            out = net(Xt[b], mtorch)
            loss = Fn.cross_entropy(out, yt[b], label_smoothing=tp['label_smoothing'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), tp['grad_clip']); opt.step()
            tl += float(loss); nb += 1
        sched.step()
        Pv = predict(net, Xva, dev, mtorch)
        vloss = float(-np.log(Pv[np.arange(len(yva)), yva] + 1e-9).mean())
        vf1 = macro_f1(yva, Pv.argmax(1)); vacc = float((Pv.argmax(1) == yva).mean())
        hist.append({'epoch': ep + 1, 'train_loss': tl / nb, 'val_loss': vloss, 'val_macro_f1': vf1, 'val_acc': vacc,
                     'lr': opt.param_groups[0]['lr'], 'sec': time.time() - t0})
        if vf1 > best[0]:
            best = (vf1, copy.deepcopy(net.state_dict()), ep + 1); bad = 0
        else:
            bad += 1
        if (ep + 1) % log_every == 0:
            log(f'  epoch {ep + 1}: train loss {tl / nb:.3f}  val macro-F1 {vf1:.3f}')
        if bad >= tp['patience']:
            break
    net.load_state_dict(best[1])
    return net, hist, best[2]


def full_metrics(y, Pm, K, vids, classes):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score,
                                 confusion_matrix, roc_auc_score, top_k_accuracy_score)
    pred = Pm.argmax(1); labels = list(range(K))
    m = {'window': {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=labels, zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', labels=labels, zero_division=0)),
        'macro_precision': float(precision_score(y, pred, average='macro', labels=labels, zero_division=0)),
        'macro_recall': float(recall_score(y, pred, average='macro', labels=labels, zero_division=0)),
        'top3_accuracy': float(top_k_accuracy_score(y, Pm, k=3, labels=labels)) if K > 3 else None,
        'kappa': float(cohen_kappa_score(y, pred)),
        'n': int(len(y))}}
    try:
        m['window']['macro_roc_auc'] = float(roc_auc_score(y, Pm, multi_class='ovr', average='macro', labels=labels))
    except Exception:
        m['window']['macro_roc_auc'] = None
    f1c = f1_score(y, pred, average=None, labels=labels, zero_division=0)
    pc = precision_score(y, pred, average=None, labels=labels, zero_division=0)
    rc = recall_score(y, pred, average=None, labels=labels, zero_division=0)
    m['per_class'] = {classes[i]: {'precision': float(pc[i]), 'recall': float(rc[i]), 'f1': float(f1c[i]),
                                   'support': int((y == i).sum())} for i in range(K)}
    m['confusion'] = confusion_matrix(y, pred, labels=labels).tolist()
    # video level: average window probabilities per video
    uv = np.unique(vids)
    vy, vp = [], []
    for v in uv:
        sel = vids == v
        vy.append(y[sel][0]); vp.append(Pm[sel].mean(0))
    vy = np.array(vy); vp = np.array(vp); vpred = vp.argmax(1)
    m['video'] = {'accuracy': float((vpred == vy).mean()),
                  'macro_f1': float(f1_score(vy, vpred, average='macro', labels=labels, zero_division=0)),
                  'kappa': float(cohen_kappa_score(vy, vpred)), 'n': int(len(vy))}
    return m


def net_to_np(net):
    return NPM.from_torch_state_dict(net.state_dict())


def np_to_net(net, p, dev):
    import torch
    sd = {k: torch.as_tensor(v, device=dev) for k, v in NPM.to_torch_arrays(p).items()}
    net.load_state_dict(sd)


# stage: central
def prepare_sets(classes, feat_mode='46'):
    vids = load_videos()
    fn = (lambda s: FT.paper_features(s)) if feat_mode == '9' else None
    tr = [v for v in vids if v['split'] == 'train']; va = [v for v in vids if v['split'] == 'val']; te = [v for v in vids if v['split'] == 'test']
    Xtr, ytr, gtr = make_windows(tr, classes, fn, aug=True)
    Xva, yva, gva = make_windows(va, classes, fn)
    Xte, yte, gte = make_windows(te, classes, fn)
    Xo, _, _ = make_windows(tr, classes, fn)     # statistics from original (non-augmented) training windows
    mu, sd = norm_stats(Xo)
    z = lambda X: ((X - mu) / sd).astype(np.float32)
    return dict(Xtr=z(Xtr), ytr=ytr, Xva=z(Xva), yva=yva, gva=gva, Xte=z(Xte), yte=yte, gte=gte, mu=mu, sd=sd)


def window_stats(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], 1)


def stage_central():
    torch, dev = torch_setup()
    log('device:', dev)
    results = {}
    configs = [('14', CLASSES14, 'bilstm_attn', 'lstm', 'attn', '46'), ('14', CLASSES14, 'birnn_attn', 'rnn', 'attn', '46'),
               ('14', CLASSES14, 'bilstm_maxpool', 'lstm', 'max', '46'), ('14', CLASSES14, 'bilstm_paper9d', 'lstm', 'attn', '9'),
               ('15', CLASSES15, 'bilstm_attn', 'lstm', 'attn', '46'), ('15', CLASSES15, 'birnn_attn', 'rnn', 'attn', '46')]
    sets = {}
    for ncls, classes, name, cell, pool, fm in configs:
        key = (ncls, fm)
        if key not in sets:
            sets[key] = prepare_sets(classes, fm)
            s = sets[key]
            log(f'{ncls} classes, {fm}-D: train windows {len(s["ytr"])}, val {len(s["yva"])}, test {len(s["yte"])}')
            np.savez(os.path.join(CACHE, f'norm_{ncls}_{fm}.npz'), mu=s['mu'], sd=s['sd'])
        s = sets[key]; K = len(classes)
        for seed in P['seed_list']:
            tag = f'{name}_{ncls}cls_seed{seed}'
            log('training', tag)
            t0 = time.time()
            net, hist, best_ep = train_net(s['Xtr'], s['ytr'], s['Xva'], s['yva'], K, seed, cell, pool)
            Pm = predict(net, s['Xte'], dev)
            met = full_metrics(s['yte'], Pm, K, s['gte'], classes)
            met.update(history=hist, best_epoch=best_ep, train_minutes=(time.time() - t0) / 60,
                       params=sum(p.numel() for p in net.parameters()))
            np.save(os.path.join(CACHE, f'pred_{tag}.npy'), Pm)
            torch.save(net.state_dict(), os.path.join(CACHE, f'net_{tag}.pt'))
            results[tag] = met
            log(f'  test window macro-F1 {met["window"]["macro_f1"]:.3f}  video acc {met["video"]["accuracy"]:.3f}  ({met["train_minutes"]:.1f} min)')
    # baselines
    from sklearn.ensemble import RandomForestClassifier
    for ncls, classes in (('14', CLASSES14), ('15', CLASSES15)):
        s = sets[(ncls, '46')]; K = len(classes)
        maj = np.bincount(s['ytr'], minlength=K).argmax()
        Pm = np.zeros((len(s['yte']), K)); Pm[:, maj] = 1
        results[f'majority_{ncls}cls'] = full_metrics(s['yte'], Pm, K, s['gte'], classes)
        log('random forest', ncls)
        n_tr = len(s['ytr']); sub = np.random.default_rng(0).choice(n_tr, size=min(n_tr, 20000), replace=False)
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, class_weight='balanced')
        rf.fit(window_stats(s['Xtr'][sub]), s['ytr'][sub])
        Pm = np.zeros((len(s['yte']), K)); Pm[:, rf.classes_] = rf.predict_proba(window_stats(s['Xte']))
        results[f'random_forest_{ncls}cls'] = full_metrics(s['yte'], Pm, K, s['gte'], classes)
    # McNemar Bi-LSTM vs Bi-RNN (14 classes, seed 0, windows)
    from scipy.stats import binomtest
    s = sets[('14', '46')]
    pa = np.load(os.path.join(CACHE, 'pred_bilstm_attn_14cls_seed0.npy')).argmax(1)
    pb = np.load(os.path.join(CACHE, 'pred_birnn_attn_14cls_seed0.npy')).argmax(1)
    b = int(((pa == s['yte']) & (pb != s['yte'])).sum()); c = int(((pa != s['yte']) & (pb == s['yte'])).sum())
    results['mcnemar_bilstm_vs_birnn'] = {'b_lstm_right_rnn_wrong': b, 'c_rnn_right_lstm_wrong': c,
                                         'p_value_exact': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}
    # efficiency: parameters, size, latency (torch CPU and NumPy runtime)
    eff = {}
    for name, cell, pool, F in (('bilstm_attn', 'lstm', 'attn', 46), ('birnn_attn', 'rnn', 'attn', 46), ('bilstm_paper9d', 'lstm', 'attn', 9)):
        net = build_net(F, 14, cell, pool).eval()
        x = torch.randn(1, P['data']['T'], F)
        with torch.no_grad():
            for _ in range(10): net(x)
            t0 = time.time()
            for _ in range(100): net(x)
        npar = sum(p.numel() for p in net.parameters())
        eff[name] = {'params': npar, 'size_mb_float32': npar * 4 / 1e6, 'torch_cpu_ms_per_window': (time.time() - t0) * 10}
    pnp = NPM.from_torch_state_dict(build_net(46, 14).state_dict())
    xw = np.random.randn(1, P['data']['T'], 46).astype(np.float32)
    t0 = time.time()
    for _ in range(50): NPM.forward(pnp, xw)
    eff['bilstm_attn']['numpy_runtime_ms_per_window'] = (time.time() - t0) * 20
    results['efficiency'] = eff
    save_json(os.path.join(OUT, 'metrics', 'central.json'), results)
    log('central stage done')


# incremental learning
class TorchIncLearner:
    """Mirror of the node's incremental learner: buffer of 16 windows -> one Adam step;
    optional replay (20% of each batch from a reservoir) and EWC."""

    def __init__(self, net, dev, replay=False, ewc=False, lam=1000.0, reservoir=None, mask=None, lr=None):
        import torch
        self.torch = torch; self.net = net; self.dev = dev
        self.opt = torch.optim.Adam(net.parameters(), lr=lr or P['incremental']['lr'])
        self.replay, self.ewc, self.lam = replay, ewc, lam
        self.res = reservoir; self.mask = mask
        self.fisher = None; self.anchor = None

    def consolidate(self, X, y):
        """Fisher information on (X, y) at the current weights; accumulate."""
        torch = self.torch
        n = min(len(X), P['incremental']['fisher_samples'])
        idx = np.random.default_rng(0).choice(len(X), n, replace=False)
        fis = {k: torch.zeros_like(p) for k, p in self.net.named_parameters()}
        self.net.eval()
        # cuDNN refuses RNN backward in eval mode; the native kernel allows it (and keeps dropout off)
        with torch.backends.cudnn.flags(enabled=False):
            for i in idx:
                out = self.net(torch.as_tensor(X[i:i + 1], device=self.dev), self.mask)
                loss = torch.nn.functional.cross_entropy(out, torch.as_tensor(y[i:i + 1], device=self.dev))
                self.net.zero_grad(); loss.backward()
                for k, p in self.net.named_parameters():
                    if p.grad is not None:
                        fis[k] += p.grad.detach() ** 2 / n
        if self.fisher is None:
            self.fisher = fis
        else:
            for k in fis: self.fisher[k] += fis[k]
        self.anchor = {k: p.detach().clone() for k, p in self.net.named_parameters()}

    def learn(self, X, y, passes=1):
        torch = self.torch
        B = P['incremental']['buffer']; nrep = int(round(P['incremental']['replay_ratio'] * B / (1 - P['incremental']['replay_ratio'])))
        rng = np.random.default_rng(len(X))
        for _ in range(passes):
            order = rng.permutation(len(X))
            for i in range(0, len(order) - B + 1, B):
                bi = order[i:i + B]
                xb, yb = X[bi], y[bi]
                if self.replay and self.res is not None and len(self.res) > 0:
                    rx, ry = self.res.sample(nrep)
                    xb = np.concatenate([xb, rx]); yb = np.concatenate([yb, ry])
                self.net.train()
                out = self.net(torch.as_tensor(xb, device=self.dev), self.mask)
                loss = torch.nn.functional.cross_entropy(out, torch.as_tensor(yb, device=self.dev), label_smoothing=P['train']['label_smoothing'])
                if self.ewc and self.fisher is not None:
                    pen = sum((self.fisher[k] * (p - self.anchor[k]) ** 2).sum() for k, p in self.net.named_parameters())
                    loss = loss + 0.5 * self.lam * pen
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); self.opt.step()


def cl_scores(acc_matrix, rand_acc, joint_acc):
    """Continual-learning metrics from an accuracy matrix R[k][j] (after task k, on task j)."""
    R = np.asarray(acc_matrix, float); K = len(R)
    A = [float(R[k, :k + 1].mean()) for k in range(K)]
    F = []
    for k in range(1, K):
        f = [R[:k, j].max() - R[k, j] for j in range(k)]
        F.append(float(np.mean(f)))
    bwt = float(np.mean([R[K - 1, j] - R[j, j] for j in range(K - 1)])) if K > 1 else 0.0
    fwt = float(np.mean([R[j - 1, j] - rand_acc[j] for j in range(1, K)])) if K > 1 else 0.0
    intr = [float(joint_acc[k] - R[k, k]) for k in range(K)]
    return {'average_accuracy': A, 'forgetting': F, 'forgetting_final': F[-1] if F else 0.0, 'bwt': bwt, 'fwt': fwt,
            'intransigence': intr, 'intransigence_mean': float(np.mean(intr[1:])) if K > 1 else 0.0}


def stage_cl():
    torch, dev = torch_setup()
    from btpcore.learn import Reservoir
    vids = load_videos()
    nz = np.load(os.path.join(CACHE, 'norm_14_46.npz')); mu, sd = nz['mu'], nz['sd']
    z = lambda X: ((X - mu) / sd).astype(np.float32)
    K = 14; inc = P['incremental']; out = {}

    def acc_f1(net, X, y, mask=None):
        if len(y) == 0:
            return 0.0, 0.0
        mt = None if mask is None else torch.as_tensor(mask, device=dev)
        pr = predict(net, X, dev, mt).argmax(1)
        return float((pr == y).mean()), macro_f1(y, pr, labels=sorted(set(y.tolist())))

    # Experiment A: a new client arrives
    log('CL experiment A (new client)')
    base_v = [v for v in vids if v.get('client') in (1, 2) and v.get('client_split') == 'train']
    new_v = [v for v in vids if v.get('client') == 3 and v.get('client_split') == 'train']
    new_te = [v for v in vids if v.get('client') == 3 and v.get('client_split') == 'test']
    val_v = [v for v in vids if v['split'] == 'val']; test_v = [v for v in vids if v['split'] == 'test']
    Xb, yb, _ = make_windows(base_v, CLASSES14, aug=True); Xb = z(Xb)
    Xn, yn, _ = make_windows(new_v, CLASSES14); Xn = z(Xn)
    Xnt, ynt, _ = make_windows(new_te, CLASSES14); Xnt = z(Xnt)
    Xva, yva, _ = make_windows(val_v, CLASSES14); Xva = z(Xva)
    Xte, yte, _ = make_windows(test_v, CLASSES14); Xte = z(Xte)
    base_net, _, _ = train_net(Xb, yb, Xva, yva, K, 0)
    base_sd = copy.deepcopy(base_net.state_dict())
    rand_net = build_net(46, K).to(dev)
    rand_acc = [acc_f1(rand_net, Xte, yte)[0], acc_f1(rand_net, Xnt, ynt)[0]]
    Xj = np.concatenate([Xb, Xn]); yj = np.concatenate([yb, yn])
    joint_net, _, _ = train_net(Xj, yj, Xva, yva, K, 0)
    joint_acc = [acc_f1(base_net, Xte, yte)[0], acc_f1(joint_net, Xnt, ynt)[0]]
    chunks = np.array_split(np.random.default_rng(0).permutation(len(yn)), inc['chunks_expA'])
    variants = {'no_update': None, 'naive': (False, False), 'replay': (True, False), 'ewc': (False, True), 'replay_ewc': (True, True)}
    expA = {}
    for vname, flags in variants.items():
        net = build_net(46, K).to(dev); net.load_state_dict(base_sd)
        res = Reservoir(inc['reservoir'], 46, P['data']['T']); res.add(Xb[np.random.default_rng(1).permutation(len(yb))], yb)
        L = TorchIncLearner(net, dev, *(flags or (False, False)), lam=inc['lambda_ewc'], reservoir=res)
        if flags and flags[1]:
            L.consolidate(Xb, yb)
        a0 = acc_f1(net, Xte, yte); n0 = acc_f1(net, Xnt, ynt)
        curve = [{'windows_seen': 0, 'base_acc': a0[0], 'base_f1': a0[1], 'new_acc': n0[0], 'new_f1': n0[1]}]
        seen = 0
        for ch in chunks:
            if flags is not None:
                L.learn(Xn[ch], yn[ch], passes=inc['passes_per_chunk'])
            seen += len(ch)
            a = acc_f1(net, Xte, yte); n = acc_f1(net, Xnt, ynt)
            curve.append({'windows_seen': seen, 'base_acc': a[0], 'base_f1': a[1], 'new_acc': n[0], 'new_f1': n[1]})
        R = [[a0[0], n0[0]], [curve[-1]['base_acc'], curve[-1]['new_acc']]]
        Rf = [[a0[1], n0[1]], [curve[-1]['base_f1'], curve[-1]['new_f1']]]
        sc = cl_scores(R, rand_acc, joint_acc)
        sc.update(accuracy_matrix=R, f1_matrix=Rf, curve=curve,
                  adaptation_gain_f1=curve[-1]['new_f1'] - curve[0]['new_f1'])
        expA[vname] = sc
        log(f'  {vname}: base acc {R[1][0]:.3f} (from {R[0][0]:.3f}), new-client acc {R[1][1]:.3f} (from {R[0][1]:.3f}), forgetting {sc["forgetting_final"]:.3f}')
    a = acc_f1(joint_net, Xte, yte); n = acc_f1(joint_net, Xnt, ynt)
    expA['joint_upper_bound'] = {'base_acc': a[0], 'new_acc': n[0], 'base_f1': a[1], 'new_f1': n[1]}
    expA['random_init_acc'] = rand_acc
    # EWC sensitivity
    sens = {}
    for lam in inc['lambda_ewc_sensitivity']:
        net = build_net(46, K).to(dev); net.load_state_dict(base_sd)
        res = Reservoir(inc['reservoir'], 46, P['data']['T']); res.add(Xb, yb)
        L = TorchIncLearner(net, dev, True, True, lam=lam, reservoir=res); L.consolidate(Xb, yb)
        for ch in chunks:
            L.learn(Xn[ch], yn[ch], passes=inc['passes_per_chunk'])
        sens[str(lam)] = {'base_acc': acc_f1(net, Xte, yte)[0], 'new_acc': acc_f1(net, Xnt, ynt)[0]}
    expA['ewc_lambda_sensitivity'] = sens
    out['A_new_client'] = expA

    # Experiment B: new classes arrive (class-incremental)
    log('CL experiment B (new classes)')
    order = list(np.random.default_rng(0).permutation(K))
    sizes = inc['class_tasks_expB']; tasks = []; s0 = 0
    for sz in sizes:
        tasks.append(order[s0:s0 + sz]); s0 += sz
    tr_v = [v for v in vids if v['split'] == 'train']
    Xall, yall, _ = make_windows(tr_v, CLASSES14, aug=True); Xall = z(Xall)
    Xo, yo, _ = make_windows(tr_v, CLASSES14); Xo = z(Xo)
    def sel(X, y, cls):
        m = np.isin(y, cls); return X[m], y[m]
    task_tr = [sel(Xall, yall, t) for t in tasks]; task_stream = [sel(Xo, yo, t) for t in tasks]
    task_te = [sel(Xte, yte, t) for t in tasks]; task_va = [sel(Xva, yva, t) for t in tasks]
    seen_masks = []
    for k in range(len(tasks)):
        m = np.zeros(K, bool); m[np.concatenate(tasks[:k + 1])] = True; seen_masks.append(m)
    base_net, _, _ = train_net(*task_tr[0], *task_va[0], K, 0, mask=seen_masks[0], max_epochs=min(40, P['train']['max_epochs']))
    base_sd = copy.deepcopy(base_net.state_dict())
    rand_net = build_net(46, K).to(dev)
    rand_acc = [acc_f1(rand_net, *task_te[j], seen_masks[j])[0] for j in range(len(tasks))]
    joint_acc = [acc_f1(base_net, *task_te[0], seen_masks[0])[0]]
    for k in range(1, len(tasks)):
        Xj = np.concatenate([task_tr[j][0] for j in range(k + 1)]); yj = np.concatenate([task_tr[j][1] for j in range(k + 1)])
        Xvj = np.concatenate([task_va[j][0] for j in range(k + 1)]); yvj = np.concatenate([task_va[j][1] for j in range(k + 1)])
        jn, _, _ = train_net(Xj, yj, Xvj, yvj, K, 0, mask=seen_masks[k], max_epochs=min(40, P['train']['max_epochs']))
        joint_acc.append(acc_f1(jn, *task_te[k], seen_masks[k])[0])
    expB = {'task_classes': [[CLASSES14[c] for c in t] for t in tasks]}
    for vname, flags in variants.items():
        net = build_net(46, K).to(dev); net.load_state_dict(base_sd)
        res = Reservoir(inc['reservoir'], 46, P['data']['T']); res.add(*task_stream[0])
        R = []; Rf = []
        for k in range(len(tasks)):
            mt = torch.as_tensor(seen_masks[k], device=dev)
            if k > 0:
                # new output units start as the average seen-class unit; random rows otherwise
                # dominate the larger trained features and every prediction collapses onto one new class
                with torch.no_grad():
                    old = torch.as_tensor(np.concatenate(tasks[:k]), device=dev); new = torch.as_tensor(tasks[k], device=dev)
                    net.out.weight[new] = net.out.weight[old].mean(0); net.out.bias[new] = net.out.bias[old].mean()
            if k > 0 and flags is not None:
                L.mask = mt
                L.learn(*task_stream[k], passes=inc['expB_passes'])
                res.add(*task_stream[k])
            if flags is not None:
                if k == 0:
                    L = TorchIncLearner(net, dev, flags[0], flags[1], lam=inc['lambda_ewc'], reservoir=res, mask=mt, lr=inc['expB_lr'])
                if flags[1]:
                    L.consolidate(*task_stream[k])
            row = [acc_f1(net, *task_te[j], seen_masks[k]) for j in range(len(tasks))]
            R.append([r[0] for r in row]); Rf.append([r[1] for r in row])
        sc = cl_scores(R, rand_acc, joint_acc); sc.update(accuracy_matrix=R, f1_matrix=Rf)
        expB[vname] = sc
        log(f'  {vname}: average accuracy {sc["average_accuracy"][-1]:.3f}, forgetting {sc["forgetting_final"]:.3f}, BWT {sc["bwt"]:.3f}')
    expB['joint_upper_bound_acc'] = joint_acc; expB['random_init_acc'] = rand_acc
    out['B_new_classes'] = expB
    save_json(os.path.join(OUT, 'metrics', 'incremental.json'), out)
    log('CL stage done')


# federated simulation
def stage_fl():
    torch, dev = torch_setup()
    import torch.nn.functional as Fn
    vids = load_videos(); fed = P['federated']; K = 14
    nz = np.load(os.path.join(CACHE, 'norm_14_46.npz')); mu, sd = nz['mu'], nz['sd']
    z = lambda X: ((X - mu) / sd).astype(np.float32)
    C = fed['clients']
    ctr, cte = {}, {}
    for k in range(1, C + 1):
        X, y, _ = make_windows([v for v in vids if v.get('client') == k and v.get('client_split') == 'train'], CLASSES14, aug=True)
        ctr[k] = (z(X), y)
        X, y, _ = make_windows([v for v in vids if v.get('client') == k and v.get('client_split') == 'test'], CLASSES14)
        cte[k] = (z(X), y)
    Xva, yva, _ = make_windows([v for v in vids if v['split'] == 'val'], CLASSES14); Xva = z(Xva)
    Xte, yte, gte = make_windows([v for v in vids if v['split'] == 'test'], CLASSES14); Xte = z(Xte)
    # server validation subset (stratified, capped) used for Shapley and before/after scores
    rng = np.random.default_rng(0); cap = fed['server_val_windows']
    per = max(1, cap // K); sv = []
    for c in range(K):
        ids = np.where(yva == c)[0]; rng.shuffle(ids); sv += list(ids[:per])
    sv = np.array(sorted(sv)); Xsv, ysv = Xva[sv], yva[sv]
    np.savez_compressed(os.path.join(CACHE, 'server_val.npz'), X=Xsv.astype(np.float16), y=ysv)
    evnet = build_net(46, K).to(dev)

    def eval_np(p, X, y):
        np_to_net(evnet, p, dev); pr = predict(evnet, X, dev).argmax(1)
        return macro_f1(y, pr, labels=list(range(K))), float((pr == y).mean())
    eval_fn = lambda p: eval_np(p, Xsv, ysv)[0]

    def local_train(p_start, X, y, epochs, anchor=None, coef=0.0, seed=0):
        net = build_net(46, K).to(dev); np_to_net(net, p_start, dev)
        opt = torch.optim.Adam(net.parameters(), lr=fed['local_lr'])
        Xt = torch.as_tensor(X, device=dev); yt = torch.as_tensor(y, device=dev)
        anc = None
        if anchor is not None and coef > 0:
            tmp = build_net(46, K).to(dev); np_to_net(tmp, anchor, dev)
            anc = {k: p.detach().clone() for k, p in tmp.named_parameters()}
        g = torch.Generator(device='cpu').manual_seed(seed)
        losses = []
        for _ in range(epochs):
            net.train(); perm = torch.randperm(len(y), generator=g)
            for i in range(0, len(perm), fed['local_batch']):
                b = perm[i:i + fed['local_batch']].to(dev)
                loss = Fn.cross_entropy(net(Xt[b]), yt[b], label_smoothing=P['train']['label_smoothing'])
                if anc is not None:
                    loss = loss + 0.5 * coef * sum(((p - anc[k]) ** 2).sum() for k, p in net.named_parameters())
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                losses.append(float(loss))
        return net_to_np(net), float(np.mean(losses)) if losses else None

    def run(method, eta=None, rounds=None, init=None, start_round=0, state=None, personal=None, history=None):
        rounds = rounds or fed['rounds']
        w = init if init is not None else NPM.from_torch_state_dict(build_net(46, K).state_dict())
        ours = method == 'ours'
        cfg = {'algorithm': 'fedadam' if ours else 'fedavg', 'eta': eta, 'beta1': fed['beta1'], 'beta2': fed['beta2'], 'tau': fed['tau']}
        state = state or (FL.FedAdamState(w) if ours else None)
        personal = personal or ({k: copy.deepcopy(w) for k in range(1, C + 1)} if ours else None)
        history = history or []
        for r in range(start_round + 1, start_round + rounds + 1):
            t0 = time.time(); updates = []
            for k in range(1, C + 1):
                wk, loss = local_train(w, *ctr[k], fed['local_epochs'], anchor=w, coef=fed['mu'] if ours else 0.0, seed=r * 10 + k)
                updates.append({'node': k, 'n': int(len(ctr[k][1])), 'weights': wk, 'metrics': {'local_loss': loss}})
                if ours:
                    personal[k], _ = local_train(personal[k], *ctr[k], fed['local_epochs'], anchor=w, coef=fed['ditto_lambda'], seed=r * 10 + k + 5)
            w, rec = FL.aggregate(w, updates, cfg, state=state, eval_fn=eval_fn, round_no=r)
            gte_f1, gte_acc = eval_np(w, Xte, yte)
            rec['global_test_macro_f1'] = gte_f1; rec['global_test_acc'] = gte_acc
            rec['client_eval'] = {}
            for k in range(1, C + 1):
                g_f1 = eval_np(w, *cte[k])[0]
                p_f1 = eval_np(personal[k], *cte[k])[0] if ours else g_f1
                rec['client_eval'][k] = {'global_f1': g_f1, 'personal_f1': p_f1}
            nbytes = sum(v.size for v in w.values()) * 4
            rec['bytes_round'] = nbytes * 2 * C
            rec['seconds'] = time.time() - t0
            for cl in rec['clients']:
                cl.pop('sent_first_values', None)
            for kk in ('delta_first_values', 'step_first_values', 'old_first_values', 'new_first_values'):
                rec.pop(kk, None)
            history.append(rec)
            log(f'  {method} eta={eta} round {r}: test macro-F1 {gte_f1:.3f}, shares ' + ', '.join(f'{c["share"]:.2f}' for c in rec['clients']) +
                ', Shapley ' + ', '.join(f'{c["shapley"]:+.3f}' for c in rec['clients']))
        return w, state, personal, history

    results = {}
    log('FL: server learning-rate search')
    cands = {}
    for eta in fed['eta_grid']:
        cands[eta] = run('ours', eta=eta, rounds=fed['eta_search_rounds'])
    best_eta = max(cands, key=lambda e: cands[e][3][-1]['val_after'])
    P['federated']['eta_selected'] = best_eta
    results['eta_search'] = {str(e): [h['val_after'] for h in cands[e][3]] for e in cands}
    log('selected eta', best_eta)
    w, state, personal, hist = cands[best_eta]
    w, state, personal, hist = run('ours', eta=best_eta, rounds=fed['rounds'] - fed['eta_search_rounds'], init=w,
                                   start_round=fed['eta_search_rounds'], state=state, personal=personal, history=hist)
    results['ours'] = hist
    log('FL: FedAvg baseline')
    wa, _, _, hist_a = run('fedavg')
    results['fedavg'] = hist_a
    # local-only lower bound
    log('FL: local-only')
    local = {}
    for k in range(1, C + 1):
        net, _, _ = train_net(*ctr[k], Xva, yva, K, 0, max_epochs=fed['rounds'] * fed['local_epochs'])
        p = net_to_np(net)
        local[k] = {'global_test_f1': eval_np(p, Xte, yte)[0], 'own_test_f1': eval_np(p, *cte[k])[0]}
    results['local_only'] = local
    central = load_json(os.path.join(OUT, 'metrics', 'central.json'))
    cf = [central[f'bilstm_attn_14cls_seed{s}']['window']['macro_f1'] for s in P['seed_list']]
    results['centralised_test_macro_f1_mean'] = float(np.mean(cf))
    for name in ('ours', 'fedavg'):
        target = 0.9 * results['centralised_test_macro_f1_mean']
        rr = [h['round'] for h in results[name] if h['global_test_macro_f1'] >= target]
        results[f'rounds_to_90pct_centralised_{name}'] = rr[0] if rr else None
    # sanity check: a client that sends noise
    log('FL: noisy-client sanity check')
    cfg = {'algorithm': 'fedadam', 'eta': best_eta, 'beta1': fed['beta1'], 'beta2': fed['beta2'], 'tau': fed['tau']}
    st = FL.FedAdamState(m={k: v.copy() for k, v in state.m.items()}, v={k: v.copy() for k, v in state.v.items()})
    wn = copy.deepcopy(w); noisy = []
    for r in range(3):
        ups = []
        for k in range(1, C + 1):
            wk, _ = local_train(wn, *ctr[k], 1, anchor=wn, coef=fed['mu'], seed=900 + r * 10 + k)
            if k == C:
                ref = FL.flat_norm(FL.sub(wk, wn))
                noise = {kk: np.random.default_rng(r).standard_normal(v.shape).astype(np.float32) for kk, v in wn.items()}
                sc = 3 * ref / FL.flat_norm(noise)
                wk = {kk: wn[kk] + sc * noise[kk] for kk in wn}
            ups.append({'node': k, 'n': int(len(ctr[k][1])), 'weights': wk})
        wn, rec = FL.aggregate(wn, ups, cfg, state=st, eval_fn=eval_fn, round_no=r + 1)
        noisy.append({'round': r + 1, 'clients': [{kk: c[kk] for kk in ('node', 'share', 'alignment', 'shapley', 'update_norm')} for c in rec['clients']],
                      'val_before': rec['val_before'], 'val_after': rec['val_after']})
    results['noisy_client_check'] = noisy
    save_json(os.path.join(OUT, 'metrics', 'federated.json'), results)
    # artefacts for deployment
    NPM.save(os.path.join(CACHE, 'fl_global.npz'), w)
    np.savez_compressed(os.path.join(CACHE, 'fl_fedadam_state.npz'), **state.to_npz_dict())
    for k in range(1, C + 1):
        NPM.save(os.path.join(CACHE, f'fl_personal_{k}.npz'), personal[k])
    log('FL stage done')


# export
def fisher_numpy(p, X, y, n=500):
    """Diagonal Fisher of the NumPy model (per-sample squared gradients), for the node's EWC term."""
    idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
    F = {k: np.zeros_like(v) for k, v in p.items()}
    for i in idx:
        _, g, _ = NPM.loss_and_grads(p, X[i:i + 1], y[i:i + 1], smoothing=0.0)
        for k in F:
            F[k] += g[k] ** 2 / len(idx)
    return F


def pack_videos(vids, classes, mu, sd):
    """Feature sequences (not windows) to keep node files small."""
    cidx = {c: i for i, c in enumerate(classes)}
    frames, offs, labels, names = [], [0], [], []
    for v in vids:
        if v['action'] not in cidx:
            continue
        for s in v['feat']:
            frames.append(((s - mu) / sd).astype(np.float16)); offs.append(offs[-1] + len(s))
            labels.append(cidx[v['action']]); names.append(v['vid'])
    return dict(frames=np.concatenate(frames) if frames else np.zeros((0, 46), np.float16),
                offsets=np.array(offs), labels=np.array(labels), videos=np.array(names))


def make_figures():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fdir = os.path.join(OUT, 'figures'); os.makedirs(fdir, exist_ok=True)
    central = load_json(os.path.join(OUT, 'metrics', 'central.json'))
    h = central['bilstm_attn_14cls_seed0']['history']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([x['train_loss'] for x in h], label='train'); ax[0].plot([x['val_loss'] for x in h], label='validation')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].legend()
    ax[1].plot([x['val_macro_f1'] for x in h], color='tab:green'); ax[1].set_xlabel('epoch'); ax[1].set_ylabel('validation macro-F1')
    fig.suptitle('Bi-LSTM training curves (14 classes, seed 0)'); fig.tight_layout(); fig.savefig(os.path.join(fdir, 'training_curves.png'), dpi=150); plt.close(fig)
    cm = np.array(central['bilstm_attn_14cls_seed0']['confusion'], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(14)); ax.set_xticklabels(CLASSES14, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(14)); ax.set_yticklabels(CLASSES14, fontsize=8)
    for i in range(14):
        for j in range(14):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if cmn[i, j] > 0.5 else 'black')
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); fig.colorbar(im, fraction=0.046); ax.set_title('Confusion matrix (row-normalised), Bi-LSTM')
    fig.tight_layout(); fig.savefig(os.path.join(fdir, 'confusion_matrix.png'), dpi=150); plt.close(fig)
    pcs = central['bilstm_attn_14cls_seed0']['per_class']
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.bar(range(14), [pcs[c]['f1'] for c in CLASSES14], color='tab:blue')
    ax.set_xticks(range(14)); ax.set_xticklabels(CLASSES14, rotation=60, ha='right', fontsize=8); ax.set_ylabel('F1'); ax.set_ylim(0, 1)
    ax.set_title('Per-class F1, Bi-LSTM'); fig.tight_layout(); fig.savefig(os.path.join(fdir, 'per_class_f1.png'), dpi=150); plt.close(fig)
    fed = load_json(os.path.join(OUT, 'metrics', 'federated.json'))
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, lab in (('ours', 'FedProx + FedAdam + Ditto (global)'), ('fedavg', 'FedAvg')):
        ax.plot([r['round'] for r in fed[name]], [r['global_test_macro_f1'] for r in fed[name]], marker='o', ms=3, label=lab)
    ax.axhline(fed['centralised_test_macro_f1_mean'], color='k', ls='--', label='centralised (upper bound)')
    lo = np.mean([v['global_test_f1'] for v in fed['local_only'].values()])
    ax.axhline(lo, color='gray', ls=':', label='local-only, mean of clients (lower bound)')
    ax.set_xlabel('round'); ax.set_ylabel('test window macro-F1'); ax.legend(fontsize=8); ax.set_title('Federated training')
    fig.tight_layout(); fig.savefig(os.path.join(fdir, 'fl_convergence.png'), dpi=150); plt.close(fig)
    rounds = fed['ours']
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    S = np.array([[c['share'] for c in r['clients']] for r in rounds]); bottom = np.zeros(len(rounds))
    cols = ['#2563eb', '#0f766e', '#6d28d9']
    for k in range(S.shape[1]):
        ax[0].bar([r['round'] for r in rounds], S[:, k], bottom=bottom, color=cols[k % 3], label=f'client {k + 1}'); bottom += S[:, k]
    ax[0].set_ylabel('share of aggregate update'); ax[0].set_xlabel('round'); ax[0].legend(fontsize=8); ax[0].set_title('Update share per round')
    Sh = np.array([[c['shapley'] for c in r['clients']] for r in rounds])
    for k in range(Sh.shape[1]):
        ax[1].plot([r['round'] for r in rounds], Sh[:, k], marker='o', ms=3, color=cols[k % 3], label=f'client {k + 1}')
    ax[1].axhline(0, color='k', lw=0.5); ax[1].set_ylabel('Shapley value (validation macro-F1)'); ax[1].set_xlabel('round'); ax[1].legend(fontsize=8)
    ax[1].set_title('Shapley value per round'); fig.tight_layout(); fig.savefig(os.path.join(fdir, 'fl_contributions.png'), dpi=150); plt.close(fig)
    inc = load_json(os.path.join(OUT, 'metrics', 'incremental.json'))
    for exp, title in (('A_new_client', 'Experiment A: new client'), ('B_new_classes', 'Experiment B: new classes')):
        vs = ['naive', 'replay', 'ewc', 'replay_ewc']
        fig, axs = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3))
        for ax, v in zip(axs, vs):
            R = np.array(inc[exp][v]['accuracy_matrix'])
            ax.imshow(R, cmap='Greens', vmin=0, vmax=1)
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=8)
            ax.set_title(v.replace('_', ' + ')); ax.set_xlabel('test task j'); ax.set_ylabel('after task k')
            ax.set_xticks(range(R.shape[1])); ax.set_yticks(range(R.shape[0]))
            ax.set_xticklabels(range(1, R.shape[1] + 1)); ax.set_yticklabels(range(1, R.shape[0] + 1))
        fig.suptitle(title + ': accuracy matrices'); fig.tight_layout()
        fig.savefig(os.path.join(fdir, f'cl_matrix_{exp}.png'), dpi=150); plt.close(fig)
    A = inc['A_new_client']
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for v in ['naive', 'replay', 'ewc', 'replay_ewc']:
        cv = A[v]['curve']
        ax.plot([c['windows_seen'] for c in cv], [c['new_f1'] for c in cv], marker='o', label=f'{v} (new client)')
    ax.plot([c['windows_seen'] for c in A['replay_ewc']['curve']], [c['base_f1'] for c in A['replay_ewc']['curve']], ls='--', label='replay_ewc (base test)')
    ax.set_xlabel('labelled windows from the new client'); ax.set_ylabel('macro-F1'); ax.legend(fontsize=7); ax.set_title('Adaptation curve')
    fig.tight_layout(); fig.savefig(os.path.join(fdir, 'cl_adaptation_curve.png'), dpi=150); plt.close(fig)


def ms(vals):
    vals = [v for v in vals if v is not None]
    return f'{100 * np.mean(vals):.1f} ± {100 * np.std(vals):.1f}' if vals else '—'


def make_tables():
    tdir = os.path.join(OUT, 'tables'); os.makedirs(tdir, exist_ok=True)
    c = load_json(os.path.join(OUT, 'metrics', 'central.json'))
    seeds = P['seed_list']; lines = []
    lines.append('## Main results (test set, mean ± std over seeds, %)\n')
    lines.append('| Model | Classes | Window acc | Window macro-F1 | Weighted F1 | Top-3 | ROC-AUC | Kappa | Video acc | Video macro-F1 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for ncls in ('14', '15'):
        names = ['bilstm_attn', 'birnn_attn'] + (['bilstm_maxpool', 'bilstm_paper9d'] if ncls == '14' else [])
        for n in names:
            rs = [c[f'{n}_{ncls}cls_seed{s}'] for s in seeds]
            lines.append(f'| {n} | {ncls} | ' + ' | '.join(ms([r['window'][k] for r in rs]) for k in ('accuracy', 'macro_f1', 'weighted_f1', 'top3_accuracy', 'macro_roc_auc', 'kappa'))
                         + ' | ' + ms([r['video']['accuracy'] for r in rs]) + ' | ' + ms([r['video']['macro_f1'] for r in rs]) + ' |')
        for b in ('majority', 'random_forest'):
            r = c[f'{b}_{ncls}cls']
            lines.append(f'| {b} | {ncls} | ' + ' | '.join(ms([r['window'][k]]) for k in ('accuracy', 'macro_f1', 'weighted_f1', 'top3_accuracy', 'macro_roc_auc', 'kappa'))
                         + ' | ' + ms([r['video']['accuracy']]) + ' | ' + ms([r['video']['macro_f1']]) + ' |')
    mc = c['mcnemar_bilstm_vs_birnn']
    lines.append(f"\nMcNemar (Bi-LSTM vs Bi-RNN, 14 classes, seed 0, test windows): b = {mc['b_lstm_right_rnn_wrong']}, c = {mc['c_rnn_right_lstm_wrong']}, exact p = {mc['p_value_exact']:.3g}\n")
    lines.append('## Efficiency\n\n| Model | Parameters | Size (MB, float32) | PyTorch CPU ms/window | NumPy runtime ms/window |\n|---|---|---|---|---|')
    for n, e in c['efficiency'].items():
        lines.append(f"| {n} | {e['params']:,} | {e['size_mb_float32']:.2f} | {e['torch_cpu_ms_per_window']:.2f} | {e.get('numpy_runtime_ms_per_window', float('nan')):.2f} |")
    pcs = c['bilstm_attn_14cls_seed0']['per_class']
    lines.append('\n## Per-class results (Bi-LSTM, 14 classes, seed 0)\n\n| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|')
    for k, v in pcs.items():
        lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    inc = load_json(os.path.join(OUT, 'metrics', 'incremental.json'))
    for exp, title in (('A_new_client', 'Incremental experiment A (new client)'), ('B_new_classes', 'Incremental experiment B (new classes)')):
        lines.append(f'\n## {title}\n\n| Variant | Final average accuracy | Forgetting score | BWT | FWT | Intransigence (mean) |\n|---|---|---|---|---|---|')
        for v in ('no_update', 'naive', 'replay', 'ewc', 'replay_ewc'):
            s = inc[exp][v]
            lines.append(f"| {v} | {s['average_accuracy'][-1]:.3f} | {s['forgetting_final']:.3f} | {s['bwt']:+.3f} | {s['fwt']:+.3f} | {s['intransigence_mean']:+.3f} |")
    fed = load_json(os.path.join(OUT, 'metrics', 'federated.json'))
    lines.append('\n## Federated learning\n')
    lo = fed['local_only']
    last_o, last_a = fed['ours'][-1], fed['fedavg'][-1]
    lines.append('| Method | Test macro-F1 (global, final round) | Mean of last 5 rounds | Mean personal F1 on own client test | Worst client F1 | Rounds to 90% of centralised |\n|---|---|---|---|---|---|')
    pf = [v['personal_f1'] for v in last_o['client_eval'].values()]; gf = [v['global_f1'] for v in last_a['client_eval'].values()]
    l5 = lambda h: f"{np.mean([r['global_test_macro_f1'] for r in h[-5:]]):.3f} ± {np.std([r['global_test_macro_f1'] for r in h[-5:]]):.3f}"
    lines.append(f"| FedProx + FedAdam + Ditto (eta {P['federated'].get('eta_selected')}) | {last_o['global_test_macro_f1']:.3f} | {l5(fed['ours'])} | {np.mean(pf):.3f} | {min(pf):.3f} | {fed.get('rounds_to_90pct_centralised_ours')} |")
    lines.append(f"| FedAvg | {last_a['global_test_macro_f1']:.3f} | {l5(fed['fedavg'])} | {np.mean(gf):.3f} | {min(gf):.3f} | {fed.get('rounds_to_90pct_centralised_fedavg')} |")
    lines.append(f"| Local-only (mean of clients) | {np.mean([v['global_test_f1'] for v in lo.values()]):.3f} | — | {np.mean([v['own_test_f1'] for v in lo.values()]):.3f} | {min(v['own_test_f1'] for v in lo.values()):.3f} | — |")
    lines.append(f"| Centralised (upper bound) | {fed['centralised_test_macro_f1_mean']:.3f} | — | — | — | — |")
    lines.append(f"\nBytes per round (3 clients, up and down): {last_o['bytes_round'] / 1e6:.2f} MB\n")
    lines.append('\n### Noisy-client sanity check (client 3 sends noise)\n\nShare and alignment measure influence on the aggregate, so a large random update scores high on both; Shapley measures usefulness on validation data.\n\n| Round | Client | Share | Alignment | Shapley |\n|---|---|---|---|---|')
    for r in fed['noisy_client_check']:
        for cl in r['clients']:
            lines.append(f"| {r['round']} | {cl['node']} | {cl['share']:+.3f} | {cl['alignment']:+.3f} | {cl['shapley']:+.4f} |")
    # comparison with published results
    v15 = [c[f'bilstm_attn_15cls_seed{s}']['video']['accuracy'] for s in seeds]
    lines.append('\n## Comparison with published Penn Action results (15 classes, video-level accuracy %)\n')
    lines.append('| Method | Input | Accuracy |\n|---|---|---|')
    for row in [('ACPS (Iqbal et al., FG 2017)', 'Pose', '79.0'), ('ST-AOG (Nie et al., CVPR 2015)', 'Pose + video', '85.5'), ('C3D (Tran et al., 2015)', 'Video', '86.0'),
                ('ST-GCN, from scratch', 'Skeleton', '89.6'), ('IDT-FV (Wang and Schmid, 2013)', 'Video', '92.0'), ('2s-AGCN, from scratch', 'Skeleton', '93.1'),
                ('PoTion (Choutas et al., CVPR 2018)', 'Pose heat maps', '93.6'), ('UNIK, from scratch', 'Skeleton', '94.0'), ('P-CNN (Cheron et al., ICCV 2015)', 'Pose + video', '95.3'),
                ('Luvizon et al. (CVPR 2018)', 'Pose', '97.4')]:
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append(f'| **Ours: Bi-LSTM + attention on MediaPipe skeleton** | MediaPipe skeleton | **{ms(v15)}** |')
    with open(os.path.join(tdir, 'report_tables.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log('tables written')


def demo_videos(meta, n_per_class=1):
    """A few Penn Action test clips as .mp4 for trying the upload flow (needs ffmpeg)."""
    if shutil.which('ffmpeg') is None:
        log('ffmpeg not found; skipping demo videos'); return
    d = os.path.join(OUT, 'demo_videos'); os.makedirs(d, exist_ok=True)
    done = {}
    for m in meta:
        if m['train'] or m['action'] not in CLASSES14 or done.get(m['action'], 0) >= n_per_class or m['nframes'] < 60:
            continue
        outp = os.path.join(d, f"{m['action']}_{m['vid']}.mp4")
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', os.path.join(m['frames_dir'], '%06d.jpg'),
                            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', outp])
        if r.returncode == 0:
            done[m['action']] = done.get(m['action'], 0) + 1


def stage_export():
    vids = load_videos(); meta = load_meta()
    nz = np.load(os.path.join(CACHE, 'norm_14_46.npz')); mu, sd = nz['mu'], nz['sd']
    mdir = os.path.join(OUT, 'models'); ddir = os.path.join(OUT, 'node_data'); os.makedirs(mdir, exist_ok=True); os.makedirs(ddir, exist_ok=True)
    gw = NPM.load(os.path.join(CACHE, 'fl_global.npz'))
    NPM.save(os.path.join(mdir, 'global.npz'), gw)
    shutil.copy(os.path.join(CACHE, 'fl_fedadam_state.npz'), os.path.join(mdir, 'fedadam_state.npz'))
    for k in range(1, P['federated']['clients'] + 1):
        shutil.copy(os.path.join(CACHE, f'fl_personal_{k}.npz'), os.path.join(mdir, f'personal_{k}.npz'))
    import torch
    central = load_json(os.path.join(OUT, 'metrics', 'central.json'))
    best_seed = max(P['seed_list'], key=lambda s: central[f'bilstm_attn_14cls_seed{s}']['window']['macro_f1'])
    sdict = torch.load(os.path.join(CACHE, f'net_bilstm_attn_14cls_seed{best_seed}.pt'), map_location='cpu')
    pc = NPM.from_torch_state_dict(sdict)
    NPM.save(os.path.join(mdir, 'central.npz'), pc)
    # check NumPy runtime == PyTorch on real windows
    net = build_net(46, 14); net.load_state_dict(sdict); net.eval()
    te = [v for v in vids if v['split'] == 'test'][:20]
    X, y, _ = make_windows(te, CLASSES14); X = ((X - mu) / sd).astype(np.float32)[:64]
    with torch.no_grad():
        pt = torch.softmax(net(torch.as_tensor(X)), 1).numpy()
    pn = NPM.predict_proba(pc, X)
    diff = float(np.abs(pt - pn).max())
    log('NumPy vs PyTorch max probability difference:', diff)
    assert diff < 1e-3, 'NumPy runtime does not match PyTorch'
    # node data
    for k in range(1, P['federated']['clients'] + 1):
        for part in ('train', 'test'):
            sel = [v for v in vids if v.get('client') == k and v.get('client_split') == part]
            np.savez_compressed(os.path.join(ddir, f'client_{k}_{part}.npz'), **pack_videos(sel, CLASSES14, mu, sd))
    shutil.copy(os.path.join(CACHE, 'server_val.npz'), os.path.join(ddir, 'server_val.npz'))
    # Fisher for EWC on the nodes (global model, union of client training data)
    tr = [v for v in vids if v.get('client_split') == 'train']
    Xf, yf, _ = make_windows(tr, CLASSES14); Xf = ((Xf - mu) / sd).astype(np.float32)
    NPM.save(os.path.join(mdir, 'fisher.npz'), fisher_numpy(gw, Xf, yf, P['incremental']['fisher_samples']))
    fed = load_json(os.path.join(OUT, 'metrics', 'federated.json'))
    cfg = {'classes': CLASSES14, 'feature_names': FT.FEATURE_NAMES, 'T': P['data']['T'], 'stride': P['data']['stride'], 'fps': FT.FPS,
           'norm_mean': mu.tolist(), 'norm_std': sd.tolist(), 'arch': dict(zip('FHADK', NPM.dims(gw))),
           'fedadam': {'eta': P['federated']['eta_selected'], 'beta1': P['federated']['beta1'], 'beta2': P['federated']['beta2'], 'tau': P['federated']['tau']},
           'mu': P['federated']['mu'], 'ditto_lambda': P['federated']['ditto_lambda'], 'lambda_ewc': P['incremental']['lambda_ewc'],
           'incremental': P['incremental'], 'start_round': P['federated']['rounds'],
           'global_test_macro_f1': fed['ours'][-1]['global_test_macro_f1'], 'central_best_seed': best_seed,
           'numpy_vs_torch_max_diff': diff, 'created': time.strftime('%Y-%m-%d %H:%M')}
    save_json(os.path.join(mdir, 'config.json'), cfg)
    # the last few FL rounds, so the admin page has history on first start
    save_json(os.path.join(mdir, 'fl_history.json'), fed['ours'])
    try:
        make_figures()
    except Exception as e:
        log('figure error:', e)
    make_tables()
    try:
        demo_videos(meta)
    except Exception as e:
        log('demo video error:', e)
    import platform, sklearn
    P['environment'] = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}
    times = load_json(os.path.join(CACHE, 'stage_times.json')) if os.path.exists(os.path.join(CACHE, 'stage_times.json')) else {}
    P['stage_minutes'] = times
    save_json(os.path.join(OUT, 'params.json'), P)
    zp = os.path.join(WORK, 'outputs.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT):
            for fn in files:
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, WORK))
    log(f'wrote {zp} ({os.path.getsize(zp) / 1e6:.1f} MB)')


STAGES = {'meta': stage_meta, 'data': stage_data, 'central': stage_central, 'cl': stage_cl, 'fl': stage_fl, 'export': stage_export}


def run_stage(name):
    t0 = time.time()
    STAGES[name]()
    tf = os.path.join(CACHE, 'stage_times.json')
    times = load_json(tf) if os.path.exists(tf) else {}
    times[name] = round((time.time() - t0) / 60, 1)
    save_json(tf, times)
    # keep parameters that stages fill in (e.g. selected eta) for later stages
    save_json(os.path.join(CACHE, 'params_partial.json'), P)


def restore_params():
    f = os.path.join(CACHE, 'params_partial.json')
    if os.path.exists(f):
        old = load_json(f)
        for k in ('federated',):
            for kk, vv in old.get(k, {}).items():
                if kk in ('eta_selected', 'partition_seed'):
                    P[k][kk] = vv


restore_params()

if __name__ == '__main__':
    for s in sys.argv[1:]:
        run_stage(s)
