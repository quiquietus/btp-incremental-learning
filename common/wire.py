"""Encoding of model weights for HTTP transfer, and shared loading helpers."""
import base64, json, os
import numpy as np

from btpcore import model as NPM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get('MODELS_DIR', os.path.join(ROOT, 'models'))
NODE_DATA = os.environ.get('NODE_DATA_DIR', os.path.join(ROOT, 'models', 'node_data'))

CLASSES14 = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowling', 'clean_and_jerk', 'golf_swing',
             'jump_rope', 'jumping_jacks', 'pull_ups', 'push_ups', 'sit_ups', 'squats', 'tennis_forehand', 'tennis_serve']


def encode_weights(p):
    return {k: {'shape': list(v.shape), 'b64': base64.b64encode(np.ascontiguousarray(v, np.float32).tobytes()).decode()} for k, v in p.items()}


def decode_weights(d):
    return {k: np.frombuffer(base64.b64decode(v['b64']), np.float32).reshape(v['shape']).copy() for k, v in d.items()}


def payload_bytes(p):
    return int(sum(v.size for v in p.values()) * 4)


def load_config():
    """config.json from the Kaggle outputs, or a placeholder so the system runs before training."""
    f = os.path.join(MODELS, 'config.json')
    if os.path.exists(f):
        with open(f) as fh:
            cfg = json.load(fh)
        cfg['placeholder'] = False
        return cfg
    return {'classes': CLASSES14, 'T': 48, 'stride': 4, 'fps': 30.0, 'norm_mean': [0.0] * 46, 'norm_std': [1.0] * 46,
            'arch': {'F': 46, 'H': 64, 'A': 64, 'D': 64, 'K': 14},
            'fedadam': {'eta': 0.01, 'beta1': 0.9, 'beta2': 0.99, 'tau': 0.001},
            'mu': 0.01, 'ditto_lambda': 0.1, 'lambda_ewc': 1000.0, 'start_round': 0, 'placeholder': True}


def load_model(name, cfg):
    f = os.path.join(MODELS, name)
    if os.path.exists(f):
        return NPM.load(f)
    a = cfg['arch']
    return NPM.init_params(a['F'], a['H'], a['A'], a['D'], a['K'], seed=0)


def load_sequences(path):
    """Feature sequences packed by the notebook -> list of (frames, label)."""
    if not os.path.exists(path):
        return []
    z = np.load(path)
    fr, off, lab = z['frames'].astype(np.float32), z['offsets'], z['labels']
    return [(fr[off[i]:off[i + 1]], int(lab[i])) for i in range(len(lab))]


def seq_windows(seqs, T=48, stride=8):
    from btpcore.features import windows
    X, y = [], []
    for f, l in seqs:
        w = windows(f, T, stride); X.append(w); y += [l] * len(w)
    if not X:
        return np.zeros((0, T, 46), np.float32), np.zeros(0, np.int64)
    return np.concatenate(X), np.array(y, np.int64)
