"""Bidirectional LSTM classifier with attention pooling, in pure NumPy.

Forward pass for live inference and a full backward pass (BPTT) so the client
node can fine-tune the model without PyTorch. This keeps the runtime at a few
tens of MB of RAM. Weights are trained on Kaggle with PyTorch and converted
with `from_torch_state_dict`; the notebook checks that both give the same
outputs.

Architecture (same as the PyTorch model):
  x (B,T,F) -> BiLSTM(H) -> BiLSTM(H) -> attention pooling -> Dense(D)+ReLU -> Dense(K)
Gate order is PyTorch's: input, forget, cell (g), output.
"""
import numpy as np

LAYERS = 2
DIRS = ('f', 'b')


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # numerically stable sigmoid


def param_names():
    names = []
    for l in range(LAYERS):
        for d in DIRS:
            names += [f'l{l}{d}_Wx', f'l{l}{d}_Wh', f'l{l}{d}_b']
    return names + ['att_W', 'att_b', 'att_v', 'fc1_W', 'fc1_b', 'out_W', 'out_b']


def init_params(F, H, A, D, K, seed=0):
    rng = np.random.default_rng(seed)
    p = {}
    for l in range(LAYERS):
        fin = F if l == 0 else 2 * H
        s = 1.0 / np.sqrt(H)
        for d in DIRS:
            p[f'l{l}{d}_Wx'] = rng.uniform(-s, s, (fin, 4 * H)).astype(np.float32)
            p[f'l{l}{d}_Wh'] = rng.uniform(-s, s, (H, 4 * H)).astype(np.float32)
            p[f'l{l}{d}_b'] = np.zeros(4 * H, np.float32)
    def lin(i, o):
        s = 1.0 / np.sqrt(i)
        return rng.uniform(-s, s, (i, o)).astype(np.float32), np.zeros(o, np.float32)
    p['att_W'], p['att_b'] = lin(2 * H, A)
    p['att_v'] = rng.uniform(-1 / np.sqrt(A), 1 / np.sqrt(A), A).astype(np.float32)
    p['fc1_W'], p['fc1_b'] = lin(2 * H, D)
    p['out_W'], p['out_b'] = lin(D, K)
    return p


def dims(p):
    F = p['l0f_Wx'].shape[0]; H = p['l0f_Wh'].shape[0]
    A = p['att_W'].shape[1]; D = p['fc1_W'].shape[1]; K = p['out_W'].shape[1]
    return F, H, A, D, K


def n_params(p):
    return int(sum(v.size for v in p.values()))


# LSTM pieces
def _lstm_dir(X, Wx, Wh, b, cache):
    B, T, _ = X.shape
    H = Wh.shape[0]
    XW = X @ Wx + b                       # (B,T,4H)
    h = np.zeros((B, H), np.float32); c = np.zeros((B, H), np.float32)
    out = np.empty((B, T, H), np.float32)
    if cache is not None:
        cache.update(X=X, gates=np.empty((B, T, 4 * H), np.float32),
                     cs=np.empty((B, T + 1, H), np.float32), hs=np.empty((B, T + 1, H), np.float32))
        cache['cs'][:, 0] = 0; cache['hs'][:, 0] = 0
    for t in range(T):
        z = XW[:, t] + h @ Wh
        i = _sig(z[:, :H]); f = _sig(z[:, H:2 * H]); g = np.tanh(z[:, 2 * H:3 * H]); o = _sig(z[:, 3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[:, t] = h
        if cache is not None:
            cache['gates'][:, t] = np.concatenate([i, f, g, o], 1)
            cache['cs'][:, t + 1] = c; cache['hs'][:, t + 1] = h
    return out


def _lstm_dir_back(dOut, Wx, Wh, cache):
    X, G, cs, hs = cache['X'], cache['gates'], cache['cs'], cache['hs']
    B, T, H = dOut.shape
    dXW = np.empty((B, T, 4 * H), np.float32)
    dWh = np.zeros_like(Wh)
    dh_next = np.zeros((B, H), np.float32); dc_next = np.zeros((B, H), np.float32)
    for t in range(T - 1, -1, -1):
        i, f, g, o = G[:, t, :H], G[:, t, H:2 * H], G[:, t, 2 * H:3 * H], G[:, t, 3 * H:]
        c, c_prev, h_prev = cs[:, t + 1], cs[:, t], hs[:, t]
        tc = np.tanh(c)
        dh = dOut[:, t] + dh_next
        do = dh * tc
        dc = dh * o * (1 - tc * tc) + dc_next
        di = dc * g; dg = dc * i; df = dc * c_prev
        dc_next = dc * f
        dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g * g), do * o * (1 - o)], 1)
        dXW[:, t] = dz
        dWh += h_prev.T @ dz
        dh_next = dz @ Wh.T
    F = X.shape[2]
    dWx = X.reshape(-1, F).T @ dXW.reshape(-1, 4 * H)
    db = dXW.sum((0, 1))
    dX = dXW @ Wx.T
    return dX, dWx, dWh, db


# full model
def forward(p, X, cache=None):
    """X: (B,T,F) standardised features. Returns logits (B,K)."""
    X = np.asarray(X, np.float32)
    inp = X
    for l in range(LAYERS):
        cf = {} if cache is not None else None
        cb = {} if cache is not None else None
        of = _lstm_dir(inp, p[f'l{l}f_Wx'], p[f'l{l}f_Wh'], p[f'l{l}f_b'], cf)
        ob = _lstm_dir(inp[:, ::-1], p[f'l{l}b_Wx'], p[f'l{l}b_Wh'], p[f'l{l}b_b'], cb)[:, ::-1]
        if cache is not None:
            cache[f'l{l}f'] = cf; cache[f'l{l}b'] = cb
        inp = np.concatenate([of, ob], 2)
    O = inp                                                   # (B,T,2H)
    U = np.tanh(O @ p['att_W'] + p['att_b'])                  # (B,T,A)
    e = U @ p['att_v']                                        # (B,T)
    e = e - e.max(1, keepdims=True)
    a = np.exp(e); a /= a.sum(1, keepdims=True)
    z = (a[..., None] * O).sum(1)                             # (B,2H)
    pre1 = z @ p['fc1_W'] + p['fc1_b']
    h1 = np.maximum(pre1, 0)
    logits = h1 @ p['out_W'] + p['out_b']
    if cache is not None:
        cache.update(O=O, U=U, alpha=a, z=z, pre1=pre1, h1=h1)
    return logits


def softmax(logits):
    e = logits - logits.max(1, keepdims=True)
    e = np.exp(e)
    return e / e.sum(1, keepdims=True)


def predict_proba(p, X, batch=256):
    X = np.asarray(X, np.float32)
    return np.concatenate([softmax(forward(p, X[i:i + batch])) for i in range(0, len(X), batch)], 0)


def attention_weights(p, X):
    cache = {}
    forward(p, X, cache)
    return cache['alpha']


def backward(p, cache, dlogits):
    """Gradients of the loss w.r.t. every parameter, given dL/dlogits."""
    g = {}
    h1, pre1, z, O, U, a = cache['h1'], cache['pre1'], cache['z'], cache['O'], cache['U'], cache['alpha']
    g['out_W'] = h1.T @ dlogits; g['out_b'] = dlogits.sum(0)
    dh1 = dlogits @ p['out_W'].T
    dpre1 = dh1 * (pre1 > 0)
    g['fc1_W'] = z.T @ dpre1; g['fc1_b'] = dpre1.sum(0)
    dz = dpre1 @ p['fc1_W'].T                                 # (B,2H)
    da = np.einsum('btd,bd->bt', O, dz)
    dO = a[..., None] * dz[:, None, :]
    de = a * (da - (a * da).sum(1, keepdims=True))
    g['att_v'] = np.einsum('bta,bt->a', U, de)
    dU = de[..., None] * p['att_v']
    dpre = dU * (1 - U * U)
    D2 = O.shape[2]
    g['att_W'] = O.reshape(-1, D2).T @ dpre.reshape(-1, dpre.shape[2])
    g['att_b'] = dpre.sum((0, 1))
    dO = dO + dpre @ p['att_W'].T
    dinp = dO
    for l in range(LAYERS - 1, -1, -1):
        H = p[f'l{l}f_Wh'].shape[0]
        dF, dB = dinp[..., :H], dinp[..., H:]
        dXf, g[f'l{l}f_Wx'], g[f'l{l}f_Wh'], g[f'l{l}f_b'] = _lstm_dir_back(dF, p[f'l{l}f_Wx'], p[f'l{l}f_Wh'], cache[f'l{l}f'])
        dXb, g[f'l{l}b_Wx'], g[f'l{l}b_Wh'], g[f'l{l}b_b'] = _lstm_dir_back(np.ascontiguousarray(dB[:, ::-1]), p[f'l{l}b_Wx'], p[f'l{l}b_Wh'], cache[f'l{l}b'])
        dinp = dXf + dXb[:, ::-1]
    return {k: v.astype(np.float32) for k, v in g.items()}


def loss_and_grads(p, X, y, smoothing=0.05, class_weight=None):
    """Cross-entropy with label smoothing. Returns (loss, grads, probs)."""
    cache = {}
    logits = forward(p, X, cache)
    B, K = logits.shape
    P = softmax(logits)
    q = np.full((B, K), smoothing / K, np.float32)
    q[np.arange(B), y] += 1.0 - smoothing
    w = np.ones(B, np.float32) if class_weight is None else class_weight[y]
    loss = float(-(w * (q * np.log(P + 1e-9)).sum(1)).sum() / w.sum())
    dlogits = ((P - q) * w[:, None] / w.sum()).astype(np.float32)
    return loss, backward(p, cache, dlogits), P


# conversion
def from_torch_state_dict(sd):
    """Map a PyTorch HARNet state_dict (tensors or arrays) to NumPy parameter names."""
    t = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)).astype(np.float32) for k, v in sd.items()}
    p = {}
    for l in range(LAYERS):
        for d, suf in (('f', ''), ('b', '_reverse')):
            p[f'l{l}{d}_Wx'] = t[f'rnn{l + 1}.weight_ih_l0{suf}'].T.copy()
            p[f'l{l}{d}_Wh'] = t[f'rnn{l + 1}.weight_hh_l0{suf}'].T.copy()
            p[f'l{l}{d}_b'] = t[f'rnn{l + 1}.bias_ih_l0{suf}'] + t[f'rnn{l + 1}.bias_hh_l0{suf}']
    p['att_W'] = t['att_W.weight'].T.copy(); p['att_b'] = t['att_W.bias']
    p['att_v'] = t['att_v.weight'][0].copy()
    p['fc1_W'] = t['fc1.weight'].T.copy(); p['fc1_b'] = t['fc1.bias']
    p['out_W'] = t['out.weight'].T.copy(); p['out_b'] = t['out.bias']
    return p


def to_torch_arrays(p):
    """Inverse of from_torch_state_dict (biases go to bias_ih; bias_hh = 0)."""
    sd = {}
    for l in range(LAYERS):
        for d, suf in (('f', ''), ('b', '_reverse')):
            sd[f'rnn{l + 1}.weight_ih_l0{suf}'] = p[f'l{l}{d}_Wx'].T.copy()
            sd[f'rnn{l + 1}.weight_hh_l0{suf}'] = p[f'l{l}{d}_Wh'].T.copy()
            sd[f'rnn{l + 1}.bias_ih_l0{suf}'] = p[f'l{l}{d}_b'].copy()
            sd[f'rnn{l + 1}.bias_hh_l0{suf}'] = np.zeros_like(p[f'l{l}{d}_b'])
    sd['att_W.weight'] = p['att_W'].T.copy(); sd['att_W.bias'] = p['att_b'].copy()
    sd['att_v.weight'] = p['att_v'][None].copy()
    sd['fc1.weight'] = p['fc1_W'].T.copy(); sd['fc1.bias'] = p['fc1_b'].copy()
    sd['out.weight'] = p['out_W'].T.copy(); sd['out.bias'] = p['out_b'].copy()
    return sd


def save(path, p):
    np.savez_compressed(path, **p)


def load(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}
