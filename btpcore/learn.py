"""NumPy training utilities for the client node: Adam, FedProx local training,
Ditto personal updates, replay + EWC incremental learning, and metrics."""
import numpy as np
from . import model as M


class Adam:
    def __init__(self, params, lr=1e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, clip=1.0):
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
        scale = min(1.0, clip / (gn + 1e-12)) if clip else 1.0
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = (params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)
        return gn


def regularised_step(params, opt, X, y, anchor=None, mu=0.0, fisher=None, lam_ewc=0.0, smoothing=0.05):
    """One optimiser step on (X, y) with optional proximal term mu/2 ||w - anchor||^2
    (FedProx for the global copy, Ditto for the personal model) and EWC term
    lam/2 sum F (w - anchor)^2. Returns the loss parts."""
    loss, g, _ = M.loss_and_grads(params, X, y, smoothing)
    prox = ewc = 0.0
    if anchor is not None and mu > 0:
        for k in g:
            diff = params[k] - anchor[k]
            g[k] = g[k] + mu * diff
            prox += 0.5 * mu * float((diff ** 2).sum())
    if anchor is not None and fisher is not None and lam_ewc > 0:
        for k in g:
            diff = params[k] - anchor[k]
            g[k] = g[k] + lam_ewc * fisher[k] * diff
            ewc += 0.5 * lam_ewc * float((fisher[k] * diff ** 2).sum())
    gn = opt.step(params, g)
    return {'ce': loss, 'prox': prox, 'ewc': ewc, 'grad_norm': gn}


def macro_f1(y_true, y_pred, K):
    f1s = []
    for c in range(K):
        tp = np.sum((y_pred == c) & (y_true == c)); fp = np.sum((y_pred == c) & (y_true != c)); fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(params, X, y, K):
    if len(X) == 0:
        return {'acc': None, 'macro_f1': None, 'n': 0}
    P = M.predict_proba(params, X)
    pred = P.argmax(1)
    return {'acc': float((pred == y).mean()), 'macro_f1': macro_f1(y, pred, K), 'n': int(len(y))}


class Reservoir:
    """Fixed-size uniform sample of past windows (reservoir sampling)."""
    def __init__(self, size, F, T, seed=0):
        self.size, self.n_seen = size, 0
        self.X = np.zeros((size, T, F), np.float32); self.y = np.zeros(size, np.int64)
        self.rng = np.random.default_rng(seed)

    def add(self, X, y):
        for xi, yi in zip(X, y):
            if self.n_seen < self.size:
                self.X[self.n_seen] = xi; self.y[self.n_seen] = yi
            else:
                j = self.rng.integers(0, self.n_seen + 1)
                if j < self.size:
                    self.X[j] = xi; self.y[j] = yi
            self.n_seen += 1

    def __len__(self):
        return min(self.n_seen, self.size)

    def sample(self, k):
        n = len(self)
        if n == 0 or k == 0:
            return None, None
        idx = self.rng.choice(n, size=min(k, n), replace=False)
        return self.X[idx], self.y[idx]
