"""Server-side aggregation (FedAvg / FedAdam) and per-client contribution measures.

Pure NumPy, used both by the Kaggle simulation and by the live server, so the
numbers in the report and on the admin page come from the same code.

Notation (as in the plan, Section 7.7):
  w_t     current global weights
  w_k     weights returned by client k, d_k = w_k - w_t
  p_k     n_k / N (data weight)
  c_k     p_k * d_k (weighted contribution)
  Delta   sum_k c_k (aggregate update, FedAvg's step)
  share_k <c_k, Delta> / ||Delta||^2   (shares sum to 1 exactly)
  align_k cos(d_k, Delta)
  phi_k   exact Shapley value of client k for validation macro-F1
"""
import itertools
import math
import numpy as np


def flat_norm(d):
    return float(math.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in d.values())))


def dot(a, b):
    return float(sum(float((a[k].astype(np.float64) * b[k].astype(np.float64)).sum()) for k in a))


def sub(a, b):
    return {k: (a[k] - b[k]).astype(np.float32) for k in a}


def sample_values(arr, n=6):
    """A few concrete numbers of a tensor, for display (first n of the flattened array)."""
    return [round(float(x), 6) for x in np.asarray(arr).ravel()[:n]]


def tensor_summary(params, n=6):
    """Per-tensor summary shown on the pages: shape, L2 norm, mean, and first values."""
    out = {}
    for k, v in params.items():
        out[k] = {'shape': list(v.shape), 'norm': round(float(np.linalg.norm(v)), 6),
                  'mean': round(float(v.mean()), 6), 'first': sample_values(v, n)}
    return out


def layer_groups(names):
    """Group tensor names into human layers for per-layer views."""
    groups = {'Bi-LSTM 1': [], 'Bi-LSTM 2': [], 'Attention': [], 'Dense + output': []}
    for k in names:
        if k.startswith('l0'): groups['Bi-LSTM 1'].append(k)
        elif k.startswith('l1'): groups['Bi-LSTM 2'].append(k)
        elif k.startswith('att'): groups['Attention'].append(k)
        else: groups['Dense + output'].append(k)
    return groups


class FedAdamState:
    def __init__(self, params=None, m=None, v=None):
        self.m = m if m is not None else {k: np.zeros_like(x) for k, x in params.items()}
        self.v = v if v is not None else {k: np.zeros_like(x) for k, x in params.items()}

    def to_npz_dict(self):
        d = {f'm__{k}': x for k, x in self.m.items()}
        d.update({f'v__{k}': x for k, x in self.v.items()})
        return d

    @classmethod
    def from_npz_dict(cls, d):
        m = {k[3:]: np.asarray(x, np.float32) for k, x in d.items() if k.startswith('m__')}
        v = {k[3:]: np.asarray(x, np.float32) for k, x in d.items() if k.startswith('v__')}
        return cls(m=m, v=v)


def weighted_delta(w_t, updates, subset=None):
    """Delta for the given subset of client indices (weights renormalised)."""
    idx = list(range(len(updates))) if subset is None else list(subset)
    N = sum(updates[i]['n'] for i in idx)
    delta = {k: np.zeros_like(v) for k, v in w_t.items()}
    for i in idx:
        pk = updates[i]['n'] / N
        for k in delta:
            delta[k] += (pk * (updates[i]['weights'][k] - w_t[k])).astype(np.float32)
    return delta


def adam_step(w_t, delta, state, eta, b1, b2, tau, commit=False):
    """FedAdam (Reddi et al., ICLR 2021), no bias correction. Returns new weights,
    the step, and the new moments. Moments are written to `state` only if commit."""
    new_w, step, m_new, v_new = {}, {}, {}, {}
    for k in w_t:
        m = b1 * state.m[k] + (1 - b1) * delta[k]
        v = b2 * state.v[k] + (1 - b2) * delta[k] * delta[k]
        s = eta * m / (np.sqrt(v) + tau)
        new_w[k] = (w_t[k] + s).astype(np.float32)
        step[k] = s.astype(np.float32); m_new[k] = m.astype(np.float32); v_new[k] = v.astype(np.float32)
    if commit:
        state.m, state.v = m_new, v_new
    return new_w, step, m_new, v_new


def aggregate(w_t, updates, cfg, state=None, eval_fn=None, round_no=None, progress=None):
    """One aggregation round.

    updates: list of {'node': id, 'n': n_k, 'weights': dict, 'metrics': {...}}
    cfg: {'algorithm': 'fedadam'|'fedavg', 'eta', 'beta1', 'beta2', 'tau'}
    eval_fn: optional callable(weights) -> validation macro-F1, used for the
             before/after score and the exact Shapley values.
    Returns (new_weights, record). `record` holds every number the admin and
    client pages display for this round.
    """
    names = list(w_t.keys())
    groups = layer_groups(names)
    K = len(updates)
    N = sum(u['n'] for u in updates)
    d = [sub(u['weights'], w_t) for u in updates]
    p = [u['n'] / N for u in updates]
    c = [{k: (p[i] * d[i][k]).astype(np.float32) for k in names} for i in range(K)]
    delta = {k: sum(c[i][k] for i in range(K)).astype(np.float32) for k in names}
    dd = dot(delta, delta)

    clients = []
    for i, u in enumerate(updates):
        share = dot(c[i], delta) / dd if dd > 0 else 1.0 / K
        nd = flat_norm(d[i]); ndel = math.sqrt(dd)
        align = dot(d[i], delta) / (nd * ndel) if nd > 0 and ndel > 0 else 0.0
        per_layer = {}
        for gname, ks in groups.items():
            dl = sum(float((delta[k].astype(np.float64) ** 2).sum()) for k in ks)
            cl = sum(float((c[i][k].astype(np.float64) * delta[k]).sum()) for k in ks)
            per_layer[gname] = cl / dl if dl > 0 else 1.0 / K
        clients.append({
            'node': u['node'], 'n': int(u['n']), 'p': p[i],
            'update_norm': nd,
            'layer_update_norms': {g: math.sqrt(sum(float((d[i][k] ** 2).sum()) for k in ks)) for g, ks in groups.items()},
            'share': share, 'layer_share': per_layer, 'alignment': align,
            'metrics': u.get('metrics', {}),
            'sent_first_values': {k: sample_values(u['weights'][k], 4) for k in names[:3]},
        })

    algo = cfg.get('algorithm', 'fedadam')
    if algo == 'fedadam':
        if state is None:
            state = FedAdamState(w_t)
        new_w, step, m_new, v_new = adam_step(w_t, delta, state, cfg['eta'], cfg['beta1'], cfg['beta2'], cfg['tau'], commit=False)
    else:
        new_w = {k: (w_t[k] + delta[k]).astype(np.float32) for k in names}
        step = delta; m_new = v_new = None

    rec = {
        'round': round_no, 'algorithm': algo, 'N': int(N), 'clients': clients,
        'delta_norm': math.sqrt(dd),
        'layer_delta_norms': {g: math.sqrt(sum(float((delta[k] ** 2).sum()) for k in ks)) for g, ks in groups.items()},
        'layer_step_mean_abs': {g: float(np.mean(np.concatenate([np.abs(step[k]).ravel() for k in ks]))) for g, ks in groups.items()},
        'step_norm': flat_norm(step),
        'delta_first_values': {k: sample_values(delta[k], 4) for k in names[:3]},
        'step_first_values': {k: sample_values(step[k], 4) for k in names[:3]},
        'old_first_values': {k: sample_values(w_t[k], 4) for k in names[:3]},
        'new_first_values': {k: sample_values(new_w[k], 4) for k in names[:3]},
        'shares_sum': float(sum(cl['share'] for cl in clients)),
    }
    # one parameter followed through every step, so the arithmetic can be shown with real numbers
    tk, ti = cfg.get('trace', ('out_b', 0))
    if tk in w_t:
        f = lambda arr: float(np.asarray(arr).ravel()[ti])
        tr = {'tensor': tk, 'index': ti, 'w_t': f(w_t[tk]),
              'clients': [{'node': u['node'], 'w_k': f(u['weights'][tk]), 'd_k': f(d[i][tk]), 'p_k': p[i], 'c_k': f(c[i][tk])}
                          for i, u in enumerate(updates)],
              'delta': f(delta[tk]), 'step': f(step[tk]), 'w_new': f(new_w[tk])}
        if algo == 'fedadam':
            tr.update(m_old=f(state.m[tk]), v_old=f(state.v[tk]), m_new=f(m_new[tk]), v_new=f(v_new[tk]))
        rec['trace'] = tr
    if algo == 'fedadam':
        rec['adam'] = {'eta': cfg['eta'], 'beta1': cfg['beta1'], 'beta2': cfg['beta2'], 'tau': cfg['tau'],
                       'm_norm': flat_norm(m_new), 'v_mean': float(np.mean(np.concatenate([x.ravel() for x in v_new.values()])))}

    if eval_fn is not None:
        cache = {}
        def F(S):
            S = tuple(sorted(S))
            if S not in cache:
                if progress:
                    progress([updates[j]['node'] for j in S])
                if not S:
                    cache[S] = eval_fn(w_t)
                elif len(S) == K:
                    cache[S] = eval_fn(new_w)
                else:
                    dS = weighted_delta(w_t, updates, S)
                    if algo == 'fedadam':
                        wS, _, _, _ = adam_step(w_t, dS, state, cfg['eta'], cfg['beta1'], cfg['beta2'], cfg['tau'], commit=False)
                    else:
                        wS = {k: w_t[k] + dS[k] for k in names}
                    cache[S] = eval_fn(wS)
            return cache[S]
        phis = []
        for i in range(K):
            others = [j for j in range(K) if j != i]
            phi = 0.0
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    wgt = math.factorial(len(S)) * math.factorial(K - len(S) - 1) / math.factorial(K)
                    phi += wgt * (F(S + (i,)) - F(S))
            phis.append(phi)
        for i, cl in enumerate(clients):
            cl['shapley'] = phis[i]
            cl['loo'] = F(tuple(range(K))) - F(tuple(j for j in range(K) if j != i))
        rec['val_before'] = F(())
        rec['val_after'] = F(tuple(range(K)))
        rec['shapley_sum'] = float(sum(phis))
        rec['subset_scores'] = {','.join(str(updates[j]['node']) for j in S) or 'none': v for S, v in cache.items()}
        if progress:
            progress(None)

    if algo == 'fedadam':
        state.m, state.v = m_new, v_new
    return new_w, rec
