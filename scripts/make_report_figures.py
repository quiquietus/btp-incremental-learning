"""Test the trained models and draw every figure used in the report and slides.

Usage (from the repository root, after installing outputs.zip into models/ and report/):
    python scripts/make_report_figures.py                     # all figures
    python scripts/make_report_figures.py --skip-video        # no MediaPipe / OpenCV needed
    python scripts/make_report_figures.py --out my_figures --dpi 300

Needs numpy and matplotlib; the pose figures also need opencv-python and mediapipe.
Writes PNG files to --out and the test results to --out/test_results.{json,csv}.
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btpcore import model as M  # noqa: E402
from btpcore import features as FT  # noqa: E402

NAVY, SKY, OR, TEAL, PUR, GR = '#0a1f3c', '#1d9bd7', '#f0652f', '#0e9384', '#7c3aed', '#9aa5b1'
SOFT, NAVYB = '#eef4fb', '#18406f'
CLIENT_COLORS = ['#1d6fd6', '#0e9384', '#8b3fd9']
VARIANTS = ['naive', 'replay', 'ewc', 'replay_ewc']
VNAME = {'naive': 'Fine-tuning', 'replay': 'Replay', 'ewc': 'EWC', 'replay_ewc': 'Replay + EWC'}
POSE_TASK_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
                 'pose_landmarker_full/float16/latest/pose_landmarker_full.task')

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight'})


# ---------------------------------------------------------------- helpers
def jload(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def pretty(name):
    return name.replace('_', ' ')


def save(fig, name):
    fig.savefig(os.path.join(ARGS.out, name), dpi=ARGS.dpi)
    plt.close(fig)
    print('  wrote', name)


def seeds(central, prefix):
    return [v for k, v in central.items() if k.startswith(prefix + '_seed')]


def mean_std(runs, level, key):
    v = np.array([r[level][key] for r in runs]) * 100
    return float(v.mean()), float(v.std())


def load_videos(split_names):
    """Return list of (raw feature array [frames, 46], label) from models/node_data."""
    out = []
    for n in split_names:
        d = np.load(os.path.join(ROOT, 'models', 'node_data', n + '.npz'))
        fr = d['frames'].astype(np.float32) * SD + MU  # undo the standardisation
        for i, lab in enumerate(d['labels']):
            out.append((fr[d['offsets'][i]:d['offsets'][i + 1]], int(lab)))
    return out


def make_windows(videos, T=48, stride=4):
    X, y, vid = [], [], []
    for v, (f, lab) in enumerate(videos):
        f = (f - MU) / SD
        if len(f) < T:
            f = np.concatenate([f, np.repeat(f[-1:], T - len(f), 0)])
        for s in range(0, len(f) - T + 1, stride):
            X.append(f[s:s + T]); y.append(lab); vid.append(v)
    return np.array(X, np.float32), np.array(y), np.array(vid)


def macro_f1(y, p, labels):
    """Mean F1 over the given classes; a class with no true positive counts as 0."""
    f = []
    for k in labels:
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


# ---------------------------------------------------------------- 1. test the models
def test_models(central):
    """Evaluate the federated models on the client test sets.

    Two macro-F1 values are reported. f1_present averages over the classes that occur in that
    test set. f1_14 averages over all 14 classes (absent classes count as 0), which is how
    federated.json scores clients, so it matches the numbers in the metrics files.
    The central model is not run here: the client data comes from its training split, so its
    scores are taken from central.json (Penn Action test split) instead.
    """
    print('Testing models on the client test sets ...')
    K = len(CL)
    tests = {k: load_videos([f'client_{k}_test']) for k in (1, 2, 3)}
    tests['all'] = tests[1] + tests[2] + tests[3]
    models = ['global', 'personal_1', 'personal_2', 'personal_3']
    rows = []
    for runs, name in ((seeds(central, 'bilstm_attn_14cls'), 'central (3 seeds)'), (seeds(central, 'birnn_attn_14cls'), 'Bi-RNN (3 seeds)')):
        rows.append({'model': name, 'test_set': 'penn_action_test', 'windows': runs[0]['window']['n'],
                     'window_acc': round(mean_std(runs, 'window', 'accuracy')[0] / 100, 4),
                     'f1_present': round(mean_std(runs, 'window', 'macro_f1')[0] / 100, 4),
                     'f1_14': round(mean_std(runs, 'window', 'macro_f1')[0] / 100, 4),
                     'videos': runs[0]['video']['n'], 'video_acc': round(mean_std(runs, 'video', 'accuracy')[0] / 100, 4)})
    for mname in models:
        p = M.load(os.path.join(ROOT, 'models', mname + '.npz'))
        for tname, vids in tests.items():
            X, y, v = make_windows(vids)
            P = M.predict_proba(p, X)
            pred = P.argmax(1)
            vp = np.array([P[v == i].mean(0).argmax() for i in range(len(vids))])
            vy = np.array([lab for _, lab in vids])
            rows.append({'model': mname, 'test_set': f'client_{tname}' if tname != 'all' else 'all_clients',
                         'windows': len(y), 'window_acc': round(float((pred == y).mean()), 4),
                         'f1_present': round(macro_f1(y, pred, sorted(set(y.tolist()))), 4),
                         'f1_14': round(macro_f1(y, pred, range(K)), 4),
                         'videos': len(vids), 'video_acc': round(float((vp == vy).mean()), 4)})
    with open(os.path.join(ARGS.out, 'test_results.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(ARGS.out, 'test_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  {'model':18s} {'test set':17s} {'win acc':>8s} {'F1 present':>11s} {'F1 (14)':>8s} {'video acc':>10s}")
    for r in rows:
        print(f"  {r['model']:18s} {r['test_set']:17s} {r['window_acc']:8.3f} {r['f1_present']:11.3f} {r['f1_14']:8.3f} {r['video_acc']:10.3f}")
    return rows


# ---------------------------------------------------------------- 2. dataset and EDA figures
def fig_class_distribution():
    def count(part):  # unique videos per class (a video can hold several pose segments)
        seen = set()
        for k in (1, 2, 3):
            d = np.load(os.path.join(ROOT, 'models', 'node_data', f'client_{k}_{part}.npz'))
            seen |= set(zip(d['videos'].tolist(), d['labels'].tolist()))
        return np.bincount([l for _, l in seen], minlength=len(CL))
    ctr, cte = count('train'), count('test')
    o = np.argsort(ctr + cte)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(CL))
    ax.bar(x, ctr[o], color=NAVY, label=f'Client training ({ctr.sum()} videos)')
    ax.bar(x, cte[o], bottom=ctr[o], color=SKY, label=f'Client test ({cte.sum()} videos)')
    for i, j in enumerate(o):
        ax.text(i, ctr[j] + cte[j] + 1, str(ctr[j] + cte[j]), ha='center', fontsize=7.5)
    ax.set_xticks(x, [pretty(CL[j]) for j in o], rotation=40, ha='right', fontsize=8.5)
    ax.set_ylabel('Number of videos'); ax.legend(frameon=False)
    save(fig, 'class_distribution.png')


def fig_client_partition():
    D = jload(ROOT, 'report', 'metrics', 'data_stats.json'); cc = D['client_class_counts']
    cls = list(next(iter(cc.values())))
    fig, ax = plt.subplots(figsize=(8, 3.2)); x = np.arange(len(cls)); w = .27
    for i, k in enumerate(['1', '2', '3']):
        ax.bar(x + (i - 1) * w, [cc[k][n] for n in cls], w, color=CLIENT_COLORS[i],
               label=f'Client {k} ({D["client_sizes"][k]["train"]} training clips)')
    ax.set_xticks(x, [pretty(n) for n in cls], rotation=40, ha='right', fontsize=8.5)
    ax.set_ylabel('Training clips'); ax.legend(frameon=False, fontsize=8)
    save(fig, 'client_partition.png')


def fig_pck():
    D = jload(ROOT, 'report', 'metrics', 'data_stats.json')['pck']; pj = D['per_joint']
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.bar(range(len(pj)), list(pj.values()), color=TEAL)
    ax.set_xticks(range(len(pj)), [pretty(k) for k in pj], rotation=40, ha='right', fontsize=8.5)
    ax.set_ylim(.8, 1); ax.set_ylabel('PCK@0.2')
    ax.axhline(D['swapped'], ls='--', color=OR, lw=1)
    ax.text(len(pj) - .5, D['swapped'] + .004, f"mean {D['swapped']:.3f}", ha='right', fontsize=8, color=OR)
    save(fig, 'pck.png')


def fig_feature_distributions(videos):
    """Box plots of six features per class (the 'feature statistics' figure)."""
    feats = ['ang_r_knee', 'ang_r_elbow', 'ang_r_shoulder', 'ang_r_hip', 'speed_r_wrist', 'torso_incl']
    labels = ['Right knee angle (deg)', 'Right elbow angle (deg)', 'Right shoulder angle (deg)',
              'Right hip angle (deg)', 'Right wrist speed (torso lengths/s)', 'Torso inclination (deg)']
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for ax, f, lab in zip(axs.flat, feats, labels):
        j = FN.index(f)
        data = [np.concatenate([v[:, j] for v, l in videos if l == k]) for k in range(len(CL))]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.6, medianprops={'color': OR})
        for b in bp['boxes']:
            b.set(facecolor=SOFT, edgecolor=NAVYB)
        ax.set_title(lab, fontsize=9.5)
    for ax in axs[1]:
        ax.set_xticks(range(1, len(CL) + 1), [pretty(c) for c in CL], rotation=60, ha='right', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'feature_distributions.png')


def fig_feature_correlation(videos):
    X = np.concatenate([v for v, _ in videos])
    X = X[~np.isnan(X).any(1)]
    C = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(FN)), FN, rotation=90, fontsize=6); ax.set_yticks(range(len(FN)), FN, fontsize=6)
    for b in (24, 32, 40, 44):
        ax.axhline(b - .5, color='k', lw=.6); ax.axvline(b - .5, color='k', lw=.6)
    fig.colorbar(im, fraction=.046, label='Pearson correlation')
    save(fig, 'feature_correlation.png')


def fig_angle_trajectories(videos, n=50):
    """Mean knee and elbow angle over normalised time, one panel per class."""
    jk, je = FN.index('ang_r_knee'), FN.index('ang_r_elbow')
    fig, axs = plt.subplots(3, 5, figsize=(12, 6.6), sharex=True, sharey=True)
    t = np.linspace(0, 1, n)
    for k, ax in enumerate(axs.flat):
        if k >= len(CL):
            ax.axis('off'); continue
        for j, c, lab in ((jk, TEAL, 'knee'), (je, OR, 'elbow')):
            R = np.array([np.interp(t, np.linspace(0, 1, len(v)), v[:, j]) for v, l in videos if l == k and len(v) > 4])
            m, s = R.mean(0), R.std(0)
            ax.plot(t, m, color=c, label=f'right {lab}'); ax.fill_between(t, m - s, m + s, color=c, alpha=.18)
        ax.set_title(pretty(CL[k]), fontsize=9)
    axs[0, 0].legend(frameon=False, fontsize=7.5)
    for ax in axs[:, 0]:
        ax.set_ylabel('angle (deg)')
    for ax in axs[-1]:
        ax.set_xlabel('normalised time')
    fig.tight_layout()
    save(fig, 'angle_trajectories.png')


def fig_phase_portraits(videos):
    """Knee angle against knee angular velocity for each class."""
    ja, jv = FN.index('ang_r_knee'), FN.index('angvel_r_knee')
    fig, axs = plt.subplots(3, 5, figsize=(12, 6.6), sharex=True, sharey=True)
    for k, ax in enumerate(axs.flat):
        if k >= len(CL):
            ax.axis('off'); continue
        vs = [v for v, l in videos if l == k][:12]
        for v in vs:
            ax.plot(v[:, ja], v[:, jv], color=CLIENT_COLORS[k % 3], lw=.6, alpha=.55)
        ax.set_title(pretty(CL[k]), fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel('knee ang. vel. (deg/s)')
    for ax in axs[-1]:
        ax.set_xlabel('knee angle (deg)')
    axs[0, 0].set_ylim(-1200, 1200)
    fig.tight_layout()
    save(fig, 'phase_portraits.png')


def fig_temporal_autocorr(videos, max_lag=60):
    """Average autocorrelation of feature groups against lag, which motivates T = 48."""
    groups = {'Joint positions': range(0, 24), 'Joint angles': range(24, 32), 'Angular velocities': range(32, 40),
              'Limb speeds': range(40, 44), 'Torso inclination': range(44, 46)}
    lags = np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (g, idx), c in zip(groups.items(), [NAVY, TEAL, OR, PUR, SKY]):
        acs = []
        for v, _ in videos:
            if len(v) <= max_lag + 5:
                continue
            for j in idx:
                x = v[:, j] - np.nanmean(v[:, j]); den = np.nansum(x * x)
                if den > 0:
                    acs.append([np.nansum(x[:len(x) - L] * x[L:]) / den for L in lags])
        ax.plot(lags / 30, np.mean(acs, 0), color=c, label=g)
    ax.axvline(48 / 30, color=GR, ls='--', lw=1); ax.text(48 / 30 + .03, .9, 'window length\nT = 48 (1.6 s)', fontsize=8)
    ax.axhline(0, color='k', lw=.5)
    ax.set_xlabel('lag (s)'); ax.set_ylabel('mean autocorrelation'); ax.legend(frameon=False, fontsize=8)
    save(fig, 'temporal_autocorr.png')


def fig_feature_profiles(videos):
    """Class-by-feature heat map of z-scored means."""
    Mn = np.array([np.nanmean(np.concatenate([v for v, l in videos if l == k]), 0) for k in range(len(CL))])
    Z = (Mn - Mn.mean(0)) / (Mn.std(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
    ax.set_yticks(range(len(CL)), [pretty(c) for c in CL], fontsize=8)
    ax.set_xticks(range(len(FN)), FN, rotation=90, fontsize=6.5)
    fig.colorbar(im, fraction=.025, label='class mean (z-score across classes)')
    save(fig, 'feature_profiles.png')


# ---------------------------------------------------------------- 3. results figures from the metrics
def fig_training_curves(central):
    h = central['bilstm_attn_14cls_seed0']['history']; hr = central['birnn_attn_14cls_seed0']['history']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot([x['train_loss'] for x in h], color=NAVY, label='Bi-LSTM, training')
    ax[0].plot([x['val_loss'] for x in h], color=NAVY, ls='--', label='Bi-LSTM, validation')
    ax[0].plot([x['train_loss'] for x in hr], color=OR, label='Bi-RNN, training')
    ax[0].plot([x['val_loss'] for x in hr], color=OR, ls='--', label='Bi-RNN, validation')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot([x['val_macro_f1'] for x in h], color=NAVY, label='Bi-LSTM')
    ax[1].plot([x['val_macro_f1'] for x in hr], color=OR, label='Bi-RNN')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('validation macro-F1'); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'training_curves.png')


def fig_confusion(central):
    cm = np.array(central['bilstm_attn_14cls_seed0']['confusion'], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1); K = len(cm)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K), [pretty(c) for c in CL], rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(K), [pretty(c) for c in CL], fontsize=8)
    for i in range(K):
        for j in range(K):
            if cmn[i, j] >= .05:
                ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if cmn[i, j] > .5 else 'black')
    ax.set_xlabel('predicted class'); ax.set_ylabel('true class')
    fig.colorbar(im, fraction=.046)
    save(fig, 'confusion_matrix.png')


def fig_per_class_f1(central):
    runs = seeds(central, 'bilstm_attn_14cls')
    f = np.array([[r['per_class'][c]['f1'] for c in CL] for r in runs])
    o = np.argsort(f.mean(0))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(range(len(CL)), f.mean(0)[o], yerr=f.std(0)[o], capsize=2.5,
           color=[OR if f.mean(0)[j] < .7 else SKY for j in o])
    ax.set_xticks(range(len(CL)), [pretty(CL[j]) for j in o], rotation=40, ha='right', fontsize=8.5)
    ax.set_ylabel('F1 (mean of 3 seeds)'); ax.set_ylim(0, 1)
    save(fig, 'per_class_f1.png')


def fig_ablation(central):
    rows = [('Majority\nclass', 'majority_14cls'), ('Bi-LSTM\n(9 features)', 'bilstm_paper9d_14cls'),
            ('Bi-RNN\n+ attention', 'birnn_attn_14cls'), ('Bi-LSTM\n+ attention', 'bilstm_attn_14cls'),
            ('Bi-LSTM\n+ max-pool', 'bilstm_maxpool_14cls'), ('Random\nforest', 'random_forest_14cls')]
    f1, f1s, va, vas = [], [], [], []
    for _, key in rows:
        runs = seeds(central, key) or [central[key]]
        a, b = mean_std(runs, 'window', 'macro_f1'); c, d = mean_std(runs, 'video', 'accuracy')
        f1.append(a); f1s.append(b); va.append(c); vas.append(d)
    x = np.arange(len(rows)); w = .38
    fig, ax = plt.subplots(figsize=(8, 3.5))
    b1 = ax.bar(x - w / 2, f1, w, yerr=f1s, color=SKY, label='Window macro-F1', capsize=3)
    b2 = ax.bar(x + w / 2, va, w, yerr=vas, color=NAVY, label='Video accuracy', capsize=3)
    b1[3].set_color(OR)
    for r, v in zip(list(b1) + list(b2), f1 + va):
        ax.text(r.get_x() + r.get_width() / 2, v + 2, f'{v:.1f}', ha='center', fontsize=7.5)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5); ax.set_ylabel('% on the test set (14 classes)')
    ax.set_ylim(0, 100); ax.legend(frameon=False, loc='upper left')
    save(fig, 'ablation.png')


def fig_published(central):
    ours = mean_std(seeds(central, 'bilstm_attn_15cls'), 'video', 'accuracy')[0]
    pub = [('ACPS', 79.0), ('ST-AOG', 85.5), ('C3D', 86.0), ('Ours', ours), ('ST-GCN', 89.6), ('IDT-FV', 92.0),
           ('2s-AGCN', 93.1), ('PoTion', 93.6), ('UNIK', 94.0), ('P-CNN', 95.3), ('Luvizon et al.', 97.4)]
    pub.sort(key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=(7, 3.6)); ys = np.arange(len(pub))
    ax.barh(ys, [p[1] for p in pub], color=[OR if p[0] == 'Ours' else GR for p in pub])
    for y, (_, v) in zip(ys, pub):
        ax.text(v + .3, y, f'{v:.1f}', va='center', fontsize=8)
    ax.set_yticks(ys, [p[0] for p in pub]); ax.set_xlim(70, 100)
    ax.set_xlabel('Video accuracy on Penn Action, 15 classes (%)')
    save(fig, 'published.png')


def fig_cl_matrices(inc):
    for exp in ('A_new_client', 'B_new_classes'):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3.1))
        for ax, v in zip(axs, VARIANTS):
            R = np.array(inc[exp][v]['accuracy_matrix'])
            ax.imshow(R, cmap='Greens', vmin=0, vmax=1)
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                            color='white' if R[i, j] > .6 else 'black')
            ax.set_title(VNAME[v], fontsize=10); ax.set_xlabel('test task $j$'); ax.set_ylabel('after task $k$')
            ax.set_xticks(range(R.shape[1]), range(1, R.shape[1] + 1)); ax.set_yticks(range(R.shape[0]), range(1, R.shape[0] + 1))
        fig.tight_layout()
        save(fig, f'cl_matrix_{exp}.png')


def fig_cl_adaptation(inc):
    A = inc['A_new_client']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for v, c in zip(VARIANTS, [GR, SKY, PUR, OR]):
        cv = A[v]['curve']; n = [p['windows_seen'] for p in cv]
        ax[0].plot(n, [p['new_f1'] for p in cv], marker='o', ms=3.5, color=c, label=VNAME[v])
        ax[1].plot(n, [p['base_f1'] for p in cv], marker='o', ms=3.5, color=c, label=VNAME[v])
    ax[0].set_title('New client', fontsize=10); ax[1].set_title('Original test data', fontsize=10)
    for a in ax:
        a.set_xlabel('labelled windows from the new client'); a.set_ylabel('macro-F1')
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'cl_adaptation_curve.png')


def fig_fl_convergence(fed):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, lab, c in (('ours', 'FedProx + FedAdam + Ditto', OR), ('fedavg', 'FedAvg', NAVY)):
        ax.plot([r['round'] for r in fed[name]], [r['global_test_macro_f1'] for r in fed[name]],
                marker='o', ms=3, color=c, label=lab)
    ax.axhline(fed['centralised_test_macro_f1_mean'], color='k', ls='--', lw=1, label='Centralised training')
    lo = np.mean([v['global_test_f1'] for v in fed['local_only'].values()])
    ax.axhline(lo, color=GR, ls=':', label='Local training only (mean of clients)')
    ax.set_xlabel('round'); ax.set_ylabel('global test macro-F1'); ax.legend(frameon=False, fontsize=8)
    save(fig, 'fl_convergence.png')


def fig_fl_contributions(fed):
    rounds = fed['ours']; r_ = [r['round'] for r in rounds]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    S = np.array([[c['share'] for c in r['clients']] for r in rounds]); bottom = np.zeros(len(rounds))
    for k in range(S.shape[1]):
        ax[0].bar(r_, S[:, k], bottom=bottom, color=CLIENT_COLORS[k], label=f'Client {k + 1}'); bottom += S[:, k]
    ax[0].set_ylabel('update share'); ax[0].set_title('Update share', fontsize=10)
    for k in range(S.shape[1]):
        ax[1].plot(r_, [r['clients'][k]['alignment'] for r in rounds], color=CLIENT_COLORS[k], marker='o', ms=2.5)
        ax[2].plot(r_, [r['clients'][k]['shapley'] for r in rounds], color=CLIENT_COLORS[k], marker='o', ms=2.5)
    ax[1].set_ylabel('cosine'); ax[1].set_title('Alignment with the aggregate', fontsize=10)
    ax[2].axhline(0, color='k', lw=.5); ax[2].set_ylabel('validation macro-F1'); ax[2].set_title('Shapley value', fontsize=10)
    for a in ax:
        a.set_xlabel('round')
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'fl_contributions.png')


def fig_eta_search(fed):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (e, v), c in zip(fed['eta_search'].items(), [GR, SKY, OR, PUR]):
        ax.plot(range(1, len(v) + 1), v, marker='o', ms=3.5, color=c, lw=2.2 if e == '0.03' else 1.4, label=f'$\\eta$ = {e}')
    ax.set_xlabel('round'); ax.set_ylabel('validation macro-F1'); ax.legend(frameon=False); ax.set_ylim(0, .8)
    save(fig, 'eta_search.png')


def fig_noisy_client(fed):
    N = fed['noisy_client_check']
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True)
    for ax, key, t in zip(axs, ['share', 'alignment', 'shapley'], ['Update share', 'Alignment (cosine)', 'Shapley value']):
        for node, c in zip([1, 2, 3], [CLIENT_COLORS[0], CLIENT_COLORS[1], OR]):
            ax.plot([r['round'] for r in N], [next(cc[key] for cc in r['clients'] if cc['node'] == node) for r in N],
                    marker='o', color=c, label=f'Client {node}' + (' (noisy labels)' if node == 3 else ''))
        ax.set_title(t, fontsize=10); ax.set_xlabel('round'); ax.axhline(0, color='k', lw=.5)
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, 'noisy_client.png')


def fig_personalisation(test_rows):
    g = {r['test_set']: r['f1_present'] for r in test_rows if r['model'] == 'global'}
    p = {r['test_set']: r['f1_present'] for r in test_rows if r['model'] == f"personal_{r['test_set'][-1]}"}
    keys = ['client_1', 'client_2', 'client_3']; x = np.arange(3); w = .36
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(x - w / 2, [g[k] for k in keys], w, color=NAVY, label='Global model')
    ax.bar(x + w / 2, [p[k] for k in keys], w, color=OR, label='Personal (Ditto) model')
    for i, k in enumerate(keys):
        ax.text(i - w / 2, g[k] + .01, f'{g[k]:.3f}', ha='center', fontsize=8)
        ax.text(i + w / 2, p[k] + .01, f'{p[k]:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x, ['Client 1', 'Client 2', 'Client 3']); ax.set_ylabel("macro-F1, classes in the client's test set")
    ax.set_ylim(0, 1.15); ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=2)
    save(fig, 'personalisation.png')


# ---------------------------------------------------------------- 4. figures that run the model on data
def fig_embedding(params):
    """Global model embeddings of client test windows (unseen by the global model)."""
    vids = load_videos([f'client_{k}_test' for k in (1, 2, 3)])
    X, y, _ = make_windows(vids, stride=8)
    cache = {}; M.forward(params, X, cache); Z = cache['z'] - cache['z'].mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False); E = Z @ Vt[:2].T
    cm = plt.get_cmap('tab20'); fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in range(len(CL)):
        m = y == k
        ax.scatter(E[m, 0], E[m, 1], s=8, color=cm(k), label=pretty(CL[k]), alpha=.75)
    ax.legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1, 1), loc='upper left', markerscale=2)
    ax.set_xlabel('principal component 1'); ax.set_ylabel('principal component 2')
    save(fig, 'embedding_pca.png')


def run_pose(video, task):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    opt = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=task),
                                       running_mode=vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(video); frames, L, i = [], [], 0
    with vision.PoseLandmarker.create_from_options(opt) as lm:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / 30)); i += 1
            frames.append(rgb)
            L.append(np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]]) if r.pose_landmarks
                     else np.full((33, 4), np.nan))
    cap.release()
    return frames, np.array(L)


def video_features(frames, L):
    h, w = frames[0].shape[:2]
    segs = FT.segments(L)
    s = max(segs, key=len) if segs else L[~np.isnan(L[:, 0, 0])]
    return FT.frame_features(s, aspect=w / h)


EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def video_figures(params):
    task = ARGS.pose_task
    if not os.path.exists(task):
        print('  downloading the MediaPipe pose model ...')
        os.makedirs(os.path.dirname(task), exist_ok=True)
        urllib.request.urlretrieve(POSE_TASK_URL, task)
    V = os.path.join(ROOT, 'report', 'demo_videos')
    vids = {'golf': 'golf_swing_0790.mp4', 'squat': 'squats_1659.mp4', 'tennis': 'tennis_serve_2141.mp4',
            'pitch': 'baseball_pitch_0001.mp4'}
    data = {k: run_pose(os.path.join(V, v), task) for k, v in vids.items()}

    fr, L = data['golf']; idx = np.linspace(5, len(fr) - 6, 5).astype(int)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2.9))
    h, w = fr[0].shape[:2]
    xs, ys = L[idx][:, :, 0] * w, L[idx][:, :, 1] * h; cx = np.nanmedian(xs)
    for ax, i in zip(axs, idx):
        ax.imshow(fr[i]); l = L[i]
        for a, b in EDGES:
            ax.plot([l[a, 0] * w, l[b, 0] * w], [l[a, 1] * h, l[b, 1] * h], color=OR, lw=2)
        ax.scatter(l[FT.JOINTS, 0] * w, l[FT.JOINTS, 1] * h, s=10, color=SKY, zorder=3)
        ax.set_title(f'frame {i}', fontsize=9); ax.axis('off')
        ax.set_xlim(cx - h * .45, cx + h * .45); ax.set_ylim(min(h, np.nanmax(ys) + h * .08), max(0, np.nanmin(ys) - h * .12))
    fig.tight_layout()
    save(fig, 'pose_strip.png')

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, (k, lab) in zip(axs, [('squat', 'Squats'), ('golf', 'Golf swing'), ('tennis', 'Tennis serve')]):
        f = video_features(*data[k]); t = np.arange(len(f)) / 30
        for nm, c in [('ang_r_knee', TEAL), ('ang_r_elbow', OR), ('ang_r_shoulder', PUR)]:
            ax.plot(t, f[:, FN.index(nm)], color=c, label=pretty(nm[4:]))
        ax.set_title(lab, fontsize=10); ax.set_xlabel('time (s)')
    axs[0].set_ylabel('joint angle (deg)'); axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, 'feature_traces.png')

    fig, axs = plt.subplots(2, 2, figsize=(9, 4.6), gridspec_kw={'width_ratios': [2.2, 1]})
    for row, k in enumerate(['golf', 'pitch']):
        f = video_features(*data[k]); s = max(0, (len(f) - 48) // 2); x = ((f[s:s + 48] - MU) / SD)[None]
        a = M.attention_weights(params, x)[0]; pr = M.predict_proba(params, x)[0]
        axs[row, 0].bar(np.arange(48), a, color=SKY); axs[row, 0].set_ylabel(r'attention $\alpha_t$')
        axs[row, 0].set_title(pretty(vids[k][:-9]) + ' clip, one 48-frame window', fontsize=9)
        top = np.argsort(pr)[::-1][:4]
        axs[row, 1].barh(range(4)[::-1], pr[top], color=[OR] + [GR] * 3)
        axs[row, 1].set_yticks(range(4)[::-1], [pretty(CL[i]) for i in top], fontsize=8)
        axs[row, 1].set_xlim(0, 1); axs[row, 1].set_title('top-4 class probabilities', fontsize=9)
    axs[1, 0].set_xlabel('frame $t$ in the window')
    fig.tight_layout()
    save(fig, 'attention_example.png')


# ---------------------------------------------------------------- 5. method diagrams (matplotlib)
def box(ax, x, y, w, h, text, fc=SOFT, ec=NAVYB, tc='black', fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, c=NAVYB, ls='-', text=None, off=(0, .12), fs=7.5, rad=0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.4, ls=ls, mutation_scale=12,
                                connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha='center', va='center', fontsize=fs, color=c)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    return fig, ax


def diagram_feature_pipeline():
    fig, ax = canvas(13, 3.9)
    steps = [('Camera frame\n(browser)', NAVY, 'white'), ('MediaPipe Pose\n33 landmarks\n+ visibility', SOFT, 'black'),
             ('Quality check\nleft/right swap,\nvisible joints', SOFT, 'black'), ('Normalise\nhip-centred,\ntorso-scaled', SOFT, 'black'),
             ('46 features\nper frame', '#fde3d7', 'black'), ('Windows\nT = 48, stride 4,\nstandardised', SOFT, 'black'),
             ('Bi-LSTM +\nattention', NAVY, 'white')]
    xs = np.linspace(.9, 12.1, len(steps))
    for x, (t, fc, tc) in zip(xs, steps):
        box(ax, x, 2.9, 1.5, 1.05, t, fc=fc, ec=OR if fc == '#fde3d7' else NAVYB, tc=tc, bold=tc == 'white')
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + .77, 2.9, b - .77, 2.9)
    groups = ['24 joint positions (12 joints, x and y)', '8 joint angles', '8 angular velocities',
              '4 limb speeds (wrists, ankles)', 'torso inclination and its velocity']
    gx = xs[4]
    box(ax, gx, 1.05, 3.6, 1.55, '', fc='white', ec=OR)
    for i, g in enumerate(groups):
        ax.text(gx - 1.65, 1.6 - i * .28, '\u2022 ' + g, fontsize=8, color=NAVY, va='center')
    arrow(ax, gx, 2.36, gx, 1.85, c=OR)
    ax.text((xs[0] + xs[1]) / 2, 3.65, 'only landmarks leave the camera page', fontsize=7.8, color=OR, ha='center', style='italic')
    save(fig, 'diagram_feature_pipeline.png')


def diagram_model(total_params):
    fig, ax = canvas(13.5, 3.4)
    layers = [('Input window\n48 \u00d7 46', SOFT, 'black', ''),
              ('Bi-LSTM layer 1\n64 + 64 units', NAVY, 'white', '57,344'),
              ('Bi-LSTM layer 2\n64 + 64 units', NAVY, 'white', '99,328'),
              ('Additive attention\n$e_t = v^\\top\\tanh(Wo_t+b)$\n$z=\\sum_t\\alpha_t o_t$', '#fde3d7', 'black', '8,320'),
              ('Dense 64\nReLU, dropout 0.3', SOFT, 'black', '8,256'),
              ('Softmax\n14 classes', TEAL, 'white', '910')]
    xs = np.linspace(.95, 12.55, len(layers))
    for x, (t, fc, tc, n) in zip(xs, layers):
        box(ax, x, 2.0, 1.7, 1.2, t, fc=fc, ec=OR if fc == '#fde3d7' else NAVYB, tc=tc, fs=8.3, bold=tc == 'white')
        if n:
            ax.text(x, 1.15, f'{n} parameters', ha='center', fontsize=7.5, color=GR)
    shapes = ['48\u00d746', '48\u00d7128', '48\u00d7128', '128', '64']
    for a, b, s in zip(xs[:-1], xs[1:], shapes):
        arrow(ax, a + .87, 2.0, b - .87, 2.0, text=s, off=(0, .2), fs=7.2)
    ax.text(xs[2] + 1.16, .5, f'Total {total_params:,} trainable parameters, {total_params * 4 / 1e6:.2f} MB in float32',
            ha='center', fontsize=9, color=NAVY)
    ax.text(xs[3], 2.95, r'weights $\alpha_t$ show which frames the decision used', fontsize=7.5, color=OR,
            ha='center', style='italic')
    save(fig, 'diagram_model.png')


def diagram_fl_round():
    fig, ax = canvas(11, 5.9)
    box(ax, 5.5, 4.9, 6.2, 1.6, '', fc=SOFT)
    ax.text(5.5, 5.5, 'Aggregation server', ha='center', fontsize=10, fontweight='bold', color=NAVY)
    for x, t in zip([3.3, 5.5, 7.7], ['1. weighted update\n$\\Delta_t=\\sum_k p_k(w_k-w_t)$',
                                       '2. FedAdam step\n$w_{t+1}=w_t+\\eta\\,m_t/(\\sqrt{v_t}+\\tau)$',
                                       '3. contributions\nshare, alignment,\nShapley (val. set)']):
        box(ax, x, 4.7, 2.0, .95, t, fc='white', fs=7.6)
    for i, x in enumerate([1.9, 5.5, 9.1]):
        box(ax, x, 1.6, 2.9, 1.5, '', fc='white', ec=CLIENT_COLORS[i])
        ax.text(x, 2.05, f'Client {i + 1}', ha='center', fontsize=9.5, fontweight='bold', color=CLIENT_COLORS[i])
        ax.text(x, 1.4, 'local data (never sent)\nFedProx training of $w_k$\n'
                        r'$F_k(w)+\frac{\mu}{2}\|w-w_t\|^2$' '\nDitto personal model $v_k$', ha='center', va='center', fontsize=7.4)
        arrow(ax, 5.5 + (x - 5.5) * .25 - .25, 4.1, x - .3, 2.42, c=NAVYB)
        arrow(ax, x + .3, 2.42, 5.5 + (x - 5.5) * .25 + .25, 4.1, c=OR, ls='--')
    ax.text(2.2, 3.35, 'send $w_t$', fontsize=8, color=NAVYB); ax.text(3.9, 3.0, 'return $w_k$, $n_k$', fontsize=8, color=OR)
    arrow(ax, 4.3, 4.7, 4.5, 4.7); arrow(ax, 6.5, 4.7, 6.7, 4.7)
    save(fig, 'diagram_fl_round.png')


def diagram_incremental():
    fig, ax = canvas(11, 4)
    nodes = {'pred': (1.3, 3.0, 'Prediction on a\nlive window'), 'corr': (3.9, 3.0, 'User confirms\nor corrects label'),
             'buf': (6.5, 3.0, 'Buffer\n16 windows'), 'mix': (9.4, 3.0, 'Mix with 20 % replay\nfrom 2,000-window\nreservoir'),
             'step': (9.4, 1.0, 'Adam step, lr $10^{-4}$\ncross-entropy\n+ EWC penalty ($\\lambda=1000$)'),
             'model': (5.2, 1.0, 'Updated personal model\nreservoir updated'), }
    for k, (x, y, t) in nodes.items():
        box(ax, x, y, 2.2 if k not in ('mix', 'step') else 2.6, 1.05, t,
            fc=NAVY if k == 'model' else ('#fde3d7' if k == 'step' else SOFT),
            ec=OR if k == 'step' else NAVYB, tc='white' if k == 'model' else 'black', bold=k == 'model', fs=8.2)
    arrow(ax, 2.42, 3.0, 2.78, 3.0); arrow(ax, 5.02, 3.0, 5.38, 3.0)
    arrow(ax, 7.62, 3.0, 8.08, 3.0, text='full', off=(0, .18))
    arrow(ax, 9.4, 2.46, 9.4, 1.55); arrow(ax, 8.08, 1.0, 6.32, 1.0)
    arrow(ax, 4.1, 1.0, 1.3, 2.46, rad=-.25)
    ax.text(5.2, .2, 'Between federated rounds the Fisher information is recomputed and the personal model is anchored to the global one (Ditto).',
            ha='center', fontsize=7.8, color=GR, style='italic')
    save(fig, 'diagram_incremental.png')


# ---------------------------------------------------------------- main
def main():
    global ARGS, CL, FN, MU, SD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'report', 'figures'))
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--skip-video', action='store_true', help='skip figures that need MediaPipe and OpenCV')
    ap.add_argument('--pose-task', default=os.path.join(ROOT, '.cache', 'pose_landmarker_full.task'))
    ARGS = ap.parse_args()
    os.makedirs(ARGS.out, exist_ok=True)

    cfg = jload(ROOT, 'models', 'config.json')
    CL, FN = cfg['classes'], cfg['feature_names']
    MU, SD = np.array(cfg['norm_mean'], np.float32), np.array(cfg['norm_std'], np.float32)
    mdir = os.path.join(ROOT, 'report', 'metrics')
    central, fed, inc = jload(mdir, 'central.json'), jload(mdir, 'federated.json'), jload(mdir, 'incremental.json')
    params = M.load(os.path.join(ROOT, 'models', 'central.npz'))

    rows = test_models(central)

    print('Dataset and exploratory figures ...')
    videos = load_videos([f'client_{k}_{s}' for k in (1, 2, 3) for s in ('train', 'test')])
    for f in (fig_class_distribution, fig_client_partition, fig_pck):
        f()
    for f in (fig_feature_distributions, fig_feature_correlation, fig_angle_trajectories, fig_phase_portraits,
              fig_temporal_autocorr, fig_feature_profiles):
        f(videos)

    print('Result figures ...')
    for f in (fig_training_curves, fig_confusion, fig_per_class_f1, fig_ablation, fig_published):
        f(central)
    fig_cl_matrices(inc); fig_cl_adaptation(inc)
    for f in (fig_fl_convergence, fig_fl_contributions, fig_eta_search, fig_noisy_client):
        f(fed)
    fig_personalisation(rows)
    fig_embedding(M.load(os.path.join(ROOT, 'models', 'global.npz')))

    print('Method diagrams ...')
    diagram_feature_pipeline(); diagram_model(central['efficiency']['bilstm_attn']['params']); diagram_fl_round(); diagram_incremental()

    if ARGS.skip_video:
        print('Skipped pose_strip, feature_traces and attention_example (--skip-video).')
    else:
        print('Pose figures (MediaPipe) ...')
        try:
            video_figures(params)
        except ImportError as e:
            print(f'  skipped: {e}. Install opencv-python and mediapipe, or pass --skip-video.')
    print('Done. Figures are in', ARGS.out)


if __name__ == '__main__':
    main()
