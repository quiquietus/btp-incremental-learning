"""Write every result in report/metrics/*.json as CSV tables (report/csv/).

Usage: python scripts/export_csv.py [report_dir]
"""
import csv, json, os, re, sys
from statistics import mean, pstdev as stdev  # population std, as in the report tables

REP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report')
OUT = os.path.join(REP, 'csv')
os.makedirs(OUT, exist_ok=True)
load = lambda n: json.load(open(os.path.join(REP, 'metrics', n + '.json')))
written = []


def write(name, rows, cols=None):
    if not rows:
        return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    written.append((name, len(rows)))


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return mean(xs), (stdev(xs) if len(xs) > 1 else 0.0)


# central recognition
C = load('central')
runs = {k: v for k, v in C.items() if isinstance(v, dict) and 'window' in v}
WIN = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'top3_accuracy', 'kappa', 'macro_roc_auc', 'n']


def split(name):
    m = re.match(r'(.+)_(\d+)cls(?:_seed(\d+))?$', name)
    return (m.group(1), int(m.group(2)), None if m.group(3) is None else int(m.group(3))) if m else (name, None, None)


rows = []
for k, v in runs.items():
    model, ncls, seed = split(k)
    r = {'model': model, 'classes': ncls, 'seed': seed}
    r.update({f'window_{m}': v['window'].get(m) for m in WIN})
    vid = v.get('video') or {}
    r.update({f'video_{m}': vid.get(m) for m in ('accuracy', 'macro_f1', 'kappa', 'n')})
    r.update({'best_epoch': v.get('best_epoch'), 'train_minutes': v.get('train_minutes'), 'params': v.get('params')})
    rows.append(r)
write('recognition_runs.csv', rows)

groups = {}
for r in rows:
    groups.setdefault((r['model'], r['classes']), []).append(r)
summ = []
for (model, ncls), rs in groups.items():
    s = {'model': model, 'classes': ncls, 'seeds': len(rs)}
    for m in ('window_macro_f1', 'window_accuracy', 'window_top3_accuracy', 'window_kappa', 'video_accuracy', 'video_macro_f1'):
        mu, sd = ms([r[m] for r in rs])
        s[m + '_mean'], s[m + '_std'] = mu, sd
    summ.append(s)
write('recognition_summary.csv', summ)

pc = []
for k, v in runs.items():
    model, ncls, seed = split(k)
    for cls, d in v.get('per_class', {}).items():
        pc.append({'model': model, 'classes': ncls, 'seed': seed, 'class': cls, **d})
write('recognition_per_class.csv', pc)

hist = []
for k, v in runs.items():
    model, ncls, seed = split(k)
    for h in v.get('history', []):
        hist.append({'model': model, 'classes': ncls, 'seed': seed, **h})
write('training_history.csv', hist)

ref = runs.get('bilstm_attn_14cls_seed0')
if ref and ref.get('confusion'):
    names = list(ref['per_class'])
    write('confusion_bilstm_attn_14cls_seed0.csv', [{'true_class': names[i], **{n: ref['confusion'][i][j] for j, n in enumerate(names)}} for i in range(len(names))],
          ['true_class'] + names)

if 'mcnemar_bilstm_vs_birnn' in C:
    write('mcnemar_bilstm_vs_birnn.csv', [C['mcnemar_bilstm_vs_birnn']])
if 'efficiency' in C:
    write('efficiency.csv', [{'model': m, **d} for m, d in C['efficiency'].items()])

# published comparison (from the report tables)
tab = open(os.path.join(REP, 'tables', 'report_tables.md')).read()
if 'Comparison with published' in tab:
    part = tab.split('Comparison with published', 1)[1]
    pub = []
    for line in part.splitlines():
        cells = [c.strip().strip('*') for c in line.strip().strip('|').split('|')]
        if len(cells) == 3 and cells[0] not in ('Method',) and not set(cells[0]) <= set('-'):
            pub.append({'method': cells[0], 'input': cells[1], 'video_accuracy_percent': cells[2]})
        elif pub and not line.strip():
            break
    write('published_comparison_15cls.csv', pub)

# federated learning
F = load('federated')


def fl_rows(name, recs):
    out = []
    for r in recs:
        row = {'method': name, 'round': r['round'], 'algorithm': r.get('algorithm'), 'N': r.get('N'),
               'global_test_macro_f1': r.get('global_test_macro_f1'), 'global_test_acc': r.get('global_test_acc'),
               'val_before': r.get('val_before'), 'val_after': r.get('val_after'), 'delta_norm': r.get('delta_norm'),
               'step_norm': r.get('step_norm'), 'shares_sum': r.get('shares_sum'), 'shapley_sum': r.get('shapley_sum'),
               'bytes_round': r.get('bytes_round'), 'seconds': r.get('seconds')}
        for k, e in (r.get('client_eval') or {}).items():
            row[f'client{k}_global_f1'] = e.get('global_f1'); row[f'client{k}_personal_f1'] = e.get('personal_f1')
        out.append(row)
    return out


write('fl_rounds.csv', fl_rows('ours_fedprox_fedadam_ditto', F['ours']) + fl_rows('fedavg', F['fedavg']))

contrib = []
for name in ('ours', 'fedavg'):
    for r in F[name]:
        for c in r['clients']:
            row = {'method': name, 'round': r['round'], 'node': c['node'], 'n': c.get('n'), 'p': c.get('p'),
                   'update_norm': c.get('update_norm'), 'share': c.get('share'), 'alignment': c.get('alignment'),
                   'shapley': c.get('shapley'), 'loo': c.get('loo')}
            for L, v in (c.get('layer_share') or {}).items():
                row[f'layer_share_{L}'] = v
            contrib.append(row)
write('fl_client_contributions.csv', contrib)

eta = []
for e, vals in F['eta_search'].items():
    for i, v in enumerate(vals, 1):
        eta.append({'eta': float(e), 'round': i, 'val_macro_f1': v})
write('fl_eta_search.csv', eta)

summary = []
for name, key in (('ours_fedprox_fedadam_ditto', 'ours'), ('fedavg', 'fedavg')):
    recs = F[key]
    f1 = [r['global_test_macro_f1'] for r in recs]
    last5 = f1[-5:]
    pers = [ms([e['personal_f1'] for e in recs[-1]['client_eval'].values()])[0]] if recs[-1].get('client_eval') else [None]
    worst = min(e['personal_f1'] for e in recs[-1]['client_eval'].values()) if recs[-1].get('client_eval') else None
    summary.append({'method': name, 'final_round_f1': f1[-1], 'last5_mean_f1': mean(last5), 'last5_std_f1': stdev(last5),
                    'best_round_f1': max(f1), 'mean_personal_f1_final': pers[0], 'worst_client_personal_f1_final': worst,
                    'rounds_to_90pct_centralised': F.get(f'rounds_to_90pct_centralised_{key}')})
lo = F['local_only']
summary.append({'method': 'local_only_mean', 'final_round_f1': mean(v['global_test_f1'] for v in lo.values()),
                'mean_personal_f1_final': mean(v['own_test_f1'] for v in lo.values()),
                'worst_client_personal_f1_final': min(v['own_test_f1'] for v in lo.values())})
summary.append({'method': 'centralised_upper_bound', 'final_round_f1': F['centralised_test_macro_f1_mean']})
write('fl_summary.csv', summary)
write('fl_local_only.csv', [{'node': k, **v} for k, v in lo.items()])

noisy = []
for r in F.get('noisy_client_check', []):
    for c in r['clients']:
        noisy.append({'round': r['round'], 'node': c['node'], 'noise_client': c['node'] == 3, 'share': c.get('share'),
                      'alignment': c.get('alignment'), 'shapley': c.get('shapley'), 'update_norm': c.get('update_norm'),
                      'val_before': r.get('val_before'), 'val_after': r.get('val_after')})
write('fl_noisy_client_check.csv', noisy)

# incremental learning
I = load('incremental')
VARS = ('no_update', 'naive', 'replay', 'ewc', 'replay_ewc')
for exp, tag in (('A_new_client', 'expA_new_client'), ('B_new_classes', 'expB_new_classes')):
    E = I[exp]
    rows, mat, curve = [], [], []
    for v in VARS:
        d = E[v]
        rows.append({'variant': v, 'final_average_accuracy': d['average_accuracy'][-1], 'forgetting': d['forgetting_final'],
                     'bwt': d['bwt'], 'fwt': d['fwt'], 'intransigence_mean': d['intransigence_mean'],
                     'adaptation_gain_f1': d.get('adaptation_gain_f1')})
        for i, (ra, rf) in enumerate(zip(d['accuracy_matrix'], d['f1_matrix'])):
            for j in range(len(ra)):
                mat.append({'variant': v, 'after_task': i, 'eval_task': j, 'accuracy': ra[j], 'macro_f1': rf[j]})
        for c in d.get('curve', []):
            curve.append({'variant': v, **c})
    write(f'{tag}_summary.csv', rows)
    write(f'{tag}_accuracy_matrix.csv', mat)
    if curve:
        write(f'{tag}_adaptation_curve.csv', curve)
A = I['A_new_client']
if 'ewc_lambda_sensitivity' in A:
    write('expA_ewc_lambda_sensitivity.csv', [{'lambda': float(k), **v} for k, v in A['ewc_lambda_sensitivity'].items()])
if 'joint_upper_bound' in A:
    write('expA_joint_upper_bound.csv', [A['joint_upper_bound']])
B = I['B_new_classes']
write('expB_task_classes.csv', [{'task': i, 'classes': ' '.join(t)} for i, t in enumerate(B['task_classes'])])

# data
D = load('data_stats')
write('data_split.csv', [{'split': k, 'videos': v} for k, v in D['by_split'].items()] + [{'split': 'total', 'videos': D['videos']}])
cc = D.get('client_class_counts', {})
cls = list(next(iter(cc.values()))) if cc else []
write('data_client_partitions.csv', [{'node': k, 'train_videos': D['client_sizes'][k]['train'], 'test_videos': D['client_sizes'][k]['test'], **cc.get(k, {})}
                                     for k in D.get('client_sizes', {})], ['node', 'train_videos', 'test_videos'] + cls)

for n, r in written:
    print(f'{n}: {r} rows')
