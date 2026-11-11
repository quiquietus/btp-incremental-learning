"""Copy the Kaggle results into the build.

Usage: python scripts/install_outputs.py path/to/outputs.zip

Takes outputs/models/* and outputs/node_data/* from the zip into ./models and the
report material (figures, tables, params.json, demo videos) into ./report.
Then checks that every file the server and the nodes need is present.
"""
import os, sys, zipfile, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = ['global.npz', 'fedadam_state.npz', 'personal_1.npz', 'personal_2.npz', 'personal_3.npz', 'config.json',
        'fl_history.json', 'node_data/server_val.npz'] + [f'node_data/client_{k}_{s}.npz' for k in (1, 2, 3) for s in ('train', 'test')]


def main(zp):
    z = zipfile.ZipFile(zp)
    n_models = n_report = 0
    for info in z.infolist():
        name = info.filename
        if info.is_dir() or '..' in name.split('/') or name.startswith('/'):
            continue
        rel = name[len('outputs/'):] if name.startswith('outputs/') else name
        if rel.startswith('models/'):
            dest = os.path.join(ROOT, rel); n_models += 1
        elif rel.startswith('node_data/'):
            dest = os.path.join(ROOT, 'models', rel); n_models += 1
        else:
            dest = os.path.join(ROOT, 'report', rel); n_report += 1
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(info) as src, open(dest, 'wb') as out:
            out.write(src.read())
    print(f'copied {n_models} model files into models/ and {n_report} report files into report/')
    missing = [f for f in NEED if not os.path.exists(os.path.join(ROOT, 'models', f))]
    if missing:
        print('MISSING:', ', '.join(missing)); sys.exit(1)
    cfg = json.load(open(os.path.join(ROOT, 'models', 'config.json')))
    print('classes:', len(cfg['classes']), '| window T =', cfg.get('T'), '| features =', cfg.get('F'))
    print('all files present. Commit models/ and push to deploy.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
