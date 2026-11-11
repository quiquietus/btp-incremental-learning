"""Builds kaggle/btp_train.ipynb with the project code embedded, so the notebook
is a single file you upload to Kaggle. Run: python kaggle/make_notebook.py"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def src(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n').splitlines(True)}


def writefile(path, content):
    return code(f'%%writefile /kaggle/working/btp/{path}\n' + content)


cells = [
    md('''
# BTP: train the client model, run the incremental and federated experiments

**What this notebook does.** It reads the Penn Action dataset (University of Pennsylvania, ICCV 2013), extracts the body skeleton from every frame with MediaPipe Pose, trains the Bi-LSTM and Bi-RNN client models with baselines and ablations, runs the incremental-learning experiments (forgetting score, BWT, FWT, intransigence) and the federated simulation (FedAvg against FedProx + FedAdam + Ditto, with per-client contribution and Shapley values), and packs everything the website needs into **`outputs.zip`**.

## Before you press "Run all"
1. In the right-hand **Settings** panel: **Accelerator → GPU T4 x1** (or T4 x2), and **Internet → On**.
2. Data: the notebook downloads Penn Action itself (about 3 GB). If the download fails, click **Add Data**, search **"Penn Action"**, add any copy that contains `frames/` and `labels/` folders, and run again. It finds the data automatically.
3. Press **Run all**. Expected time: **about 1.5 to 2.5 hours**, mostly the skeleton extraction. You can close the tab only if you use **Save Version → Save & Run All (Commit)**, which runs it in the background.

## When it finishes
Open the **Output** tab (or the folder icon at `/kaggle/working`), download **`outputs.zip`** and install it with `python scripts/install_outputs.py outputs.zip`.

Every stage saves its progress in `/kaggle/working/cache`; if a cell fails, fix the cause and re-run from that cell.
'''),
    md('## 1. Install MediaPipe in its own folder and download the pose model'),
    code('''
import os, subprocess, sys
os.makedirs('/kaggle/working/btp/btpcore', exist_ok=True)
os.makedirs('/kaggle/working/btp/kaggle', exist_ok=True)
os.makedirs('/kaggle/working/cache/lm', exist_ok=True)
# MediaPipe goes into a separate folder so its NumPy/OpenCV versions cannot break PyTorch or scikit-learn here.
!pip install -q --no-warn-conflicts --target /kaggle/tmp/mp mediapipe
# MediaPipe needs these system graphics libraries even on CPU.
!apt-get install -y -q libegl1 libgles2 libgl1 > /dev/null 2>&1 || (apt-get update -q > /dev/null 2>&1 && apt-get install -y -q libegl1 libgles2 libgl1 > /dev/null 2>&1)
!wget -q -O /kaggle/working/cache/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
print('pose model bytes:', os.path.getsize('/kaggle/working/cache/pose_landmarker_full.task'))
'''),
    md('## 2. Project code (the same files the website uses)'),
    writefile('btpcore/__init__.py', src('btpcore/__init__.py')),
    writefile('btpcore/features.py', src('btpcore/features.py')),
    writefile('btpcore/model.py', src('btpcore/model.py')),
    writefile('btpcore/fl_core.py', src('btpcore/fl_core.py')),
    writefile('btpcore/learn.py', src('btpcore/learn.py')),
    writefile('kaggle/extract.py', src('kaggle/extract.py')),
    writefile('kaggle/pipeline.py', src('kaggle/pipeline.py')),
    md('## 3. Find or download Penn Action and read its labels'),
    code('''
%cd /kaggle/working/btp/kaggle
!python pipeline.py meta
'''),
    md('## 4. Skeleton extraction with MediaPipe (longest step, about 30 to 90 minutes)\nRuns on all CPU cores. Progress is printed every 50 videos. Safe to re-run: finished videos are skipped.'),
    code('''
!PYTHONPATH=/kaggle/tmp/mp python extract.py 2>&1 | grep -v -E "inference_feedback_manager|^W0000|^I0000"
'''),
    md('## 5. Features, splits and the three federated client partitions'),
    code('!python pipeline.py data'),
    md('## 6. Centralised training: Bi-LSTM, Bi-RNN, ablations and baselines (14 and 15 classes, 3 seeds)'),
    code('!python pipeline.py central 2>&1 | grep -v -E "UserWarning|tl \\+="'),
    md('## 7. Incremental learning: experiment A (new client) and B (new sports)'),
    code('!python pipeline.py cl 2>&1 | grep -v -E "UserWarning|tl \\+="'),
    md('## 8. Federated simulation: FedAvg vs FedProx + FedAdam + Ditto, contributions and Shapley values'),
    code('!python pipeline.py fl 2>&1 | grep -v -E "UserWarning|tl \\+="'),
    md('## 9. Export models, node data, figures, tables and `params.json` into `outputs.zip`'),
    code('!python pipeline.py export 2>&1 | grep -v -E "UserWarning|tl \\+="'),
    md('## 10. Results summary'),
    code('''
from IPython.display import Markdown, display, FileLink
display(Markdown(open('/kaggle/working/outputs/tables/report_tables.md').read()))
os.chdir('/kaggle/working')
FileLink('outputs.zip')
'''),
]

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}, 'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': True}},
      'nbformat': 4, 'nbformat_minor': 5}
out = os.path.join(ROOT, 'kaggle', 'btp_train.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', out)
