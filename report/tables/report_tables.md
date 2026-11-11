## Main results (test set, mean ± std over seeds, %)

| Model | Classes | Window acc | Window macro-F1 | Weighted F1 | Top-3 | ROC-AUC | Kappa | Video acc | Video macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| bilstm_attn | 14 | 87.5 ± 1.4 | 81.1 ± 0.8 | 87.8 ± 1.3 | 95.4 ± 0.3 | 97.7 ± 0.5 | 83.9 ± 1.7 | 88.1 ± 0.4 | 86.0 ± 1.2 |
| birnn_attn | 14 | 86.5 ± 0.1 | 79.6 ± 0.9 | 86.7 ± 0.1 | 94.5 ± 0.3 | 97.6 ± 0.2 | 82.6 ± 0.1 | 87.3 ± 0.5 | 85.0 ± 1.2 |
| bilstm_maxpool | 14 | 88.8 ± 1.0 | 81.8 ± 1.3 | 89.1 ± 0.9 | 95.8 ± 0.5 | 97.5 ± 0.3 | 85.4 ± 1.3 | 89.4 ± 0.3 | 87.5 ± 0.3 |
| bilstm_paper9d | 14 | 78.3 ± 1.3 | 64.4 ± 1.9 | 78.9 ± 1.2 | 92.6 ± 0.1 | 93.8 ± 1.0 | 72.1 ± 1.6 | 74.7 ± 0.8 | 70.5 ± 0.6 |
| majority | 14 | 43.5 ± 0.0 | 4.3 ± 0.0 | 26.4 ± 0.0 | 50.5 ± 0.0 | 50.0 ± 0.0 | 0.0 ± 0.0 | 5.6 ± 0.0 | 0.8 ± 0.0 |
| random_forest | 14 | 87.8 ± 0.0 | 82.0 ± 0.0 | 87.3 ± 0.0 | 96.6 ± 0.0 | 98.5 ± 0.0 | 83.7 ± 0.0 | 84.8 ± 0.0 | 82.5 ± 0.0 |
| bilstm_attn | 15 | 86.7 ± 1.5 | 78.5 ± 1.2 | 87.1 ± 1.3 | 95.2 ± 0.6 | 97.5 ± 0.1 | 83.0 ± 1.8 | 87.8 ± 1.2 | 85.7 ± 1.3 |
| birnn_attn | 15 | 86.6 ± 0.5 | 77.9 ± 0.4 | 86.9 ± 0.4 | 95.0 ± 0.3 | 97.8 ± 0.1 | 82.7 ± 0.5 | 86.7 ± 0.6 | 84.1 ± 0.5 |
| majority | 15 | 43.2 ± 0.0 | 4.0 ± 0.0 | 26.1 ± 0.0 | 50.2 ± 0.0 | 50.0 ± 0.0 | 0.0 ± 0.0 | 5.5 ± 0.0 | 0.7 ± 0.0 |
| random_forest | 15 | 87.5 ± 0.0 | 81.8 ± 0.0 | 87.0 ± 0.0 | 96.7 ± 0.0 | 98.8 ± 0.0 | 83.5 ± 0.0 | 83.9 ± 0.0 | 81.5 ± 0.0 |

McNemar (Bi-LSTM vs Bi-RNN, 14 classes, seed 0, test windows): b = 318, c = 167, exact p = 6.59e-12

## Efficiency

| Model | Parameters | Size (MB, float32) | PyTorch CPU ms/window | NumPy runtime ms/window |
|---|---|---|---|---|
| bilstm_attn | 174,158 | 0.70 | 2.42 | 12.40 |
| birnn_attn | 56,654 | 0.23 | 3.16 | nan |
| bilstm_paper9d | 155,214 | 0.62 | 2.28 | nan |

## Per-class results (Bi-LSTM, 14 classes, seed 0)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| baseball_pitch | 0.876 | 0.967 | 0.919 | 628 |
| baseball_swing | 0.856 | 0.960 | 0.905 | 99 |
| bench_press | 0.960 | 0.768 | 0.853 | 314 |
| bowling | 0.521 | 0.896 | 0.659 | 96 |
| clean_and_jerk | 0.945 | 0.960 | 0.953 | 2204 |
| golf_swing | 0.980 | 0.883 | 0.929 | 163 |
| jump_rope | 0.894 | 0.933 | 0.913 | 45 |
| jumping_jacks | 1.000 | 0.963 | 0.981 | 54 |
| pull_ups | 0.946 | 0.702 | 0.806 | 322 |
| push_ups | 0.234 | 0.367 | 0.286 | 30 |
| sit_ups | 0.937 | 0.663 | 0.776 | 178 |
| squats | 0.809 | 0.825 | 0.817 | 576 |
| tennis_forehand | 0.796 | 0.759 | 0.777 | 108 |
| tennis_serve | 0.922 | 0.959 | 0.940 | 246 |

## Incremental experiment A (new client)

| Variant | Final average accuracy | Forgetting score | BWT | FWT | Intransigence (mean) |
|---|---|---|---|---|---|
| no_update | 0.753 | 0.000 | +0.000 | +0.632 | +0.182 |
| naive | 0.848 | 0.009 | -0.009 | +0.632 | -0.016 |
| replay | 0.848 | 0.013 | -0.013 | +0.632 | -0.020 |
| ewc | 0.839 | -0.002 | +0.002 | +0.632 | +0.012 |
| replay_ewc | 0.841 | 0.007 | -0.007 | +0.632 | +0.000 |

## Incremental experiment B (new classes)

| Variant | Final average accuracy | Forgetting score | BWT | FWT | Intransigence (mean) |
|---|---|---|---|---|---|
| no_update | 0.078 | 0.842 | -0.842 | +0.000 | +0.225 |
| naive | 0.245 | 0.842 | -0.842 | +0.000 | -0.024 |
| replay | 0.180 | 0.842 | -0.842 | +0.000 | +0.073 |
| ewc | 0.245 | 0.842 | -0.842 | +0.000 | -0.024 |
| replay_ewc | 0.238 | 0.842 | -0.842 | +0.000 | -0.015 |

## Federated learning

| Method | Test macro-F1 (global, final round) | Mean personal F1 on own client test | Worst client F1 | Rounds to 90% of centralised |
|---|---|---|---|---|
| FedProx + FedAdam + Ditto (eta 0.1) | 0.690 | 0.552 | 0.414 | 9 |
| FedAvg | 0.749 | 0.533 | 0.398 | 12 |
| Local-only (mean of clients) | 0.518 | 0.472 | 0.343 | — |
| Centralised (upper bound) | 0.811 | — | — | — |

Bytes per round (3 clients, up and down): 4.16 MB


### Noisy-client sanity check (client 3 sends noise)

| Round | Client | Share | Alignment | Shapley |
|---|---|---|---|---|
| 1 | 1 | +0.030 | +0.177 | +0.0109 |
| 1 | 2 | +0.267 | +0.519 | -0.0401 |
| 1 | 3 | +0.703 | +0.838 | +0.0011 |
| 2 | 1 | +0.022 | +0.145 | +0.0172 |
| 2 | 2 | +0.224 | +0.473 | -0.0319 |
| 2 | 3 | +0.754 | +0.868 | +0.0102 |
| 3 | 1 | +0.019 | +0.140 | +0.0167 |
| 3 | 2 | +0.190 | +0.438 | -0.0148 |
| 3 | 3 | +0.791 | +0.889 | +0.0065 |

## Comparison with published Penn Action results (15 classes, video-level accuracy %)

| Method | Input | Accuracy |
|---|---|---|
| ACPS (Iqbal et al., FG 2017) | Pose | 79.0 |
| ST-AOG (Nie et al., CVPR 2015) | Pose + video | 85.5 |
| C3D (Tran et al., 2015) | Video | 86.0 |
| ST-GCN, from scratch | Skeleton | 89.6 |
| IDT-FV (Wang and Schmid, 2013) | Video | 92.0 |
| 2s-AGCN, from scratch | Skeleton | 93.1 |
| PoTion (Choutas et al., CVPR 2018) | Pose heat maps | 93.6 |
| UNIK, from scratch | Skeleton | 94.0 |
| P-CNN (Cheron et al., ICCV 2015) | Pose + video | 95.3 |
| Luvizon et al. (CVPR 2018) | Pose | 97.4 |
| **Ours: Bi-LSTM + attention on MediaPipe skeleton** | MediaPipe skeleton | **87.8 ± 1.2** |
