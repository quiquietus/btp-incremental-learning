# Incremental Learning in a Federated Setup for Health IoT Devices

BTech project, Department of Computer Science and Engineering, The LNM Institute of Information
Technology, Jaipur, 2025 to 2026.

Team: Aditya Sharma, Hussain Haidary, Dhruv Sangal and Sarrah Sadriwala.
Guides: Dr. Anubhav Shivhare and Dr. Lal Upendra Pratap Singh.

## What this project does

The system recognises sports and exercise movements from body pose. It keeps the video private,
keeps learning on each device, and lets several devices improve one shared model without sending
their data anywhere.

A web browser opens the camera or a video file and runs MediaPipe Pose on every frame. Only the 33
body landmarks of each frame leave the camera page. A client node turns the landmarks into 46
kinematic features per frame and classifies windows of 48 frames, about 1.6 seconds, with a small
bidirectional LSTM with attention. When the prediction is wrong, the user picks the right activity.
After 16 corrected windows the node updates its personal model, using experience replay and Elastic
Weight Consolidation so that it does not forget what it already knew.

Three client nodes also train together with federated learning. Each node trains the global model on
its own data with FedProx and sends back only the weights. The server combines them with FedAdam,
and each node keeps a personal model with Ditto. In every round the server measures how much each
client contributed, with three measures: update share, alignment and the exact Shapley value.

Everything is reached from one address. A single sign-in page sends the administrator to the
aggregation console and each client operator to their own node.

## Main results

All numbers come from the Penn Action benchmark with 14 classes, averaged over three seeds unless
stated otherwise. The full tables are in report/tables and report/metrics.

| What | Result |
|---|---|
| Window macro-F1, Bi-LSTM with attention | 81.1% (std 0.8) |
| Video accuracy, 14 classes | 88.1% (std 0.4) |
| Video accuracy, standard 15-class protocol | 87.8% (std 1.2) |
| Bi-LSTM against Bi-RNN on paired test windows | significantly better, McNemar p = 6.6 × 10⁻¹² |
| Model size and speed | 174,158 parameters, 0.70 MB, 12.7 ms per window in NumPy |
| New client joins, accuracy on its data | from 68.4% to 88.3%, with under 1.3 points forgotten |
| New classes, average accuracy | 0.51 with replay, against 0.28 with fine-tuning |
| Federated global macro-F1 after 30 rounds | 0.776, against 0.744 for FedAvg and 0.811 for pooled training |

On those paired windows the Bi-LSTM is right where the Bi-RNN is wrong on 318
test windows, and the reverse happens on 167.

## How it works

### From camera to features

MediaPipe Pose runs in the browser in video mode. It returns 33 landmarks per frame, each with a
position and a visibility score. A frame is used only if both shoulders, both hips and at least one
knee are visible. Gaps of up to five frames are filled by interpolation. On Penn Action, MediaPipe
reaches a PCK@0.2 of 0.917 against the hand-annotated joints, once the left and right sides are
swapped to match the dataset's convention.

The node centres the landmarks on the hips and divides them by the torso length. A child and an adult
doing the same squat, anywhere in the frame, then give similar numbers. From these it computes 46
features per frame:

| Group | Count |
|---|---|
| Joint positions of shoulders, elbows, wrists, hips, knees and ankles | 24 |
| Elbow, shoulder, hip and knee angles, left and right | 8 |
| Angular velocities of those angles | 8 |
| Speeds of both wrists and both ankles | 4 |
| Torso inclination and its velocity | 2 |

### The model

Two bidirectional LSTM layers with 64 units in each direction read the 48-frame window. Additive
attention then weights the 48 outputs and averages them, so the model can focus on the decisive part
of a movement. A dense layer of 64 units and a softmax over 14 classes follow. The weights are
trained in PyTorch on Kaggle, and the deployed system runs them in plain NumPy. Both versions agree
to within 2.4 × 10⁻⁷ in every class probability, so the server and the nodes need no deep learning
framework.

### Learning on the device

Each node keeps a buffer of corrected windows. When it holds 16, the node adds 4 windows replayed from
a reservoir of 2,000 earlier windows and takes one Adam step at a learning rate of 10⁻⁴. The loss is
cross-entropy plus an EWC penalty with λ = 1000 and a Ditto pull towards the latest global model with
λ = 0.1. Replay keeps old classes in the gradient. EWC protects the weights that mattered most for
earlier data.

### Learning together

In a federated round the server sends the global model to every online node. Each node trains it for
two local epochs with the FedProx term, μ = 0.01, and uploads its weights and its number of training
windows. The server forms the weighted update Δ = Σ p_k (w_k − w_t), with p_k = n_k / N, and applies a
FedAdam step with η = 0.03, β₁ = 0.9, β₂ = 0.99 and τ = 10⁻³. The rate η was picked from 0.001,
0.01, 0.03 and 0.1 by validation macro-F1. One round moves 4.16 MB in total.

### Who contributed

For each client the server computes three measures. The update share ⟨c_k, Δ⟩ / ‖Δ‖² tells how much of
the step points in the client's direction, and the shares sum to one. The alignment cos(d_k, Δ) tells
whether the client pushes the same way as the group. The exact Shapley value asks how much the
client's update improves validation macro-F1, averaged over all subsets of the other clients. With
three clients this needs only eight evaluations. A test in which one client sends random noise shows
why the Shapley value matters. The noisy client gets the largest share and alignment, because its
update is large, but its Shapley value is close to zero.

## What is where

| Folder | Contents |
|---|---|
| kaggle | The training pipeline and the notebook that runs it on a Kaggle GPU and writes outputs.zip |
| btpcore | Feature extraction, the NumPy model with its backward pass, learning utilities, and aggregation with the contribution measures. The notebook, the server and the nodes all use this code. |
| portal | The single entry point: the server plus three mounted client nodes |
| server | The aggregation server: users and roles, rounds, FedAdam, Shapley values, the admin console |
| node | A client node: live prediction, incremental learning, the federated client |
| web | The sign-in page, the admin console, the client page and the shared layout |
| models | Trained weights, normalisation statistics and the data partition of each client |
| report | Metrics as JSON, tables, CSV exports, figures and short demo clips |
| scripts | Installing the Kaggle outputs, exporting CSV files, drawing the report figures, running locally |

The whole system runs in one process that needs about 250 MB of memory, and under 400 MB at the peak
of a round. It fits the free tier on Render or an old laptop.

## Reproducing the results

### 1. Train on Kaggle

1. Upload kaggle/btp_train.ipynb to Kaggle with New Notebook, then File, then Import Notebook.
2. In the settings, choose a GPU accelerator and turn the internet on.
3. Run all cells. The notebook downloads Penn Action from its official page. If that fails, attach the
   Penn Action dataset with Add Data and run it again. The full run takes about 46 minutes.
4. Download outputs.zip from the Output panel.

### 2. Install the results

```bash
python scripts/install_outputs.py ~/Downloads/outputs.zip
```

This fills models with the weights and node data, and report with the metrics, tables and parameters.

### 3. Test the models and draw the figures

```bash
pip install numpy matplotlib opencv-python mediapipe
python scripts/make_report_figures.py
```

The script first tests the federated models on each client's test set and prints a table. It writes
the same table to report/figures/test_results.csv. It then draws every figure in the report: the
dataset and data analysis plots, the results, the contribution plots and four method diagrams. The
pose figures run MediaPipe on the demo clips and download its model file on the first run. Use
--skip-video to leave them out, and --out to write the figures somewhere else.

The test table gives two versions of each client's macro-F1. The first averages over the classes
that occur in that client's test set. The second averages over all 14 classes and counts absent
classes as zero, which is how the training pipeline logged it in federated.json. Each client holds
only 8 to 11 classes, so the second version is much lower. Both rank the models in the same order.

The central model is not tested on client data, because the clients' data comes from its training
split. Its scores on the official test split are read from central.json instead.

### 4. Export the results as CSV

```bash
python scripts/export_csv.py
```

## Running the system

### On one computer

```bash
pip install -r requirements.txt
bash scripts/run_local.sh
```

Docker works too, with docker compose up --build. Open http://localhost:8000 and sign in as admin or
as client1, client2 or client3. The default passwords are set by ADMIN_PASSWORD and DEMO_PASSWORD
below. Browsers allow the camera only on localhost and on https addresses.

### On Render

1. In Render, choose New, then Blueprint, then this repository. The file render.yaml creates one
   service.
2. When Render asks, set ADMIN_PASSWORD and DEMO_PASSWORD.
3. Open the service's address. That single link is the whole system.

A free service sleeps after 15 minutes without traffic, and the first request then takes about a
minute. Open the link a minute before a demo. The free plan does not keep disks between restarts, so a
restart puts the global model and the users back to their deployed state.

### One address, separate nodes

The portal serves the sign-in page and the aggregation server at the root address and mounts the
three client nodes at /node/1/, /node/2/ and /node/3/. Each node is its own copy of the node
application, with its own models, data, buffer and training thread. It talks to the server only
through the server's HTTP interface, exactly as it would from another machine. The same code can also
run as four separate services when the nodes really are on different devices. Set ROLE to server or
node in the Dockerfile for that.

| Sign in as | Opens |
|---|---|
| admin | the aggregation console |
| client1, client2, client3 | that client node |
| viewer | the console, read-only, with links to every node |

## Giving a demo

1. Sign in as a client and choose Use camera or Upload a video. The skeleton is drawn live. The label
   and the top three classes come from this node's personal model, with the global model's opinion
   beside it.
2. If the label is wrong, choose how much to label, the last 3 seconds, the last 6 seconds or the
   whole clip, and click the right class. After 16 windows the personal model takes one step. The card
   shows the loss terms and the batch accuracy before and after.
3. On the admin console, press Start a round. Every online node downloads the global model, trains it
   and uploads its weights. The console and every client page then show the five aggregation steps,
   one weight followed through every step with its real values, and each client's share, alignment,
   Shapley value and leave-one-out score.
4. The client page lists what it sent to the server and what it received. The console lists the same
   numbers, so both sides can be matched.

## Roles and security

| Role | Can |
|---|---|
| admin | start rounds, change settings, manage users and read the audit trail |
| operator | use one node's client page and teach its personal model |
| viewer | watch every page and change nothing |

Sign-in tokens are JSON Web Tokens signed with JWT_SECRET, so one sign-in works for the console and
every node. Passwords are hashed with scrypt. Nodes prove who they are to the server with a token
derived from the same secret. The weights travel over HTTPS but are not encrypted beyond that. Secure
aggregation and differential privacy are not implemented.

## Settings

| Variable | Used by | Default |
|---|---|---|
| JWT_SECRET | all | dev-secret-change-me |
| ADMIN_PASSWORD, DEMO_PASSWORD | server | admin123, demo123 |
| SERVER_URL, SERVER_PUBLIC_URL, NODE_ID | separate-node mode only | set by the portal |
| MALLOC_ARENA_MAX | portal, keeps memory low | 2 |
| STEP_DELAY | server, pause between live aggregation steps in seconds | 1.2 |
| LOCAL_STEPS, ROUND_TIMEOUT | server | 12, 240 |
| SERVER_VAL_WINDOWS | server, validation windows used for the Shapley values | 240 |

## Data and references

Penn Action: W. Zhang, M. Zhu and K. G. Derpanis, "From actemes to action: A strongly-supervised
representation for detailed action understanding", ICCV 2013. The dataset is available at
https://dreamdragon.github.io/PennAction/.

MediaPipe Pose: V. Bazarevsky et al., "BlazePose: On-device real-time body pose tracking", 2020.

The full list of references is in the project report.
