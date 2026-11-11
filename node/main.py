"""Client node: live activity recognition from browser landmarks, incremental
learning from labelled clips, and the federated-learning client.

Run locally:  NODE_ID=1 SERVER_URL=http://localhost:8000 uvicorn node.main:app --port 8001
"""
import asyncio, json, os, threading, time, copy
# BLAS threads fight each other on a 0.1-CPU host; local rounds ran 40x slower without this
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
from collections import deque

import httpx
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from btpcore import features as FT
from btpcore import fl_core as FL
from btpcore import model as NPM
from btpcore.learn import Adam, Reservoir, evaluate, regularised_step
from common import auth
from common.wire import (NODE_DATA, decode_weights, encode_weights, load_config, load_model, load_sequences,
                         payload_bytes, seq_windows)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_ID = int(os.environ.get('NODE_ID', '1'))
SERVER_URL = os.environ.get('SERVER_URL', 'http://localhost:8000').rstrip('/')
if SERVER_URL and not SERVER_URL.startswith('http'):
    SERVER_URL = 'https://' + SERVER_URL
PUBLIC_URL = os.environ.get('RENDER_EXTERNAL_URL') or os.environ.get('PUBLIC_URL')
NODE_TOKEN = auth.node_token(NODE_ID)

app = FastAPI(title=f'BTP client node {NODE_ID}')
CFG = load_config()
CLASSES = CFG['classes']; K = len(CLASSES); T = CFG['T']
MU = np.array(CFG['norm_mean'], np.float32); SD = np.array(CFG['norm_std'], np.float32)

# models and data
LOCK = threading.RLock()
PERSONAL = load_model(f'personal_{NODE_ID}.npz', CFG)      # Ditto personal model v: serves predictions
GLOBAL = load_model('global.npz', CFG)                     # latest global model w (Ditto anchor)
_fisher_path = os.path.join(os.environ.get('MODELS_DIR', os.path.join(ROOT, 'models')), 'fisher.npz')
FISHER = NPM.load(_fisher_path) if os.path.exists(_fisher_path) else None
VERSIONS = {'personal': 0, 'global': int(CFG.get('start_round', 0))}

train_seqs = load_sequences(os.path.join(NODE_DATA, f'client_{NODE_ID}_train.npz'))
test_seqs = load_sequences(os.path.join(NODE_DATA, f'client_{NODE_ID}_test.npz'))
XTR, YTR = seq_windows(train_seqs, T, int(os.environ.get('TRAIN_STRIDE', '8')))
XTE, YTE = seq_windows(test_seqs, T, 8)
MAX_TEST = int(os.environ.get('MAX_TEST_WINDOWS', '200'))
if len(YTE) > MAX_TEST:
    _i = np.random.default_rng(0).permutation(len(YTE))[:MAX_TEST]; XTE, YTE = XTE[_i], YTE[_i]
RES = Reservoir(int(os.environ.get('RESERVOIR', '600')), 46, T, seed=NODE_ID)
if len(YTR):
    RES.add(XTR, YTR)
LABELLED_X, LABELLED_Y = [], []         # windows the operator labelled in the browser
PENDING_X, PENDING_Y = [], []           # buffer for the next incremental step
INC = {'steps': 0, 'last': None}
OPT_PERSONAL = Adam(PERSONAL, lr=CFG.get('incremental', {}).get('lr', 1e-4))

# node events (for the client page)
EVENTS = deque(maxlen=300)
SUBS = set()
LOOP = None
FLSTATE = {'status': 'starting', 'last_sent': None, 'last_received': None, 'server_ok': False, 'server_error': None,
           'eval': None}


def emit(kind, **data):
    ev = {'ts': time.time(), 'kind': kind, 'node': NODE_ID, **data}
    EVENTS.append(ev)
    if LOOP is not None:
        for q in list(SUBS):
            LOOP.call_soon_threadsafe(q.put_nowait, ev)
    return ev


def local_eval():
    with LOCK:
        p, g = copy.deepcopy(PERSONAL), copy.deepcopy(GLOBAL)
    if not len(YTE):
        return None
    return {'personal': evaluate(p, XTE, YTE, K), 'global': evaluate(g, XTE, YTE, K), 'n': int(len(YTE))}


# incremental learning
def incremental_step():
    """One update of the personal model from 16 labelled windows (+20% replay),
    with Ditto's pull towards the global model and the EWC term."""
    global PERSONAL
    inc = CFG.get('incremental', {})
    B = inc.get('buffer', 16)
    with LOCK:
        if len(PENDING_Y) < B:
            return None
        X = np.stack(PENDING_X[:B]); y = np.array(PENDING_Y[:B])
        del PENDING_X[:B]; del PENDING_Y[:B]
        nrep = int(round(inc.get('replay_ratio', 0.2) * B / (1 - inc.get('replay_ratio', 0.2))))
        rx, ry = RES.sample(nrep)
        if rx is not None:
            X = np.concatenate([X, rx]); y = np.concatenate([y, ry])
        p = copy.deepcopy(PERSONAL); anchor = copy.deepcopy(GLOBAL)
    before = NPM.predict_proba(p, X[:B]).argmax(1)
    t0 = time.time()
    st = regularised_step(p, OPT_PERSONAL, X, y, anchor=anchor, mu=CFG.get('ditto_lambda', 0.1),
                          fisher=FISHER, lam_ewc=CFG.get('lambda_ewc', 0.0) if FISHER is not None else 0.0)
    after = NPM.predict_proba(p, X[:B]).argmax(1)
    with LOCK:
        PERSONAL = p
        VERSIONS['personal'] += 1
        RES.add(X[:B], y[:B])
        INC['steps'] += 1
    INC['last'] = {'ce_loss': st['ce'], 'ditto_term': st['prox'], 'ewc_term': st['ewc'], 'grad_norm': st['grad_norm'],
                   'acc_on_batch_before': float((before == y[:B]).mean()), 'acc_on_batch_after': float((after == y[:B]).mean()),
                   'windows': B, 'replay': int(len(y) - B), 'ms': (time.time() - t0) * 1000, 'personal_version': VERSIONS['personal']}
    emit('incremental_step', **INC['last'])
    return INC['last']


def add_labelled(frames, label):
    """frames: (n, 46) raw features; windows (stride 4) are standardised and queued."""
    if len(frames) < 16:
        return 0
    W = FT.windows(np.asarray(frames, np.float32), T, 4)
    W = ((W - MU) / SD).astype(np.float32)
    with LOCK:
        for w in W:
            LABELLED_X.append(w); LABELLED_Y.append(label)
            PENDING_X.append(w); PENDING_Y.append(label)
        del LABELLED_X[:-3000]; del LABELLED_Y[:-3000]
    return len(W)


def run_incremental_async():
    def work():
        while incremental_step() is not None:
            pass
    threading.Thread(target=work, daemon=True).start()


# federated client
def fl_headers():
    return {'X-Node-Token': NODE_TOKEN}


def local_data():
    with LOCK:
        if LABELLED_X:
            X = np.concatenate([XTR, np.stack(LABELLED_X)]) if len(XTR) else np.stack(LABELLED_X)
            y = np.concatenate([YTR, np.array(LABELLED_Y)]) if len(YTR) else np.array(LABELLED_Y)
        else:
            X, y = XTR, YTR
    return X, y


def fetch_global(client):
    global GLOBAL
    r = client.get(f'{SERVER_URL}/fl/global', params={'node_id': NODE_ID}, headers=fl_headers(), timeout=120)
    r.raise_for_status()
    d = r.json()
    w = decode_weights(d['weights'])
    with LOCK:
        old = GLOBAL
        GLOBAL = w
        VERSIONS['global'] = d['version']
    change = FL.flat_norm(FL.sub(w, old))
    FLSTATE['last_received'] = {'version': d['version'], 'bytes': d['bytes'], 'at': time.time(), 'change_norm': change,
                                'summary': {k: d['summary'][k] for k in ('l0f_b', 'att_v', 'fc1_b', 'out_b')},
                                'last_round': d.get('last_round')}
    emit('global_received', **FLSTATE['last_received'])
    return w


def do_round(client, rnd, settings):
    """FedProx local training of the global copy, a Ditto update of the personal
    model, then upload of the global copy's weights."""
    global PERSONAL
    FLSTATE['status'] = f'round {rnd}: downloading the global model'
    emit('round_start', round=rnd, settings=settings)
    w0 = fetch_global(client)
    X, y = local_data()
    if not len(y):
        emit('round_skip', round=rnd, reason='no local data'); return
    rng = np.random.default_rng(rnd * 100 + NODE_ID)
    steps, B = int(settings.get('local_steps', 12)), int(settings.get('batch', 32))
    w = copy.deepcopy(w0)
    opt = Adam(w, lr=1e-3)
    ev_idx = rng.choice(len(y), min(len(y), 128), replace=False)
    loss_before = float(NPM.loss_and_grads(w, X[ev_idx], y[ev_idx])[0])
    FLSTATE['status'] = f'round {rnd}: FedProx local training ({steps} steps)'
    emit('local_training', round=rnd, steps=steps, batch=B, mu=settings.get('mu'), windows=int(len(y)), loss_before=loss_before)
    t0 = time.time()
    for s in range(steps):
        b = rng.choice(len(y), min(B, len(y)), replace=False)
        st = regularised_step(w, opt, X[b], y[b], anchor=w0, mu=float(settings.get('mu', 0.01)))
        if s % 4 == 3 or s == steps - 1:
            emit('local_step', round=rnd, step=s + 1, of=steps, ce=st['ce'], prox=st['prox'])
    loss_after = float(NPM.loss_and_grads(w, X[ev_idx], y[ev_idx])[0])
    train_sec = time.time() - t0
    # Ditto: personal model takes a few steps towards its own data, pulled to the received global
    FLSTATE['status'] = f'round {rnd}: Ditto personal update'
    with LOCK:
        pv = copy.deepcopy(PERSONAL)
    popt = Adam(pv, lr=CFG.get('incremental', {}).get('lr', 1e-4) * 5)
    for s in range(max(1, steps // 2)):
        b = rng.choice(len(y), min(B, len(y)), replace=False)
        regularised_step(pv, popt, X[b], y[b], anchor=w0, mu=float(settings.get('ditto_lambda', 0.1)))
    with LOCK:
        PERSONAL = pv; VERSIONS['personal'] += 1
    d = FL.sub(w, w0)
    groups = FL.layer_groups(list(w.keys()))
    sent = {'round': rnd, 'n': int(len(y)), 'bytes': payload_bytes(w), 'update_norm': FL.flat_norm(d),
            'layer_update_norms': {g: float(np.sqrt(sum((d[k] ** 2).sum() for k in ks))) for g, ks in groups.items()},
            'first_values': {k: {'received': FL.sample_values(w0[k], 4), 'sending': FL.sample_values(w[k], 4)} for k in ('l0f_b', 'att_v', 'out_b')},
            'loss_before': loss_before, 'loss_after': loss_after, 'train_seconds': train_sec, 'at': time.time()}
    FLSTATE['status'] = f'round {rnd}: uploading weights'
    r = client.post(f'{SERVER_URL}/fl/update', headers=fl_headers(), timeout=120,
                    json={'node_id': NODE_ID, 'round': rnd, 'n': int(len(y)), 'weights': encode_weights(w),
                          'metrics': {'loss_before': loss_before, 'loss_after': loss_after, 'local_steps': steps,
                                      'labelled_windows': len(LABELLED_Y), 'train_seconds': round(train_sec, 1)}})
    r.raise_for_status()
    FLSTATE['last_sent'] = sent
    emit('update_sent', **sent)
    FLSTATE['status'] = f'round {rnd}: waiting for the server to aggregate'


def hb_payload():
    return {'node_id': NODE_ID, 'status': FLSTATE['status'], 'n': int(len(YTR) + len(LABELLED_Y)),
            'labelled': len(LABELLED_Y), 'url': PUBLIC_URL, 'personal_version': VERSIONS['personal'],
            'has_version': VERSIONS['global']}


BUSY = threading.Event()


def keepalive():
    """While a round is being trained, keep telling the server we are alive (the main loop is busy)."""
    c = httpx.Client()
    while True:
        time.sleep(5)
        if BUSY.is_set():
            try:
                c.post(f'{SERVER_URL}/fl/heartbeat', headers=fl_headers(), timeout=30, json=hb_payload())
            except Exception:
                pass


def fl_loop():
    client = httpx.Client()
    done_round = None
    while True:
        try:
            hb = client.post(f'{SERVER_URL}/fl/heartbeat', headers=fl_headers(), timeout=60, json=hb_payload()).json()
            if not FLSTATE['server_ok']:
                emit('server_connected', server=SERVER_URL)
            FLSTATE['server_ok'] = True; FLSTATE['server_error'] = None
            if hb.get('train') and hb.get('round') != done_round:
                done_round = hb['round']
                BUSY.set()
                try:
                    do_round(client, hb['round'], hb['settings'])
                finally:
                    BUSY.clear()
            elif hb.get('version', 0) > VERSIONS['global'] and hb.get('phase') == 'idle':
                FLSTATE['status'] = 'downloading the new global model'
                fetch_global(client)
                FLSTATE['eval'] = local_eval()
                emit('local_eval', **(FLSTATE['eval'] or {}))
                FLSTATE['status'] = 'idle: personal model keeps learning from your labels'
            elif hb.get('phase') == 'idle' and not FLSTATE['status'].startswith('idle'):
                FLSTATE['status'] = 'idle: personal model keeps learning from your labels'
        except Exception as e:
            if FLSTATE['server_ok'] or FLSTATE['server_error'] is None:
                emit('server_unreachable', error=str(e)[:200], server=SERVER_URL)
            FLSTATE['server_ok'] = False; FLSTATE['server_error'] = str(e)[:200]
            FLSTATE['status'] = 'cannot reach the server (it may be waking up)'
            time.sleep(5)
        time.sleep(3)


@app.on_event('startup')
async def _startup():
    global LOOP
    LOOP = asyncio.get_running_loop()
    threading.Thread(target=fl_loop, daemon=True).start()
    threading.Thread(target=keepalive, daemon=True).start()
    def first_eval():
        FLSTATE['eval'] = local_eval()
    threading.Thread(target=first_eval, daemon=True).start()
    emit('node_started', train_windows=int(len(YTR)), test_windows=int(len(YTE)), placeholder=CFG.get('placeholder', False))


# API
class Login(BaseModel):
    username: str
    password: str


@app.post('/api/auth/login')
def login(body: Login):
    """Checked by the central server, which owns the user list."""
    try:
        r = httpx.post(f'{SERVER_URL}/api/auth/login', json=body.model_dump(), timeout=90)
    except Exception:
        raise HTTPException(503, 'The central server is not reachable yet. On free hosting it can take a minute to wake up; try again.')
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.json().get('detail', 'Login failed'))
    d = r.json()
    if d['role'] == 'operator' and d.get('node_id') != NODE_ID:
        raise HTTPException(403, f"This account belongs to client node {d.get('node_id')}, not node {NODE_ID}.")
    return d


def node_user(user=Depends(auth.current_user)):
    if user['role'] == 'admin' or user['role'] == 'viewer' or (user['role'] == 'operator' and user.get('node') == NODE_ID):
        return user
    raise HTTPException(403, 'Not your node')


@app.get('/api/config')
def config():
    return {'role': 'node', 'node_id': NODE_ID, 'classes': CLASSES, 'server_url': os.environ.get('SERVER_PUBLIC_URL', SERVER_URL), 'T': T,
            'fps': CFG.get('fps', 30), 'placeholder_model': CFG.get('placeholder', False)}


@app.get('/api/status')
def status(user=Depends(node_user)):
    return {'node_id': NODE_ID, 'versions': VERSIONS, 'fl': FLSTATE, 'train_windows': int(len(YTR)),
            'labelled_windows': len(LABELLED_Y), 'pending': len(PENDING_Y), 'incremental': INC,
            'personal_summary': FL.tensor_summary({k: PERSONAL[k] for k in ('out_b',)}, 4)}


@app.get('/api/events')
def events(user=Depends(node_user)):
    return list(EVENTS)[-120:]


@app.websocket('/ws/events')
async def ws_events(ws: WebSocket):
    try:
        auth.decode_token(ws.query_params.get('token', ''))
    except HTTPException:
        await ws.close(code=4401); return
    await ws.accept()
    q = asyncio.Queue(); SUBS.add(q)
    try:
        while True:
            await ws.send_text(json.dumps(await q.get(), default=float))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        SUBS.discard(q)


@app.websocket('/ws/stream')
async def ws_stream(ws: WebSocket):
    """Browser sends {t, lm, ar} per frame (lm: 33 x [x,y,z,visibility] or null).
    The node answers with the model's activity prediction every few frames."""
    try:
        user = auth.decode_token(ws.query_params.get('token', ''))
        node_user(user)
    except HTTPException:
        await ws.close(code=4401); return
    await ws.accept()
    sf = FT.StreamFeaturizer(T=T)
    history = deque(maxlen=int(30 * 20))        # last 20 s of raw feature frames, for labelling
    session = []                                # all frames since the clip started (upload mode)
    probs = deque(maxlen=5)
    since = 0
    every = 8
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            typ = msg.get('type', 'frame')
            if typ == 'frame':
                lm = msg.get('lm')
                n_new = sf.push(float(msg['t']), None if lm is None else np.asarray(lm, np.float32), float(msg.get('ar', 1.0)))
                if n_new:
                    new = sf.buf[-n_new:]
                    history.extend(new); session.extend(new)
                    since += n_new
                if lm is None or len(sf.buf) == 0:
                    if msg.get('report_missing'):
                        await ws.send_text(json.dumps({'type': 'nopose'}))
                    continue
                if since >= every and len(sf.buf) >= T:
                    since = 0
                    win = ((sf.window() - MU) / SD).astype(np.float32)
                    t0 = time.time()
                    with LOCK:
                        p = NPM.predict_proba(PERSONAL, win)[0]
                        g = NPM.predict_proba(GLOBAL, win)[0]
                    probs.append(p)
                    ps = np.mean(probs, 0)
                    top = np.argsort(-ps)[:3]
                    last = sf.buf[-1]
                    angles = {FT.FEATURE_NAMES[24 + i][4:]: round(float(last[24 + i]), 1) for i in range(8)}
                    await ws.send_text(json.dumps({
                        'type': 'pred', 'label': CLASSES[int(top[0])], 'p': float(ps[top[0]]),
                        'top3': [[CLASSES[int(i)], float(ps[i])] for i in top],
                        'global_label': CLASSES[int(np.argmax(g))], 'global_p': float(np.max(g)),
                        'ms': (time.time() - t0) * 1000, 'personal_version': VERSIONS['personal'],
                        'global_version': VERSIONS['global'], 'angles': angles,
                        'frames_buffered': len(sf.buf)}))
                elif len(sf.buf) < T:
                    await ws.send_text(json.dumps({'type': 'warming', 'frames_buffered': len(sf.buf), 'need': T}))
            elif typ == 'reset':
                sf.reset(); session.clear(); probs.clear()
            elif typ == 'label':
                if user['role'] != 'operator':
                    await ws.send_text(json.dumps({'type': 'error', 'error': 'Only the node operator can label clips'})); continue
                if msg.get('label') not in CLASSES:
                    await ws.send_text(json.dumps({'type': 'error', 'error': f"Unknown class {msg.get('label')!r}"})); continue
                label = CLASSES.index(msg['label'])
                frames = list(session) if msg.get('scope') == 'clip' else list(history)[-int(30 * float(msg.get('seconds', 3))):]
                n = add_labelled(frames, label)
                await ws.send_text(json.dumps({'type': 'labelled', 'label': msg['label'], 'windows': n, 'pending': len(PENDING_Y),
                                               'total': len(LABELLED_Y)}))
                emit('labelled', label=msg['label'], windows=n, total=len(LABELLED_Y))
                run_incremental_async()
    except (WebSocketDisconnect, RuntimeError):
        pass


# pages
WEB = os.path.join(ROOT, 'web')
app.mount('/shared', StaticFiles(directory=os.path.join(WEB, 'shared')), name='shared')


@app.get('/healthz')
def health():
    return {'ok': True, 'role': 'node', 'node_id': NODE_ID}


@app.get('/')
def index():
    return FileResponse(os.path.join(WEB, 'client', 'index.html'))


@app.get('/app')
def app_page():
    return FileResponse(os.path.join(WEB, 'client', 'client.html'))
