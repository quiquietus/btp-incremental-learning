"""Central server: users and roles, federated rounds, FedAdam aggregation with
per-client contributions, and the live admin console.

Run locally:  uvicorn server.main:app --port 8000
"""
import asyncio, json, os, sqlite3, threading, time, math
# Shapley scoring is many tiny matmuls; one BLAS thread is faster than four here
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
from collections import deque

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from btpcore import fl_core as FL
from btpcore import model as NPM
from btpcore.learn import macro_f1
from common import auth
from common.wire import (MODELS, NODE_DATA, decode_weights, encode_weights, load_config, load_model, payload_bytes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(ROOT, 'data'))
os.makedirs(DATA_DIR, exist_ok=True)
N_NODES = int(os.environ.get('N_NODES', '3'))
STEP_DELAY = float(os.environ.get('STEP_DELAY', '1.2'))   # pause between aggregation steps so people can follow them live

app = FastAPI(title='BTP federated server')
from fastapi.middleware.cors import CORSMiddleware
# node-hosted client pages fetch round history from here (Bearer auth, no cookies)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
CFG = load_config()

# users (SQLite)
DB = os.path.join(DATA_DIR, 'server.db')


def db():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, pw TEXT, role TEXT, node_id INTEGER, created REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS audit (ts REAL, user TEXT, action TEXT, detail TEXT)')
        if c.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0:
            admin_pw = os.environ.get('ADMIN_PASSWORD', 'admin123')
            demo_pw = os.environ.get('DEMO_PASSWORD', 'demo123')
            rows = [('admin', admin_pw, 'admin', None), ('viewer', demo_pw, 'viewer', None)]
            rows += [(f'client{k}', demo_pw, 'operator', k) for k in range(1, N_NODES + 1)]
            for u, pw, r, n in rows:
                c.execute('INSERT INTO users VALUES (?,?,?,?,?)', (u, auth.hash_password(pw), r, n, time.time()))


def audit(user, action, detail=''):
    with db() as c:
        c.execute('INSERT INTO audit VALUES (?,?,?,?)', (time.time(), user, action, detail))


init_db()

# federated state
LOCK = threading.RLock()
STATE = {
    'version': int(CFG.get('start_round', 0)),       # global model version = number of finished rounds
    'phase': 'idle',                                  # idle | collecting | aggregating
    'round': None, 'expected': [], 'received': {}, 'opened_at': None,
    'nodes': {k: {'last_seen': None, 'status': 'never connected', 'n': 0, 'labelled': 0, 'url': None,
                  'personal_version': None, 'has_version': None} for k in range(1, N_NODES + 1)},
    'settings': {'algorithm': 'fedadam', 'eta': CFG['fedadam']['eta'], 'beta1': CFG['fedadam']['beta1'],
                 'beta2': CFG['fedadam']['beta2'], 'tau': CFG['fedadam']['tau'], 'mu': CFG.get('mu', 0.01),
                 'ditto_lambda': CFG.get('ditto_lambda', 0.1), 'local_steps': int(os.environ.get('LOCAL_STEPS', '12')),
                 'batch': 32, 'round_timeout': int(os.environ.get('ROUND_TIMEOUT', '240')), 'auto_interval': 0,
                 'shapley': True},
}
GLOBAL = load_model('global.npz', CFG)
if os.path.exists(os.path.join(MODELS, 'fedadam_state.npz')):
    with np.load(os.path.join(MODELS, 'fedadam_state.npz')) as z:
        ADAM = FL.FedAdamState.from_npz_dict({k: z[k] for k in z.files})
else:
    ADAM = FL.FedAdamState(GLOBAL)
HISTORY = []          # live round records
SIM_HISTORY = []      # rounds from the Kaggle simulation, shown for context
_fh = os.path.join(MODELS, 'fl_history.json')
if os.path.exists(_fh):
    with open(_fh) as f:
        SIM_HISTORY = json.load(f)
SERVER_VAL = None
_sv = os.path.join(NODE_DATA, 'server_val.npz')
if os.path.exists(_sv):
    with np.load(_sv) as z:
        cap = int(os.environ.get('SERVER_VAL_WINDOWS', '240'))
        X, y = z['X'].astype(np.float32), z['y']
        idx = np.random.default_rng(0).permutation(len(y))[:cap]
        SERVER_VAL = (X[idx], y[idx])
K = len(CFG['classes'])


def eval_val(p):
    if SERVER_VAL is None:
        return None
    X, y = SERVER_VAL
    pred = NPM.predict_proba(p, X, batch=64).argmax(1)
    return macro_f1(y, pred, K)


# live events
EVENTS = deque(maxlen=400)
SUBSCRIBERS = set()
LOOP = None


def emit(kind, **data):
    ev = {'ts': time.time(), 'kind': kind, **data}
    EVENTS.append(ev)
    if LOOP is not None:
        for q in list(SUBSCRIBERS):
            LOOP.call_soon_threadsafe(q.put_nowait, ev)
    return ev


@app.on_event('startup')
async def _startup():
    global LOOP
    LOOP = asyncio.get_running_loop()
    asyncio.create_task(_watchdog())
    emit('server_started', version=STATE['version'], placeholder=CFG.get('placeholder', False),
         val_windows=0 if SERVER_VAL is None else len(SERVER_VAL[1]),
         global_summary=FL.tensor_summary({k: GLOBAL[k] for k in ('out_b',)}, 6))


async def _watchdog():
    """Closes a round on timeout and starts automatic rounds."""
    last_auto = time.time()
    while True:
        await asyncio.sleep(2)
        with LOCK:
            st = STATE
            if st['phase'] == 'collecting' and time.time() - st['opened_at'] > st['settings']['round_timeout']:
                if st['received']:
                    emit('round_timeout', round=st['round'], received=list(st['received'].keys()))
                    _start_aggregation()
                else:
                    emit('round_cancelled', round=st['round'], reason='no updates before the timeout')
                    st['phase'] = 'idle'; st['round'] = None
            ai = st['settings'].get('auto_interval') or 0
            if ai and st['phase'] == 'idle' and time.time() - last_auto > ai:
                last_auto = time.time()
                _open_round('auto')


# helpers
def online_nodes():
    now = time.time()
    return [k for k, n in STATE['nodes'].items() if n['last_seen'] and now - n['last_seen'] < 30]


def _open_round(who):
    st = STATE
    if st['phase'] != 'idle':
        raise HTTPException(409, f"A round is already {st['phase']}")
    exp = online_nodes()
    if not exp:
        raise HTTPException(409, 'No client node is online. Open the client pages so the nodes wake up.')
    st.update(phase='collecting', round=st['version'] + 1, expected=exp, received={}, opened_at=time.time())
    emit('round_opened', round=st['round'], expected=exp, by=who, settings=st['settings'],
         global_version=st['version'])


def _start_aggregation():
    STATE['phase'] = 'aggregating'
    threading.Thread(target=_aggregate_thread, daemon=True).start()


def _aggregate_thread():
    global GLOBAL
    try:
        with LOCK:
            r = STATE['round']; s = dict(STATE['settings'])
            ups = [STATE['received'][k] for k in sorted(STATE['received'])]
            w_t = {k: v.copy() for k, v in GLOBAL.items()}
        N = sum(u['n'] for u in ups)
        emit('agg_step', round=r, step=1, title='Updates received',
             clients=[{'node': u['node'], 'n': u['n'], 'update_norm': FL.flat_norm(FL.sub(u['weights'], w_t)),
                       'metrics': u.get('metrics', {}), 'bytes': payload_bytes(u['weights'])} for u in ups])
        time.sleep(STEP_DELAY)
        emit('agg_step', round=r, step=2, title='Data weights p_k = n_k / N', N=N,
             clients=[{'node': u['node'], 'n': u['n'], 'p': u['n'] / N} for u in ups])
        time.sleep(STEP_DELAY)
        cfg = {'algorithm': s['algorithm'], 'eta': s['eta'], 'beta1': s['beta1'], 'beta2': s['beta2'], 'tau': s['tau']}
        def prog(S):
            if S is None:
                emit('agg_progress', round=r, text='Shapley values done')
            else:
                emit('agg_progress', round=r, text='Scoring the model built from clients ' + (', '.join(map(str, S)) or 'none (old model)'))
        new_w, rec = FL.aggregate(w_t, ups, cfg, state=ADAM, eval_fn=eval_val if (s['shapley'] and SERVER_VAL is not None) else None,
                                  round_no=r, progress=prog)
        rec['settings'] = s; rec['finished'] = time.time()
        emit('agg_step', round=r, step=3, title='Weighted sum: Delta = sum p_k d_k', delta_norm=rec['delta_norm'],
             layer_delta_norms=rec['layer_delta_norms'], delta_first_values=rec['delta_first_values'], trace=rec.get('trace'))
        time.sleep(STEP_DELAY)
        emit('agg_step', round=r, step=4, title='FedAdam server step' if s['algorithm'] == 'fedadam' else 'FedAvg step (w + Delta)',
             adam=rec.get('adam'), step_norm=rec['step_norm'], layer_step_mean_abs=rec['layer_step_mean_abs'],
             step_first_values=rec['step_first_values'], old_first_values=rec['old_first_values'],
             new_first_values=rec['new_first_values'], trace=rec.get('trace'))
        time.sleep(STEP_DELAY)
        with LOCK:
            GLOBAL = new_w
            STATE['version'] = r
            HISTORY.append(rec)
            STATE['phase'] = 'idle'; STATE['round'] = None; STATE['received'] = {}
            NPM.save(os.path.join(DATA_DIR, 'global_live.npz'), GLOBAL)
        emit('agg_step', round=r, step=5, title='New global model and contributions', record=rec)
        emit('round_done', round=r, version=r, val_before=rec.get('val_before'), val_after=rec.get('val_after'))
    except Exception as e:
        import traceback; traceback.print_exc()
        with LOCK:
            STATE['phase'] = 'idle'; STATE['round'] = None
        emit('round_failed', error=str(e))


# API: auth and users
class Login(BaseModel):
    username: str
    password: str


@app.post('/api/auth/login')
def login(body: Login):
    with db() as c:
        row = c.execute('SELECT * FROM users WHERE username=?', (body.username.strip().lower(),)).fetchone()
    if not row or not auth.verify_password(body.password, row['pw']):
        raise HTTPException(401, 'Wrong username or password')
    audit(row['username'], 'login')
    return {'token': auth.make_token(row['username'], row['role'], row['node_id']), 'role': row['role'],
            'node_id': row['node_id'], 'username': row['username']}


@app.get('/api/me')
def me(user=Depends(auth.current_user)):
    return user


class NewUser(BaseModel):
    username: str
    password: str
    role: str
    node_id: int | None = None


@app.get('/api/users')
def list_users(user=Depends(auth.require('admin'))):
    with db() as c:
        return [dict(username=r['username'], role=r['role'], node_id=r['node_id']) for r in c.execute('SELECT * FROM users ORDER BY role, username')]


@app.post('/api/users')
def create_user(body: NewUser, user=Depends(auth.require('admin'))):
    if body.role not in auth.ROLES:
        raise HTTPException(400, 'Unknown role')
    if body.role == 'operator' and not body.node_id:
        raise HTTPException(400, 'An operator needs a node number')
    if len(body.password) < 6:
        raise HTTPException(400, 'Password must have at least 6 characters')
    with db() as c:
        try:
            c.execute('INSERT INTO users VALUES (?,?,?,?,?)', (body.username.strip().lower(), auth.hash_password(body.password), body.role, body.node_id, time.time()))
        except sqlite3.IntegrityError:
            raise HTTPException(409, 'User exists')
    audit(user['sub'], 'create_user', body.username)
    return {'ok': True}


@app.delete('/api/users/{username}')
def delete_user(username: str, user=Depends(auth.require('admin'))):
    if username == user['sub']:
        raise HTTPException(400, 'You cannot delete yourself')
    with db() as c:
        c.execute('DELETE FROM users WHERE username=?', (username,))
    audit(user['sub'], 'delete_user', username)
    return {'ok': True}


@app.get('/api/audit')
def get_audit(user=Depends(auth.require('admin'))):
    with db() as c:
        return [dict(r) for r in c.execute('SELECT * FROM audit ORDER BY ts DESC LIMIT 200')]


# API: federation (people)
def public_state():
    st = STATE
    return {'version': st['version'], 'phase': st['phase'], 'round': st['round'], 'expected': st['expected'],
            'received': sorted(st['received'].keys()), 'opened_at': st['opened_at'], 'now': time.time(),
            'nodes': st['nodes'], 'online': online_nodes(), 'settings': st['settings'],
            'placeholder_model': CFG.get('placeholder', False), 'classes': CFG['classes'],
            'server_val_windows': 0 if SERVER_VAL is None else len(SERVER_VAL[1]),
            'global_summary': FL.tensor_summary(GLOBAL, 4)}


@app.get('/api/state')
def get_state(user=Depends(auth.current_user)):
    with LOCK:
        return public_state()


@app.post('/api/rounds/start')
def start_round(user=Depends(auth.require('admin'))):
    with LOCK:
        _open_round(user['sub'])
    audit(user['sub'], 'start_round', str(STATE['round']))
    return {'round': STATE['round']}


class Settings(BaseModel):
    algorithm: str | None = None
    eta: float | None = None
    mu: float | None = None
    ditto_lambda: float | None = None
    local_steps: int | None = None
    round_timeout: int | None = None
    auto_interval: int | None = None
    shapley: bool | None = None


@app.put('/api/settings')
def put_settings(body: Settings, user=Depends(auth.require('admin'))):
    ch = {k: v for k, v in body.model_dump().items() if v is not None}
    if 'algorithm' in ch and ch['algorithm'] not in ('fedadam', 'fedavg'):
        raise HTTPException(400, 'algorithm must be fedadam or fedavg')
    with LOCK:
        STATE['settings'].update(ch)
    audit(user['sub'], 'settings', json.dumps(ch))
    emit('settings_changed', by=user['sub'], settings=STATE['settings'])
    return STATE['settings']


@app.get('/api/rounds')
def rounds(user=Depends(auth.current_user)):
    def short(r, sim):
        return {'round': r['round'], 'simulated': sim, 'val_before': r.get('val_before'), 'val_after': r.get('val_after'),
                'global_test_macro_f1': r.get('global_test_macro_f1'),
                'clients': [{k: c.get(k) for k in ('node', 'n', 'p', 'share', 'alignment', 'shapley', 'update_norm')} for c in r['clients']]}
    return {'simulated': [short(r, True) for r in SIM_HISTORY], 'live': [short(r, False) for r in HISTORY]}


@app.get('/api/rounds/{n}')
def round_detail(n: int, user=Depends(auth.current_user)):
    for r in HISTORY[::-1]:
        if r['round'] == n:
            return r
    for r in SIM_HISTORY[::-1]:
        if r['round'] == n:
            return {**r, 'simulated': True}
    raise HTTPException(404, 'No such round')


@app.get('/api/events')
def events(user=Depends(auth.current_user)):
    return list(EVENTS)[-150:]


@app.websocket('/ws/events')
async def ws_events(ws: WebSocket):
    try:
        auth.decode_token(ws.query_params.get('token', ''))
    except HTTPException:
        await ws.close(code=4401); return
    await ws.accept()
    q = asyncio.Queue()
    SUBSCRIBERS.add(q)
    try:
        while True:
            ev = await q.get()
            await ws.send_text(json.dumps(ev, default=float))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        SUBSCRIBERS.discard(q)


# API: federation (nodes)
def node_auth(node_id: int, x_node_token: str):
    if node_id not in STATE['nodes'] or not auth.check_node_token(node_id, x_node_token):
        raise HTTPException(403, 'Bad node token')


class Heartbeat(BaseModel):
    node_id: int
    status: str = ''
    n: int = 0
    labelled: int = 0
    url: str | None = None
    personal_version: int | None = None
    has_version: int | None = None


@app.post('/fl/heartbeat')
def heartbeat(hb: Heartbeat, x_node_token: str = Header('')):
    node_auth(hb.node_id, x_node_token)
    with LOCK:
        n = STATE['nodes'][hb.node_id]
        was_offline = not n['last_seen'] or time.time() - n['last_seen'] > 30
        n.update(last_seen=time.time(), status=hb.status, n=hb.n, labelled=hb.labelled, url=hb.url,
                 personal_version=hb.personal_version, has_version=hb.has_version)
        if was_offline:
            emit('node_online', node=hb.node_id, url=hb.url)
        st = STATE
        must_train = st['phase'] == 'collecting' and hb.node_id in st['expected'] and hb.node_id not in st['received']
        return {'version': st['version'], 'phase': st['phase'], 'round': st['round'], 'train': must_train,
                'settings': st['settings']}


@app.get('/fl/global')
def get_global(node_id: int, x_node_token: str = Header('')):
    node_auth(node_id, x_node_token)
    with LOCK:
        w = {k: v.copy() for k, v in GLOBAL.items()}; v = STATE['version']
        last = HISTORY[-1] if HISTORY and HISTORY[-1]['round'] == v else None
    summ = FL.tensor_summary(w, 4)
    emit('global_sent', node=node_id, version=v, bytes=payload_bytes(w), first_values={k: summ[k]['first'] for k in ('l0f_b', 'att_v', 'out_b')})
    return {'version': v, 'weights': encode_weights(w), 'bytes': payload_bytes(w), 'summary': summ,
            'last_round': None if last is None else {k: last.get(k) for k in ('round', 'val_before', 'val_after', 'step_norm', 'delta_norm', 'trace', 'adam')} |
            {'mine': next((c for c in last['clients'] if c['node'] == node_id), None)}}


class Update(BaseModel):
    node_id: int
    round: int
    n: int
    weights: dict
    metrics: dict = {}


@app.post('/fl/update')
def post_update(up: Update, x_node_token: str = Header('')):
    node_auth(up.node_id, x_node_token)
    w = decode_weights(up.weights)
    with LOCK:
        st = STATE
        if st['phase'] != 'collecting' or up.round != st['round']:
            raise HTTPException(409, 'No open round with that number')
        if set(w) != set(GLOBAL) or any(w[k].shape != GLOBAL[k].shape for k in w):
            raise HTTPException(400, 'Weights do not match the model')
        st['received'][up.node_id] = {'node': up.node_id, 'n': max(1, int(up.n)), 'weights': w, 'metrics': up.metrics}
        d = FL.sub(w, GLOBAL)
        emit('update_received', round=st['round'], node=up.node_id, n=up.n, bytes=payload_bytes(w),
             update_norm=FL.flat_norm(d), metrics=up.metrics,
             first_values={k: FL.sample_values(w[k], 4) for k in ('l0f_b', 'att_v', 'out_b')},
             received=sorted(st['received']), expected=st['expected'])
        if set(st['expected']) <= set(st['received']):
            _start_aggregation()
    return {'ok': True}


# pages
WEB = os.path.join(ROOT, 'web')
app.mount('/shared', StaticFiles(directory=os.path.join(WEB, 'shared')), name='shared')


@app.get('/healthz')
def health():
    return {'ok': True, 'role': 'server', 'version': STATE['version']}


@app.get('/')
def index():
    return FileResponse(os.path.join(WEB, 'admin', 'index.html'))


@app.get('/admin')
def admin_page():
    return FileResponse(os.path.join(WEB, 'admin', 'admin.html'))


@app.get('/api/config')
def config_public():
    return {'role': 'server', 'classes': CFG['classes'], 'placeholder_model': CFG.get('placeholder', False)}
