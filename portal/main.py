"""Single-address deployment: the aggregation server at /, client nodes at /node/<k>/.

Each node is its own copy of node/main.py with its own models, data, buffer and
threads, and it still reaches the server only through the HTTP API (/fl/*), the
same as when the nodes run on separate machines. One address and one login page
serve the admin console and all client pages.

Run:  PORT=8000 uvicorn portal.main:app --port 8000
"""
import importlib.util, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PORT = int(os.environ.get('PORT', '8000'))
N_NODES = int(os.environ.setdefault('N_NODES', '3'))
os.environ['PORTAL'] = '1'

from server import main as server  # noqa: E402  (reads PORTAL, so import after setting it)


def load_node(k):
    """A separate module object per node, so every node keeps its own globals."""
    os.environ['NODE_ID'] = str(k)
    os.environ['SERVER_URL'] = f'http://127.0.0.1:{PORT}'
    os.environ['SERVER_PUBLIC_URL'] = ''
    os.environ['PUBLIC_URL'] = f'/node/{k}/'
    spec = importlib.util.spec_from_file_location(f'node_{k}', os.path.join(ROOT, 'node', 'main.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NODES = {k: load_node(k) for k in range(1, N_NODES + 1)}
app = server.app
# mounted sub-apps do not get startup events, so the server starts them itself
for k, mod in NODES.items():
    app.mount(f'/node/{k}', mod.app, name=f'node{k}')
    for h in mod.app.router.on_startup:
        app.router.on_startup.append(h)
