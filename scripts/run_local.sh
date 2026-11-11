#!/usr/bin/env bash
# Run the server (:8000) and three client nodes (:8001-8003) on this computer without Docker.
# Usage: bash scripts/run_local.sh      (Ctrl+C stops all four)
set -e
cd "$(dirname "$0")/.."
export JWT_SECRET=${JWT_SECRET:-local-demo-secret}
export SERVER_URL=http://localhost:8000
PY=${PYTHON:-python3}
pids=()
trap 'kill ${pids[@]} 2>/dev/null' EXIT INT TERM
$PY -m uvicorn server.main:app --port 8000 --log-level warning & pids+=($!)
sleep 2
for k in 1 2 3; do
  NODE_ID=$k PUBLIC_URL=http://localhost:800$k $PY -m uvicorn node.main:app --port 800$k --log-level warning & pids+=($!)
done
echo
echo "  Admin console:  http://localhost:8000   (admin / admin123)"
echo "  Client nodes:   http://localhost:8001  :8002  :8003   (client1..3 / demo123)"
echo
wait
