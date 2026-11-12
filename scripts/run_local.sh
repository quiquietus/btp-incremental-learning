#!/usr/bin/env bash
# The whole system on this computer at http://localhost:8000 (portal, server and 3 client nodes).
set -e
cd "$(dirname "$0")/.."
export JWT_SECRET=${JWT_SECRET:-local-demo-secret} MALLOC_ARENA_MAX=2 PORT=${PORT:-8000}
echo
echo "  Portal: http://localhost:$PORT   (admin / admin123, client1..3 / demo123, viewer / demo123)"
echo
exec ${PYTHON:-python3} -m uvicorn portal.main:app --port "$PORT" --log-level warning
