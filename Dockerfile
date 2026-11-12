FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000 MALLOC_ARENA_MAX=2
# ROLE unset: portal (server + 3 nodes on one port). ROLE=server or ROLE=node for separate machines.
CMD ["sh", "-c", "case \"$ROLE\" in node) exec uvicorn node.main:app --host 0.0.0.0 --port $PORT;; server) exec uvicorn server.main:app --host 0.0.0.0 --port $PORT;; *) exec uvicorn portal.main:app --host 0.0.0.0 --port $PORT;; esac"]
