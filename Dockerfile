FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
# ROLE=server or ROLE=node (with NODE_ID and SERVER_URL)
CMD ["sh", "-c", "if [ \"$ROLE\" = node ]; then exec uvicorn node.main:app --host 0.0.0.0 --port $PORT; else exec uvicorn server.main:app --host 0.0.0.0 --port $PORT; fi"]
