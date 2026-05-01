#!/bin/bash
set -e

# Cloud Run injects $PORT — Streamlit (the public-facing service) must bind to it.
# FastAPI runs internally on port 8000 and is accessed by Streamlit via localhost.
UI_PORT=${PORT:-8501}

# Start the FastAPI backend in the background on port 8000
echo "🚀 Starting FastAPI Backend (Port 8000)..."
ai-researcher --mode server &
BACKEND_PID=$!

# Wait for FastAPI to become healthy before starting Streamlit
echo "⏳ Waiting for FastAPI to be ready..."
FASTAPI_READY=false
for i in $(seq 1 15); do
  if wget -q -O - http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ FastAPI is ready!"
    FASTAPI_READY=true
    break
  fi
  echo "  Attempt $i/15 — retrying in 2s..."
  sleep 2
done

if [ "$FASTAPI_READY" = false ]; then
  echo "❌ FastAPI failed to start within 30 seconds. Exiting..."
  exit 1
fi

# Start the Streamlit UI in the foreground on Cloud Run's expected port
echo "📊 Starting Streamlit UI (Port $UI_PORT)..."
exec ai-researcher --mode ui --port "$UI_PORT"
