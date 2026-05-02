#!/bin/bash
set -e

# Cloud Run injects $PORT — Streamlit (the public-facing service) must bind to it.
# FastAPI runs internally on port 8000 and is accessed by Streamlit via localhost.
UI_PORT=${PORT:-8501}

echo "--- STARTUP DIAGNOSTICS ---"
echo "Current Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Environment: PORT=$PORT, CHECKPOINT_BACKEND=$CHECKPOINT_BACKEND"
echo "---------------------------"

# Start the FastAPI backend in the background on port 8000
echo "🚀 Starting FastAPI Backend (Port 8000)..."
python -m ai_researcher.cli --mode server > /dev/stdout 2> /dev/stderr &
BACKEND_PID=$!

# Wait for FastAPI to become healthy before starting Streamlit
echo "⏳ Waiting for FastAPI to be ready..."
FASTAPI_READY=false
for i in $(seq 1 15); do
  if wget --timeout=2 -q -O - http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ FastAPI is ready!"
    FASTAPI_READY=true
    break
  fi
  echo "  Attempt $i/15 — retrying in 2s..."
  sleep 2
done

if [ "$FASTAPI_READY" = false ]; then
  echo "❌ FastAPI failed to start within 30 seconds. Checking backend status..."
  if ! ps -p $BACKEND_PID > /dev/null; then
    echo "🚨 ERROR: FastAPI process has CRASHED. Check the logs above for Python Tracebacks."
  else
    echo "⚠️ WARNING: FastAPI is still running but not responding to /health. Check for port conflicts or network issues."
  fi
  exit 1
fi

# Start the Streamlit UI in the foreground on Cloud Run's expected port
echo "📊 Starting Streamlit UI (Port $UI_PORT)..."
exec python -m ai_researcher.cli --mode ui --port "$UI_PORT"
