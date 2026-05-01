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
for i in $(seq 1 15); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ FastAPI is ready!"
    break
  fi
  echo "  Attempt $i/15 — retrying in 2s..."
  sleep 2
done

# Start the Streamlit UI in the foreground on Cloud Run's expected port
echo "📊 Starting Streamlit UI (Port $UI_PORT)..."
exec ai-researcher --mode ui --port "$UI_PORT"
