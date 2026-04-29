#!/bin/bash

# Start the FastAPI backend in the background
echo "🚀 Starting FastAPI Backend (Port 8000)..."
ai-researcher --mode server &

# Wait for a moment to let the backend initialize
# (Streamlit handles connection retries gracefully)
sleep 2

# Start the Streamlit UI in the foreground
# Use the PORT environment variable if provided (default to 8501)
PORT=${PORT:-8501}
echo "📊 Starting Streamlit UI (Port $PORT)..."
exec ai-researcher --mode ui --port "$PORT"
