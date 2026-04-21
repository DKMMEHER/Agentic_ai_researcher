#!/bin/bash

# Start the FastAPI backend in the background
echo "🚀 Starting FastAPI Backend (Port 8000)..."
ai-researcher --mode server &

# Wait for a moment to let the backend initialize
# (Streamlit handles connection retries gracefully)
sleep 2

# Start the Streamlit UI in the foreground
echo "📊 Starting Streamlit UI (Port 8501)..."
exec ai-researcher --mode ui
