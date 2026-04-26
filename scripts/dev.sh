#!/usr/bin/env bash
# Start all three services for local development.
# Run from the project root: bash scripts/dev.sh
# Requires: .env file in project root with ABLY_API_KEY etc.

set -e

# Load environment variables from .env if present
if [ -f .env ]; then
  set -a && source .env && set +a
fi

echo "Starting FastAPI backend on http://localhost:8000 ..."
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

echo "Starting frontend dev server ..."
cd frontend
npm install --silent
npm run dev &
FRONTEND_PID=$!
cd ..

echo "Starting audio inference pipeline ..."
python -m inference.live_inference --location "Main Entrance" &
AUDIO_PID=$!

echo "Starting vision inference pipeline ..."
python -m vision.live_inference --location "Main Entrance" &
VISION_PID=$!

echo ""
echo "Services running:"
echo "  API      → http://localhost:8000  (PID $API_PID)"
echo "  Frontend → http://localhost:5173  (PID $FRONTEND_PID)"
echo "  Audio    → background             (PID $AUDIO_PID)"
echo "  Vision   → background             (PID $VISION_PID)"
echo ""
echo "Press Ctrl+C to stop all services."

trap "kill $API_PID $FRONTEND_PID $AUDIO_PID $VISION_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
