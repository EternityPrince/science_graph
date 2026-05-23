#!/bin/bash

# Cyber-Academic Brutalist Local Dev Launcher
# Colors for brutalist output styling
CYAN='\033[0;36m'
AMBER='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}===================================================${NC}"
echo -e "${CYAN}🔬 SCIENCE GRAPH EXPLORER (Local Dev Launcher)${NC}"
echo -e "${CYAN}===================================================${NC}"

# Ensure we are in the correct root directory
if [ ! -d "back" ] || [ ! -d "frontend" ]; then
    echo -e "${RED}Error: Launcher must be run from the repository root directory (where back/ and frontend/ folders reside).${NC}"
    exit 1
fi

# Track child PIDs
BACK_PID=""
FRONT_PID=""

# Helper to kill processes on ports 3000 and 8000
cleanup_ports() {
    if lsof -t -i:3000 &>/dev/null; then
        echo -e "${AMBER}[SYSTEM] Releasing port 3000...${NC}"
        kill -9 $(lsof -t -i:3000) 2>/dev/null
    fi
    if lsof -t -i:8000 &>/dev/null; then
        echo -e "${CYAN}[SYSTEM] Releasing port 8000...${NC}"
        kill -9 $(lsof -t -i:8000) 2>/dev/null
    fi
}

# Cleanup handler on exit or Ctrl+C
cleanup() {
    # Disable trap to prevent loop
    trap - SIGINT SIGTERM EXIT
    
    echo -e "\n${AMBER}[SYSTEM] Gracefully shutting down services...${NC}"
    
    # Terminate backend process
    if [ -n "$BACK_PID" ]; then
        if kill -0 "$BACK_PID" 2>/dev/null; then
            echo -e "${CYAN}[SYSTEM] Stopping Python FastAPI Backend (PID: $BACK_PID)...${NC}"
            kill "$BACK_PID" 2>/dev/null
            wait "$BACK_PID" 2>/dev/null
        fi
    fi

    # Terminate frontend process
    if [ -n "$FRONT_PID" ]; then
        if kill -0 "$FRONT_PID" 2>/dev/null; then
            echo -e "${AMBER}[SYSTEM] Stopping Next.js Frontend (PID: $FRONT_PID)...${NC}"
            kill "$FRONT_PID" 2>/dev/null
            wait "$FRONT_PID" 2>/dev/null
        fi
    fi

    # Double check ports cleanup
    cleanup_ports

    echo -e "${CYAN}[SYSTEM] All services stopped. Goodbye!${NC}"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

# Clean up any lingering processes on start to prevent port conflicts
cleanup_ports

# 1. Setup Backend
echo -e "${CYAN}[SYSTEM] Configuring Python Backend...${NC}"
cd back

# Verify package manager and dependencies
if command -v uv &> /dev/null; then
    if [ ! -d ".venv" ]; then
        echo -e "${CYAN}[SYSTEM] Virtual environment (.venv) not found. Running 'uv sync'...${NC}"
        uv sync
    fi
    # Use uv to run backend
    CMD_BACK="uv run python main.py serve --no-open --host 127.0.0.1 --port 8000"
else
    # Fallback to python venv
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        CMD_BACK="python3 main.py serve --no-open --host 127.0.0.1 --port 8000"
    else
        echo -e "${RED}[SYSTEM] Error: Neither 'uv' nor a python '.venv' was found. Please run 'uv sync' or set up a virtual environment in the 'back' folder first.${NC}"
        exit 1
    fi
fi
cd ..

# 2. Setup Frontend
echo -e "${AMBER}[SYSTEM] Configuring Next.js Frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo -e "${AMBER}[SYSTEM] node_modules not found. Running 'npm install'...${NC}"
    npm install
fi
cd ..

# Set Next.js backend proxy environment variable
export BACKEND_URL="http://localhost:8000"

echo -e "${CYAN}[SYSTEM] Launching services... (Ctrl+C to stop)${NC}"

# Start Backend in background with Cyan prefix
export PYTHONUNBUFFERED=1
cd back
$CMD_BACK 2>&1 | awk '{ print "\033[0;36m[BACK] \033[0m " $0; fflush() }' &
BACK_PID=$!
cd ..

# Start Frontend in background with Amber prefix
cd frontend
npx next dev 2>&1 | awk '{ print "\033[0;33m[FRONT]\033[0m " $0; fflush() }' &
FRONT_PID=$!
cd ..

# Wait for background jobs
wait $BACK_PID $FRONT_PID
