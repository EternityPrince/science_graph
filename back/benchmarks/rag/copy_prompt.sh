#!/usr/bin/env bash
# Science Graph — Copy Benchmark Run Results Helper Script.
# Automatically resolves the correct virtual environment python interpreter.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VENV="$SCRIPT_DIR/../../../.venv/bin/python"

if [ -f "$PYTHON_VENV" ]; then
    "$PYTHON_VENV" "$SCRIPT_DIR/copy_prompt.py" "$@"
else
    # Fallback to system python3 if venv not found
    python3 "$SCRIPT_DIR/copy_prompt.py" "$@"
fi
