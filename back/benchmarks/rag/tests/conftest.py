import sys
from pathlib import Path

# Add benchmarks/rag directory to sys.path so tests can import run_pipeline, core, etc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Add the main back directory containing the src package to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
