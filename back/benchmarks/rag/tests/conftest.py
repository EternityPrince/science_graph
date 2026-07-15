import sys
from pathlib import Path

import pytest

# Add benchmarks/rag directory to sys.path so tests can import run_pipeline, core, etc.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Add the main back directory containing the src package to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.fixture(autouse=True)
def _restore_baseline_config_patch():
    """Undo run_benchmarks CUSTOM monkeypatches left by other tests in the same session."""
    from config_creator import restore_baseline_config_patch

    restore_baseline_config_patch()
    yield
    restore_baseline_config_patch()
