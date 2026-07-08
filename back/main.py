# Suppress noisy external library output BEFORE any imports that trigger them.
import os
import warnings

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Suppress benign leaked semaphore warnings on shutdown from multiprocessing/PyTorch/MLX on macOS
warnings.filterwarnings(
    "ignore", category=UserWarning, message="resource_tracker: There appear to be"
)
if "PYTHONWARNINGS" in os.environ:
    if "ignore:resource_tracker:UserWarning" not in os.environ["PYTHONWARNINGS"]:
        os.environ["PYTHONWARNINGS"] += ",ignore:resource_tracker:UserWarning"
else:
    os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"

from src.cli import app

if __name__ == "__main__":
    app()
