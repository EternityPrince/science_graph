import os
import yaml
from pathlib import Path

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "pdf-graph-analyzer"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "pdf-graph-analyzer"
CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "db_path": str(DEFAULT_DATA_DIR / "graph.db"),
    "archive_dir": str(DEFAULT_DATA_DIR / "archive"),
    "llm": {
        "provider": "mlx",
        "api_key": "",
        "base_url": "",
        "model_path": "/Users/vladimirkasterin/models/llm/gemma-3-text-12b-it-4bit",
        "max_tokens": 1000,
        "temp": 0.1
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}

class Config:
    def __init__(self):
        self.config_dir = DEFAULT_CONFIG_DIR
        self.data_dir = DEFAULT_DATA_DIR
        self.config_file = CONFIG_FILE
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = self._load_or_create_config()
        
        # Create archive dir if defined
        Path(self.data["archive_dir"]).mkdir(parents=True, exist_ok=True)

    def _load_or_create_config(self) -> dict:
        if not self.config_file.exists():
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
            return DEFAULT_CONFIG
        
        with open(self.config_file, "r", encoding="utf-8") as f:
            try:
                loaded = yaml.safe_load(f) or {}
                # Merge defaults for missing keys
                merged = {**DEFAULT_CONFIG}
                for k, v in loaded.items():
                    if isinstance(v, dict) and k in merged:
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
                return merged
            except Exception:
                return DEFAULT_CONFIG

    def save(self) -> None:
        """Saves current configuration to the config.yaml file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, default_flow_style=False, allow_unicode=True)

    @property
    def db_path(self) -> str:
        return self.data["db_path"]

    @property
    def archive_dir(self) -> str:
        return self.data["archive_dir"]

    @property
    def llm_provider(self) -> str:
        return self.data["llm"].get("provider", "mlx")

    @property
    def llm_api_key(self) -> str:
        return self.data["llm"].get("api_key", "")

    @property
    def llm_base_url(self) -> str:
        return self.data["llm"].get("base_url", "")

    @property
    def llm_model_path(self) -> str:
        return self.data["llm"].get("model_path", "")

    @property
    def llm_max_tokens(self) -> int:
        return self.data["llm"].get("max_tokens", 1000)

    @property
    def llm_temp(self) -> float:
        return self.data["llm"].get("temp", 0.1)

    @property
    def embedding_model_name(self) -> str:
        return self.data["embedding"]["model_name"]

    @property
    def chunk_size(self) -> int:
        return self.data["embedding"]["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self.data["embedding"]["chunk_overlap"]

config = Config()
