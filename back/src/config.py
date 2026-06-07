import os
import yaml
from pathlib import Path

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "pdf-graph-analyzer"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "pdf-graph-analyzer"
CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG = {
    "db_path": str(DEFAULT_DATA_DIR / "graph.db"),
    "archive_dir": str(DEFAULT_DATA_DIR / "archive"),
    "pdf_parser": "marker",
    "hf_token": "",
    "llm": {
        "provider": "mlx",
        "max_tokens": 1000,
        "model_max_context": 4096,
        "temp": 0.1,
        "request_delay": 1.0,
        "retry_backoff": 2.0,
        "chunk_pool_size": 4,
        "max_expanded_queries": 3,
        "hyde_enabled": False,
        "hyde_max_tokens": 300,
        "hyde_count": 1,
        # Task-specific input token limits
        "extraction_input_limit": 5000,
        "clustering_input_limit": 6000,
        "synthesis_input_limit": 5000,
        # Task-specific output token limits
        "extraction_output_limit": 2048,
        "clustering_output_limit": 1500,
        "synthesis_output_limit": 1500,
        # Nested split configurations
        "local": {
            "model_path": str(Path.home() / "models" / "llm" / "gemma-3-text-12b-it-4bit"),
        },
        "cloud": {
            "provider": "openai",
            "model_name": "google/gemini-2.5-flash",
            "cheap_model_name": "google/gemini-2.5-flash",
            "api_key": "",
            "base_url": "https://openrouter.ai/api/v1",
        }
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "spacy": {
        "model_name": "en_core_web_sm"
    },
    "ner": {
        "model_name": "dslim/bert-base-NER"
    },
    "pdf_compression": {
        "enabled": True,
        "dpi_threshold": 151,
        "dpi_target": 150,
        "quality": 75
    },
    "rag_components": {
        "intent_classifier": True,
        "graph_ontology_lookup": True,
        "llm_query_expansion": True,
        "hyde": True,
        "lexical_search": True,
        "dense_search": True,
        "dynamic_alpha_blending": True,
        "rrf": True,
        "graph_expansion": True,
        "reranker": True,
        "score_blending": True,
        "context_trimming": True,
        "citation_repair": True,
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
        
        # Apply HF token to environment if set
        hf_token = self.data.get("hf_token", "")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
        # Create archive dir if defined
        Path(self.data["archive_dir"]).mkdir(parents=True, exist_ok=True)

    def _load_or_create_config(self) -> dict:
        if not self.config_file.exists():
            config_template = f"""# Configuration for PDF Graph Analyzer

# Path to the SQLite database file
db_path: "{DEFAULT_CONFIG['db_path']}"

# Directory where local archives of websites/PDFs are stored
archive_dir: "{DEFAULT_CONFIG['archive_dir']}"

# PDF parser to use: 'marker' (modern OCR/Markdown converter) or 'fitz' (legacy PyMuPDF)
pdf_parser: "{DEFAULT_CONFIG['pdf_parser']}"

# HuggingFace token for downloading gated models/embeddings (optional)
hf_token: ""

# Large Language Model (LLM) configuration
llm:
  # Provider: 'mlx' (for local Apple Silicon) or 'openai' (for OpenAI / OpenRouter / compatible APIs)
  provider: "mlx"

  # Global default maximum output tokens for LLM response
  max_tokens: 1000

  # Maximum context window size for the LLM
  model_max_context: 4096

  # Default temperature (0.0 = deterministic, 1.0 = creative)
  temp: 0.1

  # Delay (in seconds) between requests to LLM / API providers
  request_delay: 1.0

  # Wait timeout (in seconds) before retrying a failed provider request
  retry_backoff: 2.0

  # Number of concurrent chunks to process in parallel via LLM
  chunk_pool_size: 4

  # Maximum number of expanded queries for search (including original, 1 = disabled)
  max_expanded_queries: 3

  # Enable Hypothetical Document Embeddings (HyDE) for retrieval improvement
  hyde_enabled: false

  # Maximum tokens generated for the hypothetical answer
  hyde_max_tokens: 300

  # Number of hypothetical answers to generate
  hyde_count: 1

  # Local model settings (used if provider is 'mlx')
  local:
    model_path: "{DEFAULT_CONFIG['llm']['local']['model_path']}"

  # Cloud model settings (used if provider is 'openai')
  cloud:
    provider: "openai"
    model_name: "google/gemini-2.5-flash"
    api_key: ""
    base_url: "https://openrouter.ai/api/v1"

  # Task-specific input token limits (used to dynamically truncate inputs to fit context)
  extraction_input_limit: 5000
  clustering_input_limit: 6000
  synthesis_input_limit: 5000

  # Task-specific output token limits (passed to model during generation)
  extraction_output_limit: 2048
  clustering_output_limit: 1500
  synthesis_output_limit: 1500

# Embedding model configuration (used for vector search and indexing)
embedding:
  # HuggingFace model name for sentence embeddings
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

  # Number of characters per text chunk
  chunk_size: 1000

  # Number of characters overlap between consecutive chunks
  chunk_overlap: 200

# spaCy model configuration (used for lemmatization)
spacy:
  # spaCy model name (e.g. "en_core_web_sm") or path
  model_name: "en_core_web_sm"

# NER model configuration (used for name extraction)
ner:
  # NER model name or HuggingFace repo ID or local path
  model_name: "dslim/bert-base-NER"

# PDF compression settings (used to downsample high-DPI scanned PDFs)
pdf_compression:
  enabled: true
  dpi_threshold: 151
  dpi_target: 150
  quality: 75

# RAG components configuration for benchmarking (Scenario 1, 2, 3)
rag_components:
  intent_classifier: true
  graph_ontology_lookup: true
  llm_query_expansion: true
  hyde: true
  lexical_search: true
  dense_search: true
  dynamic_alpha_blending: true
  rrf: true
  graph_expansion: true
  reranker: true
  score_blending: true
  context_trimming: true
  citation_repair: true
"""
            with open(self.config_file, "w", encoding="utf-8") as f:
                f.write(config_template)
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

    def init_config(self) -> None:
        """Brings the config.yaml file up-to-date with DEFAULT_CONFIG, preserving user values."""
        loaded = {}
        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                try:
                    loaded = yaml.safe_load(f) or {}
                except Exception:
                    pass

        def sync_dict(template: dict, current: dict) -> dict:
            result = {}
            for k, v in template.items():
                if k in current:
                    if isinstance(v, dict) and isinstance(current[k], dict):
                        result[k] = sync_dict(v, current[k])
                    else:
                        result[k] = current[k]
                else:
                    result[k] = v
            return result

        self.data = sync_dict(DEFAULT_CONFIG, loaded)
        self.save()

    @property
    def hf_token(self) -> str:
        return self.data.get("hf_token", "")

    @property
    def db_path(self) -> str:
        return self.data["db_path"]

    @property
    def archive_dir(self) -> str:
        return self.data["archive_dir"]

    @property
    def pdf_parser(self) -> str:
        return self.data.get("pdf_parser", "marker")

    @property
    def llm_provider(self) -> str:
        return self.data["llm"].get("provider", "mlx")

    @property
    def llm_api_key(self) -> str:
        return self.llm_cloud_api_key

    @property
    def llm_base_url(self) -> str:
        return self.llm_cloud_base_url

    @property
    def llm_model_path(self) -> str:
        return self.llm_local_model_path

    @property
    def llm_local_model_path(self) -> str:
        local_cfg = self.data["llm"].get("local", {})
        if isinstance(local_cfg, dict):
            return local_cfg.get("model_path", self.data["llm"].get("model_path", ""))
        return self.data["llm"].get("model_path", "")

    @property
    def llm_cloud_model_name(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            return cloud_cfg.get("model_name", self.data["llm"].get("model_path", ""))
        return self.data["llm"].get("model_path", "")

    @property
    def llm_cloud_api_key(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            return cloud_cfg.get("api_key", self.data["llm"].get("api_key", ""))
        return self.data["llm"].get("api_key", "")

    @property
    def llm_cloud_base_url(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            return cloud_cfg.get("base_url", self.data["llm"].get("base_url", ""))
        return self.data["llm"].get("base_url", "")

    @property
    def llm_cheap_model_name(self) -> str:
        cloud_cfg = self.data["llm"].get("cloud", {})
        if isinstance(cloud_cfg, dict):
            return cloud_cfg.get("cheap_model_name", "google/gemini-2.5-flash")
        return "google/gemini-2.5-flash"

    @property
    def llm_max_tokens(self) -> int:
        return self.data["llm"].get("max_tokens", 1000)

    @property
    def llm_model_max_context(self) -> int:
        val = self.data["llm"].get("model_max_context")
        if val is not None:
            return int(val)
        
        provider = self.llm_provider
        if provider == "openai":
            model_name = self.llm_cloud_model_name.lower()
            if "gpt-4o" in model_name:
                return 128000
            elif "gpt-4" in model_name:
                return 8192
            elif "gpt-3.5" in model_name:
                return 16385
            elif "gemini" in model_name:
                return 32768
            elif "claude" in model_name:
                return 200000
        
        return 4096

    @property
    def llm_temp(self) -> float:
        return self.data["llm"].get("temp", 0.1)

    @property
    def llm_request_delay(self) -> float:
        return float(self.data["llm"].get("request_delay", 1.0))

    @property
    def llm_retry_backoff(self) -> float:
        return float(self.data["llm"].get("retry_backoff", 2.0))

    @property
    def llm_chunk_pool_size(self) -> int:
        return int(self.data["llm"].get("chunk_pool_size", 4))

    @property
    def max_expanded_queries(self) -> int:
        return int(self.data["llm"].get("max_expanded_queries", 3))

    @property
    def hyde_enabled(self) -> bool:
        return bool(self.data["llm"].get("hyde_enabled", False))

    @property
    def hyde_max_tokens(self) -> int:
        return int(self.data["llm"].get("hyde_max_tokens", 300))

    @property
    def hyde_count(self) -> int:
        return int(self.data["llm"].get("hyde_count", 1))

    @property
    def llm_extraction_input_limit(self) -> int:
        val = self.data["llm"].get("extraction_input_limit", 5000)
        return max(val, self.llm_max_tokens)

    @property
    def llm_extraction_output_limit(self) -> int:
        return self.data["llm"].get("extraction_output_limit", 2048)

    @property
    def llm_clustering_input_limit(self) -> int:
        val = self.data["llm"].get("clustering_input_limit", 6000)
        return max(val, self.llm_max_tokens)

    @property
    def llm_clustering_output_limit(self) -> int:
        return self.data["llm"].get("clustering_output_limit", 1500)

    @property
    def llm_synthesis_input_limit(self) -> int:
        val = self.data["llm"].get("synthesis_input_limit", 5000)
        return max(val, self.llm_max_tokens)

    @property
    def llm_synthesis_output_limit(self) -> int:
        return self.data["llm"].get("synthesis_output_limit", 1500)

    @property
    def embedding_model_name(self) -> str:
        return self.data["embedding"]["model_name"]

    @property
    def chunk_size(self) -> int:
        return self.data["embedding"]["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self.data["embedding"]["chunk_overlap"]

    @property
    def spacy_model_name(self) -> str:
        return self.data.get("spacy", {}).get("model_name", "en_core_web_sm")

    @property
    def ner_model_name(self) -> str:
        return self.data.get("ner", {}).get("model_name", "dslim/bert-base-NER")

    @property
    def pdf_compression_enabled(self) -> bool:
        return self.data.get("pdf_compression", {}).get("enabled", True)

    @property
    def pdf_compression_dpi_threshold(self) -> int:
        return self.data.get("pdf_compression", {}).get("dpi_threshold", 151)

    @property
    def pdf_compression_dpi_target(self) -> int:
        return self.data.get("pdf_compression", {}).get("dpi_target", 150)

    @property
    def pdf_compression_quality(self) -> int:
        return self.data.get("pdf_compression", {}).get("quality", 75)

    def is_component_enabled(self, name: str) -> bool:
        # Check environment variable first (e.g. RAG_HYDE=false)
        env_val = os.environ.get(f"RAG_{name.upper()}")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        # Fallback to config file
        components = self.data.get("rag_components", {})
        return bool(components.get(name, DEFAULT_CONFIG["rag_components"].get(name, True)))

    @property
    def rag_components(self) -> dict:
        defaults = DEFAULT_CONFIG.get("rag_components", {})
        return {name: self.is_component_enabled(name) for name in defaults}

    @property
    def taxonomy(self) -> dict:
        """
        Lazy-loads and caches config/taxonomy.yaml from the project root.
        Returns a dict with keys: concepts, topics, descriptions.
        Falls back to empty dicts if the file is missing or malformed.
        """
        if hasattr(self, "_taxonomy"):
            return self._taxonomy  # type: ignore[attr-defined]

        # Project root is two levels above this file (src/config.py → src → project root)
        project_root = Path(__file__).parent.parent
        taxonomy_path = project_root / "config" / "taxonomy.yaml"

        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            self._taxonomy = {  # type: ignore[attr-defined]
                "concepts": raw.get("concepts", {}),
                "topics": raw.get("topics", {}),
                "descriptions": raw.get("descriptions", {}),
            }
        except Exception:
            self._taxonomy = {"concepts": {}, "topics": {}, "descriptions": {}}  # type: ignore[attr-defined]

        return self._taxonomy

    def get_storage_stats(self) -> dict:
        import os
        from src.repository.sqlite_impl import SQLiteGraphRepository
        
        storage_dir = Path(self.data_dir)
        archive_dir = Path(self.archive_dir)
        
        total_size = 0
        extension_sizes = {}
        extension_counts = {}
        
        archive_sizes_by_source = {}
        archive_counts_by_source = {}
        
        # Get mapping of paper ID to source_type from SQLiteGraphRepository
        paper_source_types = {}
        db_path = self.db_path
        if os.path.exists(db_path):
            try:
                repo = SQLiteGraphRepository(db_path)
                paper_source_types = repo.get_paper_source_types()
            except Exception:
                pass
                
        if storage_dir.exists():
            for root, _, files in os.walk(storage_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_symlink() or not file_path.is_file():
                        continue
                    sz = file_path.stat().st_size
                    total_size += sz
                    
                    ext = file_path.suffix.lower()
                    if not ext:
                        ext = "(no extension)"
                    
                    extension_sizes[ext] = extension_sizes.get(ext, 0) + sz
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                    
                    # Check if inside archive
                    try:
                        is_in_archive = archive_dir.resolve() in file_path.resolve().parents
                    except Exception:
                        is_in_archive = str(file_path.resolve()).startswith(str(archive_dir.resolve()))
                        
                    if is_in_archive:
                        paper_id = file_path.stem
                        stype = paper_source_types.get(paper_id)
                        if not stype:
                            if ext == ".pdf":
                                stype = "paper"
                            elif ext == ".md":
                                stype = "note"
                            elif ext == ".epub":
                                stype = "book"
                            else:
                                stype = "other"
                        archive_sizes_by_source[stype] = archive_sizes_by_source.get(stype, 0) + sz
                        archive_counts_by_source[stype] = archive_counts_by_source.get(stype, 0) + 1
                        
        # Format results
        ext_list = [
            {"extension": k, "size": v, "count": extension_counts[k]}
            for k, v in sorted(extension_sizes.items(), key=lambda x: x[1], reverse=True)
        ]
        
        src_list = [
            {"source": k, "size": v, "count": archive_counts_by_source[k]}
            for k, v in sorted(archive_sizes_by_source.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "storage_dir": str(storage_dir),
            "total_size": total_size,
            "extensions": ext_list,
            "sources": src_list
        }

config = Config()
