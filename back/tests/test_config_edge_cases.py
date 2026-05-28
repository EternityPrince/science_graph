import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.config import Config, DEFAULT_CONFIG


def test_llm_model_max_context_provider_heuristics():
    """Test llm_model_max_context determines context size based on cloud model name when not configured."""
    cfg = Config()
    
    # 1. When model_max_context is explicitly configured in config
    cfg.data["llm"]["model_max_context"] = 12345
    assert cfg.llm_model_max_context == 12345
    
    # 2. When model_max_context is None (or deleted)
    cfg.data["llm"]["model_max_context"] = None
    
    # Provider: mlx
    cfg.data["llm"]["provider"] = "mlx"
    assert cfg.llm_model_max_context == 4096
    
    # Provider: openai
    cfg.data["llm"]["provider"] = "openai"
    
    # Check various model name patterns (case-insensitive)
    models_to_test = {
        "gpt-4o-mini": 128000,
        "GPT-4o": 128000,
        "gpt-4-turbo": 8192,
        "GPT-4": 8192,
        "gpt-3.5-turbo": 16385,
        "google/gemini-2.5-flash": 32768,
        "GEMINI-1.5": 32768,
        "anthropic/claude-3-opus": 200000,
        "CLAUDE-3.5-sonnet": 200000,
        "some-unknown-model": 4096,
    }
    
    for model_name, expected_context in models_to_test.items():
        cfg.data["llm"]["cloud"] = {"model_name": model_name}
        assert cfg.llm_model_max_context == expected_context


def test_get_storage_stats_edge_cases():
    """Test get_storage_stats under various edge conditions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "nonexistent.db"
        archive_dir = tmp_path / "archive"
        
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.data["db_path"] = str(db_path)
        cfg.data["archive_dir"] = str(archive_dir)
        
        # 1. DB does not exist, storage directory is empty
        stats = cfg.get_storage_stats()
        assert stats["total_size"] == 0
        assert stats["extensions"] == []
        assert stats["sources"] == []
        
        # 2. Create some files, including symlinks, directories, files with no extensions
        archive_dir.mkdir()
        
        # Regular file with extension
        f1 = archive_dir / "paper1.pdf"
        f1.write_text("dummy pdf contents") # 18 bytes
        
        # File with no extension
        f2 = archive_dir / "note1"
        f2.write_text("note") # 4 bytes
        
        # Directory (should be ignored by stats calculator)
        subdir = archive_dir / "subfolder"
        subdir.mkdir()
        
        # File in subdirectory
        f3 = subdir / "book1.epub"
        f3.write_text("epub") # 4 bytes
        
        # Symlink to another file (should be ignored to avoid double-counting or loop)
        sym = archive_dir / "link.pdf"
        sym.symlink_to(f1)
        
        stats = cfg.get_storage_stats()
        assert stats["total_size"] == 26 # 18 + 4 + 4
        
        ext_sizes = {item["extension"]: item["size"] for item in stats["extensions"]}
        assert ext_sizes[".pdf"] == 18
        assert ext_sizes["(no extension)"] == 4
        assert ext_sizes[".epub"] == 4
        
        # Source checks
        src_sizes = {item["source"]: item["size"] for item in stats["sources"]}
        # Default fallbacks based on extension:
        # .pdf -> "paper"
        # .epub -> "book"
        # (no extension) -> "other"
        assert src_sizes["paper"] == 18
        assert src_sizes["book"] == 4
        assert src_sizes["other"] == 4


def test_get_storage_stats_with_db_mapping():
    """Test get_storage_stats resolves source types using SQLiteGraphRepository mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.data["archive_dir"] = str(archive_dir)
        
        # Write files named after paper_id
        f1 = archive_dir / "id_paper_custom.pdf"
        f1.write_text("content") # 7 bytes
        
        f2 = archive_dir / "id_note_custom.md"
        f2.write_text("content") # 7 bytes
        
        # Mock SQLiteGraphRepository to return a specific mapping
        mock_repo = MagicMock()
        mock_repo.get_paper_source_types.return_value = {
            "id_paper_custom": "custom_paper_type",
            "id_note_custom": "custom_note_type"
        }
        
        with patch("src.config.os.path.exists", return_value=True):
            with patch("src.repository.sqlite_impl.SQLiteGraphRepository", return_value=mock_repo):
                stats = cfg.get_storage_stats()
                
                src_sizes = {item["source"]: item["size"] for item in stats["sources"]}
                assert src_sizes["custom_paper_type"] == 7
                assert src_sizes["custom_note_type"] == 7


def test_config_load_yaml_failures_and_sync():
    """Test config load/sync behavior with malformed YAML or missing keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_file = tmp_path / "config.yaml"
        
        # Write invalid YAML
        config_file.write_text("invalid: yaml: [", encoding="utf-8")
        
        with patch("src.config.DEFAULT_CONFIG_DIR", tmp_path):
            with patch("src.config.CONFIG_FILE", config_file):
                cfg = Config()
                # Should fall back to DEFAULT_CONFIG on safe_load exception
                assert cfg.data["embedding"]["model_name"] == DEFAULT_CONFIG["embedding"]["model_name"]
                
                # Write partial config
                partial = {
                    "db_path": "/custom/db.db",
                    "llm": {
                        "provider": "openai",
                        "cloud": {
                            "model_name": "custom-gpt"
                        }
                    }
                }
                config_file.write_text(yaml.dump(partial), encoding="utf-8")
                
                cfg2 = Config()
                # db_path should be overwritten by partial config
                assert cfg2.data["db_path"] == "/custom/db.db"
                # llm provider and cloud model name should be overwritten
                assert cfg2.data["llm"]["provider"] == "openai"
                assert cfg2.data["llm"]["cloud"]["model_name"] == "custom-gpt"
                # Other fields should stay default
                assert cfg2.data["llm"]["max_tokens"] == DEFAULT_CONFIG["llm"]["max_tokens"]
                # Due to shallow merge, "base_url" won't exist in cloud dict
                assert "base_url" not in cfg2.data["llm"]["cloud"]
                
                # Test sync_dict preserves user configurations but syncs defaults
                cfg2.init_config()
                with open(config_file, "r", encoding="utf-8") as f:
                    synced_data = yaml.safe_load(f)
                
                assert synced_data["db_path"] == "/custom/db.db"
                assert synced_data["llm"]["provider"] == "openai"
                assert synced_data["llm"]["cloud"]["model_name"] == "custom-gpt"
                # Synced file must contain all DEFAULT_CONFIG keys now
                assert "pdf_compression" in synced_data


def test_hf_token_environment_propagation():
    """Test that setting hf_token in config propagates it to environment variables."""
    # Temporarily remove existing env keys if present
    env_keys = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    orig_env = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys:
        os.environ.pop(k, None)
        
    try:
        with patch.object(Config, "_load_or_create_config", return_value={"hf_token": "my_secret_token", "archive_dir": "/tmp"}):
            with patch("src.config.Path.mkdir"):
                cfg = Config()
                assert os.environ.get("HF_TOKEN") == "my_secret_token"
                assert os.environ.get("HUGGINGFACE_HUB_TOKEN") == "my_secret_token"
    finally:
        # Restore environment variables
        for k, v in orig_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
