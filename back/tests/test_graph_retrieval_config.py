import pytest
from src.config import Config

@pytest.fixture(autouse=True)
def mock_config(tmp_path, monkeypatch):
    """Isolate config paths to a temporary directory during tests to avoid modifying the user config."""
    import src.config
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_file = config_dir / "config.yaml"
    
    monkeypatch.setattr(src.config, "DEFAULT_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.config, "DEFAULT_DATA_DIR", data_dir)
    monkeypatch.setattr(src.config, "CONFIG_FILE", config_file)

@pytest.fixture(autouse=True)
def mock_config(tmp_path, monkeypatch):
    """Isolate config paths to a temporary directory during tests to avoid modifying the user config."""
    import src.config
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_file = config_dir / "config.yaml"
    
    monkeypatch.setattr(src.config, "DEFAULT_CONFIG_DIR", config_dir)
    monkeypatch.setattr(src.config, "DEFAULT_DATA_DIR", data_dir)
    monkeypatch.setattr(src.config, "CONFIG_FILE", config_file)

def test_config_old_keys_load():
    """Verify that an old config dictionary without new graph retrieval keys loads successfully and resolves defaults."""
    import copy
    cfg = Config()
    cfg.data = copy.deepcopy(cfg.data)
    # Remove new keys from self.data to simulate old config
    if "graph_retrieval" in cfg.data:
        del cfg.data["graph_retrieval"]
    if "rag_components" in cfg.data:
        cfg.data["rag_components"].pop("graph_concept_retrieval", None)
        cfg.data["rag_components"].pop("graph_bridge_retrieval", None)
        cfg.data["rag_components"].pop("graph_selected_sources_card", None)
        cfg.data["rag_components"].pop("graph_retrieval_trace", None)

    # Properties should resolve to default-off or default values
    assert cfg.graph_concept_retrieval_enabled is False
    assert cfg.graph_bridge_retrieval_enabled is False
    assert cfg.graph_selected_sources_card_enabled is False
    assert cfg.graph_retrieval_trace_enabled is False
    assert cfg.graph_retrieval_chunks_per_graph_paper == 1
    assert cfg.graph_retrieval_max_graph_candidate_papers == "auto"

def test_config_defaults_false():
    """Verify that all new graph retrieval flags default to False."""
    cfg = Config()
    # Ensure they match the default settings
    assert cfg.graph_concept_retrieval_enabled is False
    assert cfg.graph_bridge_retrieval_enabled is False
    assert cfg.graph_selected_sources_card_enabled is False
    assert cfg.graph_retrieval_trace_enabled is False

def test_config_retrieval_param_defaults():
    """Verify default parameters for graph retrieval."""
    cfg = Config()
    assert cfg.graph_retrieval_chunks_per_graph_paper == 1
    assert cfg.graph_retrieval_max_graph_candidate_papers == "auto"

def test_config_env_vars_override(monkeypatch):
    """Verify that RAG_* environment variables override configuration parameters."""
    cfg = Config()
    
    # Enable them via env vars
    monkeypatch.setenv("RAG_GRAPH_CONCEPT_RETRIEVAL", "true")
    monkeypatch.setenv("RAG_GRAPH_BRIDGE_RETRIEVAL", "1")
    monkeypatch.setenv("RAG_GRAPH_SELECTED_SOURCES_CARD", "yes")
    monkeypatch.setenv("RAG_GRAPH_RETRIEVAL_TRACE", "on")

    assert cfg.graph_concept_retrieval_enabled is True
    assert cfg.graph_bridge_retrieval_enabled is True
    assert cfg.graph_selected_sources_card_enabled is True
    assert cfg.graph_retrieval_trace_enabled is True

    # Disable them via env vars
    monkeypatch.setenv("RAG_GRAPH_CONCEPT_RETRIEVAL", "false")
    monkeypatch.setenv("RAG_GRAPH_BRIDGE_RETRIEVAL", "0")
    monkeypatch.setenv("RAG_GRAPH_SELECTED_SOURCES_CARD", "no")
    monkeypatch.setenv("RAG_GRAPH_RETRIEVAL_TRACE", "off")

    assert cfg.graph_concept_retrieval_enabled is False
    assert cfg.graph_bridge_retrieval_enabled is False
    assert cfg.graph_selected_sources_card_enabled is False
    assert cfg.graph_retrieval_trace_enabled is False
