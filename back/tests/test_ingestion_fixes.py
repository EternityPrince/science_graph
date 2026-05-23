import os
from unittest.mock import patch
from src import console as con
from src.config import config
from src.services.normalization_pipeline import get_spacy_nlp

def test_console_show_time_prefix(capsys):
    # Enable show_time
    con.SHOW_TIME = True
    
    # Print some info
    con.info("Hello World")
    
    captured = capsys.readouterr()
    # Check that prefix matches the [HH:MM:SS (+X.XXs)] format
    assert "(+" in captured.out
    assert "s)" in captured.out
    assert "→" in captured.out
    assert "Hello World" in captured.out

    # Disable it again
    con.SHOW_TIME = False
    con.info("No time prefix")
    captured_no_time = capsys.readouterr()
    assert "[info]→[/]" in captured_no_time.out or "→" in captured_no_time.out
    assert "(+" not in captured_no_time.out

def test_config_token_limit_scaling():
    # If llm_max_tokens is 64000, extraction input limit must scale to 64000
    original_max_tokens = config.data["llm"].get("max_tokens", 1000)
    try:
        config.data["llm"]["max_tokens"] = 64000
        assert config.llm_extraction_input_limit == 64000
        assert config.llm_clustering_input_limit == 64000
        assert config.llm_synthesis_input_limit == 64000
    finally:
        config.data["llm"]["max_tokens"] = original_max_tokens

def test_spacy_only_attempts_once():
    # Set spacy model to a nonexistent name
    original_model_name = config.data.get("spacy", {}).get("model_name", "en_core_web_sm")
    import src.services.normalization_pipeline as np_mod
    orig_nlp = np_mod._nlp
    orig_attempted = np_mod._spacy_attempted
    try:
        config.data.setdefault("spacy", {})["model_name"] = "nonexistent_model_xyz"
        
        # Reset the spacy global variables
        np_mod._nlp = None
        np_mod._spacy_attempted = False
        
        # Mock spacy.load to throw OSError
        with patch("spacy.load", side_effect=OSError("not found")), \
             patch("spacy.cli.download") as mock_download:
            
            # Call get_spacy_nlp
            nlp1 = get_spacy_nlp()
            assert nlp1 is None
            assert np_mod._spacy_attempted is True
            
            # Call it a second time
            nlp2 = get_spacy_nlp()
            assert nlp2 is None
            
            # download should only have been attempted once
            mock_download.assert_called_once()
    finally:
        config.data.setdefault("spacy", {})["model_name"] = original_model_name
        np_mod._nlp = orig_nlp
        np_mod._spacy_attempted = orig_attempted


def test_spacy_download_sets_system_flags():
    original_model_name = config.data.get("spacy", {}).get("model_name", "en_core_web_sm")
    import src.services.normalization_pipeline as np_mod
    orig_nlp = np_mod._nlp
    orig_attempted = np_mod._spacy_attempted
    
    env_backup = {k: os.environ.get(k) for k in ["UV_SYSTEM_PYTHON", "PIP_BREAK_SYSTEM_PACKAGES", "VIRTUAL_ENV"]}
    
    try:
        config.data.setdefault("spacy", {})["model_name"] = "nonexistent_model_xyz"
        np_mod._nlp = None
        np_mod._spacy_attempted = False
        
        env_vars_during_download = {}
        
        def fake_download(*args, **kwargs):
            env_vars_during_download["UV_SYSTEM_PYTHON"] = os.environ.get("UV_SYSTEM_PYTHON")
            env_vars_during_download["PIP_BREAK_SYSTEM_PACKAGES"] = os.environ.get("PIP_BREAK_SYSTEM_PACKAGES")
        
        if "VIRTUAL_ENV" in os.environ:
            del os.environ["VIRTUAL_ENV"]
            
        with patch("sys.prefix", "mock_sys_prefix"), \
             patch("sys.base_prefix", "mock_sys_prefix"), \
             patch("spacy.load", side_effect=OSError("not found")), \
             patch("spacy.cli.download", side_effect=fake_download):
             
             get_spacy_nlp()
             
        assert env_vars_during_download.get("UV_SYSTEM_PYTHON") == "true"
        assert env_vars_during_download.get("PIP_BREAK_SYSTEM_PACKAGES") == "true"
        
        assert os.environ.get("UV_SYSTEM_PYTHON") == env_backup["UV_SYSTEM_PYTHON"]
        assert os.environ.get("PIP_BREAK_SYSTEM_PACKAGES") == env_backup["PIP_BREAK_SYSTEM_PACKAGES"]
        
    finally:
        config.data.setdefault("spacy", {})["model_name"] = original_model_name
        np_mod._nlp = orig_nlp
        np_mod._spacy_attempted = orig_attempted
        
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_spacy_download_does_not_set_system_flags_in_venv():
    original_model_name = config.data.get("spacy", {}).get("model_name", "en_core_web_sm")
    import src.services.normalization_pipeline as np_mod
    orig_nlp = np_mod._nlp
    orig_attempted = np_mod._spacy_attempted
    
    env_backup = {k: os.environ.get(k) for k in ["UV_SYSTEM_PYTHON", "PIP_BREAK_SYSTEM_PACKAGES", "VIRTUAL_ENV"]}
    
    try:
        config.data.setdefault("spacy", {})["model_name"] = "nonexistent_model_xyz"
        np_mod._nlp = None
        np_mod._spacy_attempted = False
        
        env_vars_during_download = {}
        
        def fake_download(*args, **kwargs):
            env_vars_during_download["UV_SYSTEM_PYTHON"] = os.environ.get("UV_SYSTEM_PYTHON")
            env_vars_during_download["PIP_BREAK_SYSTEM_PACKAGES"] = os.environ.get("PIP_BREAK_SYSTEM_PACKAGES")
        
        os.environ["VIRTUAL_ENV"] = "/mock/venv"
        
        with patch("sys.prefix", "mock_prefix"), \
             patch("sys.base_prefix", "mock_base_prefix"), \
             patch("spacy.load", side_effect=OSError("not found")), \
             patch("spacy.cli.download", side_effect=fake_download):
             
             get_spacy_nlp()
             
        assert env_vars_during_download.get("UV_SYSTEM_PYTHON") == env_backup["UV_SYSTEM_PYTHON"]
        assert env_vars_during_download.get("PIP_BREAK_SYSTEM_PACKAGES") == env_backup["PIP_BREAK_SYSTEM_PACKAGES"]
        
    finally:
        config.data.setdefault("spacy", {})["model_name"] = original_model_name
        np_mod._nlp = orig_nlp
        np_mod._spacy_attempted = orig_attempted
        
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

