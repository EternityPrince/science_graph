import os
from unittest.mock import MagicMock, patch
import pytest

import src.services.normalization_pipeline as norm_module
from src.services.normalization_pipeline import NormalizationPipeline, get_spacy_nlp
from src.llm_schemas import LLMExtractionResponse, LLMConcept, LLMDataset, LLMCitationIntent, LLMConceptRelation


@pytest.fixture(autouse=True)
def reset_spacy_state():
    """Reset spaCy global module state before and after each test."""
    orig_nlp = norm_module._nlp
    orig_attempted = norm_module._spacy_attempted
    norm_module._nlp = None
    norm_module._spacy_attempted = False
    from src.config import config
    spacy_cfg = config.data.setdefault("spacy", {})
    orig_model_name = spacy_cfg.get("model_name")
    spacy_cfg["model_name"] = "en_core_web_sm"
    yield
    norm_module._nlp = orig_nlp
    norm_module._spacy_attempted = orig_attempted
    if orig_model_name is not None:
        config.data["spacy"]["model_name"] = orig_model_name
    else:
        config.data["spacy"].pop("model_name", None)


@patch("spacy.load")
@patch("spacy.cli.download")
def test_get_spacy_nlp_venv_handling(mock_download, mock_load):
    """Test get_spacy_nlp properly handles virtual environment variables and fallback."""
    # Ensure spacy.load fails on first call (to trigger download) and succeeds on second call
    mock_nlp = MagicMock()
    mock_load.side_effect = [OSError("Model not found"), mock_nlp]
    
    # Simulate virtual environment (sys.prefix != sys.base_prefix)
    with patch("sys.prefix", "mock_prefix"):
        with patch("sys.base_prefix", "mock_base_prefix"):
            
            # Setup environment variable states
            os.environ["VIRTUAL_ENV"] = "old_venv"
            os.environ["UV_SYSTEM_PYTHON"] = "old_uv"
            os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "old_pip"
            
            try:
                nlp = get_spacy_nlp()
                
                assert nlp == mock_nlp
                mock_download.assert_called_once_with("en_core_web_sm")
                
                # Check that inside the download call (which is mock_download), the env would have been swapped.
                # Since we want to ensure environment variables were restored:
                assert os.environ.get("VIRTUAL_ENV") == "old_venv"
                assert os.environ.get("UV_SYSTEM_PYTHON") == "old_uv"
                assert os.environ.get("PIP_BREAK_SYSTEM_PACKAGES") == "old_pip"
            finally:
                os.environ.pop("VIRTUAL_ENV", None)
                os.environ.pop("UV_SYSTEM_PYTHON", None)
                os.environ.pop("PIP_BREAK_SYSTEM_PACKAGES", None)


@patch("spacy.load")
@patch("spacy.cli.download")
def test_get_spacy_nlp_system_python_handling(mock_download, mock_load):
    """Test get_spacy_nlp properly handles system python environment variables when not in venv."""
    mock_nlp = MagicMock()
    mock_load.side_effect = [OSError("Model not found"), mock_nlp]
    
    # Simulate system environment (sys.prefix == sys.base_prefix)
    with patch("sys.prefix", "same_prefix"):
        with patch("sys.base_prefix", "same_prefix"):
            
            # Setup environment variable states
            if "VIRTUAL_ENV" in os.environ:
                del os.environ["VIRTUAL_ENV"]
            
            try:
                # Capture how os.environ was modified during the download call
                modified_env = {}
                def capture_download(*args, **kwargs):
                    modified_env["VIRTUAL_ENV"] = os.environ.get("VIRTUAL_ENV")
                    modified_env["UV_SYSTEM_PYTHON"] = os.environ.get("UV_SYSTEM_PYTHON")
                    modified_env["PIP_BREAK_SYSTEM_PACKAGES"] = os.environ.get("PIP_BREAK_SYSTEM_PACKAGES")
                
                mock_download.side_effect = capture_download
                
                nlp = get_spacy_nlp()
                
                assert nlp == mock_nlp
                # Verify that environment variables were set for system pip install during download
                assert modified_env["VIRTUAL_ENV"] is None
                assert modified_env["UV_SYSTEM_PYTHON"] == "true"
                assert modified_env["PIP_BREAK_SYSTEM_PACKAGES"] == "true"
            finally:
                os.environ.pop("VIRTUAL_ENV", None)
                os.environ.pop("UV_SYSTEM_PYTHON", None)
                os.environ.pop("PIP_BREAK_SYSTEM_PACKAGES", None)


def test_title_case_edge_cases():
    """Test _title_case with hyphens, spacing, and special characters."""
    pipeline = NormalizationPipeline()
    
    # Standard title case
    assert pipeline._title_case("hello world") == "Hello World"
    # Consecutive spaces
    assert pipeline._title_case("  hello   world  ") == "Hello World"
    # Hyphenated words
    assert pipeline._title_case("self-attention") == "Self-Attention"
    # Consecutive hyphens
    assert pipeline._title_case("self--attention") == "Self--Attention"
    # Hyphen at start/end
    assert pipeline._title_case("-self-attention-") == "-Self-Attention-"
    # Empty string
    assert pipeline._title_case("") == ""


def test_normalize_description_edge_cases():
    """Test normalize_description stripping thinking/reasoning tags in various formats."""
    pipeline = NormalizationPipeline()
    
    # 1. Nesting and multiple think blocks
    desc1 = "<think>first</think> <thought>second</thought> Actual desc"
    assert pipeline.normalize_description(desc1) == "second Actual desc"
    
    # 2. Case insensitive thinking tags
    desc2 = "<THINK> reasoning here </THINK> The description."
    assert pipeline.normalize_description(desc2) == "The description."
    
    # 3. Unclosed think block
    desc3 = "<think>unclosed thought"
    assert pipeline.normalize_description(desc3) == "unclosed thought"
    
    # 4. Closing tag only (discard everything before it)
    desc4 = "Thinking process here </thought> My final text."
    assert pipeline.normalize_description(desc4) == "My final text."
    
    # 5. None or empty string input
    assert pipeline.normalize_description(None) == ""
    assert pipeline.normalize_description("  ") == ""


def test_normalize_extraction_response_all_fields():
    """Test normalize_extraction_response under edge case values (None, empty lists, duplicates)."""
    pipeline = NormalizationPipeline(aliases={"rl": "Reinforcement Learning", "gan": "Generative Adversarial Network"})
    
    resp = LLMExtractionResponse(
        authors=["alice smith", "Alice Smith", "  bob jones  "],
        concepts=[
            LLMConcept(name="rl", description="<think>t</think>Reinforcement learning desc"),
            LLMConcept(name="reinforcement learning", description="Duplicate concept but different desc"),
            LLMConcept(name="gan", description="Generative adversarial networks", aliases=["Generative Adversarial Network"])
        ],
        tags=["rl", "rl", "GAN", "NLP"],
        institutions=["stanford university", "Stanford University", "  mit  "],
        author_institutions=[
            {"author": "alice smith", "institution": "stanford university"},
            {"author": "  bob jones  ", "institution": "mit"},
            {"author": "", "institution": "missing author name"} # should be ignored
        ],
        sponsored_by=["nsf", "NSF", "  darpa  "],
        datasets=[
            LLMDataset(name="imagenet", relation="used"),
            LLMDataset(name="ImageNet", relation="used"),
            LLMDataset(name="  mnist  ", relation="introduced")
        ],
        code_repositories=[
            "https://github.com/a",
            "HTTPS://github.com/a",
            "  https://github.com/b  "
        ],
        journal_or_conference="  neurips  ",
        citation_intents=[
            LLMCitationIntent(target_title="Paper A", intent="methodology"),
            LLMCitationIntent(target_title="  ", intent="empty target title") # target_title becomes ""
        ],
        concept_relations=[
            LLMConceptRelation(source="rl", target="gan", relation_type="compares"),
            LLMConceptRelation(source="rl", target="", relation_type="missing target") # target becomes ""
        ]
    )
    
    norm = pipeline.normalize_extraction_response(resp)
    
    # Authors: normalized & deduplicated
    assert norm.authors == ["Alice Smith", "Bob Jones"]
    
    # Concepts: deduplicated by slug, description normalized, aliases normalized
    assert len(norm.concepts) == 2
    assert norm.concepts[0].name == "Reinforcement Learning"
    assert norm.concepts[0].description == "Reinforcement learning desc"
    assert norm.concepts[1].name == "Generative Adversarial Network"
    assert norm.concepts[1].aliases == ["Generative Adversarial Network"]
    
    # Tags: normalized & deduplicated
    assert norm.tags == ["Reinforcement Learning", "Generative Adversarial Network", "NLP"]
    
    # Institutions: title case & deduplicated
    assert norm.institutions == ["Stanford University", "Mit"]
    
    # Author Institutions: verified formatting & removal of incomplete items
    assert len(norm.author_institutions) == 2
    assert norm.author_institutions[0] == {"author": "Alice Smith", "institution": "Stanford University"}
    assert norm.author_institutions[1] == {"author": "Bob Jones", "institution": "Mit"}
    
    # Sponsored by
    assert norm.sponsored_by == ["Nsf", "Darpa"]
    
    # Datasets
    assert len(norm.datasets) == 2
    assert norm.datasets[0].name == "Imagenet"
    assert norm.datasets[1].name == "Mnist"
    
    # Code Repositories
    assert norm.code_repositories == ["https://github.com/a", "https://github.com/b"]
    
    # Journal or Conference
    assert norm.journal_or_conference == "neurips"
    
    # Citation Intents
    assert len(norm.citation_intents) == 2
    assert norm.citation_intents[0].target_title == "Paper A"
    assert norm.citation_intents[1].target_title == ""
    
    # Concept Relations
    assert len(norm.concept_relations) == 1
    assert norm.concept_relations[0].source == "Reinforcement Learning"
    assert norm.concept_relations[0].target == "Generative Adversarial Network"


def test_scientific_concept_normalization_and_name_validation():
    """Test new rules for concept lemmatization and human name validation."""
    pipeline = NormalizationPipeline()
    
    # 1. Concept lemmatization rules
    # Modifiers should not be lemmatized to verbs/nouns (supervised, distributed, post, action)
    # Gerunds/nouns ending in -ing should be preserved (learning, training, sampling, tuning)
    assert pipeline.normalize_concept_name("supervised learning") == "Supervised Learning"
    assert pipeline.normalize_concept_name("distributed learning") == "Distributed Learning"
    assert pipeline.normalize_concept_name("action sampling") == "Action Sampling"
    assert pipeline.normalize_concept_name("hyperparameter tuning") == "Hyperparameter Tuning"
    assert pipeline.normalize_concept_name("post-training") == "Post-Training"
    
    # Plural -ings should become singular -ing
    assert pipeline.normalize_concept_name("embeddings") == "Embedding"
    assert pipeline.normalize_concept_name("image embeddings") == "Image Embedding"
    
    # Normal plural nouns should still be lemmatized to singular
    assert pipeline.normalize_concept_name("neural networks") == "Neural Network"
    assert pipeline.normalize_concept_name("decision trees") == "Decision Tree"
    assert pipeline.normalize_concept_name("large language models") == "Large Language Model"
    
    # 2. Surnames ending in 's' should be preserved if capitalized in original text
    assert pipeline.normalize_concept_name("Cathy Williams") == "Cathy Williams"
    assert pipeline.normalize_concept_name("Alice Johnson") == "Alice Johnson"
    assert pipeline.normalize_concept_name("Bob Lee") == "Bob Lee"
    
    # 3. Test is_likely_name from ner_engine
    from src.ner_engine import is_likely_name
    
    # Plausible names
    assert is_likely_name("Ashish Vaswani") is True
    assert is_likely_name("Aidan N. Gomez") is True
    assert is_likely_name("Sam Stephens") is True
    assert is_likely_name("Cathy Williams") is True
    
    # Imposter names (institutions, concepts, locations) should be rejected
    assert is_likely_name("Scientific Inc") is False
    assert is_likely_name("San Francisco") is False
    assert is_likely_name("Stanford University") is False
    assert is_likely_name("Dementia Research Institute") is False
    assert is_likely_name("Mass Spectrometry") is False
    assert is_likely_name("Universal Verifier") is False
    assert is_likely_name("Process Reward Model") is False
    assert is_likely_name("Neural Information Processing Systems") is False

