import pytest
import sys
from unittest.mock import MagicMock, patch
from core.metrics import (
    normalize_id,
    calculate_retrieval_recall,
    calculate_context_precision,
    compute_cosine_similarity,
    calculate_semantic_accuracy,
    count_text_tokens,
    estimate_prompt_tokens,
    get_embedding_engine
)
from core.retrieval import normalize_component_scores


def test_normalize_id():
    assert normalize_id(None) == ""
    assert normalize_id("") == ""
    assert normalize_id("  ") == ""
    assert normalize_id("DOC-42") == "doc_42"
    assert normalize_id("docs/doc_42") == "doc_42"
    assert normalize_id("data/sub/DOC-42.pdf") == "doc_42"
    assert normalize_id("doc_42_chunk_3") == "doc_42"
    assert normalize_id("DATA/DOC-42#chunk-5") == "doc_42"
    assert normalize_id("10.21869/2223-1560-2025-29-2-130-145") == "10.21869_2223_1560_2025_29_2_130_145"
    assert normalize_id("docs/10.21869/2223-1560-2025-29-2-130-145_chunk_1") == "10.21869_2223_1560_2025_29_2_130_145"


def test_normalize_component_scores():
    assert normalize_component_scores([]) == []
    assert normalize_component_scores([5.0]) == [1.0]
    assert normalize_component_scores([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]
    assert normalize_component_scores([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]
    assert normalize_component_scores([-1.0, 0.0, 1.0]) == [0.0, 0.5, 1.0]


def test_calculate_retrieval_recall_chunk_normalization():
    assert calculate_retrieval_recall(["doc_42"], ["docs/doc_42_chunk_3"]) == 1.0
    assert calculate_retrieval_recall(["DOC-42"], ["data/doc-42#chunk-1"]) == 1.0


def test_calculate_context_precision_chunk_normalization():
    chunks = [{"paper_id": "docs/DOC-42#chunk_1"}]
    assert calculate_context_precision(["doc_42"], chunks) == 1.0


def test_calculate_retrieval_recall_edge_cases():
    assert calculate_retrieval_recall([], []) == 1.0
    assert calculate_retrieval_recall(["p1"], []) == 0.0
    assert calculate_retrieval_recall([], ["p1"]) == 1.0
    assert calculate_retrieval_recall(["p1"], ["p1"]) == 1.0
    assert calculate_retrieval_recall(["p1", "p2"], ["p1"]) == 0.5
    assert calculate_retrieval_recall(["p1", " "], ["p1"]) == 1.0
    # Expected set is empty after stripping:
    assert calculate_retrieval_recall([" ", "  "], ["p1"]) == 1.0

def test_calculate_context_precision_edge_cases():
    assert calculate_context_precision([], []) == 1.0
    assert calculate_context_precision(["p1"], []) == 0.0
    assert calculate_context_precision([" ", "  "], []) == 1.0
    
    chunks = [
        {"paper_id": "p1"},
        {"paper_id": "p2"},
        {"paper_id": "p1"}
    ]
    # Hits at 1 and 3
    # Precision at 1: 1/1 = 1.0
    # Precision at 3: 2/3 = 0.6667
    # MAP = (1.0 + 0.6667) / 2 = 0.8333
    assert calculate_context_precision(["p1"], chunks) == 0.8333
    assert calculate_context_precision(["p3"], chunks) == 0.0

def test_compute_cosine_similarity():
    assert compute_cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert compute_cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert compute_cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

@patch("core.metrics.get_embedding_engine")
def test_calculate_semantic_accuracy(mock_get_engine):
    mock_engine = MagicMock()
    mock_engine.get_embeddings.side_effect = [
        [[1.0, 0.0]], # golden
        [[1.0, 0.0]]  # generated
    ]
    mock_get_engine.return_value = mock_engine
    
    res = calculate_semantic_accuracy(["gold"], ["gen"])
    assert len(res) == 1
    assert res[0] == 1.0
    
    assert calculate_semantic_accuracy([], []) == []
    assert calculate_semantic_accuracy(["gold"], []) == []
    assert calculate_semantic_accuracy([], ["gen"]) == []

def test_count_text_tokens_tiktoken():
    # Test tiktoken normal execution
    assert count_text_tokens("abc") > 0
    
    mock_tt = MagicMock()
    mock_tt.get_encoding.side_effect = Exception("mock get encoding error")
    with patch("core.metrics.tiktoken", mock_tt):
        assert count_text_tokens("abc") == 1
        assert count_text_tokens("hello world") == 2 # 11 // 4 is 2
        
    # Test when tiktoken is None
    with patch("core.metrics.tiktoken", None):
        assert count_text_tokens("abc") == 1
        assert count_text_tokens("hello world") == 2

def test_estimate_prompt_tokens():
    # Test baseline B0
    b0_tokens = estimate_prompt_tokens("test query", [], "B0")
    assert b0_tokens > 0
    
    # Test other baseline
    chunks = [
        {"text_content": "some text chunk content", "paper_id": "p1", "page_number": 5}
    ]
    other_tokens = estimate_prompt_tokens("test query", chunks, "B1")
    assert other_tokens > b0_tokens

def test_get_embedding_engine_error():
    import core.metrics
    old_engine = core.metrics._embedding_engine
    core.metrics._embedding_engine = None
    try:
        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name.startswith("src"):
                raise ImportError("mock import error")
            return orig_import(name, *args, **kwargs)
            
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(Exception):
                get_embedding_engine()
    finally:
        core.metrics._embedding_engine = old_engine

def test_get_embedding_engine_success():
    import core.metrics
    old_engine = core.metrics._embedding_engine
    core.metrics._embedding_engine = None
    try:
        with patch("src.vector_search.EmbeddingEngine") as mock_engine_cls:
            engine = get_embedding_engine()
            assert engine == mock_engine_cls.return_value
            mock_engine_cls.assert_called_once()
            
            # Cached return
            engine2 = get_embedding_engine()
            assert engine2 == engine
            mock_engine_cls.assert_called_once()
    finally:
        core.metrics._embedding_engine = old_engine

def test_tiktoken_import_error():
    import importlib
    import core.metrics
    # Hide tiktoken to simulate ImportError on load
    with patch.dict(sys.modules, {"tiktoken": None}):
        importlib.reload(core.metrics)
        assert core.metrics.tiktoken is None
        
    # Reload again to restore the environment
    importlib.reload(core.metrics)

