"""E2E Test for MLX Logits Streaming, Token Entropy, and Citation Alignment.
"""

from typing import List, Dict, Any, Tuple
from unittest.mock import MagicMock, patch

import pytest

from src.llm_engine.mlx_impl import MlxLLMEngine
from core.shannon_estimator import (
    compute_generation_entropy,
    compute_citation_entropy,
    find_citation_spans,
)
from core.generation import run_query_on_baseline


class DummyResponse:
    def __init__(self, token_id: int, text: str, logprobs: Any = None):
        self.token = token_id
        self.text = text
        self.logprobs = logprobs


def mock_stream_generate(model, tokenizer, prompt, max_tokens=100, sampler=None):
    """Yields dummy stream responses simulating logits & tokens."""
    sample_data = [
        (101, "According ", None),
        (102, "to ", None),
        (103, "[sciq_paper_1]", None),
        (104, ", ", None),
        (105, "the ", None),
        (106, "energy ", None),
        (107, "is ", None),
        (108, "conserved", None),
        (109, ".", None),
    ]
    try:
        import mlx.core as mx
        fake_logprobs = mx.array([-1.0, -2.0, -3.0, -4.0])
    except ImportError:
        fake_logprobs = None

    for tid, txt, _ in sample_data:
        yield DummyResponse(tid, txt, fake_logprobs)


def test_mlx_engine_generate_response_with_logits_mocked():
    """Verify generate_response_with_logits yields non-negative entropy tokens and structured tokens_info."""
    with patch("os.path.isdir", return_value=True), \
         patch("src.llm_engine.mlx_impl.MlxLLMEngine._ensure_model_loaded") as mock_ensure, \
         patch("mlx_lm.stream_generate", side_effect=mock_stream_generate):
        
        mock_ensure.return_value = None
        engine = MlxLLMEngine(model_path="/dummy/path")
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()

        text, tokens_info = engine.generate_response_with_logits("Explain conservation of energy.")

        assert text == "According to [sciq_paper_1], the energy is conserved."
        assert len(tokens_info) == 9

        for tok in tokens_info:
            assert "token_id" in tok
            assert "token_text" in tok
            assert "char_start" in tok
            assert "char_end" in tok
            assert "entropy" in tok
            assert isinstance(tok["entropy"], float)
            assert tok["entropy"] >= 0.0

        # Check character span accuracy
        assert tokens_info[0]["char_start"] == 0
        assert tokens_info[0]["char_end"] == 10
        assert tokens_info[0]["token_text"] == "According "

        # Test citation entropy calculation end-to-end
        h_gen = compute_generation_entropy(tokens_info)
        h_cit, n_cit = compute_citation_entropy(tokens_info, text)

        assert h_gen >= 0.0
        assert h_cit >= 0.0
        assert n_cit > 0


def test_citation_alignment_end_to_end():
    """Verify regex citation span detection and overlap alignment over generated text."""
    generated_text = "Results in [Block 3] show gravity g = 9.8 m/s^2 (Smith et al., 2020)."
    spans = find_citation_spans(generated_text)
    assert len(spans) == 2
    assert generated_text[spans[0][0]:spans[0][1]] == "[Block 3]"
    assert generated_text[spans[1][0]:spans[1][1]] == "(Smith et al., 2020)"

    tokens_info = [
        {"token_id": 1, "token_text": "Results ", "char_start": 0, "char_end": 8, "entropy": 0.5},
        {"token_id": 2, "token_text": "in ", "char_start": 8, "char_end": 11, "entropy": 0.2},
        {"token_id": 3, "token_text": "[Block 3]", "char_start": 11, "char_end": 20, "entropy": 0.1},
        {"token_id": 4, "token_text": " show ", "char_start": 20, "char_end": 26, "entropy": 0.3},
        {"token_id": 5, "token_text": "gravity ", "char_start": 26, "char_end": 34, "entropy": 0.4},
        {"token_id": 6, "token_text": "g = ", "char_start": 34, "char_end": 38, "entropy": 0.2},
        {"token_id": 7, "token_text": "9.8 m/s^2 ", "char_start": 38, "char_end": 48, "entropy": 0.3},
        {"token_id": 8, "token_text": "(Smith et al., 2020).", "char_start": 48, "char_end": 69, "entropy": 0.15},
    ]

    h_cit, n_cit = compute_citation_entropy(tokens_info, generated_text)
    assert n_cit >= 2
    assert h_cit >= 0.0


def test_run_query_on_baseline_shannon_diagnostics():
    """Verify run_query_on_baseline populates shannon_diagnostics in metrics dictionary."""
    mock_rag_service = MagicMock()
    mock_engine = MagicMock()

    dummy_tokens = [
        {"token_id": 1, "token_text": "Answer: ", "char_start": 0, "char_end": 8, "entropy": 0.4},
        {"token_id": 2, "token_text": "[sciq_paper_1]", "char_start": 8, "char_end": 22, "entropy": 0.1},
    ]
    mock_engine.generate_response_with_logits.return_value = (
        "Answer: [sciq_paper_1]", dummy_tokens
    )
    mock_engine.generate_response.return_value = "Answer: [sciq_paper_1]"

    mock_rag_service.llm_engine = mock_engine
    mock_rag_service.retrieve_relevant_chunks.return_value = []

    mock_config = MagicMock()
    mock_config.is_component_enabled.side_effect = lambda k: True if k == "shannon_estimator_enabled" else False
    mock_config.rag_components = {"shannon_estimator_enabled": True}
    mock_config.data = {"rag_components": {"shannon_estimator_enabled": True}, "llm": {"hyde_enabled": False}}

    with patch("core.generation.get_baseline_config", return_value={"shannon_estimator_enabled": True}):
        answer, retrieved, metrics, chunks = run_query_on_baseline(
            mock_rag_service,
            query="What is gravity?",
            baseline="B0",
            use_cloud=False,
            config=mock_config
        )

    assert "shannon_diagnostics" in metrics
    diag = metrics["shannon_diagnostics"]
    assert "h_gen" in diag
    assert "h_citation" in diag
    assert "n_citation_tokens" in diag
    assert diag["h_gen"] >= 0.0
    assert diag["h_citation"] >= 0.0
