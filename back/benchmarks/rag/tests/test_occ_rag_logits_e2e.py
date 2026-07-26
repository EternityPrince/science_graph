"""E2E Test for MLX Logits Streaming, Token Entropy, and Citation Alignment.
"""

import json
from pathlib import Path
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
    # token_ids are in-range for the fake vocab so chosen-token logprob is populated
    sample_data = [
        (0, "According ", None),
        (1, "to ", None),
        (2, "[sciq_paper_1]", None),
        (3, ", ", None),
        (4, "the ", None),
        (5, "energy ", None),
        (0, "is ", None),
        (1, "conserved", None),
        (2, ".", None),
    ]
    try:
        import mlx.core as mx
        # vocab >= 6 so top-k=5 extraction and chosen-token indexing are valid
        fake_logprobs = mx.array([-1.0, -2.0, -0.5, -4.0, -3.0, -2.5], dtype=mx.float32)
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
        engine.tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}" if ids else ""

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
            # Compact logprob telemetry required for MSP/margin/LL/CLR
            assert "logprob" in tok
            assert isinstance(tok["logprob"], float)
            assert "msp" in tok
            assert isinstance(tok["msp"], float)
            assert 0.0 < tok["msp"] <= 1.0
            assert "logit_margin" in tok
            assert isinstance(tok["logit_margin"], float)
            assert tok["logit_margin"] >= 0.0
            assert "top_logprobs" in tok
            assert isinstance(tok["top_logprobs"], dict)
            assert 1 <= len(tok["top_logprobs"]) <= 5

        # Chosen token id=0 maps to logprob -1.0 in the fake distribution
        assert tokens_info[0]["logprob"] == pytest.approx(-1.0)
        # top1 is index 2 (-0.5), top2 is index 0 (-1.0) => margin 0.5
        assert tokens_info[0]["logit_margin"] == pytest.approx(0.5)

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


def test_openai_engine_generate_response_with_logits_mocked():
    """Verify OpenAILLMEngine parses choice logprobs.content into aligned tokens_info."""
    from src.llm_engine.openai_impl import OpenAILLMEngine
    with patch("openai.OpenAI"):
        engine = OpenAILLMEngine(api_key="test-key", model_name="gpt-4o")

        mock_choice = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.token = "Hello"
        mock_item1.logprob = -0.1
        mock_top1 = MagicMock()
        mock_top1.token = "Hello"
        mock_top1.logprob = -0.1
        mock_item1.top_logprobs = [mock_top1]

        mock_item2 = MagicMock()
        mock_item2.token = " world"
        mock_item2.logprob = -0.2
        mock_item2.top_logprobs = []

        mock_choice.logprobs.content = [mock_item1, mock_item2]
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        engine._call_completions_with_lock = MagicMock(return_value=mock_response)

        text, tokens_info = engine.generate_response_with_logits("Say Hello world")

        assert text == "Hello world"
        assert len(tokens_info) == 2
        assert tokens_info[0]["token_text"] == "Hello"
        assert tokens_info[0]["logprob"] == -0.1
        assert tokens_info[0]["top_logprobs"] == {"Hello": -0.1}
        assert tokens_info[1]["token_text"] == " world"
        assert tokens_info[1]["logprob"] == -0.2


def test_ensure_b0_entropy_caching_and_generation():
    """Verify _ensure_b0_entropy generates B0 entropy when missing and caches result."""
    from core.generation import _ensure_b0_entropy
    rag_service = MagicMock()
    mock_engine = MagicMock()
    dummy_tokens = [
        {"token_text": "Ground", "logprob": -0.5, "top_logprobs": {"Ground": -0.5, "Earth": -1.2}},
        {"token_text": " truth", "logprob": -0.3, "top_logprobs": {" truth": -0.3}}
    ]
    mock_engine.generate_response_with_logits.return_value = (
        "Ground truth response", dummy_tokens
    )
    rag_service.llm_engine = mock_engine
    mock_config = MagicMock()

    h_b0 = _ensure_b0_entropy(rag_service, "what is gravity?", mock_config)
    assert h_b0 >= 0.0
    assert rag_service._query_b0_h_gen["what is gravity?"] == h_b0

    # Ensure second call uses cached value without calling llm_engine again
    rag_service.llm_engine.generate_response_with_logits.reset_mock()
    h_b0_cached = _ensure_b0_entropy(rag_service, "what is gravity?", mock_config)
    assert h_b0_cached == h_b0
    rag_service.llm_engine.generate_response_with_logits.assert_not_called()


def test_b0_entropy_disk_cache_persistence_e2e():
    """Test that _ensure_b0_entropy writes to .cache/b0_entropy.json and a new rag_service instance reads directly from disk."""
    from core.generation import _ensure_b0_entropy
    cache_file = Path(__file__).resolve().parents[1] / ".cache" / "b0_entropy.json"
    test_query = "e2e_disk_cache_persistence_test_query"

    # Pre-test cleanup if query exists in disk cache from prior runs
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if test_query in data:
                del data[test_query]
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    rag_service1 = MagicMock()
    mock_engine1 = MagicMock()
    dummy_tokens = [
        {"token_text": "Disk", "entropy": 0.3},
        {"token_text": " cache", "entropy": 0.2},
    ]
    mock_engine1.generate_response_with_logits.return_value = (
        "Disk cache", dummy_tokens
    )
    rag_service1.llm_engine = mock_engine1
    rag_service1._query_b0_h_gen = {}
    mock_config = MagicMock()

    try:
        # 1. Generate B0 entropy and ensure it's written to disk cache
        h1 = _ensure_b0_entropy(rag_service1, test_query, mock_config)
        assert h1 >= 0.0
        mock_engine1.generate_response_with_logits.assert_called_once()
        assert cache_file.exists()

        with open(cache_file, "r", encoding="utf-8") as f:
            disk_cache = json.load(f)
        assert test_query in disk_cache
        assert float(disk_cache[test_query]) == pytest.approx(h1)

        # 2. Separate new rag_service instance with empty in-memory cache
        rag_service2 = MagicMock()
        mock_engine2 = MagicMock()
        rag_service2.llm_engine = mock_engine2
        rag_service2._query_b0_h_gen = {}

        h2 = _ensure_b0_entropy(rag_service2, test_query, mock_config)
        assert h2 == pytest.approx(h1)
        # Verify llm_engine was NOT called for rag_service2 because disk cache was hit
        mock_engine2.generate_response_with_logits.assert_not_called()
        assert rag_service2._query_b0_h_gen[test_query] == pytest.approx(h1)

    finally:
        # Clean up test query from disk cache to keep workspace clean
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if test_query in data:
                    del data[test_query]
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


def test_mlx_topk_logits_correctness_and_fallback():
    """Test MlxLLMEngine logprobs top-50 argpartition calculation, fallback for small arrays, and exception handling."""
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx is not installed")

    with patch("os.path.isdir", return_value=True), \
         patch("src.llm_engine.mlx_impl.MlxLLMEngine._ensure_model_loaded") as mock_ensure:
        mock_ensure.return_value = None
        engine = MlxLLMEngine(model_path="/dummy/path")
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()

        engine.tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}" if ids else ""

        # 1. Test large logprob array (vocab size 100,000) using top-50 argpartition
        large_logprobs = mx.full((100000,), -10.0, dtype=mx.float32)
        large_logprobs[1234] = 0.0
        large_logprobs[5678] = -1.0

        def stream_large(model, tokenizer, prompt, max_tokens=100, sampler=None):
            yield DummyResponse(1234, "large", large_logprobs)

        with patch("mlx_lm.stream_generate", side_effect=stream_large):
            text_large, tokens_large = engine.generate_response_with_logits("test large")
            assert text_large == "large"
            assert len(tokens_large) == 1
            assert isinstance(tokens_large[0]["entropy"], float)
            assert tokens_large[0]["entropy"] >= 0.0
            assert tokens_large[0]["logprob"] == pytest.approx(0.0)
            assert tokens_large[0]["msp"] > 0.0
            assert tokens_large[0]["logit_margin"] == pytest.approx(1.0)  # 0.0 - (-1.0)
            assert isinstance(tokens_large[0]["top_logprobs"], dict)
            assert len(tokens_large[0]["top_logprobs"]) == 5

        # 2. Test small logprob array (< 50 elements) triggering argpartition exception fallback
        small_logprobs = mx.array([-1.0, -2.0, -0.5, -3.0, -1.5, -2.5, -4.0, -0.1, -2.2, -3.3])

        def stream_small(model, tokenizer, prompt, max_tokens=100, sampler=None):
            yield DummyResponse(7, "small", small_logprobs)

        with patch("mlx_lm.stream_generate", side_effect=stream_small):
            text_small, tokens_small = engine.generate_response_with_logits("test small")
            assert text_small == "small"
            assert len(tokens_small) == 1
            assert isinstance(tokens_small[0]["entropy"], float)
            assert tokens_small[0]["entropy"] >= 0.0
            assert tokens_small[0]["logprob"] == pytest.approx(-0.1)
            assert tokens_small[0]["msp"] > 0.0
            assert tokens_small[0]["logit_margin"] > 0.0
            assert isinstance(tokens_small[0]["top_logprobs"], dict)
            assert 1 <= len(tokens_small[0]["top_logprobs"]) <= 5

        # 3. Test exception fallback setting entropy to 0.0 on invalid logprobs
        bad_logprobs = MagicMock()
        bad_logprobs.astype.side_effect = RuntimeError("MLX conversion failure")

        def stream_bad(model, tokenizer, prompt, max_tokens=100, sampler=None):
            yield DummyResponse(3, "bad", bad_logprobs)

        with patch("mlx_lm.stream_generate", side_effect=stream_bad):
            text_bad, tokens_bad = engine.generate_response_with_logits("test bad")
            assert text_bad == "bad"
            assert len(tokens_bad) == 1
            assert tokens_bad[0]["entropy"] == 0.0
            assert tokens_bad[0]["logprob"] == 0.0
            assert tokens_bad[0]["msp"] == 0.0
            assert tokens_bad[0]["logit_margin"] == 0.0
            assert tokens_bad[0]["top_logprobs"] == {}


def test_mlx_score_text_logprobs_teacher_force_mocked():
    """Verify score_text_logprobs teacher-forces answer tokens and returns compact telemetry."""
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx is not installed")

    with patch("os.path.isdir", return_value=True), \
         patch("src.llm_engine.mlx_impl.MlxLLMEngine._ensure_model_loaded") as mock_ensure:
        mock_ensure.return_value = None
        engine = MlxLLMEngine(model_path="/dummy/path")

        # prompt_ids=[10,11], answer_ids=[0,1] => full length 4, logits (1,4,V)
        vocab = 8
        fake_logits = mx.zeros((1, 4, vocab), dtype=mx.float32)
        # Position 1 predicts answer token 0: peak at id 0
        fake_logits[0, 1, 0] = 5.0
        fake_logits[0, 1, 1] = 1.0
        # Position 2 predicts answer token 1: peak at id 1
        fake_logits[0, 2, 1] = 4.0
        fake_logits[0, 2, 0] = 0.5

        engine.model = MagicMock(return_value=fake_logits)
        engine.tokenizer = MagicMock()
        engine.tokenizer.bos_token = None

        def _encode(text, add_special_tokens=True):
            return [10, 11] if text == "Score me" else [0, 1]

        def _decode(ids):
            ids = list(ids)
            if ids == [0]:
                return "y"
            if ids == [0, 1]:
                return "yes"
            if ids == [1]:
                return "es"
            return "".join(f"t{i}" for i in ids)

        engine.tokenizer.encode = MagicMock(side_effect=_encode)
        engine.tokenizer.decode = MagicMock(side_effect=_decode)
        engine.tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda messages, **kw: messages[0]["content"]
        )

        tokens_info = engine.score_text_logprobs("Score me", "yes")
        assert len(tokens_info) == 2
        for tok in tokens_info:
            assert "token_id" in tok
            assert "token_text" in tok
            assert "char_start" in tok
            assert "char_end" in tok
            assert "logprob" in tok
            assert "entropy" in tok
            assert "msp" in tok
            assert "logit_margin" in tok
            assert "top_logprobs" in tok
            assert isinstance(tok["top_logprobs"], dict)
            assert tok["msp"] > 0.0
            assert tok["entropy"] >= 0.0

        assert tokens_info[0]["token_id"] == 0
        assert tokens_info[1]["token_id"] == 1
        assert tokens_info[0]["token_text"] == "y"
        assert tokens_info[1]["token_text"] == "es"
        assert tokens_info[0]["char_start"] == 0
        assert tokens_info[0]["char_end"] == 1
        assert tokens_info[1]["char_start"] == 1
        assert tokens_info[1]["char_end"] == 3
        # Peaked fake logits => high chosen logprob and positive margin
        assert tokens_info[0]["logprob"] > -1.0
        assert tokens_info[1]["logprob"] > -1.0
        assert tokens_info[0]["logit_margin"] > 0.0
        engine.model.assert_called_once()


def test_b1_b2_pipeline_uses_cached_b0_entropy():
    """Test running run_query_on_baseline sequentially for B1 and B2 on the same query, verifying B0 entropy is generated only once."""
    query = "Does B1 and B2 reuse cached B0 entropy?"
    cache_file = Path(__file__).resolve().parents[1] / ".cache" / "b0_entropy.json"

    # Pre-test cleanup if query exists in disk cache
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if query in data:
                del data[query]
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    mock_rag_service = MagicMock()
    mock_engine = MagicMock()
    mock_rag_service.llm_engine = mock_engine
    mock_rag_service._query_b0_h_gen = {}

    mock_chunk = MagicMock()
    mock_chunk.paper_id = "paper_1"
    mock_chunk.text_content = "Relevant text block"
    mock_rag_service.retrieve_relevant_chunks.return_value = [(mock_chunk, 0.9)]
    mock_rag_service.ask.return_value = "Answer for baseline"

    dummy_tokens = [
        {"token_id": 1, "token_text": "Answer ", "char_start": 0, "char_end": 7, "entropy": 0.4},
        {"token_id": 2, "token_text": "text", "char_start": 7, "char_end": 11, "entropy": 0.2},
    ]
    mock_engine.generate_response_with_logits.return_value = ("Answer text", dummy_tokens)
    mock_engine.generate_response.return_value = "Answer text"

    mock_config = MagicMock()
    mock_config.is_component_enabled.side_effect = lambda k: True if k == "shannon_estimator_enabled" else False
    mock_config.rag_components = {"shannon_estimator_enabled": True}
    mock_config.data = {"rag_components": {"shannon_estimator_enabled": True}, "llm": {"hyde_enabled": False}}

    def mock_get_baseline_config(baseline, rag_comp):
        return {"shannon_estimator_enabled": True, "lexical_search": (baseline == "B1"), "dense_search": (baseline == "B2")}

    try:
        with patch("core.generation.get_baseline_config", side_effect=mock_get_baseline_config):
            # First baseline run: B1
            res_b1 = run_query_on_baseline(
                mock_rag_service, query=query, baseline="B1", use_cloud=False, config=mock_config
            )

            # Second baseline run: B2 on the same query
            res_b2 = run_query_on_baseline(
                mock_rag_service, query=query, baseline="B2", use_cloud=False, config=mock_config
            )

            # Verify B0 zero-shot generation (prompt "Ответь на основе своих общих знаний") was called ONLY ONCE
            b0_calls = [
                call_item for call_item in mock_engine.generate_response_with_logits.call_args_list
                if "Answer based on your general knowledge" in str(call_item)
            ]
            assert len(b0_calls) == 1

        # Verify both runs produced valid Shannon diagnostics
        assert "shannon_diagnostics" in res_b1[2]
        assert "shannon_diagnostics" in res_b2[2]

    finally:
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if query in data:
                    del data[query]
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass







