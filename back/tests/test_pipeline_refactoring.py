import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile
from pathlib import Path
import mlx.core as mx

from src.cli import app
from typer.testing import CliRunner

from src.llm_engine import (
    BaseLLMEngine,
    ResilientParser,
    retry_with_temp_decay,
    ConstrainedLogitsProcessor,
)
from src.services.normalization_pipeline import NormalizationPipeline
from src.services.extraction_service import ExtractionService
from src.llm_schemas import LLMExtractionResponse, LLMConcept


class DummyLLMEngineForTest(BaseLLMEngine):
    """Minimal LLM Engine for unit testing decorator and methods."""
    def __init__(self):
        self.call_count = 0
        self.temps = []

    @retry_with_temp_decay(max_retries=3)
    def test_method(self, temp: float = 0.0):
        self.call_count += 1
        self.temps.append(temp)
        if self.call_count < 3:
            raise json.JSONDecodeError("Failure", "", 0)
        return "success"


class TestPipelineRefactoring(unittest.TestCase):
    # ── 1. Resilient Parser Tests ─────────────────────────────────────────────
    def test_resilient_parser(self):
        """Verify JSON extraction with Markdown blocks, tracing, and greedy fallback."""
        # 1. Standard markdown block
        raw_markdown = "```json\n{\n  \"key\": \"value\"\n}\n```"
        self.assertEqual(ResilientParser.extract_json(raw_markdown), '{\n  "key": "value"\n}')

        # 2. Markdown block without language name
        raw_block = "```\n[\"item1\", \"item2\"]\n```"
        self.assertEqual(ResilientParser.extract_json(raw_block), '["item1", "item2"]')

        # 3. Text prefix and suffix with nested braces (tracing)
        raw_text = "Some text prefix {\"a\": {\"b\": 1}} trailing text"
        self.assertEqual(ResilientParser.extract_json(raw_text), '{"a": {"b": 1}}')

        # 4. Tracing with quotes and escapes
        raw_quotes = 'Prefix {"str": "nested } brace"} suffix'
        self.assertEqual(ResilientParser.extract_json(raw_quotes), '{"str": "nested } brace"}')

        # 5. Greedy fallback (when brackets are balanced but tracing isn't simple)
        raw_greedy = "Greedy test {\"x\": 1}"
        self.assertEqual(ResilientParser.extract_json(raw_greedy), '{"x": 1}')

    # ── 2. Retry Policy and Temp Decay ────────────────────────────────────────
    def test_retry_with_temp_decay(self):
        """Verify temperature decays properly on retry."""
        engine = DummyLLMEngineForTest()
        res = engine.test_method(temp=0.8)
        self.assertEqual(res, "success")
        self.assertEqual(engine.call_count, 3)
        # Call 1: temp = 0.8
        # Call 2: temp = 0.8 * (1.0 - 1/3) = 0.533...
        # Call 3: temp = 0.8 * (1.0 - 2/3) = 0.266...
        self.assertAlmostEqual(engine.temps[0], 0.8)
        self.assertAlmostEqual(engine.temps[1], 0.5333333, places=4)
        self.assertAlmostEqual(engine.temps[2], 0.2666666, places=4)

    # ── 3. Constrained Logits Processor ───────────────────────────────────────
    def test_constrained_logits_processor(self):
        """Verify ConstrainedLogitsProcessor masks invalid tokens."""
        mock_enforcer = MagicMock()
        # Mock allowed tokens return value
        mock_allowed = MagicMock()
        mock_allowed.allowed_tokens = [2, 5]
        mock_enforcer.get_allowed_tokens.return_value = mock_allowed

        processor = ConstrainedLogitsProcessor(mock_enforcer)
        
        # Define mock tokens [0] (prompt last token) and generated [10]
        tokens = mx.array([0, 10])
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) # logits for vocab size 6
        
        masked_logits = processor(tokens, logits)
        
        # Verify call to get_allowed_tokens with generated tokens
        mock_enforcer.get_allowed_tokens.assert_called_once_with([10])
        
        # The allowed logits indices (2 and 5) should keep their original values.
        # Other indices should be -inf.
        logits_list = masked_logits.tolist()
        self.assertEqual(logits_list[2], 3.0)
        self.assertEqual(logits_list[5], 6.0)
        self.assertEqual(logits_list[0], float("-inf"))
        self.assertEqual(logits_list[1], float("-inf"))

    # ── 4. Normalization Pipeline Tests ───────────────────────────────────────
    def test_normalization_pipeline(self):
        """Verify lemmatization, alias lookup, casing, and tag/author normalization."""
        import src.services.normalization_pipeline as norm_module
        orig_nlp = norm_module._nlp
        orig_attempted = norm_module._spacy_attempted
        norm_module._nlp = None
        norm_module._spacy_attempted = False
        try:
            pipeline = NormalizationPipeline(aliases={"ml": "Machine Learning", "cnn": "Convolutional Neural Network"})
            
            # Concept name normalization
            # 1. Alias lookup & title casing
            self.assertEqual(pipeline.normalize_concept_name("ml"), "Machine Learning")
            
            # 2. Lemmatization (using spaCy if installed, fallback otherwise)
            # "Neural Networks" -> "Neural Network"
            norm_name = pipeline.normalize_concept_name("Neural Networks")
            self.assertEqual(norm_name, "Neural Network")

            # 3. Hyphen spacing cleanup
            self.assertEqual(pipeline.normalize_concept_name("self-attention mechanism"), "Self-Attention Mechanism")
            self.assertEqual(pipeline.normalize_concept_name("convolutional neural-network"), "Convolutional Neural-Network")

            # Tag normalization: should Title Case tags to match taxonomy/aliases
            self.assertEqual(pipeline.normalize_tag("cnn"), "Convolutional Neural Network")
            self.assertEqual(pipeline.normalize_tag("deep learning"), "Deep Learning")

            # Author normalization: capitalize first letters, strip whitespace
            self.assertEqual(pipeline.normalize_author_name("  john doe  "), "John Doe")
            self.assertEqual(pipeline.normalize_author_name("alice smith-jones"), "Alice Smith-Jones")

            # Description normalization (removing LLM thinking/thought tags)
            raw_desc = "The user wants a definition. </think> Actual definition here."
            self.assertEqual(pipeline.normalize_description(raw_desc), "Actual definition here.")
            
            raw_desc2 = "<think> some thinking </think>Another definition."
            self.assertEqual(pipeline.normalize_description(raw_desc2), "Another definition.")
            
            raw_desc3 = "<thought> thinking </thought> Definition with thought."
            self.assertEqual(pipeline.normalize_description(raw_desc3), "Definition with thought.")

            # Extraction response validation
            resp = LLMExtractionResponse(
                authors=["john doe", "JOHN DOE", "Jane Doe"],
                concepts=[
                    LLMConcept(name="Neural Networks", description="Short desc"),
                    LLMConcept(name="neural network", description="Longer description here")
                ],
                tags=["cnn", "CNN", "deep learning"]
            )
            norm_resp = pipeline.normalize_extraction_response(resp)
            self.assertEqual(norm_resp.authors, ["John Doe", "Jane Doe"])
            # Concepts deduplicated by slug, keeping first seen
            self.assertEqual(len(norm_resp.concepts), 1)
            self.assertEqual(norm_resp.concepts[0].name, "Neural Network")
            # Tags deduplicated by slug
            self.assertEqual(norm_resp.tags, ["Convolutional Neural Network", "Deep Learning"])
        finally:
            norm_module._nlp = orig_nlp
            norm_module._spacy_attempted = orig_attempted

    @patch("spacy.load")
    @patch("spacy.cli.download")
    def test_spacy_load_fallback_download(self, mock_download, mock_load):
        from src.services.normalization_pipeline import get_spacy_nlp
        import src.services.normalization_pipeline as norm_module
        
        # Save original value and reset
        orig_nlp = norm_module._nlp
        orig_attempted = norm_module._spacy_attempted
        norm_module._nlp = None
        norm_module._spacy_attempted = False
        
        try:
            # Setup mock_load to fail on first attempt, succeed on second attempt
            mock_nlp = MagicMock()
            mock_load.side_effect = [OSError("Model not found"), mock_nlp]
            
            nlp = get_spacy_nlp()
            
            self.assertEqual(nlp, mock_nlp)
            mock_download.assert_called_once_with("en_core_web_sm")
            self.assertEqual(mock_load.call_count, 2)
        finally:
            norm_module._nlp = orig_nlp
            norm_module._spacy_attempted = orig_attempted

    # ── 5. Token Management & Semantic Splitting ──────────────────────────────
    def test_semantic_splitting(self):
        """Verify paragraph-aware chunking split rules."""
        engine = MagicMock()
        # Mock count_tokens to return character length // 4
        engine.count_tokens.side_effect = lambda t: len(t) // 4
        
        extractor = ExtractionService(llm_engine=engine)
        
        text = "Paragraph one is short.\n\nParagraph two is also short.\n\nParagraph three is a bit longer."
        # If we split with a budget of 10 tokens (approx 40 chars)
        chunks = extractor.split_text_semantically(text, max_chunk_tokens=10, overlap_tokens=2)
        self.assertTrue(len(chunks) > 1)
        
        # Verify it splits by paragraph boundaries if possible
        for chunk in chunks:
            self.assertTrue(chunk.startswith("Paragraph"))

    # ── 6. Map-Reduce Merging ─────────────────────────────────────────────────
    def test_map_reduce_merging(self):
        """Verify Map-Reduce chunk extraction merges and deduplicates correctly."""
        engine = MagicMock()
        engine.count_tokens.side_effect = lambda t: len(t) // 4
        
        # Chunk extraction mock data:
        # Chunk 1 returns concept A and B
        # Chunk 2 returns concept B (with longer description) and C
        chunk1_data = {
            "authors": ["Alice Smith"],
            "concepts": [
                {"name": "Concept A", "description": "Desc A"},
                {"name": "Concept B", "description": "Short B"}
            ],
            "tags": ["Deep Learning"]
        }
        chunk2_data = {
            "authors": ["Bob Jones"],
            "concepts": [
                {"name": "Concept B", "description": "Much longer description for B"},
                {"name": "Concept C", "description": "Desc C"}
            ],
            "tags": ["NLP"]
        }
        engine.extract_concepts_and_metadata.side_effect = [chunk1_data, chunk2_data]
        
        extractor = ExtractionService(llm_engine=engine)
        
        # Setup configs to trigger map-reduce (threshold = 85% of limit)
        # Let's set limit = 100, threshold = 85.
        # We pass full_text long enough to exceed 85 tokens.
        long_body = "x " * 200 # 400 chars, count_tokens will return 100
        
        with patch("src.services.extraction_service.config") as mock_config:
            mock_config.llm_extraction_input_limit = 100
            mock_config.taxonomy = {
                "concepts": {},
                "topics": {},
                "descriptions": {}
            }
            
            # Patch split_text_semantically to force it to return two chunks for the map phase.
            with patch.object(extractor, "split_text_semantically", return_value=["chunk 1", "chunk 2"]):
                res = extractor.extract("Title", "Abstract", long_body, use_llm=True)
            
            self.assertTrue(res.via_llm)
            self.assertEqual(set(res.authors), {"Alice Smith", "Bob Jones"})
            self.assertEqual(set(res.tags), {"Deep Learning", "Natural Language Processing"})
            
            # Concepts should be A, B, C, with B having the longer description
            concepts_dict = {c["name"]: c["description"] for c in res.concepts}
            self.assertIn("Concept A", concepts_dict)
            self.assertIn("Concept B", concepts_dict)
            self.assertIn("Concept C", concepts_dict)
            self.assertEqual(concepts_dict["Concept B"], "Much longer description for B")

    # ── 7. CLI Command Test ───────────────────────────────────────────────────
    @patch("src.cli.LLMEngine")
    def test_cli_extract_file(self, mock_llm_engine_cls):
        """Verify the extract-file CLI command runs successfully and prints JSON graph."""
        mock_engine = MagicMock()
        mock_llm_engine_cls.return_value = mock_engine
        
        # Mock LLM output
        mock_engine.extract_concepts_and_metadata.return_value = {
            "authors": ["Alice Smith"],
            "concepts": [{"name": "Transformer", "description": "Attention model."}],
            "tags": ["Deep Learning"]
        }
        mock_engine.count_tokens.return_value = 10
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_doc.txt"
            file_path.write_text("# My Awesome Document\n\nThis is a body paragraph about deep learning models.", encoding="utf-8")
            
            # Patch config taxonomy to avoid issues
            with patch("src.cli.config") as mock_config:
                mock_config.llm_extraction_input_limit = 4000
                mock_config.taxonomy = {
                    "concepts": {},
                    "topics": {},
                    "descriptions": {}
                }
                
                result = runner.invoke(app, ["extract-file", str(file_path)])
                
                self.assertEqual(result.exit_code, 0)
                # Verify standard output contains the extracted JSON
                stdout_data = json.loads(result.stdout)
                self.assertEqual(stdout_data["authors"], ["Alice Smith"])
                self.assertEqual(stdout_data["tags"], ["Deep Learning"])
                self.assertEqual(stdout_data["concepts"][0]["name"], "Transformer")

    # ── 8. Trace Timing and CLI Trace Flag Tests ──────────────────────────────
    @patch("src.cli.Indexer")
    @patch("src.cli.get_services")
    def test_cli_index_trace(self, mock_get_services, mock_indexer_cls):
        """Verify the index CLI command runs with -t/--trace and prints a table."""
        mock_graph_repo = MagicMock()
        mock_vector_repo = MagicMock()
        mock_emb_engine = MagicMock()
        mock_llm_engine = MagicMock()
        mock_get_services.return_value = (mock_graph_repo, mock_vector_repo, mock_emb_engine, mock_llm_engine)
        
        mock_indexer = MagicMock()
        mock_indexer_cls.return_value = mock_indexer
        
        def mock_index_pdf(path, trace_info=None):
            if trace_info is not None:
                trace_info.setdefault("stages", {})["Document Parsing"] = 0.05
                trace_info.setdefault("stages", {})["Concept & Tag Extraction"] = 1.2
                trace_info.setdefault("tokens", {})["Concept & Tag Extraction"] = 500
            return "paper_id"
            
        mock_indexer.index_pdf.side_effect = mock_index_pdf
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.pdf"
            file_path.write_text("dummy", encoding="utf-8")
            result = runner.invoke(app, ["index", str(file_path), "-t"])
            print("EXIT CODE:", result.exit_code)
            print("STDOUT IS:", repr(result.stdout))
            if result.exception:
                print("EXCEPTION:", repr(result.exception))
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Ingestion Trace", result.stdout)
            self.assertIn("Document Parsing", result.stdout)
            self.assertIn("Concept & Tag Extraction", result.stdout)
            self.assertIn("500", result.stdout)

    def test_indexer_trace_stages(self):
        """Verify that Indexer timing context manager accumulates stages accurately."""
        from src.indexer import Indexer
        import time
        
        graph_repo = MagicMock()
        vector_repo = MagicMock()
        emb_engine = MagicMock()
        llm_engine = MagicMock()
        
        indexer = Indexer(graph_repo, vector_repo, emb_engine, llm_engine)
        trace_info = {}
        
        with indexer._trace_stage("Stage A", trace_info):
            time.sleep(0.01)
            
        with indexer._trace_stage("Stage A", trace_info):
            time.sleep(0.01)
            
        self.assertIn("Stage A", trace_info["stages"])
        self.assertGreater(trace_info["stages"]["Stage A"], 0.015)


if __name__ == "__main__":
    unittest.main()
