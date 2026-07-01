import pytest
pytestmark = pytest.mark.llm

import os
import tempfile
from unittest.mock import patch
import fitz  # PyMuPDF

from src.vector_search import BM25, split_text_to_chunks, EmbeddingEngine

class TestVectorSearch:
    def test_bm25_search_scoring(self):
        corpus = [
            ("c1", "Attention is all you need for transformer networks"),
            ("c2", "Reinforcement learning is trained using reward functions"),
            ("c3", "Self-supervised contrastive learning and representation learning")
        ]
        
        bm25 = BM25(corpus)
        
        # Test query matching
        results = bm25.score("transformer attention")
        assert len(results) == 3
        # c1 should be scored highest as it contains both 'transformer' and 'attention'
        assert results[0][0] == "c1"
        assert results[0][1] > 0.0
        
        # Test query matching c2
        results_rl = bm25.score("reinforcement reward")
        assert results_rl[0][0] == "c2"
        # c3 and c1 should have 0 score
        assert results_rl[1][1] == 0.0
        assert results_rl[2][1] == 0.0

    def test_split_text_to_chunks(self):
        # Create a temp PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            # Page 1: long text
            page1 = doc.new_page()
            page1_text = "This is a very long text on page one that should be split into multiple chunks because it exceeds the chunk size threshold. We repeat this sequence to ensure it splits correctly."
            page1.insert_text((50, 50), page1_text)
            
            # Page 2: short text
            page2 = doc.new_page()
            page2_text = "Short text page two."
            page2.insert_text((50, 50), page2_text)
            
            doc.save(pdf_path)
            doc.close()

            # Split with chunk_size = 50, overlap = 10
            chunks = split_text_to_chunks(
                paper_id="paper_123",
                file_path=pdf_path,
                chunk_size=50,
                chunk_overlap=10
            )

            assert len(chunks) > 1
            # Check page numbers
            assert chunks[0].page_number == 1
            assert chunks[-1].page_number == 2
            assert chunks[-1].text_content == "Short text page two."
            assert chunks[0].id.startswith("paper_123#")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_split_text_to_chunks_structure_aware(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            
            # Markdown table & LaTeX equation within a longer text block
            text_content = (
                "Here is an introduction sentence.\n"
                "| Header 1 | Header 2 |\n"
                "|---|---|\n"
                "| Cell 1 | Cell 2 |\n"
                "Another sentence describing the table.\n"
                "$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$\n"
                "And a conclusion sentence.\n"
                "\\begin{equation} a^2 + b^2 = c^2 \\end{equation}\n"
                "Inline math: \\(x + y = z\\) and \\[a^2 = b\\] and $f(x) = y$.\n"
                "This has prices: $100 and $200."
            )
            page.insert_text((50, 50), text_content)
            doc.save(pdf_path)
            doc.close()

            # Split with chunk_size = 80, overlap = 10
            chunks = split_text_to_chunks(
                paper_id="struct_test",
                file_path=pdf_path,
                chunk_size=80,
                chunk_overlap=10
            )

            # Recombine and check that the structures exist intact in the chunks
            combined = "\n".join([c.text_content for c in chunks])
            assert "| Header 1 | Header 2 |" in combined
            assert "$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$" in combined
            assert "\\begin{equation} a^2 + b^2 = c^2 \\end{equation}" in combined
            assert "\\(x + y = z\\)" in combined
            assert "\\[a^2 = b\\]" in combined
            assert "$f(x) = y$" in combined
            assert "$100 and $200" in combined
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    @patch("sentence_transformers.SentenceTransformer")
    @patch.object(EmbeddingEngine, "_get_device", return_value="mps")
    def test_embedding_engine_device_selection_mps(self, mock_get_device, mock_sentence_transformer):
        engine = EmbeddingEngine(model_name="paraphrase-MiniLM")
        engine._ensure_model_loaded()
        
        mock_sentence_transformer.assert_called_once_with("paraphrase-MiniLM", device="mps")

    @patch("sentence_transformers.SentenceTransformer")
    @patch.object(EmbeddingEngine, "_get_device", return_value="cpu")
    def test_embedding_engine_device_selection_cpu(self, mock_get_device, mock_sentence_transformer):
        engine = EmbeddingEngine(model_name="paraphrase-MiniLM")
        engine._ensure_model_loaded()
        
        mock_sentence_transformer.assert_called_once_with("paraphrase-MiniLM", device="cpu")


    def test_bm25_incremental_indexing(self):
        corpus = [
            ("c1", "Attention is all you need for transformer networks"),
            ("c3", "Self-supervised contrastive learning and representation learning")
        ]
        bm25 = BM25(corpus)
        results = bm25.score("reward")
        assert results[0][1] == 0.0

        # Now add new documents incrementally
        bm25.add_documents([
            ("c2", "Reinforcement learning is trained using reward functions")
        ])
        results2 = bm25.score("reward")
        assert results2[0][0] == "c2"
        assert results2[0][1] > 0.0

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_engine_prefixes_non_e5(self, mock_sentence_transformer):
        mock_model = mock_sentence_transformer.return_value
        import numpy as np
        mock_model.encode.return_value = np.array([0.1, 0.2])
        
        engine = EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Single query
        emb = engine.get_embedding("hello", is_query=True)
        mock_model.encode.assert_called_with("hello", convert_to_numpy=True, show_progress_bar=False)
        assert emb == [0.1, 0.2]
        
        # Batch passage
        engine.get_embeddings(["world"], is_query=False)
        mock_model.encode.assert_called_with(["world"], convert_to_numpy=True, show_progress_bar=False)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_engine_prefixes_e5(self, mock_sentence_transformer):
        mock_model = mock_sentence_transformer.return_value
        import numpy as np
        mock_model.encode.return_value = np.array([0.3, 0.4])
        
        engine = EmbeddingEngine(model_name="intfloat/multilingual-e5-base")
        
        # Single query defaults to is_query=True -> query: prefix
        engine.get_embedding("hello")
        mock_model.encode.assert_called_with("query: hello", convert_to_numpy=True, show_progress_bar=False)
        
        # Single query with prefix already present -> no duplicate prefix
        engine.get_embedding("query: hello")
        mock_model.encode.assert_called_with("query: hello", convert_to_numpy=True, show_progress_bar=False)

        # Single passage -> passage: prefix
        engine.get_embedding("hello", is_query=False)
        mock_model.encode.assert_called_with("passage: hello", convert_to_numpy=True, show_progress_bar=False)

        # Single passage with prefix already present -> no duplicate prefix
        engine.get_embedding("passage: hello", is_query=False)
        mock_model.encode.assert_called_with("passage: hello", convert_to_numpy=True, show_progress_bar=False)

        # Batch passages defaults to is_query=False -> passage: prefix
        engine.get_embeddings(["world", "passage: already"])
        mock_model.encode.assert_called_with(["passage: world", "passage: already"], convert_to_numpy=True, show_progress_bar=False)

        # Batch queries -> query: prefix
        engine.get_embeddings(["world", "query: already"], is_query=True)
        mock_model.encode.assert_called_with(["query: world", "query: already"], convert_to_numpy=True, show_progress_bar=False)

