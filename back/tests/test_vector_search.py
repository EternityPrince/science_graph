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

    @patch("sentence_transformers.SentenceTransformer")
    @patch("torch.backends.mps.is_available")
    def test_embedding_engine_device_selection_mps(self, mock_mps_available, mock_sentence_transformer):
        mock_mps_available.return_value = True
        
        engine = EmbeddingEngine(model_name="paraphrase-MiniLM")
        engine._ensure_model_loaded()
        
        mock_sentence_transformer.assert_called_once_with("paraphrase-MiniLM", device="mps")

    @patch("sentence_transformers.SentenceTransformer")
    @patch("torch.backends.mps.is_available")
    @patch("torch.cuda.is_available")
    def test_embedding_engine_device_selection_cpu(self, mock_cuda_available, mock_mps_available, mock_sentence_transformer):
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = False
        
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
