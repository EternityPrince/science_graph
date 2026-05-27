import unittest
from unittest.mock import MagicMock
from src.models import Chunk, Paper, Concept
from src.services.graph_expander import ExperimentalGraphExpander
from src.llm_schemas import EvidenceListResponse, EvidenceItem

class TestGraphExpander(unittest.TestCase):
    def setUp(self):
        self.graph_repo = MagicMock()
        self.vector_repo = MagicMock()
        self.llm_engine = MagicMock()
        self.reranker = MagicMock()
        
        self.expander = ExperimentalGraphExpander(
            graph_repo=self.graph_repo,
            vector_repo=self.vector_repo,
            llm_engine=self.llm_engine,
            reranker=self.reranker,
            p_base=0.75,
            gamma=0.5,
            limit=5,
            top_chunks_per_paper=2
        )

    def test_adaptive_stopping(self):
        """Verify that the discovery loop stops when K_n < 1."""
        # Initial chunk from paper_1
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Intro text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        # Mock get_papers_batch for Level 0
        p1 = Paper(id="paper_1", title="Title One", properties={"summary": "Summary One"})
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        
        # Mock get_neighbors for paper_1 to return 2 neighbors
        # For Hop 1: n = 1
        # K_1 = 2 * (0.75 * 0.5^1) = 2 * 0.375 = 0.75.
        # Since K_1 = 0.75 < 1.0, the loop should break and not perform Hop 1 expansion.
        self.graph_repo.get_neighbors.return_value = [
            ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}"),
            ("paper_3", "Paper", "CITES", "paper_1", "Paper", "{}")
        ]
        
        # Mock LLM response for the evidence filtering step (Level 0 facts only)
        # Only fact_1 (the starting paper) and fact_2 (the initial chunk)
        mock_response = EvidenceListResponse(
            evidence_list=[
                EvidenceItem(id="fact_1", score=0.9, is_essential=True),
                EvidenceItem(id="fact_2", score=0.9, is_essential=True)
            ]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        result = self.expander.expand("test query", initial_chunks)
        
        # Verify that get_neighbors was called once for paper_1
        self.graph_repo.get_neighbors.assert_called_once_with("paper_1", max_depth=1)
        
        # Verify that reranker.predict was NOT called because Hop 1 stopped before building cards/reranking
        self.reranker.predict.assert_not_called()
        
        # Check final output block contains initial items
        self.assertIn("[Paper] Title One", result)
        self.assertIn("[Chunk] Title One (Page 1)", result)

    def test_bidirectional_citations(self):
        """Verify that citations (outbound) and cited_by (inbound) are gathered correctly."""
        # Starting point
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Text content", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        
        # Hop 1 (n=1): return 3 neighbors.
        # K_1 = 3 * (0.75 * 0.5^1) = 1.125 >= 1.0. Loop continues. Limit = int(1.125) = 1 selected candidate.
        # Neighbors:
        # - paper_2 (outbound CITES: paper_1 cites paper_2)
        # - paper_3 (inbound CITES: paper_3 cites paper_1)
        # - concept_1 (MENTIONS_CONCEPT)
        self.graph_repo.get_neighbors.return_value = [
            ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}"),
            ("paper_3", "Paper", "CITES", "paper_1", "Paper", "{}"),
            ("paper_1", "Paper", "MENTIONS_CONCEPT", "concept_1", "Concept", "{}")
        ]
        
        # Mock get_paper for the neighbors
        import datetime
        p2 = Paper(id="paper_2", title="Title Two", abstract="Abstract Two", year=datetime.datetime.now().year)
        p3 = Paper(id="paper_3", title="Title Three")
        concept1 = Concept(id="concept_1", name="Concept One", properties={"description": "Desc One"})
        
        def mock_get_paper(pid):
            if pid == "paper_2": return p2
            if pid == "paper_3": return p3
            return None
        self.graph_repo.get_paper.side_effect = mock_get_paper
        
        # Mock get_papers_batch for neighbor fetch
        self.graph_repo.get_papers_batch.side_effect = lambda ids: {
            pid: p for pid in ids if (p := mock_get_paper(pid)) is not None
        }
        self.graph_repo.get_concept.return_value = concept1
        
        self.reranker.predict.side_effect = [
            [0.8, 0.3, 0.5],  # Hop 1 reranking
            [0.9]             # Chunk ingestion reranking
        ]
        
        # Mock vector chunks for paper_2 (selected candidate)
        c2 = Chunk(id="chunk_2", paper_id="paper_2", text_content="Text from paper 2", page_number=1)
        self.vector_repo.get_chunks_for_paper.return_value = [c2]
        
        # Mock final LLM evidence filtering
        mock_response = EvidenceListResponse(
            evidence_list=[
                EvidenceItem(id="fact_1", score=0.9, is_essential=True),  # paper_1
                EvidenceItem(id="fact_3", score=0.8, is_essential=True)   # paper_2
            ]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        result = self.expander.expand("query", initial_chunks)
        
        # Check that paper_2 is included as essential in the result
        self.assertIn("[Paper] Title Two", result)
        # Since concept_1 was not selected (k_limit=1), it shouldn't be in the result
        self.assertNotIn("[Concept] Concept One", result)

    def test_semaphore_usage(self):
        """Verify that reranker is called once per hop, and LLM is called once at the end."""
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Intro text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        
        # Hop 1 (n=1): return 3 neighbors.
        # K_1 = 3 * (0.75 * 0.5^1) = 1.125 >= 1.0. int(1.125) = 1.
        self.graph_repo.get_neighbors.side_effect = [
            [
                ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}"),
                ("paper_1", "Paper", "CITES", "paper_3", "Paper", "{}"),
                ("paper_1", "Paper", "MENTIONS_CONCEPT", "concept_1", "Concept", "{}")
            ],
            [] # Hop 2 returns no neighbors
        ]
        
        self.graph_repo.get_paper.side_effect = lambda pid: Paper(id=pid, title=f"Title {pid}")
        self.graph_repo.get_concept.return_value = Concept(id="concept_1", name="Concept One")
        
        # Reranker returns scores for 3 candidates
        self.reranker.predict.return_value = [0.9, 0.3, 0.2]
        
        # Mock final LLM
        mock_response = EvidenceListResponse(
            evidence_list=[EvidenceItem(id="fact_1", score=0.9, is_essential=True)]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        self.expander.expand("query", initial_chunks)
        
        # Verify reranker predict called exactly once (for the single hop)
        self.assertEqual(self.reranker.predict.call_count, 1)
        # Verify LLM generate_and_validate_json called exactly once
        self.assertEqual(self.llm_engine.generate_and_validate_json.call_count, 1)

    def test_resilient_json_fallback(self):
        """Verify safety fallback if evidence filtering LLM call fails."""
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Intro text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        self.graph_repo.get_neighbors.return_value = []
        
        # LLM validation raises exception (malformed JSON/model down)
        self.llm_engine.generate_and_validate_json.side_effect = Exception("JSON Decode Failed")
        
        result = self.expander.expand("query", initial_chunks)
        
        # Check that we fell back gracefully and included all facts
        self.assertIn("[Paper] Title One", result)
        self.assertIn("[Chunk] Title One (Page 1)", result)

    def test_trace_logging(self):
        """Verify that expanding with trace=True executes successfully and outputs logs."""
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Intro text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        self.graph_repo.get_neighbors.return_value = []
        
        mock_response = EvidenceListResponse(
            evidence_list=[EvidenceItem(id="fact_1", score=0.9, is_essential=True)]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        result = self.expander.expand("test query", initial_chunks, trace=True)
        self.assertIn("[Paper] Title One", result)

if __name__ == "__main__":
    unittest.main()
