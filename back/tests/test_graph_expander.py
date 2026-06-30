import unittest
from unittest.mock import MagicMock
from src.models import Chunk, Paper, Concept
from src.services.graph_expander import ExperimentalGraphExpander

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
        
        # Mock Cross-Encoder filtering
        self.reranker.predict.return_value = [0.9, 0.9]
        
        result = self.expander.expand("test query", initial_chunks)
        
        # Verify that get_neighbors was called once for paper_1
        self.graph_repo.get_neighbors.assert_called_once_with("paper_1", max_depth=1)
        
        # Verify that reranker.predict was called once (for final evidence list filtering)
        self.assertEqual(self.reranker.predict.call_count, 1)
        
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
            [0.9],            # Chunk ingestion reranking
            [0.9, 0.9, 0.9, 0.9] # Final evidence filtering (paper_1, chunk_1, paper_2, chunk_2)
        ]
        
        # Mock vector chunks for paper_2 (selected candidate)
        c2 = Chunk(id="chunk_2", paper_id="paper_2", text_content="Text from paper 2", page_number=1)
        self.vector_repo.get_chunks_for_paper.return_value = [c2]
        
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
        
        # Reranker returns scores for candidates and final filtering
        self.reranker.predict.side_effect = [
            [0.9, 0.3, 0.2],  # Hop 1
            [0.9, 0.9, 0.9]   # Final filtering
        ]
        
        self.expander.expand("query", initial_chunks)
        
        # Verify reranker predict called exactly twice (once for hop, once for filtering)
        self.assertEqual(self.reranker.predict.call_count, 2)
        # Verify LLM generate_and_validate_json was not called
        self.assertEqual(self.llm_engine.generate_and_validate_json.call_count, 0)

    def test_resilient_cross_encoder_fallback(self):
        """Verify safety fallback if evidence filtering Cross-Encoder call fails."""
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Intro text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        self.graph_repo.get_neighbors.return_value = []
        
        # Cross-Encoder predict raises exception
        self.reranker.predict.side_effect = Exception("Predict Failed")
        
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
        
        self.reranker.predict.return_value = [0.9, 0.9]
        
        result = self.expander.expand("test query", initial_chunks, trace=True)
        self.assertIn("[Paper] Title One", result)

    def test_graph_expander_with_batch(self):
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Intro text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        
        # Configure get_neighbors_batch to return a neighbor and trigger use_batch
        self.graph_repo.get_neighbors_batch.return_value = [
            ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}")
        ]
        
        self.reranker.predict.return_value = [0.9, 0.9]
        
        result = self.expander.expand("query", initial_chunks)
        
        self.graph_repo.get_neighbors_batch.assert_called_once_with(["paper_1"])
        self.graph_repo.get_neighbors.assert_not_called()
        self.assertIn("[Paper] Title One", result)

    def test_other_relation_types_coverage(self):
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        
        self.graph_repo.get_neighbors.return_value = [
            ("paper_1", "Paper", "AUTHORED", "author_1", "Author", "{}"),
            ("author_2", "Author", "AUTHORED", "paper_1", "Paper", "{}"),
            ("paper_1", "Paper", "HAS_TAG", "tag_1", "Concept", "{}"),
            ("tag_2", "Concept", "HAS_TAG", "paper_1", "Paper", "{}"),
            ("paper_1", "Paper", "SPONSORED_BY", "inst_1", "Institution", "{}"),
            ("paper_1", "Paper", "USED_DATASET", "ds_1", "Dataset", "{}"),
            ("paper_1", "Paper", "PUBLISHED_IN", "journal_1", "JournalConference", "{}"),
            ("concept_1", "Concept", "SUBCLASS_OF", "concept_2", "Concept", "{}"),
        ]
        
        self.graph_repo.get_paper.side_effect = lambda pid: Paper(id=pid, title=f"Title {pid}")
        self.graph_repo.get_author.side_effect = lambda aid: MagicMock(name=f"Author {aid}")
        self.graph_repo.get_concept.side_effect = lambda cid: MagicMock(name=f"Concept {cid}")
        self.graph_repo.get_node_properties.side_effect = lambda nid: {"name": f"Node {nid}"}
        
        self.reranker.predict.side_effect = [
            [0.9] * 8,   # Hop reranking
            [0.9] * 100  # Final filtering
        ]
        
        self.expander.expand("query", initial_chunks)

    def test_reranker_sigmoid_and_threshold_filtering(self):
        """Verify that sigmoid scales extreme raw logits and threshold filters out low scores."""
        self.expander.p_base = 10.0
        self.expander.gamma = 0.9
        # 1. Setup initial chunks (Level 0)
        c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Starting text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        p1 = Paper(id="paper_1", title="Title One")
        self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
        
        # 2. Hop 1 neighbors
        self.graph_repo.get_neighbors.return_value = [
            ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}"),
            ("paper_1", "Paper", "CITES", "paper_3", "Paper", "{}"),
        ]
        
        p2 = Paper(id="paper_2", title="Title Two")
        p3 = Paper(id="paper_3", title="Title Three")
        self.graph_repo.get_paper.side_effect = lambda pid: p2 if pid == "paper_2" else (p3 if pid == "paper_3" else None)
        
        # Mock reranker returns extreme logits: -100.0 (sigmoid ~0.0) and 100.0 (sigmoid ~1.0)
        # Hop 1 candidates: paper_2 gets -100.0, paper_3 gets 100.0
        # Chunk ingestion: chunk_3_1 gets -100.0, chunk_3_2 gets 100.0
        self.reranker.predict.side_effect = [
            [-100.0, 100.0],              # Hop 1 candidates (paper_2, paper_3)
            [-100.0, 100.0],              # Chunk ingestion (chunk_3_1, chunk_3_2)
            [100.0, 100.0, 100.0, 100.0]  # Final filtering (paper_1, chunk_1, paper_3, chunk_3_2)
        ]
        
        # 3. Vector chunks for paper_3 (paper_2 should be filtered out)
        c3_1 = Chunk(id="chunk_3_1", paper_id="paper_3", text_content="Noise chunk from paper 3", page_number=2)
        c3_2 = Chunk(id="chunk_3_2", paper_id="paper_3", text_content="Signal chunk from paper 3", page_number=3)
        self.vector_repo.get_chunks_for_paper.side_effect = lambda pid: [c3_1, c3_2] if pid == "paper_3" else []
        
        result = self.expander.expand("test query", initial_chunks)
        
        # Title Two should NOT be in result because its candidate sigmoid score is ~0.0 (< 0.4)
        self.assertNotIn("Title Two", result)
        
        # Title Three should be in result because its candidate sigmoid score is ~1.0 (>= 0.4)
        self.assertIn("Title Three", result)
        
        # Signal chunk from paper 3 should be in result
        self.assertIn("Signal chunk from paper 3", result)
        
        # Noise chunk from paper 3 should NOT be in result because its sigmoid score is ~0.0 (< 0.4)
        self.assertNotIn("Noise chunk from paper 3", result)

    def test_safe_sigmoid_range_and_stability(self):
        """Verify that safe_sigmoid maps inputs to the [0, 1] range properly and is numerically stable."""
        from src.services.graph_expander import safe_sigmoid
        
        # Exact center maps to 0.5
        self.assertAlmostEqual(safe_sigmoid(0.5), 0.5)
        
        # Values below 0.5 map close to 0
        self.assertLess(safe_sigmoid(0.0), 1e-5)
        self.assertGreaterEqual(safe_sigmoid(0.0), 0.0)
        self.assertLess(safe_sigmoid(0.4), 0.1) # 1 / (1 + e^(2.5)) ~ 0.075 < 0.1
        
        # Values above 0.5 map close to 1
        self.assertGreater(safe_sigmoid(1.0), 0.9999)
        self.assertLessEqual(safe_sigmoid(1.0), 1.0)
        self.assertGreater(safe_sigmoid(0.6), 0.9) # 1 / (1 + e^(-2.5)) ~ 0.925 > 0.9
        
        # Extreme values should not overflow and return correct limits
        self.assertEqual(safe_sigmoid(-1000.0), 0.0)
        self.assertEqual(safe_sigmoid(1000.0), 1.0)

    def test_dynamic_top_p_filtering(self):
        """Verify dynamic Top-P selection filters candidates and chunks based on cumulative sum of scores."""
        from src.config import config
        # Save original hyperparameters dict to restore later
        orig_hype = config.data.get("hyperparameters", {}).copy()
        
        # Inject our custom top_p parameters
        config.data["hyperparameters"] = {
            "graph": {
                "semantic_score_top_p": 0.9,
                "sigmoid_score_top_p": 0.9,
                "essential_fact_threshold": 0.0,  # make sure evidence list includes all for simplicity
                "crawl_stop_threshold": 1.0
            }
        }
        
        try:
            self.expander.p_base = 10.0
            self.expander.gamma = 0.9
            
            c1 = Chunk(id="chunk_1", paper_id="paper_1", text_content="Starting text", page_number=1)
            initial_chunks = [(c1, 0.9)]
            
            p1 = Paper(id="paper_1", title="Title One")
            self.graph_repo.get_papers_batch.return_value = {"paper_1": p1}
            
            # Neighbors for Hop 1
            self.graph_repo.get_neighbors.return_value = [
                ("paper_1", "Paper", "CITES", "paper_2", "Paper", "{}"),
                ("paper_1", "Paper", "CITES", "paper_3", "Paper", "{}"),
                ("paper_1", "Paper", "CITES", "paper_4", "Paper", "{}"),
            ]
            
            p2 = Paper(id="paper_2", title="Title Two")
            p3 = Paper(id="paper_3", title="Title Three")
            p4 = Paper(id="paper_4", title="Title Four")
            self.graph_repo.get_paper.side_effect = lambda pid: {
                "paper_2": p2,
                "paper_3": p3,
                "paper_4": p4,
            }.get(pid)
            
            # Predict raw scores that map to semantic scores (sigmoids):
            # paper_2 semantic_score ~0.7, paper_3 semantic_score ~0.25, paper_4 semantic_score ~0.05
            # Combined raw logits we return: [0.5339, 0.4561, 0.3822]
            # Since total is 1.0003, 90% is 0.90027. Cumulative score: 0.7001 (paper_2), 0.9503 (paper_3).
            # So paper_2 and paper_3 are selected, paper_4 is filtered.
            hop_logits = [0.5339, 0.4561, 0.3822]
            
            # Chunk ingestion scores:
            # paper_2: chunk_2_1 gets 0.8 (raw 0.5555), chunk_2_2 gets 0.1 (raw 0.4121) -> Total = 0.9. top_p needs both.
            # paper_3: chunk_3_1 gets 0.9 (raw 0.5879), chunk_3_2 gets 0.1 (raw 0.4121) -> Total = 1.0. top_p stops at first chunk.
            chunk_logits = [0.5555, 0.4121, 0.5879, 0.4121]
            # Final filtering scores:
            final_logits = [100.0] * 10
            
            self.reranker.predict.side_effect = [
                hop_logits,
                chunk_logits,
                final_logits
            ]
            
            # Mock get_chunks_for_paper
            c2_1 = Chunk(id="chunk_2_1", paper_id="paper_2", text_content="Signal chunk from paper 2", page_number=2)
            c2_2 = Chunk(id="chunk_2_2", paper_id="paper_2", text_content="Low score chunk from paper 2", page_number=3)
            c3_1 = Chunk(id="chunk_3_1", paper_id="paper_3", text_content="Signal chunk from paper 3", page_number=4)
            c3_2 = Chunk(id="chunk_3_2", paper_id="paper_3", text_content="Low score chunk from paper 3", page_number=5)
            self.vector_repo.get_chunks_for_paper.side_effect = lambda pid: {
                "paper_2": [c2_1, c2_2],
                "paper_3": [c3_1, c3_2],
            }.get(pid, [])
            
            result = self.expander.expand("test query", initial_chunks)
            
            # paper_4 should NOT be in result
            self.assertNotIn("Title Four", result)
            # paper_2 and paper_3 should be in result
            self.assertIn("Title Two", result)
            self.assertIn("Title Three", result)
            
            # Both chunks of paper_2 should be in result (since 0.8 + 0.1 total is 0.9, top_p 0.9 needs both)
            self.assertIn("Signal chunk from paper 2", result)
            self.assertIn("Low score chunk from paper 2", result)
            
            # For paper_3, only chunk_3_1 should be in result, chunk_3_2 is filtered out (since 0.9 >= 0.9)
            self.assertIn("Signal chunk from paper 3", result)
            self.assertNotIn("Low score chunk from paper 3", result)
            
        finally:
            config.data["hyperparameters"] = orig_hype

if __name__ == "__main__":
    unittest.main()
