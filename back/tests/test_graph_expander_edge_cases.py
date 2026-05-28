import json
import unittest
from unittest.mock import MagicMock, patch

from src.models import Chunk, Paper, Concept
from src.services.graph_expander import ExperimentalGraphExpander
from src.llm_schemas import EvidenceListResponse, EvidenceItem


class TestGraphExpanderEdgeCases(unittest.TestCase):
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
            p_base=10.0,
            gamma=0.5,
            limit=5,
            top_chunks_per_paper=2
        )

    def test_empty_initial_chunks(self):
        """Verify expander handles empty initial chunks list gracefully without looping or crashing."""
        result = self.expander.expand("test query", [])
        self.assertEqual(result, "No enrichment facts gathered.")
        self.graph_repo.get_neighbors.assert_not_called()
        self.llm_engine.generate_and_validate_json.assert_not_called()

    def test_missing_papers_in_batch(self):
        """Verify behavior when papers map returned by get_papers_batch is missing some metadata."""
        chunk = Chunk(id="chunk_1", paper_id="missing_paper", text_content="Intro text", page_number=1)
        initial_chunks = [(chunk, 0.9)]
        
        # Return empty papers batch map
        self.graph_repo.get_papers_batch.return_value = {}
        self.graph_repo.get_neighbors.return_value = []
        
        # Mock final LLM evidence filtering
        mock_response = EvidenceListResponse(
            evidence_list=[EvidenceItem(id="fact_1", score=0.9, is_essential=True)]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        result = self.expander.expand("test query", initial_chunks)
        # Should fallback to paper_id since title cannot be resolved
        self.assertIn("missing_paper", result)

    def test_neighborhood_filtering_rules(self):
        """Test allowed neighbor types for each start-node type."""
        c1 = Chunk(id="chunk_1", paper_id="start_node", text_content="Text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        # Test 1: Start node label is Author
        # Allowed neighbor types for Author: Paper, UserNote, Institution
        self.graph_repo.get_papers_batch.return_value = {"start_node": Paper(id="start_node", title="Start")}
        
        # Let's mock a sequence of hops to test "Author" start node crawling rules:
        # Hop 1: Paper -> Author, Paper -> Concept, Paper -> Tag
        # Hop 2: Author -> Concept (not allowed), Author -> Institution (allowed)
        def mock_get_neighbors(node_id, max_depth=1):
            if node_id == "start_node":
                return [
                    ("start_node", "Paper", "AUTHORED", "author_1", "Author", "{}"),
                    ("start_node", "Paper", "MENTIONS_CONCEPT", "c1", "Concept", "{}"),
                    ("start_node", "Paper", "HAS_TAG", "t1", "Concept", "{}")
                ]
            elif node_id == "author_1":
                return [
                    ("author_1", "Author", "MENTIONS_CONCEPT", "concept_1", "Concept", "{}"), # forbidden
                    ("author_1", "Author", "AFFILIATED_WITH", "institution_1", "Institution", "{}") # allowed
                ]
            return []
            
        self.graph_repo.get_neighbors.side_effect = mock_get_neighbors
        
        self.graph_repo.get_author.return_value = MagicMock(name="Author One")
        self.graph_repo.get_concept.return_value = MagicMock(name="Concept One")
        self.graph_repo.get_node_properties.side_effect = lambda nid: {"name": "Institution One"} if nid == "institution_1" else {}
        
        # Hop 1 candidate scores
        self.reranker.predict.side_effect = [
            [0.9, 0.8, 0.7], # Hop 1 (author_1, c1, t1)
            [0.8], # Hop 2 (institution_1, concept_1 is filtered out so only 1 candidate)
        ]
        
        result = self.expander.expand("test query", initial_chunks)
        
        # Verify that we queried neighbors for author_1, but we did NOT query neighbors for concept_1 (since it was skipped)
        # and did not add concept_1 to candidates list.
        self.graph_repo.get_neighbors.assert_any_call("author_1", max_depth=1)
        self.assertNotIn("Concept One", result)

    def test_connection_description_construction_and_errors(self):
        """Test edge condition connection description constructing under CITES, HAS_TAG, etc., and malformed JSON properties."""
        c1 = Chunk(id="chunk_1", paper_id="p1", text_content="Text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        self.graph_repo.get_papers_batch.return_value = {"p1": Paper(id="p1", title="Title One")}
        
        # Mock various edge properties and types
        self.graph_repo.get_neighbors.return_value = [
            # CITES with valid intent/context properties
            ("p1", "Paper", "CITES", "p2", "Paper", json.dumps({"intent": "background", "context": "Prior work."})),
            # CITES with malformed properties (should fallback to no intent/context)
            ("p1", "Paper", "CITES", "p3", "Paper", "invalid_json_here{"),
            # HAS_TAG
            ("p1", "Paper", "HAS_TAG", "t1", "Concept", "{}"),
            # INTRODUCED_DATASET
            ("p1", "Paper", "INTRODUCED_DATASET", "d1", "Dataset", "{}"),
        ]
        
        self.graph_repo.get_paper.side_effect = lambda pid: Paper(id=pid, title=f"Title {pid}")
        self.graph_repo.get_concept.return_value = Concept(id="t1", name="Tag One")
        self.graph_repo.get_node_properties.side_effect = lambda nid: {"name": "Dataset One"} if nid == "d1" else {}
        
        self.reranker.predict.return_value = [0.9, 0.8, 0.7, 0.6]
        
        # LLM evidence filtering
        mock_response = EvidenceListResponse(
            evidence_list=[
                EvidenceItem(id="fact_1", score=0.9, is_essential=True), # start paper
                EvidenceItem(id="fact_2", score=0.9, is_essential=True), # p2 (Cites background)
                EvidenceItem(id="fact_3", score=0.8, is_essential=True), # p3 (Cites malformed)
                EvidenceItem(id="fact_4", score=0.7, is_essential=True), # t1 (Tag)
                EvidenceItem(id="fact_5", score=0.6, is_essential=True)  # d1 (Dataset)
            ]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        result = self.expander.expand("test query", initial_chunks)
        
        # Verify descriptions in output
        self.assertIn("Cites paper 'Title One' (Intent: background) [Context: \"Prior work.\"]", result)
        self.assertIn("Cites paper 'Title One'", result) # fallback without intent
        self.assertIn("Tag for paper/note 'Title One'", result)
        self.assertIn("Dataset introducing in paper/note 'Title One'", result)

    def test_llm_filtering_unknown_ids(self):
        """Test final context generation when LLM response contains non-existent fact IDs."""
        c1 = Chunk(id="chunk_1", paper_id="p1", text_content="Text", page_number=1)
        initial_chunks = [(c1, 0.9)]
        
        self.graph_repo.get_papers_batch.return_value = {"p1": Paper(id="p1", title="Title One")}
        self.graph_repo.get_neighbors.return_value = []
        
        # Return fact_xyz which doesn't exist
        mock_response = EvidenceListResponse(
            evidence_list=[
                EvidenceItem(id="fact_1", score=0.9, is_essential=True),
                EvidenceItem(id="fact_xyz", score=0.9, is_essential=True)
            ]
        )
        self.llm_engine.generate_and_validate_json.return_value = mock_response
        
        result = self.expander.expand("test query", initial_chunks)
        # Verify it doesn't crash and includes the valid fact_1
        self.assertIn("Title One", result)
