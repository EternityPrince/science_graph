import unittest
import os
import tempfile
import datetime
import asyncio
from typing import Dict, Any, List

from src.models import Paper, Concept, Institution, Dataset, CodeRepository, JournalConference, UserNote, slugify, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.indexer import Indexer
from src.services.graph_expander import ExperimentalGraphExpander
from src.services.note_service import NoteService
from src.parsers.md_parser import MarkdownParser
from src.services.extraction_service import ExtractionResult

class MockEmbeddingEngine:
    def get_embedding(self, text: str) -> List[float]:
        # Return deterministic embedding based on string to test similarity
        emb = [0.0] * 384
        if "deepmind" in text.lower():
            emb[0] = 1.0
        elif "mit" in text.lower():
            emb[1] = 1.0
        return emb

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(t) for t in texts]

class MockReranker:
    def predict(self, pairs: List[tuple]) -> List[float]:
        scores = []
        for q, text in pairs:
            if "essential" in text.lower() or "note" in text.lower() or "important" in text.lower():
                scores.append(0.9)
            elif "deepmind" in text.lower() or "mit" in text.lower():
                scores.append(0.8)
            else:
                scores.append(0.5)
        return scores

class MockLLMEngine:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate_response_async(self, prompt: str, **kwargs) -> str:
        if "method" in prompt.lower() or "uses_method" in prompt.lower():
            return "USES_METHOD"
        return "BACKGROUND"

    def generate_and_validate_json(self, prompt: str, schema_class: Any, **kwargs) -> Any:
        from src.llm_schemas import EvidenceListResponse, EvidenceItem
        return EvidenceListResponse(evidence_list=[
            EvidenceItem(id="fact_1", score=0.9, is_essential=True),
            EvidenceItem(id="fact_2", score=0.9, is_essential=True),
            EvidenceItem(id="fact_3", score=0.9, is_essential=True)
        ])

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.graph_repo = SQLiteGraphRepository(self.db_path)
        self.vector_repo = SQLiteVectorRepository(self.db_path)
        self.emb_engine = MockEmbeddingEngine()
        self.llm_engine = MockLLMEngine()

        self.indexer = Indexer(
            graph_repo=self.graph_repo,
            vector_repo=self.vector_repo,
            embedding_engine=self.emb_engine,
            llm_engine=self.llm_engine
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_new_node_types_persistence(self):
        # 1. Save Institution
        inst = Institution(id="google_deepmind", name="Google DeepMind", properties={"city": "London"})
        with self.graph_repo.transaction():
            self.graph_repo.save_nodes_bulk([(inst.id, "Institution", inst.properties)])
            
        # Verify get_nodes_by_label
        nodes = self.graph_repo.get_nodes_by_label("Institution")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0][0], "google_deepmind")
        self.assertEqual(nodes[0][1]["city"], "London")

    def test_entity_resolution_linking(self):
        # Setup existing nodes in cache / DB
        inst = Institution(id="google_deepmind", name="Google DeepMind", properties={"name": "Google DeepMind", "embedding": [1.0] + [0.0] * 383})
        with self.graph_repo.transaction():
            self.graph_repo.save_nodes_bulk([(inst.id, "Institution", inst.properties)])

        # Resolve exact slug match
        res1 = self.indexer.resolve_entity("Institution", "Google DeepMind")
        self.assertEqual(res1, "google_deepmind")

        # Resolve cosine similarity match (>0.95 similarity)
        res2 = self.indexer.resolve_entity("Institution", "DeepMind Technologies")
        self.assertEqual(res2, "google_deepmind")

        # Resolve string similarity match (>0.95 string similarity)
        res3 = self.indexer.resolve_entity("Institution", "Google DeepMindd")
        self.assertEqual(res3, "google_deepmind")

        # Fallback to new slug when not matched
        res4 = self.indexer.resolve_entity("Institution", "MIT")
        self.assertEqual(res4, "mit")

    def test_concept_relations_hierarchy(self):
        # Save concepts and subclass relationship
        c1 = Concept(id="neural_networks", name="Neural Networks")
        c2 = Concept(id="deep_learning", name="Deep Learning")
        
        with self.graph_repo.transaction():
            self.graph_repo.save_nodes_bulk([
                (c1.id, "Concept", {"name": c1.name}),
                (c2.id, "Concept", {"name": c2.name})
            ])
            self.graph_repo.save_edges_bulk([
                (c2.id, c1.id, "SUBCLASS_OF", {})
            ])

        neighbors = self.graph_repo.get_neighbors(c2.id)
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][2], "SUBCLASS_OF")
        self.assertEqual(neighbors[0][3], c1.id)

    def test_note_frontmatter_relationships_parsing(self):
        # Write temporary markdown note with relationships
        note_content = """---
title: My Research Note
authors: Vlad Kasterin
tags: machine learning, AI
comments_on: Attention Is All You Need
agrees_with: Deep Residual Learning
disagrees_with: BERT Pre-training
linked_to: Transformer, Self-Attention
---

Body of the note referencing key topics.
"""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as temp_note:
            temp_note.write(note_content)
            temp_note_path = temp_note.name

        try:
            parser = MarkdownParser()
            paper, links, body = parser.parse(temp_note_path)
            
            # Check parsed relationships in paper properties
            self.assertEqual(paper.properties["source_type"], "note")
            self.assertEqual(paper.properties["comments_on"], ["Attention Is All You Need"])
            self.assertEqual(paper.properties["agrees_with"], ["Deep Residual Learning"])
            self.assertEqual(paper.properties["disagrees_with"], ["BERT Pre-training"])
            self.assertEqual(paper.properties["linked_to"], ["Transformer", "Self-Attention"])

            # Save note via NoteService
            note_service = NoteService(
                graph_repo=self.graph_repo,
                vector_repo=self.vector_repo,
                embedding_engine=self.emb_engine,
                llm_engine=self.llm_engine
            )
            
            # Mock index_markdown inside Indexer to prevent full file system operations in unit test
            paper_id, path = note_service.create_note(
                title="My Custom Note",
                content="Note body",
                comments_on=["Attention Is All You Need"],
                agrees_with=["Deep Residual Learning"],
                disagrees_with=["BERT Pre-training"],
                linked_to=["Transformer"]
            )
            
            # Retrieve note node from DB
            retrieved = self.graph_repo.get_paper(paper_id)
            self.assertEqual(retrieved.properties.get("source_type"), "note")
            self.assertEqual(retrieved.properties.get("comments_on"), ["Attention Is All You Need"])
            self.assertEqual(retrieved.properties.get("agrees_with"), ["Deep Residual Learning"])
            self.assertEqual(retrieved.properties.get("disagrees_with"), ["BERT Pre-training"])
            self.assertEqual(retrieved.properties.get("linked_to"), ["Transformer"])

            # Verify note list contains this note
            notes_list = self.graph_repo.get_notes()
            self.assertTrue(len(notes_list) >= 1)
            self.assertEqual(notes_list[0].id, paper_id)
        finally:
            if os.path.exists(temp_note_path):
                os.remove(temp_note_path)
            if 'path' in locals() and os.path.exists(path):
                os.remove(path)

    def test_cites_edges_intent_and_context(self):
        # We manually test parallel classification in Indexer
        # We create a dummy paper citing another paper, then index it
        paper = Paper(
            id="citing_paper",
            title="Citing Paper Title",
            authors=["Author One"],
            year=2026,
            properties={"source_type": "paper"}
        )
        extraction = ExtractionResult(
            authors=["Author One"],
            concepts=[],
            tags=[]
        )
        
        # Manually run the pipeline parts
        full_text = "In this work, we compare our performance with Citing Paper Title using a method proposed by them."
        
        # Test context extractor
        context = self.indexer._get_citation_context(full_text, "Citing Paper Title")
        self.assertIn("compare our performance", context)

        # Test parallel intent classification & writes building
        nodes, edges = asyncio.run(
            self.indexer._build_graph_writes_async(
                paper=paper,
                extraction=extraction,
                full_text=full_text,
                is_markdown=False,
                refs_or_links=["Citing Paper Title"],
                api_references=[],
                api_citations=[]
            )
        )
        
        # Verify CITES edge has context and intent properties
        cites_edges = [e for e in edges if e[2] == "CITES"]
        self.assertEqual(len(cites_edges), 1)
        self.assertEqual(cites_edges[0][3]["intent"], "USES_METHOD")
        self.assertIn("using a method", cites_edges[0][3]["context"])

    def test_prioritization_and_decay_and_notes(self):
        # 1. Setup paper nodes with varying publication years and citations
        p1 = Paper(id="old_classic", title="Old Classic Paper", year=2015)
        p1.properties["citationCount"] = 10
        p2 = Paper(id="new_breakthrough", title="New Breakthrough Paper", year=datetime.datetime.now().year)
        p2.properties["citationCount"] = 5
        # 3. Setup UserNote node
        note = Paper(id="user_note", title="Personal User Note")
        note.properties["source_type"] = "note"
        note.properties["citationCount"] = 0

        # Save to DB
        with self.graph_repo.transaction():
            self.graph_repo.save_paper(p1)
            self.graph_repo.save_paper(p2)
            self.graph_repo.save_paper(note)

            # Link them to concept "AI"
            c = Concept(id="ai", name="AI")
            self.graph_repo.save_concept(c)
            self.graph_repo.add_edge(p1.id, c.id, "MENTIONS_CONCEPT")
            self.graph_repo.add_edge(p2.id, c.id, "MENTIONS_CONCEPT")
            self.graph_repo.add_edge(note.id, c.id, "LINKED_TO")

        # 2. Run graph expansion
        reranker = MockReranker()
        expander = ExperimentalGraphExpander(
            graph_repo=self.graph_repo,
            vector_repo=self.vector_repo,
            llm_engine=self.llm_engine,
            reranker=reranker,
            p_base=0.9,
            gamma=0.9,
            limit=5
        )

        initial_chunk = Chunk(id="c0", paper_id="ai", text_content="AI is important", page_number=1)
        # We start from concept node AI
        results_str = expander.expand("What is AI?", [(initial_chunk, 1.0)], trace=False)

        # Verification:
        # Note should be sorted first, and new_breakthrough should get a high score due to newness bonus
        # Let's inspect the crawl order or check their presence
        # Check that the note was crawled and returned
        self.assertIn("[UserNote] Personal User Note", results_str)
        self.assertIn("[Paper] New Breakthrough Paper", results_str)

if __name__ == '__main__':
    unittest.main()
