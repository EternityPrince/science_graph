import unittest
import numpy as np
from src.models import Paper, Author, Concept, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository

class TestSQLiteRepositories(unittest.TestCase):
    def setUp(self):
        # Use an in-memory database for testing
        self.db_path = ":memory:"
        self.graph_repo = SQLiteGraphRepository(self.db_path)
        # Vector repo will share the same in-memory DB tables if we initialize it
        # However, :memory: databases are unique per connection.
        # To test them together, we can use a temporary file.
        import tempfile
        import os
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        self.graph_repo_temp = SQLiteGraphRepository(self.temp_db_path)
        self.vector_repo_temp = SQLiteVectorRepository(self.temp_db_path)

    def tearDown(self):
        import os
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_graph_nodes_and_edges(self):
        # 1. Save Paper
        paper = Paper(
            id="test_doi",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            doi="test_doi",
            abstract="The dominant sequence transduction models are based..."
        )
        self.graph_repo_temp.save_paper(paper)
        
        retrieved_paper = self.graph_repo_temp.get_paper("test_doi")
        self.assertIsNotNone(retrieved_paper)
        self.assertEqual(retrieved_paper.title, "Attention Is All You Need")
        self.assertEqual(retrieved_paper.year, 2017)
        self.assertEqual(retrieved_paper.authors, ["Ashish Vaswani", "Noam Shazeer"])

        # 2. Save Author & Link
        author = Author(id="ashish_vaswani", name="Ashish Vaswani")
        self.graph_repo_temp.save_author(author)
        self.graph_repo_temp.add_edge("ashish_vaswani", "test_doi", "AUTHORED")

        # 3. Save Concept & Link
        concept = Concept(id="transformer", name="Transformer Architecture")
        self.graph_repo_temp.save_concept(concept)
        self.graph_repo_temp.add_edge("test_doi", "transformer", "MENTIONS_CONCEPT")

        # 4. Traverse neighbors
        neighbors = self.graph_repo_temp.get_neighbors("test_doi", max_depth=1)
        # Expected connections:
        # - Ashish Vaswani authored paper
        # - Paper mentions transformer
        self.assertTrue(len(neighbors) >= 2)
        
        edge_types = [n[2] for n in neighbors]
        self.assertIn("AUTHORED", edge_types)
        self.assertIn("MENTIONS_CONCEPT", edge_types)

    def test_vector_similarity_search(self):
        # 1. Save chunks
        chunk1 = Chunk(
            id="p1#0",
            paper_id="test_doi",
            text_content="We present the Transformer, a model architecture relying solely on attention.",
            page_number=1,
            embedding=[1.0, 0.0, 0.0]
        )
        chunk2 = Chunk(
            id="p1#1",
            paper_id="test_doi",
            text_content="Recurrent models typically factor computation along the symbol positions.",
            page_number=1,
            embedding=[0.0, 1.0, 0.0]
        )
        
        # Save chunks (this will auto-create placeholder node test_doi if not exists)
        self.vector_repo_temp.save_chunks([chunk1, chunk2])
        
        # 2. Query similarity
        # Query vector close to chunk1 [1.0, 0.1, 0.0]
        query_emb = [0.9, 0.1, 0.0]
        results = self.vector_repo_temp.search_similar_chunks(query_emb, limit=1)
        
        self.assertEqual(len(results), 1)
        best_chunk, score = results[0]
        self.assertEqual(best_chunk.id, "p1#0")
        self.assertGreater(score, 0.8)
        self.assertIn("Transformer", best_chunk.text_content)
        
        # Query vector close to chunk2 [0.1, 0.9, 0.0]
        query_emb2 = [0.1, 0.9, 0.0]
        results2 = self.vector_repo_temp.search_similar_chunks(query_emb2, limit=1)
        self.assertEqual(results2[0][0].id, "p1#1")

    def test_get_papers_batch(self):
        # Save two papers
        paper1 = Paper(id="p1", title="Paper 1", authors=["Author A"], year=2021)
        paper2 = Paper(id="p2", title="Paper 2", authors=["Author B"], year=2022)
        self.graph_repo_temp.save_paper(paper1)
        self.graph_repo_temp.save_paper(paper2)
        
        # Batch retrieve them
        papers = self.graph_repo_temp.get_papers_batch(["p1", "p2", "nonexistent"])
        self.assertEqual(len(papers), 2)
        self.assertIn("p1", papers)
        self.assertIn("p2", papers)
        self.assertEqual(papers["p1"].title, "Paper 1")
        self.assertEqual(papers["p2"].title, "Paper 2")

    def test_sqlite_fts5_bm25_search(self):
        # Save chunks
        chunk1 = Chunk(
            id="c1",
            paper_id="test_doi",
            text_content="Supervised learning uses labeled training datasets to train models.",
            page_number=1,
            embedding=[0.1] * 384
        )
        chunk2 = Chunk(
            id="c2",
            paper_id="test_doi",
            text_content="Unsupervised learning algorithms analyze and cluster unlabeled data.",
            page_number=1,
            embedding=[0.2] * 384
        )
        self.vector_repo_temp.save_chunks([chunk1, chunk2])
        
        # Search for "supervised"
        results = self.vector_repo_temp.search_text_bm25("supervised", limit=5)
        self.assertTrue(len(results) >= 1)
        # The best match should be chunk1 since it contains "supervised" (c2 only has "unsupervised")
        self.assertEqual(results[0][0].id, "c1")
        
        # Search for "unsupervised"
        results2 = self.vector_repo_temp.search_text_bm25("unsupervised", limit=5)
        self.assertTrue(len(results2) >= 1)
        self.assertEqual(results2[0][0].id, "c2")

if __name__ == "__main__":
    unittest.main()
