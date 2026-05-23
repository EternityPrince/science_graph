import os
import tempfile
import unittest
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.models import Paper, Author, Concept, Chunk

class TestRepositoryDelete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.graph_repo = SQLiteGraphRepository(self.tmp.name)
        self.vector_repo = SQLiteVectorRepository(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_delete_edges_by_target(self):
        """Test delete_edges_by_target deletes target edges matching specific types."""
        # Save a paper and two authors
        p1 = Paper(id="p1", title="Paper 1")
        self.graph_repo.save_paper(p1)
        
        a1 = Author(id="a1", name="Author 1")
        self.graph_repo.save_author(a1)
        
        a2 = Author(id="a2", name="Author 2")
        self.graph_repo.save_author(a2)
        
        self.graph_repo.add_edge("a1", "p1", "AUTHORED")
        self.graph_repo.add_edge("a2", "p1", "AUTHORED")
        self.graph_repo.add_edge("p1", "a1", "OTHER_EDGE")  # target is a1, source is p1
        
        # Verify edge count before deletion
        edges_before = self.graph_repo.get_all_edges()
        self.assertEqual(len(edges_before), 3)

        # Delete AUTHORED edges pointing to p1
        self.graph_repo.delete_edges_by_target("p1", ["AUTHORED"])
        
        edges_after = self.graph_repo.get_all_edges()
        self.assertEqual(len(edges_after), 1)
        self.assertEqual(edges_after[0][0], "p1")
        self.assertEqual(edges_after[0][1], "a1")
        self.assertEqual(edges_after[0][2], "OTHER_EDGE")

    def test_delete_edges_by_source(self):
        """Test delete_edges_by_source deletes source edges matching specific types."""
        p1 = Paper(id="p1", title="Paper 1")
        self.graph_repo.save_paper(p1)
        
        c1 = Concept(id="c1", name="Concept 1")
        self.graph_repo.save_concept(c1)
        
        c2 = Concept(id="c2", name="Concept 2")
        self.graph_repo.save_concept(c2)
        
        self.graph_repo.add_edge("p1", "c1", "MENTIONS_CONCEPT")
        self.graph_repo.add_edge("p1", "c2", "HAS_TAG")
        self.graph_repo.add_edge("c1", "p1", "CITED_BY")
        
        edges_before = self.graph_repo.get_all_edges()
        self.assertEqual(len(edges_before), 3)

        # Delete only MENTIONS_CONCEPT from p1
        self.graph_repo.delete_edges_by_source("p1", ["MENTIONS_CONCEPT"])
        
        edges_after = self.graph_repo.get_all_edges()
        self.assertEqual(len(edges_after), 2)
        
        edge_types = [e[2] for e in edges_after]
        self.assertIn("HAS_TAG", edge_types)
        self.assertIn("CITED_BY", edge_types)
        self.assertNotIn("MENTIONS_CONCEPT", edge_types)

    def test_delete_node_cascade_edges(self):
        """Test delete_node cascades deletion to all connected edges."""
        p1 = Paper(id="p1", title="Paper 1")
        self.graph_repo.save_paper(p1)
        
        a1 = Author(id="a1", name="Author 1")
        self.graph_repo.save_author(a1)
        
        self.graph_repo.add_edge("a1", "p1", "AUTHORED")
        self.graph_repo.add_edge("p1", "a1", "MENTIONS")
        
        # Verify node and edges exist
        self.assertIsNotNone(self.graph_repo.get_paper("p1"))
        self.assertEqual(len(self.graph_repo.get_all_edges()), 2)
        
        # Delete paper node
        self.graph_repo.delete_node("p1")
        
        # Node and edges should be gone
        self.assertIsNone(self.graph_repo.get_paper("p1"))
        self.assertEqual(len(self.graph_repo.get_all_edges()), 0)

    def test_delete_node_cascade_chunks(self):
        """Test delete_node cascades deletion to SQLiteVectorRepository chunks."""
        # Insert a paper and save chunks referencing it
        p1 = Paper(id="p1", title="Paper 1")
        self.graph_repo.save_paper(p1)
        
        chunk1 = Chunk(id="p1#0", paper_id="p1", text_content="Content chunk 1", page_number=1, embedding=[0.1, 0.2, 0.3])
        chunk2 = Chunk(id="p1#1", paper_id="p1", text_content="Content chunk 2", page_number=2, embedding=[0.1, 0.2, 0.3])
        
        self.vector_repo.save_chunks([chunk1, chunk2])
        
        # Verify chunks exist
        self.assertEqual(len(self.vector_repo.get_chunks_for_paper("p1")), 2)
        
        # Delete the paper node
        self.graph_repo.delete_node("p1")
        
        # Chunks should be deleted since there is a foreign key on paper_id to nodes(id)
        # Note: chunks table has a FOREIGN KEY(paper_id) REFERENCES nodes(id) ON DELETE CASCADE
        self.assertEqual(len(self.vector_repo.get_chunks_for_paper("p1")), 0)

    def test_delete_edges_empty_list(self):
        """Test calling delete edges with empty list does nothing and doesn't crash."""
        p1 = Paper(id="p1", title="Paper 1")
        self.graph_repo.save_paper(p1)
        a1 = Author(id="a1", name="Author 1")
        self.graph_repo.save_author(a1)
        self.graph_repo.add_edge("a1", "p1", "AUTHORED")
        
        self.graph_repo.delete_edges_by_target("p1", [])
        self.graph_repo.delete_edges_by_source("a1", [])
        
        self.assertEqual(len(self.graph_repo.get_all_edges()), 1)

    def test_delete_nonexistent_node(self):
        """Test deleting a node that doesn't exist doesn't crash."""
        try:
            self.graph_repo.delete_node("nonexistent_id")
        except Exception as e:
            self.fail(f"delete_node raised an exception on nonexistent node ID: {e}")
