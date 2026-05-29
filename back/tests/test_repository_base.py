import unittest
from unittest.mock import MagicMock
from src.repository.base import GraphRepository, VectorRepository
from src.models import Chunk


class DummyGraphRepository(GraphRepository):
    def __init__(self):
        self.added_edges = []
        
    def save_paper(self, paper): pass
    def get_paper(self, paper_id): pass
    def get_papers_batch(self, paper_ids): pass
    def find_paper_by_title(self, title): pass
    def find_paper_by_doi(self, doi): pass
    def find_paper_by_content_hash(self, content_hash): pass
    def save_author(self, author): pass
    def get_author(self, author_id): pass
    def save_concept(self, concept): pass
    def get_concept(self, concept_id): pass
    def get_neighbors(self, node_id, max_depth=1): pass
    def get_stats(self): pass
    def cleanup_orphaned_concepts(self): pass
    def get_all_nodes(self): pass
    def get_all_edges(self): pass
    def get_node_by_id(self, node_id): pass
    def get_papers_by_author(self, author_id): pass
    def get_papers_by_entity(self, entity_id, edge_type): pass
    def get_distinct_targets(self, source_ids, edge_type): pass
    def search_papers_by_title(self, query, limit=20): pass
    def get_notes(self): pass
    def delete_edges_by_target(self, target_id, edge_types): pass
    def delete_edges_by_source(self, source_id, edge_types): pass
    def delete_node(self, node_id): pass
    def get_paper_ids(self): pass
    def get_non_placeholder_paper_ids(self): pass
    def get_paper_source_types(self): pass
    def get_browse_rows(self, table, page, limit, search_query=None): pass
    def get_browse_count(self, table, search_query=None): pass
    def update_node_properties(self, node_id, properties): pass
    def get_concept_aliases(self): pass
    def get_nodes_by_label(self, label): pass
    def get_node_properties(self, node_id): pass

    def add_edge(self, source_id, target_id, edge_type, properties=None):
        self.added_edges.append((source_id, target_id, edge_type, properties))


class DummyVectorRepository(VectorRepository):
    def __init__(self):
        self.saved_chunks = []
        
    def save_chunks(self, chunks):
        self.saved_chunks.extend(chunks)
        
    def search_similar_chunks(self, query_embedding, limit=5): pass
    def search_text_fts5(self, query, limit=10): pass
    def get_chunks_for_paper(self, paper_id): pass
    def get_all_chunks(self): pass


class TestRepositoryBase(unittest.TestCase):
    def test_graph_repository_bulk_and_transaction(self):
        repo = DummyGraphRepository()
        
        # Test save_nodes_bulk does nothing
        repo.save_nodes_bulk([("n1", "Paper", {})])
        
        # Test save_edges_bulk iterates and calls add_edge
        edges = [
            ("p1", "p2", "CITES", {"weight": 1.0}),
            ("p2", "p3", "CITES", None),
        ]
        repo.save_edges_bulk(edges)
        self.assertEqual(repo.added_edges, edges)
        
        # Test transaction yield
        with repo.transaction():
            pass

    def test_graph_repository_get_neighbors_batch_fallback(self):
        repo = DummyGraphRepository()
        repo.get_neighbors = MagicMock(return_value=[("src", "Paper", "CITES", "tgt", "Paper", "{}")])
        res = repo.get_neighbors_batch(["node1", "node2"])
        self.assertEqual(len(res), 2)
        self.assertEqual(repo.get_neighbors.call_count, 2)
        repo.get_neighbors.assert_any_call("node1", max_depth=1)
        repo.get_neighbors.assert_any_call("node2", max_depth=1)

    def test_vector_repository_save_chunks_bulk(self):
        repo = DummyVectorRepository()
        chunk1 = MagicMock(spec=Chunk)
        chunk2 = MagicMock(spec=Chunk)
        
        repo.save_chunks_bulk([chunk1, chunk2])
        self.assertEqual(repo.saved_chunks, [chunk1, chunk2])
