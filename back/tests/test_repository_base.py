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

    def get_papers_mentioning_concepts(self, concept_ids): return []
    def get_concepts_for_papers(self, paper_ids): return []
    def get_concept_document_frequencies(self, concept_ids): return {}
    def get_total_paper_count(self): return 0
    def get_citation_neighbors(self, paper_ids): return []
    def search_chunks_within_papers(self, query_embedding, paper_ids, limit_per_paper=3): return []
    def get_neighbor_papers(self, seed_paper_ids, order=2, allowed_edge_types=None): return []
    def resolve_graph_nodes_to_local_papers(self, node_ids): return []
    def get_chunks_count_by_paper_ids(self, paper_ids): return {}
    def filter_papers_with_chunks(self, paper_ids): return []
    def count_total_local_papers(self): return 0
    def get_concept_idf(self, concept_ids): return {}

    # --- Bibliographic projection abstract methods ---
    def save_reference_corpus_stats(self, total_papers, doc_frequencies): pass
    def get_reference_corpus_stats(self): return (0, {})
    def save_paper_reference_vectors(self, vectors): pass
    def get_paper_reference_vectors(self, paper_ids=None): return []
    def save_chunk_reference_mentions(self, mentions): pass
    def get_chunk_reference_mentions(self, paper_id=None): return []
    def delete_derived_edges_for_papers(self, paper_ids, edge_types): pass
    def delete_derived_edges_by_types(self, edge_types): pass
    def delete_chunk_reference_mentions_for_paper(self, paper_id): pass
    def delete_chunk_nodes_for_paper(self, paper_id): pass
    def get_non_placeholder_papers(self): return []
    def find_paper_by_arxiv(self, arxiv_id): pass
    def find_paper_by_url(self, url): pass
    def remap_external_work_to_local_paper(self, external_work_id, local_paper_id): pass

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

    def test_abstract_methods_coverage(self):
        repo = DummyGraphRepository()
        GraphRepository.save_paper(repo, None)
        GraphRepository.get_paper(repo, None)
        GraphRepository.get_papers_batch(repo, None)
        GraphRepository.find_paper_by_title(repo, None)
        GraphRepository.find_paper_by_doi(repo, None)
        GraphRepository.find_paper_by_content_hash(repo, None)
        GraphRepository.save_author(repo, None)
        GraphRepository.get_author(repo, None)
        GraphRepository.save_concept(repo, None)
        GraphRepository.get_concept(repo, None)
        GraphRepository.add_edge(repo, None, None, None)
        GraphRepository.get_neighbors(repo, None)
        GraphRepository.get_stats(repo)
        GraphRepository.cleanup_orphaned_concepts(repo)
        GraphRepository.get_all_nodes(repo)
        GraphRepository.get_all_edges(repo)
        GraphRepository.get_node_by_id(repo, None)
        GraphRepository.get_papers_by_author(repo, None)
        GraphRepository.get_papers_by_entity(repo, None, None)
        GraphRepository.get_distinct_targets(repo, None, None)
        GraphRepository.search_papers_by_title(repo, None)
        GraphRepository.get_notes(repo)
        GraphRepository.delete_edges_by_target(repo, None, None)
        GraphRepository.delete_edges_by_source(repo, None, None)
        GraphRepository.delete_node(repo, None)
        GraphRepository.get_paper_ids(repo)
        GraphRepository.get_non_placeholder_paper_ids(repo)
        GraphRepository.get_paper_source_types(repo)
        GraphRepository.get_browse_rows(repo, None, None, None, None)
        GraphRepository.get_browse_count(repo, None, None)
        GraphRepository.update_node_properties(repo, None, None)
        GraphRepository.get_concept_aliases(repo)
        GraphRepository.get_nodes_by_label(repo, None)
        GraphRepository.get_node_properties(repo, None)

        v_repo = DummyVectorRepository()
        VectorRepository.save_chunks(v_repo, None)
        VectorRepository.search_similar_chunks(v_repo, None)
        VectorRepository.search_text_fts5(v_repo, None)
        VectorRepository.get_chunks_for_paper(v_repo, None)
        VectorRepository.get_all_chunks(v_repo)
