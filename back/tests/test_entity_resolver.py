import pytest
from unittest.mock import MagicMock
import threading
from src.services.entity_resolver import EntityResolver
from src.repository.base import GraphRepository
from src.vector_search import EmbeddingEngine

def test_entity_resolver_slug_match():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    resolved_id = resolver.resolve_entity("Concept", "Machine Learning")
    assert resolved_id == "machine_learning"

def test_entity_resolver_alias_match():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    # Concept specific alias mapping
    graph_repo.get_concept_aliases.return_value = {
        "ml": "Machine Learning"
    }
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    resolved_id = resolver.resolve_entity("Concept", "ML")
    assert resolved_id == "machine_learning"

def test_entity_resolver_vector_similarity_match():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    # We will return some nodes with embeddings
    # node embedding: [1.0, 0.0]
    # query embedding: [0.99, 0.1]
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning", "embedding": [1.0, 0.0]})
    ]
    emb_engine.get_embedding.return_value = [0.99, 0.1]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    # Target entity is "Machine Learning Refined", which should match "Machine Learning" by embedding
    resolved_id = resolver.resolve_entity("Concept", "Machine Learning Refined")
    assert resolved_id == "machine_learning"
    emb_engine.get_embedding.assert_called_once_with("Machine Learning Refined")

def test_entity_resolver_string_similarity_fallback():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning"})
    ]
    # Vector search fails or returns nothing
    emb_engine.get_embedding.side_effect = Exception("No embedding model")
    
    resolver = EntityResolver(graph_repo, emb_engine)
    # difflib.SequenceMatcher ratio for "Machine Learning" and "Machine Learning!" is > 0.95
    resolved_id = resolver.resolve_entity("Concept", "Machine Learning!")
    assert resolved_id == "machine_learning"

def test_entity_resolver_caching_and_invalidation():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {"ml": "Machine Learning"}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    
    # 1. Test alias caching (get_concept_aliases)
    r1 = resolver.resolve_entity("Concept", "ML")
    r2 = resolver.resolve_entity("Concept", "ML")
    assert r1 == "machine_learning"
    assert r2 == "machine_learning"
    assert graph_repo.get_concept_aliases.call_count == 1
    assert graph_repo.get_nodes_by_label.call_count == 0
    
    # 2. Test node list caching (get_nodes_by_label)
    # "Deep Learning" is not an alias, so it will search existing nodes
    r3 = resolver.resolve_entity("Concept", "Deep Learning")
    r4 = resolver.resolve_entity("Concept", "Deep Learning")
    assert r3 == "deep_learning"
    assert r4 == "deep_learning"
    assert graph_repo.get_nodes_by_label.call_count == 1
    
    # 3. Invalidate cache
    resolver.invalidate_concept_cache()
    
    # Call again, should call repository again
    r5 = resolver.resolve_entity("Concept", "ML")
    r6 = resolver.resolve_entity("Concept", "Deep Learning")
    assert r5 == "machine_learning"
    assert r6 == "deep_learning"
    assert graph_repo.get_concept_aliases.call_count == 2
    assert graph_repo.get_nodes_by_label.call_count == 2

def test_entity_resolver_add_to_cache():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    # Initial resolve
    assert resolver.resolve_entity("Concept", "Deep Learning") == "deep_learning"
    
    # Add Deep Learning to cache
    resolver.add_resolved_entity_to_cache("Concept", "deep_learning", "Deep Learning", [0.1, 0.2])
    
    # Try resolving again - now it should find it in cache
    # (Since it's in the cache, and we resolve "Deep Learning", it should match slug or exact name in the cache)
    assert resolver.resolve_entity("Concept", "Deep Learning") == "deep_learning"

def test_entity_resolver_embedding_dim_mismatch():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning", "embedding": [1.0, 0.0]})
    ]
    # Query embedding has 3 elements, but node embedding has 2 elements (dimension mismatch)
    emb_engine.get_embedding.return_value = [0.9, 0.1, 0.2]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    # The dimension mismatch should be caught safely in ValueError block, falling back to string similarity
    # difflib.SequenceMatcher ratio for "Machine Learning" and "Machine Learning!" is > 0.95, which should match
    resolved_id = resolver.resolve_entity("Concept", "Machine Learning!")
    assert resolved_id == "machine_learning"

def test_entity_resolver_thread_safety():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine_learning", {"name": "Machine Learning"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    
    num_threads = 10
    iterations = 50
    exceptions = []
    
    def worker(tid):
        try:
            for i in range(iterations):
                if i % 3 == 0:
                    resolver.resolve_entity("Concept", f"Machine Learning {i}")
                elif i % 3 == 1:
                    resolver.add_resolved_entity_to_cache("Concept", f"concept_{tid}_{i}", f"Concept {tid} {i}")
                else:
                    resolver.invalidate_concept_cache()
        except Exception as e:
            exceptions.append(e)
            
    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
        
    assert not exceptions, f"Concurrent operations raised exceptions: {exceptions}"


def test_entity_resolver_double_checked_locking_race():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    # We want to simulate the scenario where:
    # 1. Thread reads _aliases_cache -> not None.
    # 2. Invalidation happens.
    # 3. Thread reads _aliases_cache again to assign to aliases_map -> None.
    # In our fixed code, we only read self._aliases_cache ONCE (at the start of the method).
    # In the unfixed code, it read it TWICE (once for checking None, and once to assign aliases_map).
    
    reads = []
    class RaceResolver(EntityResolver):
        @property
        def _aliases_cache(self):
            reads.append(True)
            if len(reads) == 1:
                return {"ml": "Machine Learning"}
            return None
            
        @_aliases_cache.setter
        def _aliases_cache(self, value):
            pass
            
    resolver = RaceResolver(graph_repo, emb_engine)
    # This should execute safely without raising TypeError
    resolved_id = resolver.resolve_entity("Concept", "ML")
    assert resolved_id == "machine_learning"


def test_entity_resolver_no_caching_on_exceptions():
    """Verify that repository exceptions are not cached and cause retries."""
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    # 1. Alias query raises exception
    graph_repo.get_concept_aliases.side_effect = Exception("DB error")
    resolver = EntityResolver(graph_repo, emb_engine)
    
    with pytest.raises(Exception, match="DB error"):
        resolver.resolve_entity("Concept", "ML")
        
    # Second try should call the repo again
    with pytest.raises(Exception, match="DB error"):
        resolver.resolve_entity("Concept", "ML")
        
    assert graph_repo.get_concept_aliases.call_count == 2

    # 2. Node query raises exception
    graph_repo.get_concept_aliases.side_effect = None
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.side_effect = Exception("DB node error")
    
    with pytest.raises(Exception, match="DB node error"):
        resolver.resolve_entity("Concept", "Machine Learning")
        
    with pytest.raises(Exception, match="DB node error"):
        resolver.resolve_entity("Concept", "Machine Learning")
        
    assert graph_repo.get_nodes_by_label.call_count == 2


def test_entity_resolver_best_match_selection():
    """Verify that best match above 0.95 is selected among multiple matches."""
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_concept_aliases.return_value = {}
    # Multiple candidates
    graph_repo.get_nodes_by_label.return_value = [
        ("ml_96", {"name": "ML 96", "embedding": [1.0, 0.0]}),
        ("ml_99", {"name": "ML 99", "embedding": [0.0, 1.0]}),
    ]
    # Query: close to ml_99 (dot product with ml_99 is 0.99, ml_96 is 0.1)
    emb_engine.get_embedding.return_value = [0.1, 0.99]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    res = resolver.resolve_entity("Concept", "Machine Learning")
    assert res == "ml_99"

    # For string similarity fallback:
    # We clear cache so it re-reads
    resolver.invalidate_cache()
    graph_repo.get_nodes_by_label.return_value = [
        ("ml_alpha", {"name": "Introduction to Machine Learning Alpha"}),
        ("ml_beta", {"name": "Introduction to Machine Learning Beta"}),
    ]
    emb_engine.get_embedding.side_effect = Exception("No embedding")
    res_str = resolver.resolve_entity("Concept", "Introduction to Machine Learning Bet")
    assert res_str == "ml_beta"



def test_entity_resolver_non_concept_labels():
    """Verify cache invalidation and queries for non-concept labels."""
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    graph_repo.get_nodes_by_label.return_value = [
        ("john_doe", {"name": "John Doe"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    # Query for "Author"
    res1 = resolver.resolve_entity("Author", "John Doe")
    res2 = resolver.resolve_entity("Author", "John Doe")
    assert res1 == "john_doe"
    assert res2 == "john_doe"
    # Should only call repo once
    assert graph_repo.get_nodes_by_label.call_count == 1
    
    # Generic cache invalidation for specific label
    resolver.invalidate_cache("Author")
    res3 = resolver.resolve_entity("Author", "John Doe")
    assert res3 == "john_doe"
    assert graph_repo.get_nodes_by_label.call_count == 2

    # Clear all caches
    resolver.invalidate_cache(None)
    res4 = resolver.resolve_entity("Author", "John Doe")
    assert res4 == "john_doe"
    assert graph_repo.get_nodes_by_label.call_count == 3


def test_entity_resolver_invalid_empty_names():
    """Verify behavior with invalid, empty, or None names."""
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    resolver = EntityResolver(graph_repo, emb_engine)
    assert resolver.resolve_entity("Concept", "") == ""
    assert resolver.resolve_entity("Concept", "   ") == ""
    assert resolver.resolve_entity("Concept", None) == ""


def test_entity_resolver_invalid_embeddings_and_edge_cases():
    """Verify behavior with different invalid embedding formats (zero vector, wrong type, ValueError)."""
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    # 1. Zero vector
    item1 = EntityResolver(graph_repo, emb_engine)._prepare_cache_item("id1", "Name 1", [0.0, 0.0])
    # emb_arr should be None since norm is 0
    assert item1[3] is None
    assert item1[4] is None

    # 2. Wrong type (e.g. dict)
    item2 = EntityResolver(graph_repo, emb_engine)._prepare_cache_item("id2", "Name 2", {"invalid": "type"})
    assert item2[3] is None

    # 3. ValueError in array creation
    item3 = EntityResolver(graph_repo, emb_engine)._prepare_cache_item("id3", "Name 3", ["string_inside_list"])
    assert item3[3] is None

