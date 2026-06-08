import pytest
from unittest.mock import MagicMock
import numpy as np
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
