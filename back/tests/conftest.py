import os
import tempfile
from unittest.mock import MagicMock
import pytest

from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.indexer import Indexer

@pytest.fixture
def temp_db():
    """Generates a temporary SQLite database file path and cleans up both db and usearch index files on teardown."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield db_path
    
    # Clean up SQLite DB
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
            
    # Clean up USearch index file
    usearch_path = db_path.replace(".db", ".usearch")
    if os.path.exists(usearch_path):
        try:
            os.remove(usearch_path)
        except OSError:
            pass

@pytest.fixture
def graph_repo(temp_db):
    """Returns a SQLiteGraphRepository instance connected to the temporary test database."""
    return SQLiteGraphRepository(temp_db)

@pytest.fixture
def vector_repo(temp_db):
    """Returns a SQLiteVectorRepository instance connected to the temporary test database."""
    return SQLiteVectorRepository(temp_db)

@pytest.fixture
def mock_embedding_engine():
    """Returns a mocked EmbeddingEngine to prevent real model loads during testing."""
    engine = MagicMock()
    # Sensible default mocks that return 384-dimensional embeddings (the default ndim in SQLiteVectorRepository)
    engine.get_embedding.return_value = [0.1] * 384
    engine.get_embeddings.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
    return engine

@pytest.fixture
def mock_llm_engine():
    """Returns a mocked LLM engine."""
    engine = MagicMock()
    engine.generate_response.return_value = "Mock LLM Summary Response"
    engine.extract_concepts_and_metadata.return_value = None
    return engine

@pytest.fixture
def indexer(graph_repo, vector_repo, mock_embedding_engine, mock_llm_engine):
    """Returns an Indexer instance pre-configured with test repositories and mocked engines."""
    return Indexer(
        graph_repo=graph_repo,
        vector_repo=vector_repo,
        embedding_engine=mock_embedding_engine,
        llm_engine=mock_llm_engine,
    )
