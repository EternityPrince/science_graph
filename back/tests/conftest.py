import os
import tempfile
import sys
import types
from unittest.mock import MagicMock
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Functional in-memory usearch.Index shim
# ---------------------------------------------------------------------------
# The native usearch compiled extension is broken in this environment
# (`_nk_capabilities` symbol not found).  We substitute a pure-Python
# implementation that covers the interface used by SQLiteVectorRepository:
#   Index(ndim, metric), add(key, vec), key in index, len(index),
#   search(vec, n) -> Matches(keys, distances), load(path), save(path)
# ---------------------------------------------------------------------------

class _Matches:
    """Lightweight container mirroring usearch.index.Matches."""
    def __init__(self, keys, distances):
        self.keys = keys
        self.distances = distances

    def __len__(self):
        return len(self.keys)


class _InMemoryIndex:
    """Pure-numpy usearch.Index replacement supporting cosine similarity."""

    def __init__(self, ndim: int = 384, metric: str = "cos"):
        self.ndim = ndim
        self._metric = metric
        self._keys: list[int] = []
        self._vecs: list[np.ndarray] = []

    # --- usearch API ---

    def add(self, key: int, vec: np.ndarray) -> None:
        key = int(key)
        if key in self:
            return
        self._keys.append(key)
        self._vecs.append(vec.astype(np.float32))

    def __contains__(self, key) -> bool:
        return int(key) in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def search(self, query: np.ndarray, n: int) -> _Matches:
        if not self._vecs:
            return _Matches([], [])

        q = query.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        scores = []
        for i, v in enumerate(self._vecs):
            v_norm = v / (np.linalg.norm(v) + 1e-10)
            cos_sim = float(np.dot(q_norm, v_norm))
            dist = 1.0 - cos_sim          # cosine distance (usearch convention)
            scores.append((dist, self._keys[i]))

        scores.sort(key=lambda x: x[0])
        top = scores[:n]
        keys = [k for _, k in top]
        dists = [d for d, _ in top]
        return _Matches(keys, dists)

    def load(self, path: str) -> None:
        """No-op: tests use ephemeral databases."""
        pass

    def save(self, path: str) -> None:
        """No-op: tests use ephemeral databases."""
        pass


# Register the shim so that `from usearch.index import Index` just works
_usearch_index_module = types.ModuleType("usearch.index")
_usearch_index_module.Index = _InMemoryIndex
_usearch_module = types.ModuleType("usearch")
sys.modules["usearch"] = _usearch_module
sys.modules["usearch.index"] = _usearch_index_module
sys.modules["usearch.compiled"] = MagicMock()

# Monkeypatch huggingface_hub.dataclasses.strict and mock missing transformers module
# to ensure compatibility between marker-pdf 0.1.3 and transformers v5
import sys
import types
if "transformers.utils.model_parallel_utils" not in sys.modules:
    mod = types.ModuleType("transformers.utils.model_parallel_utils")
    mod.get_device_map = lambda *a, **k: None
    mod.assert_device_map = lambda *a, **k: None
    sys.modules["transformers.utils.model_parallel_utils"] = mod

try:
    import huggingface_hub.dataclasses
    huggingface_hub.dataclasses.strict = lambda cls=None, *args, **kwargs: (lambda c: c) if cls is None else cls
except ImportError:
    pass

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
    engine.get_embeddings.side_effect = lambda texts, *args, **kwargs: [[0.1] * 384 for _ in texts]
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


@pytest.fixture(autouse=True)
def reset_config():
    """Resets the global config.data to default values and cleans up the environment for each test."""
    from src.config import config, DEFAULT_CONFIG
    import copy
    import os
    
    original_data = copy.deepcopy(config.data)
    config.data = copy.deepcopy(DEFAULT_CONFIG)
    
    # Backup environment variables
    original_env = dict(os.environ)
    
    # Clear test-contaminating environment variables
    os.environ.pop("SCIENCE_GRAPH_USE_CLOUD", None)
    for key in list(os.environ.keys()):
        if key.startswith("RAG_"):
            os.environ.pop(key, None)
            
    yield
    
    # Restore original data and environment
    config.data = original_data
    os.environ.clear()
    os.environ.update(original_env)


