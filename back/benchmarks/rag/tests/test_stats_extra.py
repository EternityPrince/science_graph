import pytest
from unittest.mock import MagicMock
from core.stats import BenchmarkStatsCollector

class DummyReranker:
    def predict(self, *args, **kwargs):
        return [0.9]

class DummyLLMEngine:
    def __init__(self):
        self.should_fail_count = False

    def generate_response(self, prompt):
        return "response"

    def count_tokens(self, text):
        if self.should_fail_count:
            raise Exception("token count error")
        return len(text)

class DummyEmbeddingEngine:
    def get_embedding(self, text):
        return [0.1]

class DummyVectorRepo:
    def search_similar_chunks(self, *args, **kwargs):
        return []
    def search_text_fts5(self, *args, **kwargs):
        return []

class DummyGraphRepo:
    def get_neighbors(self, *args, **kwargs):
        return []
    def get_paper(self, *args, **kwargs):
        return {}
    def get_author(self, *args, **kwargs):
        return {}
    def get_concept(self, *args, **kwargs):
        return {}
    def get_papers_batch(self, *args, **kwargs):
        return []

class DummyExpander:
    def expand(self, *args, **kwargs):
        return []

class DummyRAGService:
    def __init__(self):
        self.emb_engine = DummyEmbeddingEngine()
        self.vector_repo = DummyVectorRepo()
        self.graph_repo = DummyGraphRepo()
        self.llm_engine = DummyLLMEngine()
        self.expander = DummyExpander()
        self._reranker = None # Start with None to test lazy loading
        
    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = DummyReranker()
        return self._reranker
        
    def _validate_and_repair_citations(self, *args, **kwargs):
        return "repaired"

def test_stats_collector():
    rag_service = DummyRAGService()
    collector = BenchmarkStatsCollector(rag_service)
    
    # Test wrap_method on None/missing method (line 29)
    collector.wrap_method(None, "non_existent", "dense_retrieval")
    collector.wrap_method(rag_service, "non_existent", "dense_retrieval")
    
    # 1. Start interception
    collector.start()
    
    # 2. Trigger methods
    rag_service.emb_engine.get_embedding("hello")
    rag_service.vector_repo.search_similar_chunks()
    rag_service.vector_repo.search_text_fts5()
    rag_service.graph_repo.get_neighbors()
    rag_service.graph_repo.get_paper()
    
    # Lazy load reranker (runs wrapped_get_reranker code block)
    reranker = rag_service._get_reranker()
    reranker.predict()
    
    # Test LLM generation normal
    rag_service.llm_engine.generate_response("generate this")
    assert collector.prompt_tokens == len("generate this")
    
    # Test LLM generation error (lines 40-41)
    rag_service.llm_engine.should_fail_count = True
    rag_service.llm_engine.generate_response("another prompt")
    assert collector.prompt_tokens == len("another prompt") // 4
    
    rag_service._validate_and_repair_citations()
    rag_service.expander.expand()
    
    # 3. Verify metrics
    metrics = collector.get_metrics()
    assert metrics["total_io_calls"] == 10
    assert metrics["components"]["embedding"]["calls"] == 1
    assert metrics["components"]["dense_retrieval"]["calls"] == 1
    assert metrics["components"]["lexical_retrieval"]["calls"] == 1
    assert metrics["components"]["graph_neighbors"]["calls"] == 1
    assert metrics["components"]["db_lookups"]["calls"] == 1
    assert metrics["components"]["reranking"]["calls"] == 1
    assert metrics["components"]["llm_generation"]["calls"] == 2
    assert metrics["components"]["citation_repair"]["calls"] == 1
    assert metrics["components"]["graph_expansion"]["calls"] == 1
    
    # 5. Stop
    collector.stop()

def test_stats_collector_reranker_preloaded():
    rag_service = DummyRAGService()
    rag_service._reranker = DummyReranker() # Preloaded!
    collector = BenchmarkStatsCollector(rag_service)
    collector.start()
    rag_service._reranker.predict()
    metrics = collector.get_metrics()
    assert metrics["components"]["reranking"]["calls"] == 1
    collector.stop()
