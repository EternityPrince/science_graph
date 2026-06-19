"""
LLM Engine Factory.
"""

import os
from src.config import config
from src.llm_engine.base import BaseLLMEngine

def LLMEngine(use_cloud: bool = False, *args, **kwargs) -> BaseLLMEngine:
    """Factory for returning the correct LLM Engine based on config/parameters."""
    import src.llm_engine
    is_cloud = use_cloud or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1" or config.llm_provider == "openai"
    if is_cloud:
        if src.llm_engine._cloud_engine_singleton is None:
            from src.llm_engine import OpenAILLMEngine
            src.llm_engine._cloud_engine_singleton = OpenAILLMEngine()
        return src.llm_engine._cloud_engine_singleton
    else:
        if src.llm_engine._local_engine_singleton is None:
            provider = config.llm_provider
            if provider == "gguf":
                from src.llm_engine import GgufLLMEngine
                src.llm_engine._local_engine_singleton = GgufLLMEngine(*args, **kwargs)
            else:
                from src.llm_engine import MlxLLMEngine
                src.llm_engine._local_engine_singleton = MlxLLMEngine(*args, **kwargs)
        return src.llm_engine._local_engine_singleton
