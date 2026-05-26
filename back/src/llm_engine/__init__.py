"""
LLM Engine Package.
"""

from src.llm_engine.base import (
    BaseLLMEngine,
    strip_thinking_tokens,
    ResilientParser,
    retry_with_temp_decay,
    retry_with_temp_decay_async,
)
from src.llm_engine.mlx_impl import MlxLLMEngine, ConstrainedLogitsProcessor, build_mlx_tokenizer_data
from src.llm_engine.openai_impl import OpenAILLMEngine, AsyncRateLimiter
from src.llm_engine.factory import LLMEngine

_local_engine_singleton = None
_cloud_engine_singleton = None

__all__ = [
    "LLMEngine",
    "BaseLLMEngine",
    "MlxLLMEngine",
    "OpenAILLMEngine",
    "strip_thinking_tokens",
    "ResilientParser",
    "retry_with_temp_decay",
    "retry_with_temp_decay_async",
    "ConstrainedLogitsProcessor",
    "build_mlx_tokenizer_data",
    "AsyncRateLimiter",
]
