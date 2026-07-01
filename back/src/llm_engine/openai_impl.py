"""
OpenAI LLM Engine implementation for cloud model provider.
"""

import asyncio
import time
from typing import Optional, Type

from pydantic import BaseModel
from src.config import config
from src import console as con
from src.llm_engine.base import BaseLLMEngine, strip_thinking_tokens


class AsyncRateLimiter:
    """Enforces a minimum interval between requests to avoid overloading providers."""
    def __init__(self, delay: float):
        self.delay = delay
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.delay <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.monotonic()


class OpenAILLMEngine(BaseLLMEngine):
    def __init__(self, model_name: str = None):
        import openai
        api_key = config.llm_cloud_api_key
        if not api_key and config.llm_provider == "openai-compatible":
            api_key = "dummy"

        base_url = config.llm_cloud_base_url
        self.model_name = model_name or config.llm_cloud_model_name

        if not api_key:
            con.error("API key is not configured for OpenAI/OpenRouter.")
            raise ValueError("Missing API key for OpenAI provider")

        # Determine MTP details
        mtp_requested = config.llm_enable_mtp
        mtp_file_found = config.llm_mtp_file_found
        mtp_effective = config.llm_effective_mtp_mode
        model_path = config.llm_model_path
        actual_base_url = base_url or "https://api.openai.com/v1"

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MTP requested: {str(mtp_requested).lower()}")
        logger.info(f"MTP file found: {str(mtp_file_found).lower()}")
        logger.info(f"MTP effective mode: {'enabled' if mtp_effective else 'disabled'}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Backend base URL: {actual_base_url}")
        logger.info("Request mode: one-shot")
        logger.info("Token limits source: existing project config")
        logger.info("Token estimation source: existing tiktoken mechanism")

        if mtp_requested and not mtp_file_found:
            logger.warning("MTP file (mtp.safetensors) is missing in model path. MTP will be disabled.")
            con.warning("MTP file (mtp.safetensors) is missing in model path. MTP will be disabled.")

        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        self.client = openai.OpenAI(**client_args)
        self.rate_limiter = AsyncRateLimiter(config.llm_request_delay)

        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self.tokenizer = None

        con.success(f"OpenAI API LLM ready: [bold]{self.model_name}[/bold]")

    def _truncate_to_context(self, text: str, max_input_tokens: int) -> str:
        if self.tokenizer is None:
            return text[:max_input_tokens * 4]
        try:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) <= max_input_tokens:
                return text
            return self.tokenizer.decode(token_ids[:max_input_tokens])
        except Exception:
            return text[:max_input_tokens * 4]

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            if task == "extraction":
                resolved_max_tokens = config.llm_extraction_output_limit
            elif task == "clustering":
                resolved_max_tokens = config.llm_clustering_output_limit
            elif task == "synthesis":
                resolved_max_tokens = config.llm_synthesis_output_limit
            
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp
        model_to_use = model if model else self.model_name

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=resolved_max_tokens,
            temperature=temp,
        )
        return strip_thinking_tokens(response.choices[0].message.content)

    async def generate_response_async(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            if task == "extraction":
                resolved_max_tokens = config.llm_extraction_output_limit
            elif task == "clustering":
                resolved_max_tokens = config.llm_clustering_output_limit
            elif task == "synthesis":
                resolved_max_tokens = config.llm_synthesis_output_limit
            
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp
        model_to_use = model if model else self.model_name

        max_retries = 3
        backoff = config.llm_retry_backoff
        last_err = None

        for attempt in range(max_retries):
            await self.rate_limiter.wait()
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                return strip_thinking_tokens(response.choices[0].message.content)
            except Exception as e:
                last_err = e
                if attempt == max_retries - 1:
                    raise e
                sleep_time = backoff * (2 ** attempt)
                con.warning(f"LLM API request failed: {e}. Retrying in {sleep_time}s...")
                await asyncio.sleep(sleep_time)
        raise last_err

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        messages = [
            {
                "role": "system",
                "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema_class.model_json_schema()}"
            },
            {"role": "user", "content": prompt}
        ]

        # Try structured outputs
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=resolved_max_tokens,
                temperature=temp,
                response_format=schema_class,
            )
            return strip_thinking_tokens(response.choices[0].message.content)
        except Exception:
            # Fallback to JSON mode
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                    response_format={"type": "json_object"},
                )
                return strip_thinking_tokens(response.choices[0].message.content)
            except Exception:
                # Basic fallback
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=resolved_max_tokens,
                    temperature=temp,
                )
                return strip_thinking_tokens(response.choices[0].message.content)

    async def generate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        messages = [
            {
                "role": "system",
                "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema_class.model_json_schema()}"
            },
            {"role": "user", "content": prompt}
        ]

        async def _call_with_format(response_format):
            max_retries = 3
            backoff = config.llm_retry_backoff
            last_err = None
            for attempt in range(max_retries):
                await self.rate_limiter.wait()
                try:
                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": resolved_max_tokens,
                        "temperature": temp,
                    }
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    response = await asyncio.to_thread(self.client.chat.completions.create, **kwargs)
                    return strip_thinking_tokens(response.choices[0].message.content)
                except Exception as e:
                    last_err = e
                    if attempt == max_retries - 1:
                        raise e
                    sleep_time = backoff * (2 ** attempt)
                    con.warning(f"LLM API JSON request failed: {e}. Retrying in {sleep_time}s...")
                    await asyncio.sleep(sleep_time)
            raise last_err

        # Try structured outputs
        try:
            return await _call_with_format(schema_class)
        except Exception:
            # Fallback to JSON mode
            try:
                return await _call_with_format({"type": "json_object"})
            except Exception:
                # Basic fallback
                return await _call_with_format(None)
