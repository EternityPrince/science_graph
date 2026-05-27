"""
Base classes and helpers for LLM Engines.
"""

import json
import re
import functools
import inspect
import asyncio
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from src.config import config
from src import console as con
from src.prompts import prompts


def strip_thinking_tokens(text: str) -> str:
    """
    Strips thinking tokens (<think>...</think> and unclosed trailing <think>...)
    and technical/special/formatting tokens (like <|im_start|>, <|im_end|>, etc.)
    from the LLM output.
    """
    if not text:
        return text
    # Remove closed think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed think blocks at the end
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)

    # Patterns for technical formatting tokens.
    # Note: we escape regex special characters.
    technical_patterns = [
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\|im_sep\|>",
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
        r"<\|eot_id\|>",
        r"<\|eom_id\|>",
        r"<\|endoftext\|>",
        r"<\|assistant\|>",
        r"<\|user\|>",
        r"<\|system\|>",
        r"<\|end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<s>",
        r"</s>",
        r"<start_of_turn>",
        r"<end_of_turn>",
        r"<<SYS>>",
        r"<</SYS>>",
        r"<pad>",
        r"<unk>",
    ]
    
    # Remove these tokens
    pattern = "|".join(technical_patterns)
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean up role prefix headers (e.g. "assistant\n" or "assistant:") if they leak at the start
    text = text.strip()
    text = re.sub(r"^(?:assistant|user|system)(?:\n|:\s*)", "", text, flags=re.IGNORECASE)

    return text.strip()


class ResilientParser:
    """
    Extracts the JSON block (first '{' or '[' to its matching closing character,
    or via fallback regex) from raw LLM responses.
    """
    @staticmethod
    def extract_json(text: str) -> str:
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()

        # Find first opening character '{' or '['
        start_idx = -1
        open_char = None
        close_char = None
        for i, char in enumerate(clean):
            if char in ('{', '['):
                start_idx = i
                open_char = char
                close_char = '}' if char == '{' else ']'
                break

        if start_idx != -1:
            # Trace matching closing character
            count = 0
            in_string = False
            escape = False
            for i in range(start_idx, len(clean)):
                char = clean[i]
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == open_char:
                        count += 1
                    elif char == close_char:
                        count -= 1
                        if count == 0:
                            return clean[start_idx:i+1]

        # Greedy fallback if tracing fails
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"(\[.*\])", clean, re.DOTALL)
        if match:
            return match.group(1)
        return clean


def retry_with_temp_decay(max_retries: int = 3):
    """
    Decorator to retry LLM JSON extraction on failure, decaying the temperature
    toward 0.0 with each retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            initial_temp = bound.arguments.get("temp", 0.0)
            
            last_err = None
            for attempt in range(max_retries + 1):
                if attempt == 0:
                    current_temp = initial_temp
                elif attempt == max_retries:
                    current_temp = 0.0
                else:
                    current_temp = initial_temp * (1.0 - attempt / max_retries)
                
                bound.arguments["temp"] = current_temp
                
                try:
                    return func(*bound.args, **bound.kwargs)
                except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as e:
                    last_err = e
                    con.warning(
                        f"JSON extraction attempt {attempt + 1} failed: {e}. "
                        f"Retrying with decayed temperature {current_temp:.2f}..."
                    )
            raise last_err
        return wrapper
    return decorator


def retry_with_temp_decay_async(max_retries: int = 3):
    """
    Decorator to retry LLM JSON extraction asynchronously on failure, decaying the temperature
    toward 0.0 with each retry.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            
            initial_temp = bound.arguments.get("temp", 0.0)
            
            last_err = None
            for attempt in range(max_retries + 1):
                if attempt == 0:
                    current_temp = initial_temp
                elif attempt == max_retries:
                    current_temp = 0.0
                else:
                    current_temp = initial_temp * (1.0 - attempt / max_retries)
                
                bound.arguments["temp"] = current_temp
                
                try:
                    return await func(*bound.args, **bound.kwargs)
                except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as e:
                    last_err = e
                    con.warning(
                        f"JSON extraction attempt {attempt + 1} failed: {e}. "
                        f"Retrying with decayed temperature {current_temp:.2f}..."
                    )
            raise last_err
        return wrapper
    return decorator


class BaseLLMEngine:
    @staticmethod
    def extract_json(text: str) -> str:
        return ResilientParser.extract_json(text)

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
        raise NotImplementedError

    async def generate_response_async(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
        return await asyncio.to_thread(self.generate_response, prompt, max_tokens, temp, task, model=model)

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

    async def generate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await asyncio.to_thread(self.generate_json, prompt, schema_class, temp, max_tokens)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "encode"):
                    return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return len(text) // 4

    @retry_with_temp_decay(max_retries=3)
    def generate_and_validate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        response = self.generate_json(
            prompt=prompt,
            schema_class=schema_class,
            temp=temp,
            max_tokens=max_tokens,
        )
        clean_json = ResilientParser.extract_json(response)
        try:
            parsed = json.loads(clean_json)
        except Exception as e:
            raise json.JSONDecodeError(
                f"JSON decode failed: {e}\nRaw output: {response}",
                doc=clean_json,
                pos=0
            ) from e
            
        try:
            return schema_class.model_validate(parsed)
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}") from e

    @retry_with_temp_decay_async(max_retries=3)
    async def generate_and_validate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        response = await self.generate_json_async(
            prompt=prompt,
            schema_class=schema_class,
            temp=temp,
            max_tokens=max_tokens,
        )
        clean_json = ResilientParser.extract_json(response)
        try:
            parsed = json.loads(clean_json)
        except Exception as e:
            raise json.JSONDecodeError(
                f"JSON decode failed: {e}\nRaw output: {response}",
                doc=clean_json,
                pos=0
            ) from e
            
        try:
            return schema_class.model_validate(parsed)
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}") from e

    def _clean_json_response(self, response: str) -> str:
        return ResilientParser.extract_json(response)

    def _truncate_to_context(self, text: str, max_input_tokens: int) -> str:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return text[:max_input_tokens * 4]
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_input_tokens:
                return text
            return self.tokenizer.decode(tokens[:max_input_tokens])
        except Exception:
            return text[:max_input_tokens * 4]

    def extract_concepts_and_metadata(self, text: str) -> Optional[dict]:
        """Extracts authors, scientific concepts, and topic tags from text with Pydantic-based validation."""
        max_input = config.llm_extraction_input_limit
        safe_text = self._truncate_to_context(text, max_input)
        prompt = prompts.get_prompt("extraction", "concepts_metadata", safe_text=safe_text)

        try:
            from src.llm_schemas import LLMExtractionResponse, validate_extraction_response
            validated_model = self.generate_and_validate_json(
                prompt=prompt,
                schema_class=LLMExtractionResponse,
                temp=0.3,  # Start with slight temp to allow diversity, will decay to 0.0
                max_tokens=config.llm_extraction_output_limit,
            )
            
            validated, warnings = validate_extraction_response(validated_model.model_dump())
            
            if warnings:
                con.warning("LLM extraction output validated with warnings:")
                for w in warnings:
                    con.warning(f"  - {w}")
            else:
                con.success("LLM extraction output validated successfully.")
                
            orig_concepts = len(validated_model.concepts)
            val_concepts = len(validated.concepts)
            score = val_concepts / orig_concepts if orig_concepts > 0 else 1.0
            con.info(f"LLM Concept Extraction Quality Score: {val_concepts}/{orig_concepts} ({score:.0%})")
            
            return validated.model_dump()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM concept extraction failed: {e}")
            con.warning(f"LLM concept extraction validation failed: {e}")
            return None

    async def extract_concepts_and_metadata_async(self, text: str) -> Optional[dict]:
        """Extracts authors, scientific concepts, and topic tags asynchronously from text with Pydantic-based validation."""
        max_input = config.llm_extraction_input_limit
        safe_text = self._truncate_to_context(text, max_input)
        prompt = prompts.get_prompt("extraction", "concepts_metadata", safe_text=safe_text)

        try:
            from src.llm_schemas import LLMExtractionResponse, validate_extraction_response
            validated_model = await self.generate_and_validate_json_async(
                prompt=prompt,
                schema_class=LLMExtractionResponse,
                temp=0.3,  # Start with slight temp to allow diversity, will decay to 0.0
                max_tokens=config.llm_extraction_output_limit,
            )
            
            validated, warnings = validate_extraction_response(validated_model.model_dump())
            
            if warnings:
                con.warning("LLM extraction output validated with warnings:")
                for w in warnings:
                    con.warning(f"  - {w}")
            else:
                con.success("LLM extraction output validated successfully.")
                
            orig_concepts = len(validated_model.concepts)
            val_concepts = len(validated.concepts)
            score = val_concepts / orig_concepts if orig_concepts > 0 else 1.0
            con.info(f"LLM Concept Extraction Quality Score: {val_concepts}/{orig_concepts} ({score:.0%})")
            
            return validated.model_dump()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Async LLM concept extraction failed: {e}")
            con.warning(f"Async LLM concept extraction validation failed: {e}")
            return None

    def cluster_chunks_by_topic(self, chunks_summary: str, topic: str) -> Optional[dict]:
        """Groups text chunks into thematic sections for the review report with Pydantic-based validation."""
        max_input = config.llm_clustering_input_limit
        safe_chunks = self._truncate_to_context(chunks_summary, max_input)
        prompt = prompts.get_prompt("synthesis", "cluster_chunks", topic=topic, safe_chunks=safe_chunks)

        try:
            from src.llm_schemas import LLMClusteringResponse, validate_clustering_response
            validated_model = self.generate_and_validate_json(
                prompt=prompt,
                schema_class=LLMClusteringResponse,
                temp=0.0,
                max_tokens=config.llm_clustering_output_limit,
            )
            
            validated, warnings = validate_clustering_response(validated_model.model_dump())
            
            if warnings:
                con.warning("LLM clustering output validated with warnings:")
                for w in warnings:
                    con.warning(f"  - {w}")
            else:
                con.success("LLM clustering output validated successfully.")
                
            return validated.model_dump()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM clustering failed: {e}")
            con.warning(f"LLM clustering validation failed: {e}")
            return None

    def synthesize_section(
        self,
        section_name: str,
        chunks_text: str,
        topic: str,
        max_tokens: int = None,
    ) -> str:
        """Generates a Markdown section for the review from a cluster of text chunks."""
        max_input = config.llm_synthesis_input_limit
        safe_chunks = self._truncate_to_context(chunks_text, max_input)
        prompt = prompts.get_prompt("synthesis", "synthesize_section", topic=topic, section_name=section_name, safe_chunks=safe_chunks)

        try:
            return self.generate_response(prompt, max_tokens=max_tokens, temp=0.2, task="synthesis")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Section synthesis failed for '{section_name}': {e}")
            return f"*[Generation failed for this section: {e}]*"
