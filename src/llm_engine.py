"""
LLM Engine — abstracts generation to support both local MLX models and OpenAI-compatible APIs.
All output goes through src.console for consistent styled formatting.
"""

import os
import sys
import json
import re
import functools
import logging
from pathlib import Path
from typing import Optional, Type, List, Tuple, Callable, Union, Any

from pydantic import BaseModel, ValidationError
import mlx.core as mx

from src.config import config
from src import console as con

logger = logging.getLogger(__name__)


def strip_thinking_tokens(text: str) -> str:
    """
    Strips thinking tokens (<think>...</think> and unclosed trailing <think>...)
    from the LLM output.
    """
    if not text:
        return text
    # Remove closed think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove unclosed think blocks at the end
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
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
            import inspect
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


def build_mlx_tokenizer_data(tokenizer) -> Any:
    """Builds TokenEnforcerTokenizerData for mlx-lm tokenization wrapper."""
    from lmformatenforcer import TokenEnforcerTokenizerData
    
    hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
    vocab_size = len(hf_tokenizer)
    
    all_special_ids = set(getattr(hf_tokenizer, "all_special_ids", []))
    eos_token_id = getattr(hf_tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    else:
        eos_token_id = list(eos_token_id)
        
    try:
        token_0 = hf_tokenizer.encode("0", add_special_tokens=False)[-1]
    except Exception:
        token_0 = hf_tokenizer.encode("0")[-1]
        
    regular_tokens = []
    for token_idx in range(vocab_size):
        if token_idx in all_special_ids:
            continue
        try:
            decoded_after_0 = hf_tokenizer.decode([token_0, token_idx])[1:]
            decoded_regular = hf_tokenizer.decode([token_idx])
            is_word_start_token = len(decoded_after_0) > len(decoded_regular)
            regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
        except Exception:
            continue
            
    def decode_fn(tokens: List[int]) -> str:
        return hf_tokenizer.decode(tokens).rstrip('')
        
    return TokenEnforcerTokenizerData(
        regular_tokens=regular_tokens,
        decoder=decode_fn,
        eos_token_id=eos_token_id,
        use_bitmask=False,
        vocab_size=vocab_size
    )


class ConstrainedLogitsProcessor:
    """Logits processor that filters tokens to match a given TokenEnforcer schema."""
    def __init__(self, token_enforcer: Any):
        self.token_enforcer = token_enforcer

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # tokens[1:] are the generated tokens (skipping the last prompt token)
        generated_tokens = tokens.tolist()[1:]
        allowed_tokens = self.token_enforcer.get_allowed_tokens(generated_tokens).allowed_tokens
        
        if not allowed_tokens:
            return logits
            
        import numpy as np
        vocab_size = logits.shape[-1]
        mask = np.full(vocab_size, -np.inf, dtype=np.float32)
        mask[allowed_tokens] = 0.0
        
        return logits + mx.array(mask)


class BaseLLMEngine:
    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

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
        # Generate JSON text using the configured LLM engine
        response = self.generate_json(
            prompt=prompt,
            schema_class=schema_class,
            temp=temp,
            max_tokens=max_tokens,
        )
        
        # Extract the JSON block
        clean_json = ResilientParser.extract_json(response)
        
        # Parse and validate the JSON
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
        # Kept for compatibility
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
        prompt = (
            "You are a strict scientific text analyzer. Analyze the following paper text (abstract and introduction).\n"
            "Extract:\n"
            "1. Authors: A list of ONLY actual human names (e.g. \"Jane Doe\", \"John Smith\"). Do NOT include paper titles or citations here.\n"
            "2. Concepts: A list of scientific concepts, algorithms, frameworks, and key formulas. These MUST be short noun phrases (1-3 words max, e.g. \"Self-Attention\", \"Transformer\"). Do NOT extract full sentences or citations as concepts.\n"
            "3. Tags: A list of 3-7 high-level topic tags/keywords (e.g., \"statistics\", \"probability theory\", \"gradient descent\", \"optimization methods\", \"deep learning\"). These should represent the main fields and tools used in the paper.\n\n"
            "You MUST format the output as a valid JSON object with the following schema:\n"
            "{\n"
            "  \"authors\": [\"Author Name 1\", \"Author Name 2\"],\n"
            "  \"concepts\": [\n"
            "    {\"name\": \"Concept Name\", \"description\": \"1 concise sentence description\"}\n"
            "  ],\n"
            "  \"tags\": [\"tag1\", \"tag2\", \"tag3\"]\n"
            "}\n\n"
            "Do NOT include any markdown code blocks, text outside JSON, or conversational filler. Output ONLY the raw JSON string.\n\n"
            f"Paper text:\n{safe_text}"
        )
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

    def cluster_chunks_by_topic(self, chunks_summary: str, topic: str) -> Optional[dict]:
        """Groups text chunks into thematic sections for the review report with Pydantic-based validation."""
        max_input = config.llm_clustering_input_limit
        safe_chunks = self._truncate_to_context(chunks_summary, max_input)
        prompt = (
            f"You are a scientific editor preparing a literature review on the topic: '{topic}'.\n"
            f"You are given a list of text chunks from papers (each has an id, paper title, and excerpt).\n"
            "Group these chunks into 3-6 thematic SECTIONS for the review.\n\n"
            "Rules:\n"
            "- Each section must have a clear, descriptive title.\n"
            "- Assign each chunk_id to exactly one section.\n"
            "- Output ONLY a valid JSON object: {\"Section Title\": [\"chunk_id1\", \"chunk_id2\"], ...}\n"
            "- Do NOT include any text outside the JSON.\n\n"
            f"Chunks:\n{safe_chunks}"
        )
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
        prompt = (
            f"You are a scientific writer synthesizing a section of a literature review.\n"
            f"Review topic: '{topic}'\n"
            f"Section: '{section_name}'\n\n"
            "Using ONLY the provided text fragments, write a coherent, well-structured section (3-5 paragraphs).\n"
            "- Cite papers by their titles in parentheses, e.g. (Vaswani et al., 2017).\n"
            "- Highlight agreements and disagreements between approaches.\n"
            "- Use precise, academic language. Do NOT make up facts not in the fragments.\n"
            "- Output pure Markdown (no code blocks, no headers — those are added externally).\n\n"
            f"Text fragments:\n{safe_chunks}"
        )
        try:
            return self.generate_response(prompt, max_tokens=max_tokens, temp=0.2, task="synthesis")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Section synthesis failed for '{section_name}': {e}")
            return f"*[Generation failed for this section: {e}]*"


class MlxLLMEngine(BaseLLMEngine):
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.llm_model_path
        self._tokenizer_data = None
        self.model = None
        self.tokenizer = None

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Local MLX model path not found: {self.model_path}\n"
                f"  Run: python3 main.py config  to see configured paths."
            )

    def _ensure_model_loaded(self):
        if self.model is None:
            model_name = Path(self.model_path).name
            con.model_msg(f"Loading MLX LLM [bold]{model_name}[/bold] …")

            from mlx_lm import load
            with con.suppress_stderr(), con.suppress_stdout():
                self.model, self.tokenizer = load(self.model_path)

            con.success(f"MLX LLM ready: [bold]{model_name}[/bold]")

    def count_tokens(self, text: str) -> int:
        self._ensure_model_loaded()
        return super().count_tokens(text)

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None) -> str:
        self._ensure_model_loaded()
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

        formatted_prompt = prompt
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temp)

        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=resolved_max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return strip_thinking_tokens(response)

    def generate_json(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._ensure_model_loaded()
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None:
            resolved_max_tokens = config.llm_max_tokens

        temp = temp if temp is not None else config.llm_temp

        if self._tokenizer_data is None:
            self._tokenizer_data = build_mlx_tokenizer_data(self.tokenizer)

        from lmformatenforcer import TokenEnforcer, JsonSchemaParser
        parser = JsonSchemaParser(schema_class.model_json_schema())
        enforcer = TokenEnforcer(self._tokenizer_data, parser)
        logits_processor = ConstrainedLogitsProcessor(enforcer)

        formatted_prompt = prompt
        is_formatted = any(
            tag in prompt
            for tag in [
                "<|im_start|>",
                "<|start_header_id|>",
                "[INST]",
                "<start_of_turn>",
                "<|im_end|>"
            ]
        )

        if not is_formatted and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a strict JSON extractor. Output ONLY valid JSON matching this schema: {schema_class.model_json_schema()}"
                    },
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temp)

        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=resolved_max_tokens,
            sampler=sampler,
            verbose=False,
            logits_processors=[logits_processor],
        )
        return strip_thinking_tokens(response)


class OpenAILLMEngine(BaseLLMEngine):
    def __init__(self):
        import openai
        api_key = config.llm_api_key
        base_url = config.llm_base_url
        self.model_name = config.llm_model_path

        if not api_key:
            con.error("API key is not configured for OpenAI/OpenRouter.")
            raise ValueError("Missing API key for OpenAI provider")

        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        self.client = openai.OpenAI(**client_args)

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

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None) -> str:
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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=resolved_max_tokens,
            temperature=temp,
        )
        return strip_thinking_tokens(response.choices[0].message.content)

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


def LLMEngine(*args, **kwargs) -> BaseLLMEngine:
    """Factory for returning the correct LLM Engine based on config."""
    provider = config.llm_provider.lower()
    if provider == "openai":
        return OpenAILLMEngine()
    else:
        return MlxLLMEngine(*args, **kwargs)
