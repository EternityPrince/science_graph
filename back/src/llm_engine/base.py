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
        prompt = (
            "You are a strict scientific text analyzer. Analyze the following paper text (abstract and introduction).\n"
            "Extract the following scientific entities, properties, and relationships:\n"
            "1. Authors: A list of ONLY actual human names (e.g. \"Jane Doe\", \"John Smith\"). Do NOT include paper titles or citations here.\n"
            "2. Concepts: A list of scientific concepts, algorithms, frameworks, and key formulas. These MUST be short noun phrases (1-3 words max, e.g. \"Self-Attention\", \"Transformer\"). For each concept, extract a list of synonyms and abbreviations as 'aliases' (e.g., [\"LLM\", \"Large Language Model\"]).\n"
            "3. Tags: A list of 3-7 high-level topic tags/keywords (e.g., \"statistics\", \"deep learning\").\n"
            "4. Institutions: A list of organizations, universities, or companies mentioned in the text (e.g., \"MIT\", \"Google DeepMind\").\n"
            "5. Author Institutions: Mapping of extracted authors to their affiliated institutions, formatted as a list of {\"author\": \"Author Name\", \"institution\": \"Institution Name\"}.\n"
            "6. Sponsored By: List of institutions or companies that sponsored, funded, or supported the paper.\n"
            "7. Datasets: List of benchmarks or datasets mentioned in the text, formatted as a list of {\"name\": \"Dataset Name\", \"relation\": \"USED_DATASET\" or \"INTRODUCED_DATASET\"}.\n"
            "8. Code Repositories: List of code repository URLs (like GitHub links) implementing the paper.\n"
            "9. Journal or Conference: The name of the journal or conference where the paper was published (e.g., \"NeurIPS\", \"ICML\", \"Nature\").\n"
            "10. Citation Intents: List of referenced papers/authors mentioned in the text, and classify their citation intent, formatted as a list of {\"target_title\": \"Cited Paper Title\", \"intent\": \"USES_METHOD\" or \"EXTENDS\" or \"COMPARES_WITH\" or \"DISPUTES\" or \"BACKGROUND\"}.\n"
            "11. Concept Relations: List of relationships between extracted concepts, formatted as a list of {\"source\": \"Concept A\", \"target\": \"Concept B\", \"relation_type\": \"SUBCLASS_OF\" or \"IS_A\" or \"PREREQUISITE_FOR\"}.\n\n"
            "You MUST format the output as a valid JSON object with the following schema:\n"
            "{\n"
            "  \"authors\": [\"Author Name 1\", \"Author Name 2\"],\n"
            "  \"concepts\": [\n"
            "    {\"name\": \"Concept Name\", \"description\": \"1 concise sentence description\", \"aliases\": [\"alias1\", \"alias2\"]}\n"
            "  ],\n"
            "  \"tags\": [\"tag1\", \"tag2\"],\n"
            "  \"institutions\": [\"MIT\", \"Google DeepMind\"],\n"
            "  \"author_institutions\": [{\"author\": \"John Doe\", \"institution\": \"MIT\"}],\n"
            "  \"sponsored_by\": [\"Google DeepMind\"],\n"
            "  \"datasets\": [{\"name\": \"GSM8k\", \"relation\": \"USED_DATASET\"}],\n"
            "  \"code_repositories\": [\"https://github.com/...\"],\n"
            "  \"journal_or_conference\": \"NeurIPS\",\n"
            "  \"citation_intents\": [{\"target_title\": \"Attention Is All You Need\", \"intent\": \"USES_METHOD\"}],\n"
            "  \"concept_relations\": [{\"source\": \"Self-Attention\", \"target\": \"Transformer\", \"relation_type\": \"PREREQUISITE_FOR\"}]\n"
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

    async def extract_concepts_and_metadata_async(self, text: str) -> Optional[dict]:
        """Extracts authors, scientific concepts, and topic tags asynchronously from text with Pydantic-based validation."""
        max_input = config.llm_extraction_input_limit
        safe_text = self._truncate_to_context(text, max_input)
        prompt = (
            "You are a strict scientific text analyzer. Analyze the following paper text (abstract and introduction).\n"
            "Extract the following scientific entities, properties, and relationships:\n"
            "1. Authors: A list of ONLY actual human names (e.g. \"Jane Doe\", \"John Smith\"). Do NOT include paper titles or citations here.\n"
            "2. Concepts: A list of scientific concepts, algorithms, frameworks, and key formulas. These MUST be short noun phrases (1-3 words max, e.g. \"Self-Attention\", \"Transformer\"). For each concept, extract a list of synonyms and abbreviations as 'aliases' (e.g., [\"LLM\", \"Large Language Model\"]).\n"
            "3. Tags: A list of 3-7 high-level topic tags/keywords (e.g., \"statistics\", \"deep learning\").\n"
            "4. Institutions: A list of organizations, universities, or companies mentioned in the text (e.g., \"MIT\", \"Google DeepMind\").\n"
            "5. Author Institutions: Mapping of extracted authors to their affiliated institutions, formatted as a list of {\"author\": \"Author Name\", \"institution\": \"Institution Name\"}.\n"
            "6. Sponsored By: List of institutions or companies that sponsored, funded, or supported the paper.\n"
            "7. Datasets: List of benchmarks or datasets mentioned in the text, formatted as a list of {\"name\": \"Dataset Name\", \"relation\": \"USED_DATASET\" or \"INTRODUCED_DATASET\"}.\n"
            "8. Code Repositories: List of code repository URLs (like GitHub links) implementing the paper.\n"
            "9. Journal or Conference: The name of the journal or conference where the paper was published (e.g., \"NeurIPS\", \"ICML\", \"Nature\").\n"
            "10. Citation Intents: List of referenced papers/authors mentioned in the text, and classify their citation intent, formatted as a list of {\"target_title\": \"Cited Paper Title\", \"intent\": \"USES_METHOD\" or \"EXTENDS\" or \"COMPARES_WITH\" or \"DISPUTES\" or \"BACKGROUND\"}.\n"
            "11. Concept Relations: List of relationships between extracted concepts, formatted as a list of {\"source\": \"Concept A\", \"target\": \"Concept B\", \"relation_type\": \"SUBCLASS_OF\" or \"IS_A\" or \"PREREQUISITE_FOR\"}.\n\n"
            "You MUST format the output as a valid JSON object with the following schema:\n"
            "{\n"
            "  \"authors\": [\"Author Name 1\", \"Author Name 2\"],\n"
            "  \"concepts\": [\n"
            "    {\"name\": \"Concept Name\", \"description\": \"1 concise sentence description\", \"aliases\": [\"alias1\", \"alias2\"]}\n"
            "  ],\n"
            "  \"tags\": [\"tag1\", \"tag2\"],\n"
            "  \"institutions\": [\"MIT\", \"Google DeepMind\"],\n"
            "  \"author_institutions\": [{\"author\": \"John Doe\", \"institution\": \"MIT\"}],\n"
            "  \"sponsored_by\": [\"Google DeepMind\"],\n"
            "  \"datasets\": [{\"name\": \"GSM8k\", \"relation\": \"USED_DATASET\"}],\n"
            "  \"code_repositories\": [\"https://github.com/...\"],\n"
            "  \"journal_or_conference\": \"NeurIPS\",\n"
            "  \"citation_intents\": [{\"target_title\": \"Attention Is All You Need\", \"intent\": \"USES_METHOD\"}],\n"
            "  \"concept_relations\": [{\"source\": \"Self-Attention\", \"target\": \"Transformer\", \"relation_type\": \"PREREQUISITE_FOR\"}]\n"
            "}\n\n"
            "Do NOT include any markdown code blocks, text outside JSON, or conversational filler. Output ONLY the raw JSON string.\n\n"
            f"Paper text:\n{safe_text}"
        )
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
