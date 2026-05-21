"""
LLM Engine — abstracts generation to support both local MLX models and OpenAI-compatible APIs.
All output goes through src.console for consistent styled formatting.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Optional

from src.config import config
from src import console as con


class BaseLLMEngine:
    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None) -> str:
        raise NotImplementedError

    def _clean_json_response(self, response: str) -> str:
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
        return clean

    def extract_concepts_and_metadata(self, text: str) -> Optional[dict]:
        """Extracts authors, scientific concepts, and topic tags from text."""
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
            f"Paper text:\n{text[:5000]}"
        )
        try:
            response = self.generate_response(prompt, max_tokens=1000, temp=0.0)
            clean_resp = self._clean_json_response(response)
            return json.loads(clean_resp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM concept extraction failed: {e}")
            return None

    def cluster_chunks_by_topic(self, chunks_summary: str, topic: str) -> Optional[dict]:
        """Groups text chunks into thematic sections for the review report."""
        prompt = (
            f"You are a scientific editor preparing a literature review on the topic: '{topic}'.\n"
            "You are given a list of text chunks from papers (each has an id, paper title, and excerpt).\n"
            "Group these chunks into 3-6 thematic SECTIONS for the review.\n\n"
            "Rules:\n"
            "- Each section must have a clear, descriptive title.\n"
            "- Assign each chunk_id to exactly one section.\n"
            "- Output ONLY a valid JSON object: {\"Section Title\": [\"chunk_id1\", \"chunk_id2\"], ...}\n"
            "- Do NOT include any text outside the JSON.\n\n"
            f"Chunks:\n{chunks_summary[:6000]}"
        )
        try:
            response = self.generate_response(prompt, max_tokens=800, temp=0.0)
            clean = self._clean_json_response(response)
            return json.loads(clean)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM clustering failed: {e}")
            return None

    def synthesize_section(
        self,
        section_name: str,
        chunks_text: str,
        topic: str,
        max_tokens: int = 800,
    ) -> str:
        """Generates a Markdown section for the review from a cluster of text chunks."""
        prompt = (
            f"You are a scientific writer synthesizing a section of a literature review.\n"
            f"Review topic: '{topic}'\n"
            f"Section: '{section_name}'\n\n"
            "Using ONLY the provided text fragments, write a coherent, well-structured section (3-5 paragraphs).\n"
            "- Cite papers by their titles in parentheses, e.g. (Vaswani et al., 2017).\n"
            "- Highlight agreements and disagreements between approaches.\n"
            "- Use precise, academic language. Do NOT make up facts not in the fragments.\n"
            "- Output pure Markdown (no code blocks, no headers — those are added externally).\n\n"
            f"Text fragments:\n{chunks_text[:5000]}"
        )
        try:
            return self.generate_response(prompt, max_tokens=max_tokens, temp=0.2)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Section synthesis failed for '{section_name}': {e}")
            return f"*[Generation failed for this section: {e}]*"


class MlxLLMEngine(BaseLLMEngine):
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.llm_model_path

        if not os.path.isdir(self.model_path):
            fallback_path = "/Users/vladimirkasterin/models/llm/qwen3-8b-4bit"
            if os.path.isdir(fallback_path):
                con.warning(f"Model path not found, using fallback: {Path(fallback_path).name}")
                self.model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Local MLX model path not found: {self.model_path}\n"
                    f"  Run: python3 main.py config  to see configured paths."
                )

        model_name = Path(self.model_path).name
        con.model_msg(f"Loading MLX LLM [bold]{model_name}[/bold] …")

        from mlx_lm import load
        with con.suppress_stderr(), con.suppress_stdout():
            self.model, self.tokenizer = load(self.model_path)

        con.success(f"MLX LLM ready: [bold]{model_name}[/bold]")

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None) -> str:
        max_tokens = max_tokens or config.llm_max_tokens
        temp = temp if temp is not None else config.llm_temp

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temp)

        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return response.strip()


class OpenAILLMEngine(BaseLLMEngine):
    def __init__(self):
        import openai
        api_key = config.llm_api_key
        base_url = config.llm_base_url
        self.model_name = config.llm_model_path  # We'll use model_path field for the model name

        if not api_key:
            con.error("API key is not configured for OpenAI/OpenRouter.")
            raise ValueError("Missing API key for OpenAI provider")

        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        self.client = openai.OpenAI(**client_args)
        con.success(f"OpenAI API LLM ready: [bold]{self.model_name}[/bold]")

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None) -> str:
        max_tokens = max_tokens or config.llm_max_tokens
        temp = temp if temp is not None else config.llm_temp

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return response.choices[0].message.content.strip()


def LLMEngine(*args, **kwargs) -> BaseLLMEngine:
    """Factory for returning the correct LLM Engine based on config."""
    provider = config.llm_provider.lower()
    if provider == "openai":
        return OpenAILLMEngine()
    else:
        return MlxLLMEngine(*args, **kwargs)
