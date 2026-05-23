import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.llm_engine import LLMEngine

print("1. Loading LLM Engine...")
engine = LLMEngine()

text = """Example Domain

This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.

More information..."""

print("2. Preparing extraction prompt...")
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
    f"Paper text:\n{text}"
)

formatted_prompt = prompt
if hasattr(engine.tokenizer, "apply_chat_template"):
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Chat template error: {e}")

print("3. Generating response (verbose=True via mlx_lm.generate)...")
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

sampler = make_sampler(temp=0.0)
start_time = time.time()
raw_resp = generate(
    model=engine.model,
    tokenizer=engine.tokenizer,
    prompt=formatted_prompt,
    max_tokens=config.llm_extraction_output_limit,
    sampler=sampler,
    verbose=True
)
print(f"\nGeneration completed in {time.time() - start_time:.2f}s")

print("=== RAW RESPONSE ===")
print(repr(raw_resp))
print("====================")

cleaned = engine._clean_json_response(raw_resp)
print("=== CLEANED RESPONSE ===")
print(repr(cleaned))
print("========================")

import json
try:
    parsed = json.loads(cleaned)
    print("Parsed JSON successfully:")
    print(json.dumps(parsed, indent=2))
except Exception as e:
    print(f"JSON Parsing failed: {e}")
