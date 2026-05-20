import os
import sys
from mlx_lm import load, generate
from src.config import config

class LLMEngine:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.llm_model_path
        
        # Verify model directory exists
        if not os.path.isdir(self.model_path):
            print(f"[!] Warning: Model path '{self.model_path}' not found.", file=sys.stderr)
            # Try fallback to Qwen model if Gemma is not present
            fallback_path = "/Users/vladimirkasterin/models/llm/qwen3-8b-4bit"
            if os.path.isdir(fallback_path):
                print(f"[*] Found fallback model: {fallback_path}", file=sys.stderr)
                self.model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Local MLX model path not found. Please verify config.yaml "
                    f"or place a model at {config.llm_model_path}."
                )

        print(f"[*] Loading local MLX model from {self.model_path}...")
        self.model, self.tokenizer = load(self.model_path)
        print("[+] Model loaded successfully.")

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None) -> str:
        """Generates text completion using the loaded MLX model."""
        max_tokens = max_tokens or config.llm_max_tokens
        temp = temp if temp is not None else config.llm_temp
        
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temp)
        
        # Generates output via mlx-lm
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        return response.strip()

    def extract_concepts_and_metadata(self, text: str) -> Optional[dict]:
        """Uses the local LLM to extract clean author names and scientific concepts/formulas from text."""
        import json
        import re
        
        prompt = (
            "You are a scientific text analyzer. Analyze the following paper text (abstract and introduction).\n"
            "Extract:\n"
            "1. Clean list of authors.\n"
            "2. Scientific concepts, algorithms, frameworks, and key formulas mentioned in the text.\n\n"
            "You MUST format the output as a valid JSON object with the following schema:\n"
            "{\n"
            "  \"authors\": [\"Author Name 1\", \"Author Name 2\"],\n"
            "  \"concepts\": [\n"
            "    {\"name\": \"Concept Name\", \"description\": \"1 sentence description\"}\n"
            "  ]\n"
            "}\n\n"
            "Do NOT include any markdown code blocks, text outside JSON, or conversational filler. Output ONLY the raw JSON string.\n\n"
            f"Paper text:\n{text[:5000]}"
        )
        try:
            response = self.generate_response(prompt, max_tokens=1000, temp=0.0)
            clean_resp = response.strip()
            if clean_resp.startswith("```"):
                clean_resp = re.sub(r"^```(?:json)?\n?", "", clean_resp)
                clean_resp = re.sub(r"\n?```$", "", clean_resp)
            clean_resp = clean_resp.strip()
            
            # Simple fallback to find JSON block if it has prefix/suffix text
            if not (clean_resp.startswith("{") and clean_resp.endswith("}")):
                match = re.search(r"\{.*\}", clean_resp, re.DOTALL)
                if match:
                    clean_resp = match.group(0)
            
            return json.loads(clean_resp)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[!] LLM concept extraction failed: {e}")
            return None
