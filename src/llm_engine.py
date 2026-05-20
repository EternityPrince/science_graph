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
