"""
MLX LLM Engine implementation for Apple Silicon devices.
"""

import os
from pathlib import Path
from typing import Optional, Type, List, Any

from pydantic import BaseModel
import mlx.core as mx

from src.config import config
from src import console as con
from src.llm_engine.base import BaseLLMEngine, strip_thinking_tokens


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


class MlxLLMEngine(BaseLLMEngine):
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.llm_local_model_path
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
                self.model, self.tokenizer = load(self.model_path, tokenizer_config={"fix_mistral_regex": True})

            con.success(f"MLX LLM ready: [bold]{model_name}[/bold]")

    def count_tokens(self, text: str) -> int:
        self._ensure_model_loaded()
        return super().count_tokens(text)

    def generate_response(self, prompt: str, max_tokens: int = None, temp: float = None, task: str = None, model: Optional[str] = None) -> str:
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
