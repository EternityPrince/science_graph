"""
MLX LLM Engine implementation for Apple Silicon devices.
"""

import os
from pathlib import Path
from typing import Optional, Type, List, Any

from pydantic import BaseModel
try:
    import mlx.core as mx
except ImportError:
    mx = None

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
        self.generated_tokens = []

    def __call__(self, tokens: Any, logits: Any) -> Any:
        num_tokens = tokens.shape[0]
        current_len = len(self.generated_tokens)
        
        if num_tokens > current_len + 1:
            for idx in range(current_len + 1, num_tokens):
                self.generated_tokens.append(tokens[idx].item())
                
        allowed_tokens = self.token_enforcer.get_allowed_tokens(self.generated_tokens).allowed_tokens
        
        if not allowed_tokens:
            return logits
            
        # Create logits mask directly in MLX to avoid CPU allocations and copies
        vocab_size = logits.shape[-1]
        mask = mx.full((vocab_size,), float("-inf"), dtype=mx.float32)
        mask[mx.array(allowed_tokens)] = 0.0
        
        return logits + mask


class MlxLLMEngine(BaseLLMEngine):
<<<<<<< HEAD
    def __init__(self, model_path: str = ""):
=======
    def __init__(self, model_path: str = None):
        if mx is None:
            raise ImportError(
                "MLX is not installed. MlxLLMEngine is only supported on Apple Silicon macOS with the 'mlx' package installed."
            )
>>>>>>> 4756785 (fix test)
        self.model_path = model_path or config.llm_local_model_path
        self._tokenizer_data = None
        self.model = None
        self.tokenizer = None

        print("MLX_INIT", {
            "pid": os.getpid(),
            "self_id": id(self),
            "model_path": self.model_path,
        })

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Local MLX model path not found: {self.model_path}\n"
                f"  Run: python3 main.py config  to see configured paths."
            )

        # Force-import mlx_lm.generate on the current (main) thread so that
        # its module-level `generation_stream = mx.new_stream(...)` is bound
        # to the main thread.  Without this, the Marker PDF parser can trigger
        # the first import inside an asyncio.to_thread worker, which creates
        # the stream on a short-lived thread; subsequent generate() calls on
        # the main thread then crash with "There is no Stream(gpu, N) in
        # current thread" (MLX 0.31+ enforces thread-local streams).
        import mlx_lm.generate  # noqa: F401

    def unload_model(self):
        print("MLX_UNLOAD", {
            "pid": os.getpid(),
            "self_id": id(self),
            "model_path": self.model_path,
            "model_was_none": self.model is None,
        })
        
        if self.model is not None:
            import gc
            self.model = None
            self.tokenizer = None
            self._tokenizer_data = None
            gc.collect()
            try:
                import mlx.core as mx
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except ImportError:
                pass
            con.success("MLX LLM model unloaded and GPU cache cleared")

    def _ensure_model_loaded(self):
        print("MLX_ENSURE_BEFORE", {
            "pid": os.getpid(),
            "self_id": id(self),
            "model_path": self.model_path,
            "model_is_none": self.model is None,
            "tokenizer_is_none": self.tokenizer is None,
        })
        
        if self.model is None:
            model_name = Path(self.model_path).name
            con.model_msg(f"Loading MLX LLM [bold]{model_name}[/bold] …")

            print("MLX_REAL_LOAD_START", {
                "pid": os.getpid(),
                "self_id": id(self),
                "model_path": self.model_path,
            })

            from mlx_lm import load
            with con.suppress_stderr(), con.suppress_stdout():
                self.model, self.tokenizer = load(self.model_path, tokenizer_config={"fix_mistral_regex": True})

            print("MLX_REAL_LOAD_DONE", {
                "pid": os.getpid(),
                "self_id": id(self),
                "model_path": self.model_path,
            })

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

    async def generate_response_async(
        self,
        prompt: str,
        max_tokens: int = None,
        temp: float = None,
        task: str = None,
        model: Optional[str] = None,
    ) -> str:
        import asyncio
        await asyncio.sleep(0)
        return self.generate_response(prompt, max_tokens, temp, task, model=model)

    async def generate_json_async(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        temp: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        import asyncio
        await asyncio.sleep(0)
        return self.generate_json(prompt, schema_class, temp, max_tokens)
