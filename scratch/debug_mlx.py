import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config

print("1. Loading mlx_lm.load...")
from mlx_lm import load, generate

model_path = "/Users/vladimirkasterin/models/llm/Qwopus3.5-9B-v3-4bit"
print(f"2. Loading model from {model_path}...")
start_time = time.time()
model, tokenizer = load(model_path)
print(f"3. Model & tokenizer loaded in {time.time() - start_time:.2f}s")

prompt = "Hello! What is your name?"
print(f"4. Original prompt: {prompt}")

if hasattr(tokenizer, "apply_chat_template"):
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"5. Formatted prompt with chat template:\n{repr(formatted_prompt)}")
    except Exception as e:
        print(f"5. Chat template formatting failed: {e}")
        formatted_prompt = prompt
else:
    print("5. Tokenizer has no apply_chat_template")
    formatted_prompt = prompt

print("6. Calling generate with verbose=True...")
start_time = time.time()
try:
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=50,
        verbose=True
    )
    print(f"\n7. Generation finished in {time.time() - start_time:.2f}s")
    print(f"=== RESPONSE ===\n{response}\n================")
except Exception as e:
    print(f"7. Generation failed: {e}")
