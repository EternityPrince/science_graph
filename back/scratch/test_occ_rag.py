import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

print("1. Loading mlx_lm.load...")
from mlx_lm import load, generate

model_path = "/Users/vladimirkasterin/models/llm/OCC-RAG-1.7B"
print(f"2. Loading model from {model_path}...")
start_time = time.time()
model, tokenizer = load(model_path)
print(f"3. Model & tokenizer loaded in {time.time() - start_time:.2f}s")

prompt = "How do limitations in assessing language skills affect diagnosis?"
print(f"4. Original prompt: {prompt}")

if hasattr(tokenizer, "apply_chat_template"):
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"5. Formatted prompt:\n{repr(formatted_prompt)}")
    except Exception as e:
        print(f"5. Chat template formatting failed: {e}")
        formatted_prompt = prompt
else:
    formatted_prompt = prompt

print("6. Calling generate...")
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
