import traceback
from mlx_lm import load

model_path = "/Users/vladimirkasterin/models/llm/gemma-3-text-12b-it-4bit"
print(f"Attempting to load model from {model_path}...")
try:
    model, tokenizer = load(model_path)
    print("Success!")
except Exception as e:
    print("Failed with exception:")
    traceback.print_exc()
