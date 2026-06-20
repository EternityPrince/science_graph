import sys
import re
from pathlib import Path
from datasets import load_dataset
import yaml

def parse_text(text):
    pattern = re.compile(
        r"\[context\]:\s*(.*?)\s*(?:\n|\s)*\[question\]:\s*(.*?)\s*(?:\n|\s)*\[answer\]:\s*(.*)",
        re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return {
            "c": match.group(1).strip(),
            "q": match.group(2).strip(),
            "a": match.group(3).strip()
        }
    return None

def main():
    script_dir = Path(__file__).resolve().parent
    dataset_cache_dir = script_dir / "hf_cache"
    
    print(f"Loading dataset from cache directory: {dataset_cache_dir}")
    ds = load_dataset("Nandini82/sciq-rag-dataset", cache_dir=str(dataset_cache_dir))
    
    # Gather all parsed examples from all splits
    all_examples = []
    for split_name in ds.keys():
        for item in ds[split_name]:
            parsed = parse_text(item["text"])
            if parsed:
                all_examples.append(parsed)

    print(f"Total examples parsed: {len(all_examples)}")
    
    # Map unique contexts to IDs
    context_to_id = {}
    unique_contexts = []
    for ex in all_examples:
        c = ex["c"]
        if c not in context_to_id:
            unique_contexts.append(c)
            context_to_id[c] = len(unique_contexts)
            
    print(f"Number of unique contexts: {len(unique_contexts)}")
    
    # Write unique contexts to markdown files
    sciq_papers_dir = script_dir / "sciq_papers"
    sciq_papers_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing markdown files for contexts to: {sciq_papers_dir}")
    
    for idx, c in enumerate(unique_contexts, start=1):
        file_path = sciq_papers_dir / f"sciq_paper_{idx}.md"
        content = f"---\ntitle: \"SciQ Paper {idx}\"\nauthors: [\"SciQ Author\"]\nyear: 2026\n---\n\n{c}\n"
        file_path.write_text(content, encoding="utf-8")
        
    # Write sciq_dataset.yaml in requested format
    yaml_data = []
    for idx, ex in enumerate(all_examples, start=1):
        yaml_data.append({
            "question": {
                "id": idx,
                "c": ex["c"],
                "q": ex["q"],
                "a": ex["a"]
            }
        })
        
    yaml_output_path = script_dir.parent / "sciq_dataset.yaml"
    print(f"Writing YAML dataset to: {yaml_output_path}")
    with open(yaml_output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
    print("Done!")

if __name__ == "__main__":
    main()
