from pathlib import Path
from typing import Dict, Any

# Map of baselines and their descriptions
BASELINES_INFO: Dict[str, str] = {
    "B0": "Zero-Shot (Чистая генерация) — оценка базовых знаний LLM без контекста.",
    "B1": "Pure Lexical (Только лексика) — поиск строго по ключевым словам через SQLite FTS5.",
    "B2": "Pure Dense (Только векторы) — классический семантический поиск по эмбеддингам.",
    "B3": "Dense + HyDE (Векторы + Гипотетический документ) — семантический поиск с гипотетическим ответом.",
    "B4": "Standard Hybrid (Базовый гибрид) — связка FTS5 + Векторы через RRF без графов.",
    "B5": "Hybrid + Graph (Базовый Граф-RAG) — гибридный поиск + статический обход графа (без реранкера/LLM-расширения).",
    "B6": "Full Pipeline (Максимальный запуск) — включены все 12 компонентов (граф, реранкер, LLM-расширение и др. без HyDE).",
    "CUSTOM": "Custom Run (Конфигурация с пользовательскими параметрами) — для тестирования влияния настроек на поиск."
}


def get_baseline_config(baseline: str, config_rag_components: dict) -> Dict[str, bool]:
    """Returns the RAG components configuration dictionary for a given baseline.
    
    Accepts config_rag_components keys to initialize the default state.
    """
    components = {k: False for k in config_rag_components.keys()}
    
    if baseline == "B0":
        # Zero-shot has no retrieval components enabled
        pass
    elif baseline == "B1":
        components["lexical_search"] = True
    elif baseline == "B2":
        components["dense_search"] = True
    elif baseline == "B3":
        components["dense_search"] = True
        components["hyde"] = True
    elif baseline == "B4":
        components["dense_search"] = True
        components["lexical_search"] = True
        components["rrf"] = True
        components["dynamic_alpha_blending"] = True
    elif baseline == "B5":
        components["dense_search"] = True
        components["lexical_search"] = True
        components["rrf"] = True
        components["dynamic_alpha_blending"] = True
        components["graph_expansion"] = True
        components["context_trimming"] = True
        components["citation_repair"] = True
    elif baseline == "B6":
        # Full pipeline has everything enabled except hyde (respecting user overrides)
        components = {k: config_rag_components.get(k, True) for k in config_rag_components.keys()}
        components["hyde"] = False
    elif baseline == "CUSTOM":
        # Custom has everything enabled except hyde by default (respecting user overrides)
        components = {k: config_rag_components.get(k, True) for k in config_rag_components.keys()}
        components["hyde"] = False
        
    if baseline not in ["B6", "CUSTOM"]:
        components["intent_classifier"] = False
    else:
        components["intent_classifier"] = config_rag_components.get("intent_classifier", False)
    return components


def get_safe_model_name(model_name: str) -> str:
    """Safely normalizes LLM/embedder model path/name to be used as directory name."""
    name = Path(model_name).name
    name = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return name


DEFAULT_LIMIT = 50

def load_benchmark_dataset(dataset_path: Path, limit: int = None, seed: int = 42) -> list:
    """Loads a dataset YAML file, formats it if it's SciQ format,
    and applies random sampling to limit count if specified or if default matches.
    """
    import yaml
    import random
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    if not data:
        return []
        
    # Check if this is the SciQ YAML format (a list of dicts with a "question" key)
    is_sciq = False
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict) and "question" in data[0]:
            is_sciq = True
            
    if is_sciq:
        # Convert SciQ format to standard benchmark cases
        converted = []
        for idx, item in enumerate(data):
            q_data = item["question"]
            q_id = q_data.get("id", idx + 1)
            converted.append({
                "id": f"sciq_{q_id}",
                "category": "sciq",
                "query": q_data["q"],
                "golden_answer": q_data["a"],
                "c": q_data["c"]  # temporary context string storage
            })
            
        # Resolve context IDs stably
        unique_contexts = []
        context_to_id = {}
        for c in [case["c"] for case in converted]:
            if c not in context_to_id:
                unique_contexts.append(c)
                context_to_id[c] = len(unique_contexts)
                
        for case in converted:
            c_id = context_to_id[case["c"]]
            case["expected_papers"] = [f"sciq_paper_{c_id}"]
            del case["c"]
            
        data = converted
        
        # Default limit is 50 for SciQ when not specified
        # If limit is explicitly -1, it means "no limit/run all"
        if limit is None:
            limit = DEFAULT_LIMIT
        elif limit == -1:
            limit = None
            
    # Apply limit if requested and dataset is larger
    if limit is not None and limit != -1 and len(data) > limit:
        rng = random.Random(seed)
        data = rng.sample(data, limit)
        try:
            data.sort(key=lambda x: x.get("id", ""))
        except Exception:
            pass
            
    return data
