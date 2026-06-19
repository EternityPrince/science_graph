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
