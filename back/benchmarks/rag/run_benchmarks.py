"""
Science Graph — RAG Quality Benchmarking Runner.
Runs a golden dataset against 7 baseline configurations (B0 to B6)
and outputs a single, copy-pasteable YAML file for browser-based AI evaluation.
"""

import sys
import time
import argparse
from pathlib import Path
import yaml

# Set up python path to resolve src imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.container import container
from src.services.rag_service import RAGService
from src.services.graph_expander import ExperimentalGraphExpander
from src.config import config
from src import console as con

# Map of baselines and their descriptions
BASELINES_INFO = {
    "B0": "Zero-Shot (Чистая генерация) — оценка базовых знаний LLM без контекста.",
    "B1": "Pure Lexical (Только лексика) — поиск строго по ключевым словам через SQLite FTS5.",
    "B2": "Pure Dense (Только векторы) — классический семантический поиск по эмбеддингам.",
    "B3": "Dense + HyDE (Векторы + Гипотетический документ) — семантический поиск с гипотетическим ответом.",
    "B4": "Standard Hybrid (Базовый гибрид) — связка FTS5 + Векторы через RRF без графов.",
    "B5": "Hybrid + Graph (Базовый Граф-RAG) — гибридный поиск + статический обход графа (без реранкера/LLM-расширения).",
    "B6": "Full Pipeline (Максимальный запуск) — включены все 13 компонентов (граф, реранкер, LLM-расширение, HyDE и др.)."
}

def get_baseline_config(baseline: str) -> dict:
    """Returns the RAG components configuration dictionary for a given baseline."""
    # Start with all False
    components = {k: False for k in config.rag_components.keys()}
    
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
        # Full pipeline has everything enabled
        components = {k: True for k in config.rag_components.keys()}
        
    return components

class BenchmarkStatsCollector:
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.stats = {
            "embedding": {"calls": 0, "time_sec": 0.0},
            "dense_retrieval": {"calls": 0, "time_sec": 0.0},
            "lexical_retrieval": {"calls": 0, "time_sec": 0.0},
            "graph_neighbors": {"calls": 0, "time_sec": 0.0},
            "db_lookups": {"calls": 0, "time_sec": 0.0},
            "reranking": {"calls": 0, "time_sec": 0.0},
            "graph_expansion": {"calls": 0, "time_sec": 0.0},
            "llm_generation": {"calls": 0, "time_sec": 0.0},
            "citation_repair": {"calls": 0, "time_sec": 0.0},
        }
        self.interceptors = []

    def wrap_method(self, obj, method_name, key):
        if not obj or not hasattr(obj, method_name):
            return
        orig_method = getattr(obj, method_name)
        stats = self.stats
        
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            stats[key]["calls"] += 1
            try:
                return orig_method(*args, **kwargs)
            finally:
                stats[key]["time_sec"] += time.perf_counter() - t0
                
        setattr(obj, method_name, wrapper)
        self.interceptors.append((obj, method_name, orig_method))

    def start(self):
        # 1. Embeddings
        if hasattr(self.rag_service, "emb_engine") and self.rag_service.emb_engine is not None:
            self.wrap_method(self.rag_service.emb_engine, "get_embedding", "embedding")
        
        # 2. Dense retrieval
        if hasattr(self.rag_service, "vector_repo") and self.rag_service.vector_repo is not None:
            self.wrap_method(self.rag_service.vector_repo, "search_similar_chunks", "dense_retrieval")
        
        # 3. Lexical retrieval
        if hasattr(self.rag_service, "vector_repo") and self.rag_service.vector_repo is not None:
            self.wrap_method(self.rag_service.vector_repo, "search_text_fts5", "lexical_retrieval")
        
        # 4. Graph neighbors
        if hasattr(self.rag_service, "graph_repo") and self.rag_service.graph_repo is not None:
            self.wrap_method(self.rag_service.graph_repo, "get_neighbors", "graph_neighbors")
        
        # 5. DB lookups
        if hasattr(self.rag_service, "graph_repo") and self.rag_service.graph_repo is not None:
            for db_method in ["get_paper", "get_author", "get_concept", "get_papers_batch"]:
                self.wrap_method(self.rag_service.graph_repo, db_method, "db_lookups")
            
        # 6. Reranking wrapper
        if hasattr(self.rag_service, "_get_reranker"):
            # If reranker is already loaded, wrap it
            if getattr(self.rag_service, "_reranker", None) is not None:
                reranker = self.rag_service._reranker
                if not hasattr(reranker.predict, "__wrapped_by_bench__"):
                    orig_predict = reranker.predict
                    def wrapped_predict(*p_args, **p_kwargs):
                        t0 = time.perf_counter()
                        self.stats["reranking"]["calls"] += 1
                        try:
                            return orig_predict(*p_args, **p_kwargs)
                        finally:
                            self.stats["reranking"]["time_sec"] += time.perf_counter() - t0
                    wrapped_predict.__wrapped_by_bench__ = True
                    reranker.predict = wrapped_predict
                    self.interceptors.append((reranker, "predict", orig_predict))

            orig_get_reranker = self.rag_service._get_reranker
            stats = self.stats
            def wrapped_get_reranker(*args, **kwargs):
                reranker = orig_get_reranker(*args, **kwargs)
                if reranker and not hasattr(reranker.predict, "__wrapped_by_bench__"):
                    orig_predict = reranker.predict
                    def wrapped_predict(*p_args, **p_kwargs):
                        t0 = time.perf_counter()
                        stats["reranking"]["calls"] += 1
                        try:
                            return orig_predict(*p_args, **p_kwargs)
                        finally:
                            stats["reranking"]["time_sec"] += time.perf_counter() - t0
                    wrapped_predict.__wrapped_by_bench__ = True
                    reranker.predict = wrapped_predict
                return reranker
            self.rag_service._get_reranker = wrapped_get_reranker
            self.interceptors.append((self.rag_service, "_get_reranker", orig_get_reranker))
        
        # 7. LLM generation
        if hasattr(self.rag_service, "llm_engine") and self.rag_service.llm_engine is not None:
            self.wrap_method(self.rag_service.llm_engine, "generate_response", "llm_generation")
        
        # 8. Citation repair
        self.wrap_method(self.rag_service, "_validate_and_repair_citations", "citation_repair")
        
        # 9. Expander wrapper
        if getattr(self.rag_service, "expander", None) is not None:
            expander = self.rag_service.expander
            if hasattr(expander, "expand"):
                self.wrap_method(expander, "expand", "graph_expansion")


    def reset(self):
        for key in self.stats:
            self.stats[key]["calls"] = 0
            self.stats[key]["time_sec"] = 0.0

    def stop(self):
        # Restore all methods
        for obj, method_name, orig_method in reversed(self.interceptors):
            setattr(obj, method_name, orig_method)
        self.interceptors.clear()

    def get_metrics(self):
        rounded_components = {}
        total_calls = 0
        for k, v in self.stats.items():
            rounded_components[k] = {
                "calls": v["calls"],
                "time_sec": round(v["time_sec"], 4)
            }
            total_calls += v["calls"]
        return {
            "components": rounded_components,
            "total_io_calls": total_calls
        }

def run_query_on_baseline(
    rag_service: RAGService, 
    query: str, 
    baseline: str, 
    use_cloud: bool
) -> tuple[str, list[str], dict]:
    """Runs a query under a temporary baseline configuration and returns (answer, retrieved_papers, metrics)."""
    
    # Save original configurations
    orig_components = {name: config.is_component_enabled(name) for name in config.rag_components.keys()}
    orig_hyde = config.data["llm"].get("hyde_enabled", False)
    
    # Configure baseline components
    components_settings = get_baseline_config(baseline)
    if "rag_components" not in config.data:
        config.data["rag_components"] = {}
    for k, v in components_settings.items():
        config.data["rag_components"][k] = v
        
    # Toggle global HyDE config in the underlying config dictionary
    config.data["llm"]["hyde_enabled"] = components_settings.get("hyde", False)
    
    # Expander setup: B6 uses advanced expander, B5 uses static neighbor graph relations
    if baseline == "B6":
        try:
            reranker = rag_service._get_reranker()
            rag_service.expander = ExperimentalGraphExpander(
                graph_repo=rag_service.graph_repo,
                vector_repo=rag_service.vector_repo,
                llm_engine=rag_service.llm_engine,
                reranker=reranker
            )
        except Exception as e:
            con.warning(f"Could not load Advanced Expander for B6: {e}. Falling back to static graph.")
            rag_service.expander = None
    else:
        rag_service.expander = None
        
    collector = BenchmarkStatsCollector(rag_service)
    collector.start()
    
    try:
        if baseline == "B0":
            # Zero-shot bypasses RAG retrieval
            prompt = f"Вопрос: {query}\nОтветь на основе своих общих знаний."
            answer = rag_service.llm_engine.generate_response(prompt)
            retrieved_papers = []
        else:
            # RAG pipelines run standard retrieve-and-generate
            # Retrieve relevant chunks first to extract paper IDs
            final_chunks = rag_service.retrieve_relevant_chunks(query, limit=5)
            retrieved_papers = list({chunk.paper_id for chunk, _ in final_chunks})
            
            # Reset collector to only measure the actual ask run
            collector.reset()
            
            if not final_chunks:
                answer = "Информация отсутствует в базе данных."
            else:
                answer = rag_service.ask(query, limit=5)
                
        metrics = collector.get_metrics()
    finally:
        collector.stop()
        # Restore configurations
        for k, v in orig_components.items():
            config.data["rag_components"][k] = v
        config.data["llm"]["hyde_enabled"] = orig_hyde
        rag_service.expander = None
        
    return answer, retrieved_papers, metrics


def merge_evaluation_data(existing_data: dict, new_data: dict) -> dict:
    """
    Merges new evaluation data into existing evaluation data.
    Preserves other baselines for each question.
    """
    if not existing_data or not isinstance(existing_data, dict):
        return new_data
        
    merged = {}
    
    # Merge metadata
    existing_meta = existing_data.get("metadata", {})
    new_meta = new_data.get("metadata", {})
    
    merged_meta = {**existing_meta, **new_meta}
    
    # baselines_evaluated should be the union of both
    existing_baselines = existing_meta.get("baselines_evaluated", [])
    if not isinstance(existing_baselines, list):
        existing_baselines = []
    new_baselines = new_meta.get("baselines_evaluated", [])
    if not isinstance(new_baselines, list):
        new_baselines = []
        
    union_baselines = sorted(list(set(existing_baselines) | set(new_baselines)))
    merged_meta["baselines_evaluated"] = union_baselines
    
    merged["metadata"] = merged_meta
    
    # Merge results
    existing_results = existing_data.get("results", [])
    if not isinstance(existing_results, list):
        existing_results = []
    new_results = new_data.get("results", [])
    
    # Map existing results by question ID for fast lookup
    existing_map = {item.get("id"): item for item in existing_results if item.get("id")}
    
    merged_results = []
    for new_item in new_results:
        new_id = new_item.get("id")
        if new_id in existing_map:
            # Merge baselines
            existing_item = existing_map[new_id]
            merged_item = {**existing_item, **new_item} # new fields overwrite old ones
            
            # Merge baselines dictionary
            existing_baselines_dict = existing_item.get("baselines", {})
            if not isinstance(existing_baselines_dict, dict):
                existing_baselines_dict = {}
            new_baselines_dict = new_item.get("baselines", {})
            
            merged_baselines = {**existing_baselines_dict, **new_baselines_dict}
            merged_item["baselines"] = merged_baselines
            merged_results.append(merged_item)
        else:
            merged_results.append(new_item)
            
    # Also keep any existing results that were NOT in the new run
    new_ids = {item.get("id") for item in new_results if item.get("id")}
    for existing_item in existing_results:
        existing_id = existing_item.get("id")
        if existing_id and existing_id not in new_ids:
            merged_results.append(existing_item)
            
    # Sort merged results by ID if possible
    try:
        merged_results.sort(key=lambda x: x.get("id", ""))
    except Exception:
        pass
        
    merged["results"] = merged_results
    return merged

def main():
    parser = argparse.ArgumentParser(description="Science Graph RAG Baselines Benchmarking Runner")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None,
        help="Path to golden dataset YAML file. Defaults to golden_dataset.yaml or golden_dataset.example.yaml"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="benchmarks/rag/reports/evaluation_results.yaml",
        help="Path to save evaluation output results."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use cloud LLM engine instead of local one."
    )
    parser.add_argument(
        "--baselines", type=str, default="all",
        help="Comma-separated baselines to run (e.g. B0,B2,B6) or 'all'."
    )
    args = parser.parse_args()

    # Determine dataset path
    dataset_path = args.dataset
    if not dataset_path:
        local_dir = Path(__file__).resolve().parent
        dataset_path = local_dir / "golden_dataset.yaml"
        if not dataset_path.exists():
            dataset_path = local_dir / "golden_dataset.example.yaml"
            con.info(f"Using default example dataset: {dataset_path}")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        con.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = yaml.safe_load(f)

    if not test_cases:
        con.error("Empty or invalid dataset file.")
        sys.exit(1)

    # Initialize RAG Service
    con.info("Initializing repositories and models...")
    try:
        rag_service = container.get_rag_service(use_cloud=args.cloud)
    except Exception as e:
        con.error(f"Failed to initialize RAG Service: {e}")
        sys.exit(1)

    # Resolve baselines to run
    if args.baselines.lower() == "all":
        baselines_to_run = list(BASELINES_INFO.keys())
    else:
        baselines_to_run = [b.strip().upper() for b in args.baselines.split(",") if b.strip()]

    con.info(f"Running evaluation on {len(test_cases)} cases for baselines: {', '.join(baselines_to_run)}")
    con.blank()

    results = []

    for idx, case in enumerate(test_cases, start=1):
        query = case.get("query")
        case_id = case.get("id", f"Q{idx:02d}")
        con.info(f"[{case_id}] Query: '{query[:60]}...'")
        
        case_result = {
            "id": case_id,
            "category": case.get("category", "general"),
            "query": query,
            "golden_answer": case.get("golden_answer", "").strip(),
            "expected_papers": case.get("expected_papers", []),
            "baselines": {}
        }
        
        for baseline in baselines_to_run:
            description = BASELINES_INFO.get(baseline, "")
            con.dim(f"  Running {baseline}: {description.split('—')[0]}")
            
            t0 = time.perf_counter()
            try:
                answer, retrieved, metrics = run_query_on_baseline(
                    rag_service, query, baseline, use_cloud=args.cloud
                )
                status = "success"
            except Exception as e:
                answer = f"Error occurred during generation: {e}"
                retrieved = []
                metrics = {
                    "components": {
                        k: {"calls": 0, "time_sec": 0.0}
                        for k in [
                            "embedding", "dense_retrieval", "lexical_retrieval",
                            "graph_neighbors", "db_lookups", "reranking",
                            "graph_expansion", "llm_generation", "citation_repair"
                        ]
                    },
                    "total_io_calls": 0
                }
                status = "error"
                con.error(f"    Baseline {baseline} failed: {e}")
                
            elapsed = time.perf_counter() - t0
            
            case_result["baselines"][baseline] = {
                "status": status,
                "latency_sec": round(elapsed, 3),
                "retrieved_papers": retrieved,
                "baseline_config": get_baseline_config(baseline),
                "metrics": metrics,
                "generated_answer": answer.strip()
            }
            
        results.append(case_result)
        con.success(f"[{case_id}] Completed.")
        con.blank()

    # Save output results to YAML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing file if it exists for merging
    existing_data = None
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = yaml.safe_load(f)
                if isinstance(existing_data, list):
                    # Handle legacy list structure
                    existing_data = {"metadata": {}, "results": existing_data}
        except Exception as e:
            con.warning(f"Could not load existing evaluation results for merging: {e}")
    
    from datetime import datetime
    llm_provider = config.data["llm"]["provider"]
    if args.cloud:
        llm_model = config.data["llm"]["cloud"]["model_name"]
        llm_provider_detail = f"cloud ({config.data['llm']['cloud'].get('provider', 'openai')})"
    else:
        llm_model = config.data["llm"]["local"]["model_path"]
        llm_provider_detail = f"local ({llm_provider})"
        
    embedding_model = config.data["embedding"]["model_name"]
    reranker_model = "mixedbread-ai/mxbai-rerank-xsmall-v1" if config.data["rag_components"].get("reranker", True) else "disabled"
    
    output_data = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm": {
                "provider": llm_provider_detail,
                "model_name": llm_model,
                "temperature": config.data["llm"].get("temp", 0.1),
                "max_tokens": config.data["llm"].get("max_tokens", 1000)
            },
            "embeddings": {
                "model_name": embedding_model
            },
            "reranker": {
                "model_name": reranker_model
            },
            "baselines_evaluated": baselines_to_run
        },
        "results": results
    }
    
    # Merge if existing data exists
    if existing_data:
        output_data = merge_evaluation_data(existing_data, output_data)
        
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    con.success(f"Benchmarking complete! Results saved to: {output_path.resolve()}")
    con.info("You can copy fragments of this file and feed them into your browser AI to analyze truthfulness and quality.")

if __name__ == "__main__":
    main()
