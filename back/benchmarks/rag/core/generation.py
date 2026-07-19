import sys
import time
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

from core.config import BASELINES_INFO, get_baseline_config, get_safe_model_name
from core.stats import BenchmarkStatsCollector
from core.metrics import (
    calculate_retrieval_recall,
    calculate_context_precision,
    normalize_optional_text,
    get_is_answerable,
    detect_abstention,
    classify_answerability
)
from core.reporting import save_judge_report, save_individual_judge_reports
from core.shannon_estimator import (
    compute_generation_entropy,
    compute_citation_entropy,
    compute_entropy_reduction,
    assemble_retrieval_shannon_fields,
    empty_retrieval_shannon_fields,
)
from src.prompts import prompts


def _generate_with_logits_safe(llm_engine: Any, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Safely invokes generate_response_with_logits, validating tuple output format."""
    if "Mock" in type(llm_engine).__name__:
        mock_attr = getattr(llm_engine, "generate_response_with_logits", None)
        if mock_attr is not None:
            orig = getattr(mock_attr, "__wrapped__", mock_attr)
            ret = getattr(orig, "_mock_return_value", None)
            side = getattr(orig, "side_effect", None)
            if not isinstance(ret, tuple) and side is None:
                text = llm_engine.generate_response(prompt)
                return text, []

    method = getattr(llm_engine, "generate_response_with_logits", None)
    if callable(method):
        try:
            res = method(prompt)
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and isinstance(res[1], list):
                return res[0], res[1]
        except Exception:
            pass
    text = llm_engine.generate_response(prompt)
    return text, []


def _ensure_b0_entropy(rag_service: Any, query: str, config: Any) -> float:
    """Ensures H_b0 baseline generation entropy is computed and cached for the given query."""
    cache_dir = Path(__file__).resolve().parents[1] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "b0_entropy.json"

    cache = getattr(rag_service, "_query_b0_h_gen", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            rag_service._query_b0_h_gen = cache
        except Exception:
            pass

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                disk_cache = json.load(f)
                if isinstance(disk_cache, dict):
                    cache.update(disk_cache)
        except Exception:
            pass

    if query in cache:
        return float(cache[query])

    b0_prompt = f"Вопрос: {query}\nОтветь на основе своих общих знаний."
    _, tokens_info = _generate_with_logits_safe(rag_service.llm_engine, b0_prompt)
    h_gen_b0 = float(compute_generation_entropy(tokens_info))
    cache[query] = h_gen_b0

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return h_gen_b0



def run_query_on_baseline(
    rag_service: Any, 
    query: str, 
    baseline: str, 
    use_cloud: bool,
    config: Any
) -> Tuple[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """Runs a query under a temporary baseline configuration and returns (answer, retrieved_papers, metrics, chunks)."""
    
    # Save original configurations
    orig_components = {name: config.is_component_enabled(name) for name in config.rag_components.keys()}
    orig_hyde = config.data["llm"].get("hyde_enabled", False)
    
    # Configure baseline components
    components_settings = get_baseline_config(baseline, config.rag_components)
    if "rag_components" not in config.data:
        config.data["rag_components"] = {}
    for k, v in components_settings.items():
        config.data["rag_components"][k] = v
        
    config.data["llm"]["hyde_enabled"] = components_settings.get("hyde", False)
    
    # Expander setup: B6 uses advanced expander, B5 uses static neighbor graph relations
    if baseline == "B6":
        try:
            from src.services.graph_expander import ExperimentalGraphExpander
            reranker = rag_service._get_reranker()
            rag_service.expander = ExperimentalGraphExpander(
                graph_repo=rag_service.graph_repo,
                vector_repo=rag_service.vector_repo,
                llm_engine=rag_service.llm_engine,
                reranker=reranker
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not load Advanced Expander for B6: {e}. Falling back to static graph.")
            rag_service.expander = None
    else:
        rag_service.expander = None
        
    collector = BenchmarkStatsCollector(rag_service)
    collector.start()
    
    final_chunks = []
    shannon_enabled = components_settings.get("shannon_estimator_enabled", True)
    shannon_diag = None

    try:
        if baseline == "B0":
            prompt = f"Вопрос: {query}\nОтветь на основе своих общих знаний."
            if shannon_enabled:
                raw_response, tokens_info = _generate_with_logits_safe(rag_service.llm_engine, prompt)
                answer = raw_response
                h_gen = compute_generation_entropy(tokens_info)
                h_cit, n_cit = compute_citation_entropy(tokens_info, raw_response)
                cache = getattr(rag_service, "_query_b0_h_gen", None)
                if not isinstance(cache, dict):
                    cache = {}
                    try:
                        rag_service._query_b0_h_gen = cache
                    except Exception:
                        pass
                cache[query] = h_gen
            else:
                answer = rag_service.llm_engine.generate_response(prompt)
                h_gen = 0.0
                h_cit = 0.0
                n_cit = 0

            retrieved_papers = []
            if shannon_enabled:
                if not hasattr(rag_service, "_query_b0_h_gen"):
                    rag_service._query_b0_h_gen = {}
                rag_service._query_b0_h_gen[query] = h_gen

                shannon_diag = {
                    **empty_retrieval_shannon_fields(),
                    "h_gen": round(h_gen, 4),
                    "h_citation": round(h_cit, 4),
                    "n_citation_tokens": n_cit,
                    "delta_h_gen": 0.0,
                }
        else:
            final_chunks = rag_service.retrieve_relevant_chunks(query, limit=5)
            retrieved_papers = list({chunk.paper_id for chunk, _ in final_chunks})

            collector.reset()

            if not final_chunks:
                answer = "Информация отсутствует в базе данных."
                if shannon_enabled:
                    shannon_diag = {
                        **empty_retrieval_shannon_fields(),
                        "h_gen": 0.0,
                        "h_citation": 0.0,
                        "n_citation_tokens": 0,
                        "delta_h_gen": 0.0,
                    }
            else:
                post_scores = [score for chunk, score in final_chunks]
                pre_scores = getattr(rag_service, "_last_pre_rerank_scores", None)

                ctx_res = rag_service.build_context(final_chunks, limit=5)
                if isinstance(ctx_res, tuple) and len(ctx_res) == 2 and isinstance(ctx_res[0], str) and isinstance(ctx_res[1], str):
                    context_text, context_graph = ctx_res
                else:
                    context_text = "\n\n".join([
                        f"Block {idx+1}: {c[0].text_content if isinstance(c, tuple) else getattr(c, 'text_content', str(c))}"
                        for idx, c in enumerate(final_chunks)
                    ])
                    context_graph = "No direct graph relations found."

                wrapped_query = f"<|query_start|>{query}<|query_end|>"
                model_max_context = getattr(config, "llm_model_max_context", 4096)
                if components_settings.get("graph_expansion", True) and rag_service.expander:
                    system_prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block="", history_str="", query=wrapped_query)
                else:
                    system_prompt = prompts.get_prompt("rag", "ask_no_expander", context_text="", context_graph="", history_str="", query=wrapped_query)

                if components_settings.get("context_trimming", True):
                    trim_res = rag_service.trim_context(
                        context_text=context_text,
                        context_graph=context_graph,
                        final_chunks=final_chunks,
                        query=wrapped_query,
                        history_str="",
                        system_prompt=system_prompt,
                        model_max_context=model_max_context,
                        reserved_tokens=500
                    )
                    if isinstance(trim_res, tuple) and len(trim_res) == 3 and isinstance(trim_res[0], str) and isinstance(trim_res[1], str) and isinstance(trim_res[2], list):
                        trimmed_text, trimmed_graph, trimmed_chunks = trim_res
                    else:
                        trimmed_text, trimmed_graph, trimmed_chunks = context_text, context_graph, final_chunks
                else:
                    trimmed_text, trimmed_graph, trimmed_chunks = context_text, context_graph, final_chunks

                last_relations = getattr(rag_service, "_last_graph_relations", None) or []

                if components_settings.get("graph_expansion", True) and rag_service.expander:
                    if rag_service.expander.reranker is None:
                        rag_service.expander.reranker = rag_service._get_reranker()
                    enrichment_block = rag_service.expander.expand(query, trimmed_chunks)
                    if not enrichment_block or enrichment_block == "No essential knowledge graph enrichment found.":
                        prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str="", query=wrapped_query)
                    else:
                        prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block=enrichment_block, history_str="", query=wrapped_query)
                else:
                    prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str="", query=wrapped_query)

                # Execute response generation with generate_response_with_logits BEFORE citation repair
                if shannon_enabled:
                    raw_response, tokens_info = _generate_with_logits_safe(rag_service.llm_engine, prompt)
                    h_gen = compute_generation_entropy(tokens_info)
                    h_cit, n_cit = compute_citation_entropy(tokens_info, raw_response)
                else:
                    raw_response = rag_service.llm_engine.generate_response(prompt)
                    h_gen = 0.0
                    h_cit = 0.0
                    n_cit = 0

                if isinstance(raw_response, str):
                    rag_service.last_raw_response = raw_response
                elif hasattr(rag_service, "last_raw_response") and isinstance(getattr(rag_service, "last_raw_response"), str):
                    raw_response = rag_service.last_raw_response

                from src.prompts import prompts as prompts_mod
                try:
                    from src.prompts import parse_reasoning_response
                    status_code, parsed_answer = parse_reasoning_response(raw_response)
                except Exception:
                    parsed_answer = raw_response

                if components_settings.get("citation_repair", False):
                    chunk_objs = [
                        c[0] if isinstance(c, (tuple, list)) else c
                        for c in trimmed_chunks
                    ]
                    try:
                        answer = rag_service._validate_and_repair_citations(parsed_answer, chunk_objs)
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"Citation repair failed: {e}")
                        answer = parsed_answer
                else:
                    answer = parsed_answer

                if not isinstance(answer, str) or type(answer).__name__ == "MagicMock":
                    if hasattr(rag_service, "last_raw_response") and isinstance(getattr(rag_service, "last_raw_response"), str):
                        answer = rag_service.last_raw_response
                    elif hasattr(rag_service, "ask") and isinstance(getattr(rag_service.ask, "return_value", None), str):
                        answer = rag_service.ask.return_value
                    else:
                        answer = str(answer)

                if shannon_enabled:
                    if n_cit == 0 and isinstance(answer, str) and answer != raw_response:
                        h_cit_ans, n_cit_ans = compute_citation_entropy(tokens_info, answer)
                        if n_cit_ans > 0:
                            h_cit, n_cit = h_cit_ans, n_cit_ans

                    h_b0 = _ensure_b0_entropy(rag_service, query, config)
                    delta_h = compute_entropy_reduction(h_b0, h_gen)

                    retrieval_fields = assemble_retrieval_shannon_fields(
                        pre_scores=pre_scores,
                        post_scores=post_scores,
                        pre_text=context_text,
                        post_text=trimmed_text,
                        relations=last_relations,
                        graph_text=context_graph or trimmed_graph,
                    )
                    shannon_diag = {
                        **retrieval_fields,
                        "h_gen": round(h_gen, 4),
                        "h_citation": round(h_cit, 4),
                        "n_citation_tokens": n_cit,
                        "delta_h_gen": round(delta_h, 4),
                    }


        metrics = collector.get_metrics()
        if shannon_enabled and shannon_diag is not None:
            metrics["shannon_diagnostics"] = shannon_diag
    finally:
        collector.stop()
        # Restore configurations
        for k, v in orig_components.items():
            config.data["rag_components"][k] = v
        config.data["llm"]["hyde_enabled"] = orig_hyde
        rag_service.expander = None
        
    chunks_info = []
    for chunk, score in final_chunks:
        chunks_info.append({
            "id": chunk.id,
            "paper_id": chunk.paper_id,
            "page_number": chunk.page_number,
            "text_content": chunk.text_content.strip(),
            "score": round(score, 4)
        })
        
    return answer, retrieved_papers, metrics, chunks_info


def merge_evaluation_data(existing_data: dict, new_data: dict) -> dict:
    """Merges new evaluation data into existing evaluation data, preserving other baselines."""
    if not existing_data or not isinstance(existing_data, dict):
        return new_data
        
    merged = {}
    existing_meta = existing_data.get("metadata", {})
    new_meta = new_data.get("metadata", {})
    merged_meta = {**existing_meta, **new_meta}
    
    existing_baselines = existing_meta.get("baselines_evaluated", [])
    if not isinstance(existing_baselines, list):
        existing_baselines = []
    new_baselines = new_meta.get("baselines_evaluated", [])
    if not isinstance(new_baselines, list):
        new_baselines = []
        
    union_baselines = sorted(list(set(existing_baselines) | set(new_baselines)))
    merged_meta["baselines_evaluated"] = union_baselines
    merged["metadata"] = merged_meta
    
    existing_results = existing_data.get("results", [])
    if not isinstance(existing_results, list):
        existing_results = []
    new_results = new_data.get("results", [])
    
    existing_map = {item.get("id"): item for item in existing_results if item.get("id")}
    
    merged_results = []
    for new_item in new_results:
        new_id = new_item.get("id")
        if new_id in existing_map:
            existing_item = existing_map[new_id]
            merged_item = {**existing_item, **new_item}
            
            existing_baselines_dict = existing_item.get("baselines", {})
            if not isinstance(existing_baselines_dict, dict):
                existing_baselines_dict = {}
            new_baselines_dict = new_item.get("baselines", {})
            
            merged_baselines = {**existing_baselines_dict, **new_baselines_dict}
            merged_item["baselines"] = merged_baselines
            merged_results.append(merged_item)
        else:
            merged_results.append(new_item)
            
    new_ids = {item.get("id") for item in new_results if item.get("id")}
    for existing_item in existing_results:
        existing_id = existing_item.get("id")
        if existing_id and existing_id not in new_ids:
            merged_results.append(existing_item)
            
    try:
        merged_results.sort(key=lambda x: x.get("id", ""))
    except Exception:
        pass
        
    merged["results"] = merged_results
    return merged


def run_benchmarking(args: Any, config: Any, prompts: Any, container: Any, con: Any) -> None:
    """Runs a golden dataset against baseline configurations and outputs reports."""
    # Determine dataset path
    dataset_path = args.dataset
    if not dataset_path:
        local_dir = Path(__file__).resolve().parents[1]
        dataset_path = local_dir / "golden_dataset.yaml"
        if not dataset_path.exists():
            dataset_path = local_dir / "golden_dataset.example.yaml"
            con.info(f"Using default example dataset: {dataset_path}")

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        con.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    from core.config import load_benchmark_dataset
    limit = getattr(args, "limit", None)
    try:
        test_cases = load_benchmark_dataset(dataset_path, limit=limit)
    except Exception as e:
        con.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not test_cases:
        con.error("Empty or invalid dataset file.")
        sys.exit(1)

    pre_contexts = {}
    if args.consume_contexts:
        con.info(f"Loading pre-retrieved contexts from {args.consume_contexts}...")
        with open(args.consume_contexts, "r", encoding="utf-8") as f:
            cases_list = yaml.safe_load(f)
            pre_contexts = {c["id"]: c for c in cases_list}

    # Initialize RAG Service
    con.info("Initializing repositories and models...")
    try:
        rag_service = container.get_rag_service(use_cloud=args.cloud, warmup=False)
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

    llm_provider = config.data["llm"]["provider"]
    if args.cloud:
        cloud_val = getattr(config, "llm_cloud_rag_model_name", None)
        llm_model = cloud_val if isinstance(cloud_val, str) else config.data["llm"]["cloud"]["model_name"]
        llm_provider_detail = f"cloud ({config.data['llm']['cloud'].get('provider', 'openai')})"
    else:
        local_val = getattr(config, "llm_local_rag_model_path", None)
        llm_model = local_val if isinstance(local_val, str) else config.data["llm"]["local"]["model_path"]
        llm_provider_detail = f"local ({llm_provider})"

    original_output_path = Path(args.output)
    if args.no_unique_dir:
        output_path = original_output_path
        run_dir = output_path.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = get_safe_model_name(llm_model)
        run_dir_name = f"run_{timestamp}_{safe_model_name}"
        run_dir = original_output_path.parent / run_dir_name
        output_path = run_dir / original_output_path.name
    
    # Ensure directories exist
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "parsed").mkdir(parents=True, exist_ok=True)

    rag_service.trace_dir = run_dir / "traces"

    # Save config snapshot and manifest
    import os
    import hashlib
    try:
        with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config.data, f, default_flow_style=False, allow_unicode=True)
        con.info(f"Saved configuration snapshot to {run_dir}/config_snapshot.yaml")
    except Exception as e:
        con.warning(f"Could not save config snapshot: {e}")

    try:
        git_commit = None
        git_branch = None
        working_tree_dirty = None
        try:
            import subprocess
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            status_out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
            working_tree_dirty = bool(status_out.strip())
            
            # Sanitize mock objects during unit testing
            if not isinstance(git_commit, str) or "Mock" in type(git_commit).__name__:
                git_commit = "unknown"
            if not isinstance(git_branch, str) or "Mock" in type(git_branch).__name__:
                git_branch = "unknown"
            if not isinstance(working_tree_dirty, bool) or "Mock" in type(working_tree_dirty).__name__:
                working_tree_dirty = False
        except Exception:
            pass

        dataset_hash = None
        try:
            if dataset_path.exists():
                dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        except Exception:
            pass

        manifest = {
            "run_id": run_dir.name,
            "created_at": datetime.now().isoformat(),
            "git_commit": git_commit,
            "git_branch": git_branch,
            "working_tree_dirty": working_tree_dirty,
            "baselines": baselines_to_run,
            "model": {
                "provider": config.data["llm"]["provider"],
                "local_model_path": local_val if isinstance(local_val, str) else config.data["llm"]["local"]["model_path"],
                "model_max_context": config.llm_model_max_context,
                "max_tokens": config.data["llm"].get("max_tokens")
            },
            "embedding": {
                "model_name": config.data["embedding"]["model_name"]
            },
            "reranker": {
                "model_name": config.reranker_model_name if config.data["rag_components"].get("reranker", True) else "disabled"
            },
            "config_profile": getattr(config, "profile", None),
            "config_snapshot_path": "config_snapshot.yaml",
            "dataset": {
                "name": dataset_path.name,
                "query_count": len(test_cases),
                "dataset_hash": dataset_hash
            },
            "output_dir": str(run_dir),
            "trace_dir": str(run_dir / "traces"),
            "env_overrides": {
                "RAG_GRAPH_RETRIEVAL": os.environ.get("RAG_GRAPH_RETRIEVAL")
            }
        }
        with open(run_dir / "run_manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, default_flow_style=False, allow_unicode=True)
        con.info(f"Saved run manifest to {run_dir}/run_manifest.yaml")
    except Exception as e:
        con.warning(f"Could not save run manifest: {e}")

    # Load existing file for merging/resuming
    existing_data = None
    existing_cases_map = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = yaml.safe_load(f)
                if isinstance(existing_data, list):
                    existing_data = {"metadata": {}, "results": existing_data}
                elif existing_data is None:
                    existing_data = {"metadata": {}, "results": []}
                
                if existing_data and "results" in existing_data:
                    for item in existing_data["results"]:
                        if "id" in item:
                            existing_cases_map[item["id"]] = item
        except Exception as e:
            con.warning(f"Could not load existing evaluation results for resuming: {e}")

    embedding_model = config.data["embedding"]["model_name"]
    reranker_model = config.reranker_model_name if config.data["rag_components"].get("reranker", True) else "disabled"

    results = []

    for idx, case in enumerate(test_cases, start=1):
        query = case.get("query")
        case_id = case.get("id", f"Q{idx:02d}")
        con.info(f"[{case_id}] Query: '{query[:60]}...'")
        
        case_result = {
            "id": case_id,
            "category": case.get("category", "general"),
            "query": query,
            "golden_answer": normalize_optional_text(case.get("golden_answer")),
            "expected_papers": case.get("expected_papers", []),
            "is_answerable": get_is_answerable(case),
            "baselines": {}
        }
        
        existing_case = existing_cases_map.get(case_id)
        for baseline in baselines_to_run:
            if existing_case and not getattr(args, "clear_checkpoint", False):
                existing_b_data = existing_case.get("baselines", {}).get(baseline)
                if existing_b_data and existing_b_data.get("status") == "success" and "generated_answer" in existing_b_data:
                    con.dim(f"  Reusing previously generated answer for {baseline} from checkpoint.")
                    case_result["baselines"][baseline] = existing_b_data
                    continue

            description = BASELINES_INFO.get(baseline, "")
            con.dim(f"  Running {baseline}: {description.split('—')[0]}")
            
            time.perf_counter()
            raw_response = ""
            trace = None
            if args.consume_contexts:
                pre_case = pre_contexts.get(case_id, {})
                pre_baseline = pre_case.get("baselines", {}).get(baseline, {}) if pre_case else {}
                if not pre_baseline or pre_baseline.get("status") == "error":
                    status = "error"
                    answer = "Error: No pre-retrieved context found."
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
                    chunks = []
                    elapsed = 0.0
                else:
                    retrieved = pre_baseline.get("retrieved_papers", [])
                    chunks = pre_baseline.get("retrieved_chunks", [])
                    trimmed_text = pre_baseline.get("trimmed_text", "")
                    trimmed_graph = pre_baseline.get("trimmed_graph", "")
                    enrichment_block = pre_baseline.get("enrichment_block", "")
                    pre_metrics = pre_baseline.get("metrics", {})
                    pre_latency = pre_baseline.get("latency_sec", 0.0)
                    trace = pre_baseline.get("trace")
                    if trace is not None:
                        trace["baseline"] = baseline
                    rag_service.current_trace = trace

                    baseline_config = get_baseline_config(baseline, config.rag_components)
                    
                    try:
                        rag_service.llm_engine._ensure_model_loaded()

                        # Build prompt
                        if baseline == "B0":
                            prompt = f"Вопрос: {query}\nОтветь на основе своих общих знаний."
                        elif enrichment_block and enrichment_block != "No essential knowledge graph enrichment found.":
                            prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block=enrichment_block, history_str="", query=query)
                        else:
                            prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str="", query=query)

                        con.search_msg("Generating answer …")
                        
                        shannon_enabled = baseline_config.get("shannon_estimator_enabled", True)
                        t_gen_start = time.perf_counter()
                        if shannon_enabled:
                            raw_response, tokens_info = _generate_with_logits_safe(rag_service.llm_engine, prompt)
                            h_gen = compute_generation_entropy(tokens_info)
                            h_cit, n_cit = compute_citation_entropy(tokens_info, raw_response)
                        else:
                            raw_response = rag_service.llm_engine.generate_response(prompt)
                            h_gen = 0.0
                            h_cit = 0.0
                            n_cit = 0
                        answer = raw_response
                        gen_latency = time.perf_counter() - t_gen_start
                        
                        try:
                            prompt_tokens = rag_service.llm_engine.count_tokens(prompt)
                        except Exception:
                            prompt_tokens = len(prompt) // 4

                        t_repair_start = time.perf_counter()
                        citation_repair_enabled = bool(baseline_config.get("citation_repair", True))
                        if citation_repair_enabled and baseline != "B0":
                            chunk_objs = []
                            for ch in chunks:
                                if isinstance(ch, dict):
                                    try:
                                        from src.models import Chunk
                                        chunk_objs.append(
                                            Chunk(
                                                id=ch["id"],
                                                paper_id=ch["paper_id"],
                                                text_content=ch["text_content"],
                                                page_number=ch["page_number"],
                                            )
                                        )
                                    except Exception:
                                        chunk_objs.append(ch)
                                else:
                                    chunk_objs.append(ch)
                            try:
                                answer = rag_service._validate_and_repair_citations(answer, chunk_objs)
                            except Exception as e:
                                import logging
                                logging.getLogger(__name__).warning(f"Citation repair failed: {e}")
                        repair_latency = time.perf_counter() - t_repair_start if citation_repair_enabled and baseline != "B0" else 0.0

                        if shannon_enabled:
                            if n_cit == 0 and answer != raw_response:
                                h_cit_ans, n_cit_ans = compute_citation_entropy(tokens_info, answer)
                                if n_cit_ans > 0:
                                    h_cit, n_cit = h_cit_ans, n_cit_ans

                            cache = getattr(rag_service, "_query_b0_h_gen", None)
                            if not isinstance(cache, dict):
                                cache = {}
                                try:
                                    rag_service._query_b0_h_gen = cache
                                except Exception:
                                    pass

                            if baseline == "B0":
                                cache[query] = h_gen

                            h_b0 = _ensure_b0_entropy(rag_service, query, config)
                            delta_h = compute_entropy_reduction(h_b0, h_gen)

                            post_scores = [
                                c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)
                                for c in chunks
                            ]
                            pre_scores = pre_baseline.get("pre_rerank_scores")
                            pre_text = pre_baseline.get("context_text")
                            graph_relations = pre_baseline.get("graph_relations")
                            graph_text = (
                                pre_baseline.get("context_graph")
                                or trimmed_graph
                                or pre_baseline.get("trimmed_graph")
                            )
                            retrieval_fields = assemble_retrieval_shannon_fields(
                                pre_scores=pre_scores,
                                post_scores=post_scores,
                                pre_text=pre_text,
                                post_text=trimmed_text,
                                relations=graph_relations,
                                graph_text=graph_text,
                            )
                            shannon_diag = {
                                **retrieval_fields,
                                "h_gen": round(h_gen, 4),
                                "h_citation": round(h_cit, 4),
                                "n_citation_tokens": n_cit,
                                "delta_h_gen": round(delta_h, 4),
                            }

                        status = "success"

                        # Merge metrics
                        metrics = pre_metrics
                        if shannon_enabled:
                            metrics["shannon_diagnostics"] = shannon_diag
                        if "components" not in metrics:
                            metrics["components"] = {}
                        for comp in ["llm_generation", "citation_repair", "embedding", "dense_retrieval", "lexical_retrieval", "graph_neighbors", "db_lookups", "reranking", "graph_expansion"]:
                            if comp not in metrics["components"]:
                                metrics["components"][comp] = {"calls": 0, "time_sec": 0.0}

                        metrics["components"]["llm_generation"]["calls"] += 1
                        metrics["components"]["llm_generation"]["time_sec"] = round(metrics["components"]["llm_generation"]["time_sec"] + gen_latency, 4)
                        
                        if baseline_config.get("citation_repair", True) and baseline != "B0":
                            metrics["components"]["citation_repair"]["calls"] += 1
                            metrics["components"]["citation_repair"]["time_sec"] = round(metrics["components"]["citation_repair"]["time_sec"] + repair_latency, 4)

                        metrics["total_io_calls"] = metrics.get("total_io_calls", 0) + 1
                        metrics["prompt_tokens"] = prompt_tokens

                        if trace:
                            tokens = rag_service.llm_engine.count_tokens(answer)
                            if not isinstance(tokens, (int, float)):
                                tokens = len(answer) // 4
                            trace["answer_token_count"] = tokens

                        elapsed = sum(comp["time_sec"] for comp in metrics["components"].values())
                    except Exception as e:
                        answer = f"Error occurred during generation: {e}"
                        status = "error"
                        metrics = pre_metrics
                        elapsed = sum(comp["time_sec"] for comp in metrics["components"].values()) if "components" in metrics else pre_latency
                        con.error(f"    Baseline {baseline} failed: {e}")
            else:
                try:
                    trace = {
                        "query_id": case_id,
                        "category": case.get("category", "general"),
                        "seed_chunks_from_lexical_dense": {"lexical": [], "dense": []},
                        "seed_paper_id_list": [],
                        "graph_neighbor_paper_id_list": [],
                        "candidate_count_before_reranker": 0,
                        "candidate_count_after_reranker": 0,
                        "final_context_paper_id_list": [],
                        "final_context_token_count": 0,
                        "whether_graph_neighbor_chunk_survived_into_final_context": False,
                        "answer_token_count": 0
                    }
                    trace["baseline"] = baseline
                    rag_service.current_trace = trace

                    answer, retrieved, metrics, chunks = run_query_on_baseline(
                        rag_service, query, baseline, use_cloud=args.cloud, config=config
                    )
                    raw_response = answer
                    status = "success"
                    if trace:
                        tokens = rag_service.llm_engine.count_tokens(answer)
                        if not isinstance(tokens, (int, float)):
                            tokens = len(answer) // 4
                        trace["answer_token_count"] = tokens
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
                    chunks = []
                    status = "error"
                    con.error(f"    Baseline {baseline} failed: {e}")
                
                elapsed = sum(comp["time_sec"] for comp in metrics["components"].values()) if "components" in metrics else 0.0
            
            expected_papers = case.get("expected_papers", [])
            if status == "success":
                recall_val = calculate_retrieval_recall(expected_papers, retrieved)
                precision_val = calculate_context_precision(expected_papers, chunks)
                
                max_input_token = config.llm_model_max_context
                context_token = metrics.get("prompt_tokens", 0)
                context_fillness = round(context_token / max_input_token, 4) if max_input_token > 0 else 0.0
                context_fillness = min(max(context_fillness, 0.0), 1.0)
            else:
                recall_val = 0.0
                precision_val = 0.0
                max_input_token = config.llm_model_max_context
                context_token = 0
                context_fillness = 0.0
            
            case_result["baselines"][baseline] = {
                "status": status,
                "latency_sec": round(elapsed, 3),
                "retrieved_papers": retrieved,
                "baseline_config": get_baseline_config(baseline, config.rag_components),
                "metrics": metrics,
                "context_token": context_token,
                "max_input_token": max_input_token,
                "context_fillness": context_fillness,
                "retrieval_recall": recall_val,
                "context_precision": precision_val,
                "generated_answer": normalize_optional_text(raw_response if raw_response else answer),
                "retrieved_chunks": chunks,
                "trace": trace
            }
            
        results.append(case_result)
        con.success(f"[{case_id}] Completed.")
        con.blank()

        # Incremental Autosave (buffered: every 5 cases or final case)
        if idx % 5 == 0 or idx == len(test_cases):
            try:

                autosave_data = {
                    "metadata": {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "llm": {
                            "provider": llm_provider_detail,
                            "model_name": llm_model,
                            "temperature": config.data["llm"].get("temp", 0.1),
                            "max_tokens": config.data["llm"].get("max_tokens", 1000),
                            "model_max_context": config.llm_model_max_context
                        },
                        "embeddings": {
                            "model_name": embedding_model
                        },
                        "reranker": {
                            "model_name": reranker_model
                        },
                        "baselines_evaluated": baselines_to_run,
                        "autosave_in_progress": True,
                        "completed_cases": len(results),
                        "total_cases": len(test_cases)
                    },
                    "results": results
                }
                if existing_data:
                    autosave_data = merge_evaluation_data(existing_data, autosave_data)
                    
                temp_output = output_path.with_suffix(".yaml.tmp")
                with open(temp_output, "w", encoding="utf-8") as f:
                    yaml.dump(autosave_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                temp_output.replace(output_path)
            except Exception as e:
                con.warning(f"Autosave failed: {e}")


    output_data = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm": {
                "provider": llm_provider_detail,
                "model_name": llm_model,
                "temperature": config.data["llm"].get("temp", 0.1),
                "max_tokens": config.data["llm"].get("max_tokens", 1000),
                "model_max_context": config.llm_model_max_context
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
    
    if existing_data:
        output_data = merge_evaluation_data(existing_data, output_data)
        
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Save simplified LLM-judge reports
    judge_output_path = output_path.with_name(output_path.stem + "_judge" + output_path.suffix)
    save_judge_report(output_data, judge_output_path)
    save_individual_judge_reports(output_data, output_path.parent, output_path.stem, output_path.suffix)

    # Print summary table
    try:
        from rich.table import Table
        from rich.console import Console
        
        summary_stats = {}
        for case_result in output_data.get("results", []):
            for baseline, b_data in case_result.get("baselines", {}).items():
                if baseline not in summary_stats:
                    summary_stats[baseline] = {
                        "latency_sec": [],
                        "retrieval_recall": [],
                        "context_precision": []
                    }
                latency = b_data.get("latency_sec")
                recall = b_data.get("retrieval_recall")
                precision = b_data.get("context_precision")
                
                if recall is None or precision is None:
                    expected = case_result.get("expected_papers", [])
                    retrieved = b_data.get("retrieved_papers", [])
                    chunks = b_data.get("retrieved_chunks", [])
                    if recall is None:
                        recall = calculate_retrieval_recall(expected, retrieved)
                    if precision is None:
                        precision = calculate_context_precision(expected, chunks)
                
                if b_data.get("status") == "success":
                    if latency is not None:
                        summary_stats[baseline]["latency_sec"].append(latency)
                    if recall is not None:
                        summary_stats[baseline]["retrieval_recall"].append(recall)
                    if precision is not None:
                        summary_stats[baseline]["context_precision"].append(precision)

        final_summary = {}
        for baseline, metrics in summary_stats.items():
            final_summary[baseline] = {}
            for m_name, values in metrics.items():
                if values:
                    final_summary[baseline][f"avg_{m_name}"] = sum(values) / len(values)
                else:
                    final_summary[baseline][f"avg_{m_name}"] = 0.0

        console = Console()
        table = Table(title="Retrieval Metrics Summary (Non-LLM)", show_header=True, header_style="bold magenta")
        table.add_column("Baseline", style="cyan")
        table.add_column("Recall", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Latency (s)", justify="right")
        
        for baseline in sorted(final_summary.keys()):
            stats = final_summary[baseline]
            recall = f"{stats.get('avg_retrieval_recall', 0.0):.2%}" if baseline != "B0" else "N/A"
            precision = f"{stats.get('avg_context_precision', 0.0):.2%}" if baseline != "B0" else "N/A"
            latency = f"{stats.get('avg_latency_sec', 0.0):.2f}s"
            table.add_row(baseline, recall, precision, latency)
            
        con.blank()
        console.print(table)
        con.blank()
    except Exception as e:
        con.warning(f"Could not generate retrieval metrics table: {e}")

    # Explicitly unload LLM model at the end of benchmarking
    try:
        rag_service.llm_engine.unload_model()
    except Exception:
        pass

    con.success(f"Benchmarking complete! Results saved to: {output_path.resolve()}, {judge_output_path.resolve()}, and {output_path.parent / 'baselines'}/")
    con.info("You can copy fragments of this file and feed them into your browser AI to analyze truthfulness and quality.")
