import sys
import time
import asyncio
import yaml
import threading
from pathlib import Path
from datetime import datetime

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from src import console as con
from core.metrics import (
    normalize_optional_text,
    get_is_answerable,
    detect_abstention,
    classify_answerability
)

thread_lock = threading.Lock()

def safe_read_modify_write_yaml(file_path: Path, modify_fn):
    import yaml
    
    def run_locked():
        existing_data = None
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f)
            except Exception:
                existing_data = None
                
        new_data = modify_fn(existing_data)
        
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            yaml.dump(new_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        temp_path.replace(file_path)

    if HAS_FCNTL:
        lock_file_path = file_path.with_suffix(file_path.suffix + ".lock")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                run_locked()
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
    else:
        with thread_lock:
            run_locked()


def save_generation_baseline_result(file_path: Path, case_id: str, case_info: dict, baseline_name: str, baseline_data: dict, metadata: dict):
    def modify_fn(existing_data):
        if not existing_data or not isinstance(existing_data, dict):
            existing_data = {"metadata": metadata, "results": []}
        else:
            existing_data["metadata"] = {**existing_data.get("metadata", {}), **metadata}
            
        results = existing_data.get("results", [])
        case_item = next((r for r in results if r.get("id") == case_id), None)
        if not case_item:
            case_item = {
                "id": case_id,
                "category": case_info.get("category", "general"),
                "query": case_info.get("query"),
                "golden_answer": normalize_optional_text(case_info.get("golden_answer")),
                "expected_papers": case_info.get("expected_papers", []),
                "is_answerable": get_is_answerable(case_info),
                "baselines": {}
            }
            results.append(case_item)
            
        case_item["baselines"][baseline_name] = baseline_data
        existing_data["results"] = results
        return existing_data
        
    safe_read_modify_write_yaml(file_path, modify_fn)


def save_evaluation_baseline_result(file_path: Path, case_id: str, case_info: dict, baseline_name: str, baseline_data: dict, eval_metrics_raw: dict, metadata: dict):
    eval_metrics = {k: v for k, v in eval_metrics_raw.items() if k != "eval_details"}
    eval_details = eval_metrics_raw.get("eval_details", {})

    def modify_fn(existing_data):
        if not existing_data or not isinstance(existing_data, dict):
            existing_data = {"metadata": metadata, "results": []}
        else:
            existing_data["metadata"] = {**existing_data.get("metadata", {}), **metadata}
            
        results = existing_data.get("results", [])
        case_item = next((r for r in results if r.get("id") == case_id), None)
        if not case_item:
            case_item = {
                "id": case_id,
                "category": case_info.get("category", "general"),
                "query": case_info.get("query"),
                "golden_answer": normalize_optional_text(case_info.get("golden_answer")),
                "expected_papers": case_info.get("expected_papers", []),
                "is_answerable": get_is_answerable(case_info),
                "baselines": {}
            }
            results.append(case_item)
            
        case_item["baselines"][baseline_name] = {
            "status": baseline_data.get("status", "success"),
            "latency_sec": baseline_data.get("latency_sec"),
            "retrieved_papers": baseline_data.get("retrieved_papers", []),
            "eval_metrics": eval_metrics,
            "eval_details": eval_details,
            "generated_answer": baseline_data.get("generated_answer", ""),
            "retrieved_chunks": baseline_data.get("retrieved_chunks", []),
            "context_token": baseline_data.get("context_token"),
            "max_input_token": baseline_data.get("max_input_token"),
            "context_fillness": baseline_data.get("context_fillness"),
            "trace": baseline_data.get("trace")
        }
        existing_data["results"] = results
        
        # Recalculate summary averages
        summary = {}
        summary_stats = {}
        for r in results:
            for b_name, b_val in r.get("baselines", {}).items():
                if b_name not in summary_stats:
                    summary_stats[b_name] = {
                        "latency_sec": [],
                        "retrieval_recall": [],
                        "context_precision": [],
                        "faithfulness": [],
                        "answer_relevance": [],
                        "citation_fidelity": [],
                        "semantic_accuracy": [],
                        "context_fillness": [],
                        "ar_sa_f1": [],
                        "token_output": [],
                        "token_answer": [],
                        "token_reasoning": [],
                    }
                latency = b_val.get("latency_sec")
                if latency is not None:
                    summary_stats[b_name]["latency_sec"].append(latency)
                
                metrics = b_val.get("eval_metrics", {})
                for k, val in metrics.items():
                    if k not in summary_stats[b_name]:
                        continue
                    if b_name == "B0" and k in ("faithfulness", "citation_fidelity", "context_precision"):
                        continue
                    if b_name != "B0" and not b_val.get("retrieved_chunks") and k in ("faithfulness", "citation_fidelity", "context_precision"):
                        continue
                    if k == "ar_sa_f1" and (not r.get("is_answerable", True) or val is None):
                        continue
                    if val is None:
                        continue
                    summary_stats[b_name][k].append(val)
                    
        for b_name, metrics in summary_stats.items():
            summary[b_name] = {}
            for m_name, values in metrics.items():
                numeric = [v for v in values if v is not None]
                if numeric:
                    summary[b_name][f"avg_{m_name}"] = round(sum(numeric) / len(numeric), 4)
                else:
                    summary[b_name][f"avg_{m_name}"] = 0.0
                    
        existing_data["summary"] = summary
        return existing_data
        
    safe_read_modify_write_yaml(file_path, modify_fn)


def generate_baseline_case(
    rag_service,
    config,
    prompts,
    case_id,
    case,
    baseline,
    args,
    pre_contexts
):
    from core.generation import run_query_on_baseline, get_baseline_config
    from core.metrics import calculate_retrieval_recall, calculate_context_precision
    
    query = case.get("query")
    trace = None
    if pre_contexts:
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
            raw_response = ""
        else:
            retrieved = pre_baseline.get("retrieved_papers", []),
            retrieved = retrieved[0] if isinstance(retrieved, tuple) else retrieved
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
                
                shannon_enabled = baseline_config.get("shannon_estimator_enabled", True)
                t_gen_start = time.perf_counter()
                if shannon_enabled:
                    from core.generation import _generate_with_logits_safe
                    from core.shannon_estimator import (
                        compute_rank_entropy,
                        compute_lexical_entropy,
                        compute_generation_entropy,
                        compute_citation_entropy,
                        compute_entropy_reduction,
                    )
                    raw_response, tokens_info = _generate_with_logits_safe(rag_service.llm_engine, prompt)
                    answer = raw_response
                    h_gen = compute_generation_entropy(tokens_info)
                    h_cit, n_cit = compute_citation_entropy(tokens_info, raw_response)
                else:
                    raw_response = rag_service.llm_engine.generate_response(prompt)
                    answer = raw_response
                    h_gen = 0.0
                    h_cit = 0.0
                    n_cit = 0

                gen_latency = time.perf_counter() - t_gen_start
                
                try:
                    prompt_tokens = rag_service.llm_engine.count_tokens(prompt)
                except Exception:
                    prompt_tokens = len(prompt) // 4

                t_repair_start = time.perf_counter()
                if baseline_config.get("citation_repair", True) and baseline != "B0":
                    from src.models import Chunk
                    chunk_objs = [
                        Chunk(
                            id=ch["id"],
                            paper_id=ch["paper_id"],
                            text_content=ch["text_content"],
                            page_number=ch["page_number"]
                        )
                        for ch in chunks
                    ]
                    try:
                        answer = rag_service._validate_and_repair_citations(answer, chunk_objs)
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(f"Citation repair failed: {e}")
                repair_latency = time.perf_counter() - t_repair_start if baseline_config.get("citation_repair", True) and baseline != "B0" else 0.0

                status = "success"

                # Merge metrics
                metrics = pre_metrics
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

                if shannon_enabled:
                    if baseline == "B0":
                        if not hasattr(rag_service, "_query_b0_h_gen"):
                            rag_service._query_b0_h_gen = {}
                        rag_service._query_b0_h_gen[query] = h_gen
                        shannon_diag = {
                            "h_rank_pre_rerank": 0.0,
                            "h_rank_post_rerank": 0.0,
                            "h_lexical_pre_trim": 0.0,
                            "h_lexical_post_trim": 0.0,
                            "h_graph_relation_type": 0.0,
                            "h_graph_degree": 0.0,
                            "h_gen": round(h_gen, 4),
                            "h_citation": round(h_cit, 4),
                            "n_citation_tokens": n_cit,
                            "delta_h_gen": 0.0,
                        }
                    else:
                        if n_cit == 0 and isinstance(answer, str) and answer != raw_response:
                            h_cit_ans, n_cit_ans = compute_citation_entropy(tokens_info, answer)
                            if n_cit_ans > 0:
                                h_cit, n_cit = h_cit_ans, n_cit_ans
                        post_scores = [c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0) for c in chunks]
                        h_rank_post = compute_rank_entropy(post_scores) if post_scores else 0.0
                        h_lex_post = compute_lexical_entropy(trimmed_text) if trimmed_text else 0.0
                        h_b0 = getattr(rag_service, "_query_b0_h_gen", {}).get(query)
                        delta_h = compute_entropy_reduction(h_b0, h_gen) if h_b0 is not None else 0.0
                        shannon_diag = {
                            "h_rank_pre_rerank": round(h_rank_post, 4),
                            "h_rank_post_rerank": round(h_rank_post, 4),
                            "h_lexical_pre_trim": round(h_lex_post, 4),
                            "h_lexical_post_trim": round(h_lex_post, 4),
                            "h_graph_relation_type": 0.0,
                            "h_graph_degree": 0.0,
                            "h_gen": round(h_gen, 4),
                            "h_citation": round(h_cit, 4),
                            "n_citation_tokens": n_cit,
                            "delta_h_gen": round(delta_h, 4),
                        }
                    metrics["shannon_diagnostics"] = shannon_diag


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
                raw_response = ""
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
            raw_response = ""
        
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
        
    return {
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


async def run_pipelined_stage_async(
    args,
    config,
    run_dir: Path,
    dataset_path: Path,
    baselines_to_run: list,
    eval_results: Path,
    metrics_results: Path,
    retrieved_contexts_file: Path,
    total_steps: int
):
    from rich.progress import ProgressColumn
    from rich.text import Text
    
    class IterationSpeedColumn(ProgressColumn):
        def render(self, task):
            speed = task.finished_speed or task.speed
            if speed is None or speed == 0:
                return Text("- sec/it", style="progress.data.speed")
            sec_per_it = 1.0 / speed
            return Text(f"{sec_per_it:.2f} sec/it", style="progress.data.speed")

    # 1. Initialize RAG service and Cloud Evaluator
    from src.services.container import container
    from src.prompts import prompts
    from core.config import load_benchmark_dataset
    from core.evaluator import CloudEvaluator, get_cloud_credentials, evaluate_baseline_case
    
    con.info("Initializing repositories and models for pipelined run...")
    rag_service = container.get_rag_service(use_cloud=args.cloud, warmup=False)
    if getattr(args, "output", None):
        rag_service.trace_dir = Path(args.output).parent / "traces"
    
    api_key, base_url, model_name = get_cloud_credentials(config)
    con.info(f"Initializing Cloud LLM Evaluator ({model_name}) for pipelined run...")
    evaluator = CloudEvaluator(api_key, base_url, model_name, args.concurrency, args.rpm, args.retries)
    
    script_dir = Path(__file__).resolve().parents[1]
    prompts_path = script_dir / "prompts" / "judge_prompts.yaml"
    if not prompts_path.exists():
        con.error(f"Judge prompts file not found: {prompts_path}")
        sys.exit(1)
        
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_dict = yaml.safe_load(f)
        
    # Load dataset
    test_cases = load_benchmark_dataset(dataset_path, limit=args.limit)
    
    pre_contexts = {}
    if retrieved_contexts_file and retrieved_contexts_file.exists():
        con.info(f"Loading pre-retrieved contexts from {retrieved_contexts_file}...")
        try:
            with open(retrieved_contexts_file, "r", encoding="utf-8") as f:
                cases_list = yaml.safe_load(f)
                if isinstance(cases_list, list):
                    pre_contexts = {c["id"]: c for c in cases_list}
                elif isinstance(cases_list, dict) and "results" in cases_list:
                    pre_contexts = {c["id"]: c for c in cases_list["results"]}
        except Exception as e:
            con.warning(f"Could not load pre-retrieved contexts: {e}")
    
    # Metadata Setup
    llm_provider = config.data["llm"]["provider"]
    if args.cloud:
        cloud_val = getattr(config, "llm_cloud_rag_model_name", None)
        llm_model = cloud_val if isinstance(cloud_val, str) else config.data["llm"]["cloud"]["model_name"]
        llm_provider_detail = f"cloud ({config.data['llm']['cloud'].get('provider', 'openai')})"
    else:
        local_val = getattr(config, "llm_local_rag_model_path", None)
        llm_model = local_val if isinstance(local_val, str) else config.data["llm"]["local"]["model_path"]
        llm_provider_detail = f"local ({llm_provider})"
        
    embedding_model = config.data["embedding"]["model_name"]
    reranker_model = config.reranker_model_name if config.data["rag_components"].get("reranker", True) else "disabled"
    
    gen_metadata = {
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
    }
    
    eval_metadata = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_metadata": gen_metadata,
        "evaluation_llm": {
            "model_name": model_name,
            "provider": base_url
        }
    }
    
    # Existing checkpoints
    existing_generation = {}
    if eval_results.exists() and not getattr(args, "clear_checkpoint", False):
        try:
            with open(eval_results, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "results" in data:
                    for r in data["results"]:
                        existing_generation[r["id"]] = r
        except Exception:
            pass
            
    checkpoint_path = metrics_results.parent / ".eval_checkpoint.json"
    if args.clear_checkpoint and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass
            
    # Load evaluation checkpoint
    from core.evaluator import load_checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    # Queue for pipelined communication
    queue = asyncio.Queue()
    
    # Progress bars
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, finished_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        IterationSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=con.console
    ) as progress:
        
        gen_task = progress.add_task("[cyan]RAG Generation", total=total_steps)
        eval_task = progress.add_task("[magenta]LLM Judge Evaluation", total=total_steps)
        
        async def generator_task():
            for case_idx, case in enumerate(test_cases, start=1):
                query = case.get("query")
                case_id = case.get("id", f"Q{case_idx:02d}")
                expected_papers = case.get("expected_papers", [])
                golden_answer = normalize_optional_text(case.get("golden_answer"))
                
                con.info(f"[{case_id}] Query: '{query[:60]}...'")
                
                for baseline in baselines_to_run:
                    reused = False
                    baseline_data = None
                    
                    if case_id in existing_generation:
                        existing_b = existing_generation[case_id].get("baselines", {}).get(baseline)
                        if existing_b and existing_b.get("status") == "success" and "generated_answer" in existing_b:
                            baseline_data = existing_b
                            reused = True
                            progress.advance(gen_task, 1)
                            con.dim(f"  Reusing previously generated answer for {baseline} from checkpoint.")
                            
                    if not reused:
                        con.dim(f"  Running {baseline}...")
                        baseline_data = await asyncio.to_thread(
                            generate_baseline_case,
                            rag_service,
                            config,
                            prompts,
                            case_id,
                            case,
                            baseline,
                            args,
                            pre_contexts
                        )
                        progress.advance(gen_task, 1)
                        
                        save_generation_baseline_result(
                            eval_results,
                            case_id,
                            case,
                            baseline,
                            baseline_data,
                            gen_metadata
                        )
                        
                    await queue.put((
                        case_id,
                        baseline,
                        query,
                        golden_answer,
                        expected_papers,
                        baseline_data,
                        case
                    ))
                con.success(f"[{case_id}] Completed generation.")
            
            for _ in range(args.concurrency):
                await queue.put(None)
                
        async def evaluator_worker():
            max_tokens_val = getattr(config, "llm_model_max_context", 4096)
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                    
                case_id, baseline, query, golden_answer, expected_papers, baseline_data, case_info = item
                
                try:
                    eval_metrics = await evaluate_baseline_case(
                        evaluator,
                        prompts_dict,
                        case_id,
                        query,
                        golden_answer,
                        expected_papers,
                        baseline,
                        baseline_data,
                        checkpoint_data,
                        checkpoint_path,
                        max_input_token=max_tokens_val,
                        is_answerable=case_info.get("is_answerable", True)
                    )
                    
                    save_evaluation_baseline_result(
                        metrics_results,
                        case_id,
                        case_info,
                        baseline,
                        baseline_data,
                        eval_metrics,
                        eval_metadata
                    )
                except Exception as e:
                    con.error(f"Error evaluating {case_id} [{baseline}]: {e}")
                finally:
                    progress.advance(eval_task, 1)
                    queue.task_done()

        eval_workers = [asyncio.create_task(evaluator_worker()) for _ in range(args.concurrency)]
        await generator_task()
        await queue.join()
        await asyncio.gather(*eval_workers)
        
    try:
        from core.reporting import save_judge_report, save_individual_judge_reports
        with open(metrics_results, "r", encoding="utf-8") as f:
            final_eval_data = yaml.safe_load(f)
        
        judge_output_path = eval_results.with_name(eval_results.stem + "_judge" + eval_results.suffix)
        save_judge_report(final_eval_data, judge_output_path)
        save_individual_judge_reports(final_eval_data, eval_results.parent, eval_results.stem, eval_results.suffix)
    except Exception as e:
        con.warning(f"Could not save judge reports: {e}")
        
    try:
        rag_service.llm_engine.unload_model()
    except Exception:
        pass
        
    con.success("Pipelined RAG Generation and Evaluation complete!")
