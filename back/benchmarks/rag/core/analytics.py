import statistics
from typing import Any
from core.metrics import (
    calculate_retrieval_recall,
    calculate_context_precision,
    estimate_prompt_tokens,
    calculate_semantic_accuracy
)
from core.models import parse_report, ReportOutput

QUALITY_METRICS = [
    "retrieval_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
    "citation_fidelity",
    "semantic_accuracy",
    "context_fillness"
]

ALL_METRICS = QUALITY_METRICS + ["latency_sec", "token_output", "token_answer", "token_reasoning"]

METRIC_LABELS = {
    "retrieval_recall": "Retrieval Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "citation_fidelity": "Citation Fidelity",
    "semantic_accuracy": "Semantic Accuracy",
    "context_fillness": "Context Fillness",
    "latency_sec": "Latency (sec)",
    "token_output": "Token Output",
    "token_answer": "Token Answer",
    "token_reasoning": "Token Reasoning"
}


DETAIL_GRAPH_FIELDS = {
    "graph_retrieval_enabled": (False, bool),
    "graph_retrieval_skip_reason": ("", str),
    
    # Concept diagnostics
    "query_concepts_all_count": (0, int),
    "query_concepts_strong_count": (0, int),
    "query_concepts_dropped_count": (0, int),
    "query_concepts_all": ([], list),
    "query_concepts_strong": ([], list),
    "query_concepts_dropped": ([], list),
    
    # Graph neighbor resolution
    "graph_neighbor_nodes_total": (0, int),
    "graph_neighbor_paper_nodes_count": (0, int),
    "graph_neighbor_local_papers_count": (0, int),
    "graph_neighbor_papers_with_chunks_count": (0, int),
    "graph_neighbor_placeholder_or_external_count": (0, int),
    "graph_neighbor_non_paper_nodes_count": (0, int),
    "graph_neighbor_chunks_retrieved_count": (0, int),
    
    # Graph candidate counts
    "graph_concept_candidate_papers_count": (0, int),
    "graph_bridge_candidate_papers_count": (0, int),
    "graph_chunks_before_rerank_count": (0, int),
    "graph_chunk_candidates_count": (0, int),
    "graph_candidate_source_breakdown": ({}, dict),
    
    # Reranker integration
    "base_candidates_count": (0, int),
    "merged_candidates_count_before_reranker": (0, int),
    "reranker_input_count_before_limit": (0, int),
    "reranker_input_count_after_limit": (0, int),
    "candidate_count_after_reranker": (0, int),
    
    # Graph survival
    "graph_candidate_rerank_positions": ([], list),
    "best_graph_candidate_rank_after_rerank": (None, int),
    "graph_chunks_survived_final_context_count": (0, int),
    "graph_survival_rate": (0.0, float),
    "graph_chunks_survived_final_context": ([], list),
    
    # Final context diversity
    "distinct_papers_in_final_context": (0, int),
    
    # Optional samples/debug lists
    "graph_neighbor_resolution_sample": ([], list),
    "graph_concept_candidate_papers": ([], list),
    "graph_bridge_candidate_papers": ([], list),
    "graph_chunks_before_rerank": ([], list),
}


def analyze_metrics(data: Any, trace_map: dict = None) -> dict:
    """Computes all summary statistics, wins, breakdowns, and difficulties from YAML data."""
    # Parsed with Pydantic for strict validation and format unification
    if isinstance(data, ReportOutput):
        report = data
    else:
        report = parse_report(data)
    
    # Mutate a dictionary format of the report to keep compatibility and support in-place mutations
    data_dict = report.model_dump()
    results = data_dict.get("results", [])
    if not results:
        raise ValueError("No results found in the benchmarking data.")
        
    # Find all baselines present in the first result
    first_result = results[0]
    baselines = list(first_result.get("baselines", {}).keys())
    if not baselines:
        for r in results:
            if r.get("baselines"):
                baselines = list(r["baselines"].keys())
                break
    baselines = sorted(baselines)

    # 0. Merge trace entries if trace_map is provided (or fallback to defaults)
    for r in results:
        q_id = r.get("id")
        query = r.get("query")
        for b in baselines:
            b_data = r.setdefault("baselines", {}).setdefault(b, {})
            
            trace_entry = None
            if trace_map:
                if q_id:
                    trace_entry = trace_map.get((b, str(q_id)))
                if not trace_entry and query:
                    trace_entry = trace_map.get((b, str(query)))
            
            for field, (default_val, _) in DETAIL_GRAPH_FIELDS.items():
                val = None
                if trace_entry:
                    if field == "query_concepts_all_count":
                        val = len(trace_entry.get("query_concepts_all", []))
                    elif field == "query_concepts_strong_count":
                        val = len(trace_entry.get("query_concepts_strong", []))
                    elif field == "query_concepts_dropped_count":
                        val = len(trace_entry.get("query_concepts_dropped", []))
                    else:
                        val = trace_entry.get(field)
                
                if val is None:
                    if field == "query_concepts_all_count" and trace_entry:
                        val = len(trace_entry.get("query_concepts_all", []))
                    elif field == "query_concepts_strong_count" and trace_entry:
                        val = len(trace_entry.get("query_concepts_strong", []))
                    elif field == "query_concepts_dropped_count" and trace_entry:
                        val = len(trace_entry.get("query_concepts_dropped", []))
                    else:
                        val = default_val
                        
                b_data[field] = val
    
    # Pre-calculate missing semantic accuracies for all results/baselines
    missing_semantics = []
    golden_list = []
    generated_list = []
    for r_idx, r in enumerate(results):
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            eval_metrics = b_data.get("eval_metrics", {})
            sem = None
            if isinstance(eval_metrics, dict):
                sem = eval_metrics.get("semantic_accuracy")
            if sem is None:
                sem = b_data.get("semantic_accuracy")
            if sem is None:
                gold = (r.get("golden_answer") or "").strip()
                gen_raw = (b_data.get("generated_answer") or "").strip()
                if gold and gen_raw:
                    from core.sanitization import extract_clean_answer
                    _, gen = extract_clean_answer(gen_raw)
                    golden_list.append(gold)
                    generated_list.append(gen)
                    missing_semantics.append((r_idx, b))
                
    if missing_semantics:
        computed_sems = calculate_semantic_accuracy(golden_list, generated_list)
        for (r_idx, b), val in zip(missing_semantics, computed_sems):
            b_data = results[r_idx]["baselines"][b]
            if "eval_metrics" not in b_data or not isinstance(b_data["eval_metrics"], dict):
                b_data["eval_metrics"] = {}
            b_data["eval_metrics"]["semantic_accuracy"] = val

    metadata = data_dict.get("metadata") or {}
    original_metadata = metadata.get("original_metadata") or metadata
    max_input_token = original_metadata.get("llm", {}).get("model_max_context")
    if max_input_token is None:
        try:
            from src.config import config
            max_input_token = getattr(config, "llm_model_max_context", 4096)
        except Exception:
            max_input_token = 4096

    # Fill in other deterministic metrics if missing
    for r in results:
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            if "eval_metrics" not in b_data or not isinstance(b_data["eval_metrics"], dict):
                b_data["eval_metrics"] = {}
            eval_metrics = b_data["eval_metrics"]
            
            # 1. retrieval_recall
            if eval_metrics.get("retrieval_recall") is None:
                rec = b_data.get("retrieval_recall")
                if rec is None:
                    rec = calculate_retrieval_recall(
                        r.get("expected_papers") or [], 
                        b_data.get("retrieved_papers") or []
                    )
                eval_metrics["retrieval_recall"] = rec
                
            # 2. context_precision
            if eval_metrics.get("context_precision") is None:
                prec = b_data.get("context_precision")
                if prec is None:
                    prec = calculate_context_precision(
                        r.get("expected_papers") or [], 
                        b_data.get("retrieved_chunks") or []
                    )
                eval_metrics["context_precision"] = prec
                
            # 3. context_fillness
            if eval_metrics.get("context_fillness") is None:
                fillness = b_data.get("context_fillness")
                if fillness is None:
                    context_token = b_data.get("context_token")
                    max_input_token_val = b_data.get("max_input_token")
                    if context_token is None:
                        context_token = estimate_prompt_tokens(
                            r.get("query") or "", 
                            b_data.get("retrieved_chunks") or [], 
                            b
                        )
                    if max_input_token_val is None:
                        max_input_token_val = max_input_token
                    fillness = round(context_token / max_input_token_val, 4) if max_input_token_val > 0 else 0.0
                    fillness = min(max(fillness, 0.0), 1.0)
                eval_metrics["context_fillness"] = fillness

            # 4. token_output, token_answer, token_reasoning
            if eval_metrics.get("token_output") is None:
                generated_answer = b_data.get("generated_answer", "")
                from core.evaluator import get_clean_judge_answer
                judge_answer = get_clean_judge_answer(generated_answer)
                from core.metrics import count_text_tokens
                token_output = count_text_tokens(generated_answer)
                token_answer = count_text_tokens(judge_answer)
                token_reasoning = max(0, token_output - token_answer)
                eval_metrics["token_output"] = token_output
                eval_metrics["token_answer"] = token_answer
                eval_metrics["token_reasoning"] = token_reasoning

    # If the input was originally a mutable dict or list, update it in-place to propagate changes
    if isinstance(data, dict):
        if "results" in data:
            data["results"] = results
        else:
            # If it's a dict representing a single case or custom structure
            data.update(data_dict)
    elif isinstance(data, list):
        # If it's a list, update elements in place
        for orig_item, new_item in zip(data, results):
            if isinstance(orig_item, dict) and isinstance(new_item, dict):
                orig_item.update(new_item)

    # Collect raw values per baseline and metric
    raw_values = {b: {m: [] for m in ALL_METRICS} for b in baselines}
    for b in baselines:
        raw_values[b]["status"] = []
    
    categories = set()
    category_values = {}
    query_scores = []

    # Dictionary to collect raw values for graph diagnostics per baseline
    graph_raw = {
        b: {
            "enabled": [],
            "skipped": [],
            "concepts_all": [],
            "concepts_strong": [],
            "concepts_dropped": [],
            "neighbor_nodes": [],
            "neighbor_paper_nodes": [],
            "neighbor_local_papers": [],
            "neighbor_papers_with_chunks": [],
            "neighbor_chunks_retrieved": [],
            "base_candidates": [],
            "graph_chunk_candidates": [],
            "merged_before": [],
            "reranker_before": [],
            "reranker_after": [],
            "candidates_after": [],
            "chunks_before_rerank": [],
            "chunks_survived": [],
            "best_rank": [],
            "neighbor_candidates": [],
            "concept_candidates": [],
            "bridge_candidates": [],
            "distinct_papers_final": []
        }
        for b in baselines
    }
    category_graph_raw = {}
    
    for r in results:
        q_id = r.get("id", "UNKNOWN")
        q_text = r.get("query", "")
        category = r.get("category", "default")
        categories.add(category)
        
        if category not in category_values:
            category_values[category] = {b: {m: [] for m in ALL_METRICS} for b in baselines}

        if category not in category_graph_raw:
            category_graph_raw[category] = {
                b: {
                    "graph_chunk_candidates": [],
                    "chunks_before_rerank": [],
                    "chunks_survived": [],
                    "concepts_strong": [],
                    "distinct_papers_final": []
                }
                for b in baselines
            }
            
        q_quality_sum = 0.0
        q_quality_count = 0
        
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
                
            status = b_data.get("status", "failed")
            raw_values[b]["status"].append(status)
            
            # Latency
            lat = b_data.get("latency_sec")
            if lat is not None:
                raw_values[b]["latency_sec"].append(lat)
                category_values[category][b]["latency_sec"].append(lat)
                
            # Quality metrics
            eval_metrics = b_data.get("eval_metrics", {})
            for m in QUALITY_METRICS:
                val = eval_metrics.get(m)
                if val is not None:
                    raw_values[b][m].append(val)
                    category_values[category][b][m].append(val)
                    q_quality_sum += val
                    q_quality_count += 1

            # Token metrics
            for m in ["token_output", "token_answer", "token_reasoning"]:
                val = eval_metrics.get(m)
                if val is not None:
                    raw_values[b][m].append(val)
                    category_values[category][b][m].append(val)

            # Collect graph retrieval diagnostics
            enabled = b_data.get("graph_retrieval_enabled", False)
            graph_raw[b]["enabled"].append(enabled)
            
            skip_reason = b_data.get("graph_retrieval_skip_reason", "")
            graph_raw[b]["skipped"].append(bool(skip_reason))
            
            graph_raw[b]["concepts_all"].append(b_data.get("query_concepts_all_count", 0))
            graph_raw[b]["concepts_strong"].append(b_data.get("query_concepts_strong_count", 0))
            graph_raw[b]["concepts_dropped"].append(b_data.get("query_concepts_dropped_count", 0))
            
            graph_raw[b]["neighbor_nodes"].append(b_data.get("graph_neighbor_nodes_total", 0))
            graph_raw[b]["neighbor_paper_nodes"].append(b_data.get("graph_neighbor_paper_nodes_count", 0))
            graph_raw[b]["neighbor_local_papers"].append(b_data.get("graph_neighbor_local_papers_count", 0))
            graph_raw[b]["neighbor_papers_with_chunks"].append(b_data.get("graph_neighbor_papers_with_chunks_count", 0))
            graph_raw[b]["neighbor_chunks_retrieved"].append(b_data.get("graph_neighbor_chunks_retrieved_count", 0))
            
            graph_raw[b]["base_candidates"].append(b_data.get("base_candidates_count", 0))
            graph_raw[b]["graph_chunk_candidates"].append(b_data.get("graph_chunk_candidates_count", 0))
            graph_raw[b]["merged_before"].append(b_data.get("merged_candidates_count_before_reranker", 0))
            graph_raw[b]["reranker_before"].append(b_data.get("reranker_input_count_before_limit", 0))
            graph_raw[b]["reranker_after"].append(b_data.get("reranker_input_count_after_limit", 0))
            graph_raw[b]["candidates_after"].append(b_data.get("candidate_count_after_reranker", 0))
            
            graph_raw[b]["chunks_before_rerank"].append(b_data.get("graph_chunks_before_rerank_count", 0))
            graph_raw[b]["chunks_survived"].append(b_data.get("graph_chunks_survived_final_context_count", 0))
            
            best_rank = b_data.get("best_graph_candidate_rank_after_rerank")
            if best_rank is not None:
                try:
                    graph_raw[b]["best_rank"].append(float(best_rank))
                except (ValueError, TypeError):
                    pass
            
            breakdown = b_data.get("graph_candidate_source_breakdown") or {}
            graph_raw[b]["neighbor_candidates"].append(breakdown.get("graph_neighbor", b_data.get("graph_neighbor_chunks_retrieved_count", 0)))
            graph_raw[b]["concept_candidates"].append(breakdown.get("graph_concept_retrieval", b_data.get("graph_concept_candidate_papers_count", 0)))
            graph_raw[b]["bridge_candidates"].append(breakdown.get("graph_bridge_retrieval", b_data.get("graph_bridge_candidate_papers_count", 0)))
            
            graph_raw[b]["distinct_papers_final"].append(b_data.get("distinct_papers_in_final_context", 0))

            # Category raw graph collection
            category_graph_raw[category][b]["graph_chunk_candidates"].append(b_data.get("graph_chunk_candidates_count", 0))
            category_graph_raw[category][b]["chunks_before_rerank"].append(b_data.get("graph_chunks_before_rerank_count", 0))
            category_graph_raw[category][b]["chunks_survived"].append(b_data.get("graph_chunks_survived_final_context_count", 0))
            category_graph_raw[category][b]["concepts_strong"].append(b_data.get("query_concepts_strong_count", 0))
            category_graph_raw[category][b]["distinct_papers_final"].append(b_data.get("distinct_papers_in_final_context", 0))
                    
        # Compute query average quality score to determine difficulty
        if q_quality_count > 0:
            avg_q_score = q_quality_sum / q_quality_count
            query_scores.append({
                "id": q_id,
                "query": q_text,
                "category": category,
                "avg_score": avg_q_score
            })
            
    # Sort queries by difficulty (hardest first, i.e., lowest score)
    query_scores.sort(key=lambda x: x["avg_score"])
    
    # Compute summary statistics
    summary_stats = {}
    for b in baselines:
        summary_stats[b] = {}
        statuses = raw_values[b]["status"]
        success_rate = (statuses.count("success") / len(statuses)) * 100 if statuses else 0.0
        summary_stats[b]["success_rate"] = success_rate
        
        for m in ALL_METRICS:
            vals = raw_values[b][m]
            if not vals:
                summary_stats[b][m] = {
                    "mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0, "count": 0
                }
                continue
            
            mean_val = statistics.mean(vals)
            min_val = min(vals)
            max_val = max(vals)
            med_val = statistics.median(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            
            summary_stats[b][m] = {
                "mean": mean_val,
                "min": min_val,
                "max": max_val,
                "median": med_val,
                "stdev": std_val,
                "count": len(vals)
            }

        # Calculate graph retrieval aggregates
        total_q = len(results)
        enabled_rate = sum(1 for e in graph_raw[b]["enabled"] if e) / total_q if total_q > 0 else 0.0
        skipped_rate = sum(1 for s in graph_raw[b]["skipped"] if s) / total_q if total_q > 0 else 0.0
        
        avg_concepts = statistics.mean(graph_raw[b]["concepts_all"]) if graph_raw[b]["concepts_all"] else 0.0
        avg_strong = statistics.mean(graph_raw[b]["concepts_strong"]) if graph_raw[b]["concepts_strong"] else 0.0
        avg_dropped = statistics.mean(graph_raw[b]["concepts_dropped"]) if graph_raw[b]["concepts_dropped"] else 0.0
        
        avg_neighbor_nodes = statistics.mean(graph_raw[b]["neighbor_nodes"]) if graph_raw[b]["neighbor_nodes"] else 0.0
        avg_neighbor_paper_nodes = statistics.mean(graph_raw[b]["neighbor_paper_nodes"]) if graph_raw[b]["neighbor_paper_nodes"] else 0.0
        avg_neighbor_local_papers = statistics.mean(graph_raw[b]["neighbor_local_papers"]) if graph_raw[b]["neighbor_local_papers"] else 0.0
        avg_neighbor_papers_with_chunks = statistics.mean(graph_raw[b]["neighbor_papers_with_chunks"]) if graph_raw[b]["neighbor_papers_with_chunks"] else 0.0
        avg_neighbor_chunks_retrieved = statistics.mean(graph_raw[b]["neighbor_chunks_retrieved"]) if graph_raw[b]["neighbor_chunks_retrieved"] else 0.0
        
        avg_base_candidates = statistics.mean(graph_raw[b]["base_candidates"]) if graph_raw[b]["base_candidates"] else 0.0
        avg_graph_chunk_candidates = statistics.mean(graph_raw[b]["graph_chunk_candidates"]) if graph_raw[b]["graph_chunk_candidates"] else 0.0
        avg_merged_before = statistics.mean(graph_raw[b]["merged_before"]) if graph_raw[b]["merged_before"] else 0.0
        avg_reranker_before = statistics.mean(graph_raw[b]["reranker_before"]) if graph_raw[b]["reranker_before"] else 0.0
        avg_reranker_after = statistics.mean(graph_raw[b]["reranker_after"]) if graph_raw[b]["reranker_after"] else 0.0
        avg_candidates_after = statistics.mean(graph_raw[b]["candidates_after"]) if graph_raw[b]["candidates_after"] else 0.0
        
        sum_before = sum(graph_raw[b]["chunks_before_rerank"])
        sum_survived = sum(graph_raw[b]["chunks_survived"])
        survival_rate = sum_survived / sum_before if sum_before > 0 else 0.0
        
        queries_with_chunks = sum(1 for c in graph_raw[b]["chunks_before_rerank"] if c > 0) / total_q if total_q > 0 else 0.0
        queries_with_chunks_survived = sum(1 for c in graph_raw[b]["chunks_survived"] if c > 0) / total_q if total_q > 0 else 0.0
        
        avg_best_rank = statistics.mean(graph_raw[b]["best_rank"]) if graph_raw[b]["best_rank"] else None
        
        avg_neighbor_cand = statistics.mean(graph_raw[b]["neighbor_candidates"]) if graph_raw[b]["neighbor_candidates"] else 0.0
        avg_concept_cand = statistics.mean(graph_raw[b]["concept_candidates"]) if graph_raw[b]["concept_candidates"] else 0.0
        avg_bridge_cand = statistics.mean(graph_raw[b]["bridge_candidates"]) if graph_raw[b]["bridge_candidates"] else 0.0
        
        avg_distinct_papers = statistics.mean(graph_raw[b]["distinct_papers_final"]) if graph_raw[b]["distinct_papers_final"] else 0.0
        
        summary_stats[b]["graph_diagnostics"] = {
            "enabled_rate": enabled_rate,
            "skipped_rate": skipped_rate,
            "avg_concepts": avg_concepts,
            "avg_strong": avg_strong,
            "avg_dropped": avg_dropped,
            "avg_neighbor_nodes": avg_neighbor_nodes,
            "avg_neighbor_paper_nodes": avg_neighbor_paper_nodes,
            "avg_neighbor_local_papers": avg_neighbor_local_papers,
            "avg_neighbor_papers_with_chunks": avg_neighbor_papers_with_chunks,
            "avg_neighbor_chunks_retrieved": avg_neighbor_chunks_retrieved,
            "avg_base_candidates": avg_base_candidates,
            "avg_graph_chunk_candidates": avg_graph_chunk_candidates,
            "avg_merged_before": avg_merged_before,
            "avg_reranker_before": avg_reranker_before,
            "avg_reranker_after": avg_reranker_after,
            "avg_candidates_after": avg_candidates_after,
            "survival_rate": survival_rate,
            "queries_with_chunks": queries_with_chunks,
            "queries_with_chunks_survived": queries_with_chunks_survived,
            "avg_best_rank": avg_best_rank,
            "avg_neighbor_cand": avg_neighbor_cand,
            "avg_concept_cand": avg_concept_cand,
            "avg_bridge_cand": avg_bridge_cand,
            "avg_distinct_papers": avg_distinct_papers
        }
            
    # Compute category statistics
    category_stats = {}
    for cat in sorted(categories):
        category_stats[cat] = {}
        for b in baselines:
            category_stats[cat][b] = {}
            for m in ALL_METRICS:
                vals = category_values[cat][b][m]
                if vals:
                    category_stats[cat][b][m] = statistics.mean(vals)
                else:
                    category_stats[cat][b][m] = 0.0

    # Compute category graph statistics
    category_graph_stats = {}
    for cat in sorted(categories):
        category_graph_stats[cat] = {}
        for b in baselines:
            raw_c = category_graph_raw[cat][b]
            avg_chunk_cand = statistics.mean(raw_c["graph_chunk_candidates"]) if raw_c["graph_chunk_candidates"] else 0.0
            
            sum_before_c = sum(raw_c["chunks_before_rerank"])
            sum_survived_c = sum(raw_c["chunks_survived"])
            survival_rate_c = sum_survived_c / sum_before_c if sum_before_c > 0 else 0.0
            
            queries_survived_c = sum(1 for x in raw_c["chunks_survived"] if x > 0) / len(raw_c["chunks_survived"]) if raw_c["chunks_survived"] else 0.0
            
            avg_strong_c = statistics.mean(raw_c["concepts_strong"]) if raw_c["concepts_strong"] else 0.0
            avg_distinct_c = statistics.mean(raw_c["distinct_papers_final"]) if raw_c["distinct_papers_final"] else 0.0
            
            category_graph_stats[cat][b] = {
                "avg_graph_chunk_candidates": avg_chunk_cand,
                "graph_survival_rate": survival_rate_c,
                "queries_with_graph_chunks_survived": queries_survived_c,
                "avg_strong_query_concepts": avg_strong_c,
                "avg_distinct_papers_in_final_context": avg_distinct_c
            }
                    
    # Pairwise Win Rate (how often baseline X > baseline Y for a given metric)
    pairwise_win_rates = {}
    for metric in QUALITY_METRICS + ["latency_sec"]:
        pairwise_win_rates[metric] = {b1: {b2: 0.0 for b2 in baselines} for b1 in baselines}
        
        for b1 in baselines:
            for b2 in baselines:
                if b1 == b2:
                    continue
                wins = 0
                total = 0
                for r in results:
                    val1 = r.get("baselines", {}).get(b1, {}).get("eval_metrics", {}).get(metric) if metric in QUALITY_METRICS else r.get("baselines", {}).get(b1, {}).get(metric)
                    val2 = r.get("baselines", {}).get(b2, {}).get("eval_metrics", {}).get(metric) if metric in QUALITY_METRICS else r.get("baselines", {}).get(b2, {}).get(metric)
                    
                    if val1 is not None and val2 is not None:
                        total += 1
                        if metric == "latency_sec":
                            if val1 < val2:
                                wins += 1
                        else:
                            if val1 > val2:
                                wins += 1
                
                pairwise_win_rates[metric][b1][b2] = (wins / total * 100) if total > 0 else 0.0

    # Collect failure cases
    failure_cases = []
    for r in results:
        q_id = r.get("id", "UNKNOWN")
        q_text = r.get("query", "")
        category = r.get("category", "default")
        
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            
            skip_reason = b_data.get("graph_retrieval_skip_reason", "")
            neighbor_nodes = b_data.get("graph_neighbor_nodes_total", 0)
            papers_with_chunks = b_data.get("graph_neighbor_papers_with_chunks_count", 0)
            chunks_before = b_data.get("graph_chunks_before_rerank_count", 0)
            chunks_survived = b_data.get("graph_chunks_survived_final_context_count", 0)
            
            cond1 = (neighbor_nodes > 0 and papers_with_chunks == 0)
            cond2 = (chunks_before > 0 and chunks_survived == 0)
            cond3 = bool(skip_reason)
            
            if cond1 or cond2 or cond3:
                failure_cases.append({
                    "query_id": q_id,
                    "query": q_text,
                    "baseline": b,
                    "category": category,
                    "skip_reason": skip_reason,
                    "neighbor_nodes": neighbor_nodes,
                    "papers_with_chunks": papers_with_chunks,
                    "chunks_before": chunks_before,
                    "chunks_survived": chunks_survived
                })
                
    def failure_sort_key(item):
        severity = 0
        if item["chunks_before"] > 0 and item["chunks_survived"] == 0:
            severity = 2
        elif item["neighbor_nodes"] > 0 and item["papers_with_chunks"] == 0:
            severity = 1
        return (severity, item["chunks_before"], item["neighbor_nodes"], item["query_id"])

    failure_cases.sort(key=failure_sort_key, reverse=True)
    top_failures = failure_cases[:5]

    has_graph_trace = bool(trace_map)

    return {
        "baselines": baselines,
        "summary": summary_stats,
        "categories": sorted(list(categories)),
        "category_stats": category_stats,
        "query_difficulty": query_scores,
        "pairwise_win_rates": pairwise_win_rates,
        "total_queries": len(results),
        "has_graph_trace": has_graph_trace,
        "category_graph_stats": category_graph_stats,
        "top_failures": top_failures
    }
