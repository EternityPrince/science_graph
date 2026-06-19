import statistics
from typing import Dict, List, Any, Tuple
from core.config import BASELINES_INFO
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

ALL_METRICS = QUALITY_METRICS + ["latency_sec"]

METRIC_LABELS = {
    "retrieval_recall": "Retrieval Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "citation_fidelity": "Citation Fidelity",
    "semantic_accuracy": "Semantic Accuracy",
    "context_fillness": "Context Fillness",
    "latency_sec": "Latency (sec)"
}


def analyze_metrics(data: Any) -> dict:
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
                gen = (b_data.get("generated_answer") or "").strip()
                if gold and gen:
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
    max_input_token = original_metadata.get("llm", {}).get("max_tokens", 10000)

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
    
    for r in results:
        q_id = r.get("id", "UNKNOWN")
        q_text = r.get("query", "")
        category = r.get("category", "default")
        categories.add(category)
        
        if category not in category_values:
            category_values[category] = {b: {m: [] for m in ALL_METRICS} for b in baselines}
            
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

    return {
        "baselines": baselines,
        "summary": summary_stats,
        "categories": sorted(list(categories)),
        "category_stats": category_stats,
        "query_difficulty": query_scores,
        "pairwise_win_rates": pairwise_win_rates,
        "total_queries": len(results)
    }
