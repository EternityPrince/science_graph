#!/usr/bin/env python3
"""
Science Graph — RAG Quality Metrics & Trace Parser and Aggregator.
Parses result_metrics.yaml and trace files inside the run directory.
"""

import sys
import argparse
import csv
import json
import statistics
import yaml
from pathlib import Path

# Increase CSV field size limit to handle large context blocks safely
max_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_limit)
        break
    except OverflowError:
        max_limit = int(max_limit / 2)


# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.analytics import analyze_metrics
from core.reporting import (
    print_rich_tables,
    generate_markdown_report,
    export_wide_csv,
    export_detailed_csv
)
from core.models import load_report_file


def load_graph_retrieval_trace(trace_path: Path) -> dict[tuple[str, str], dict]:
    """
    Robustly loads graph_retrieval_trace.jsonl.
    Returns a dictionary mapping (baseline, query_id) -> trace_entry.
    Falls back to (baseline, query) if query_id is missing.
    In case of duplicate keys, keeps the latest entry.
    """
    trace_map = {}
    if not trace_path.exists():
        return trace_map
        
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception as e:
                    print(f"Warning: Invalid JSON in {trace_path} at line {line_idx}: {e}")
                    continue
                
                baseline = entry.get("baseline", "B6")
                query_id = entry.get("query_id")
                query = entry.get("query")
                
                if query_id:
                    key = (baseline, str(query_id))
                elif query:
                    key = (baseline, str(query))
                else:
                    continue
                
                trace_map[key] = entry
    except Exception as e:
        print(f"Error loading graph retrieval trace: {e}")
        
    return trace_map


def parse_graph_retrieval_trace(traces_dir: Path, parsed_dir: Path):
    """Parses graph_retrieval_trace.jsonl to CSV and summary JSON."""
    trace_file = traces_dir / "graph_retrieval_trace.jsonl"
    if not trace_file.exists():
        trace_file = traces_dir.parent / "graph_retrieval_trace.jsonl"
    if not trace_file.exists():
        print(f"Warning: graph_retrieval_trace.jsonl not found in {traces_dir} or {traces_dir.parent}")
        trace_file = traces_dir.parent / "graph_retrieval_trace.jsonl"
        print(f"Warning: graph_retrieval_trace.jsonl not found in {traces_dir}")
        return None, None
    rows = []
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        print(f"Error reading graph_retrieval_trace.jsonl: {e}")
        return None, None

    if not rows:
        return None, None

    csv_file = parsed_dir / "graph_retrieval_trace.parsed.csv"
    headers = [
        "query_id", "baseline", "category", "query",
        "graph_retrieval_enabled", "graph_retrieval_skip_reason",
        "base_candidates_count", "graph_neighbor_paper_ids_count", "graph_chunk_candidates_count",
        "merged_candidates_count_before_reranker", "reranker_input_count_before_limit",
        "reranker_input_count_after_limit", "candidate_count_after_reranker",
        "graph_chunks_survived_final_context_count", "graph_chunks_survived_final_context",
        "graph_survival_rate", "distinct_papers_in_final_context",
        "base_candidate_paper_ids_count", "graph_chunk_candidate_paper_ids_count",
        "final_context_paper_ids_count"
    ]

    csv_rows = []
    for r in rows:
        base_cand_papers = r.get("base_candidate_paper_ids", [])
        graph_chunk_cand_papers = r.get("graph_chunk_candidate_paper_ids", [])
        final_context_papers = r.get("final_context_paper_ids", [])

        base_candidate_paper_ids_count = len(base_cand_papers) if isinstance(base_cand_papers, list) else 0
        graph_chunk_candidate_paper_ids_count = len(graph_chunk_cand_papers) if isinstance(graph_chunk_cand_papers, list) else 0
        final_context_paper_ids_count = len(final_context_papers) if isinstance(final_context_papers, list) else 0

        survived_context = r.get("graph_chunks_survived_final_context", [])
        survived_str = json.dumps(survived_context) if isinstance(survived_context, list) else str(survived_context)

        csv_rows.append({
            "query_id": r.get("query_id") or (r.get("query_concepts", [""])[0] if r.get("query_concepts") else "UNKNOWN"),
            "baseline": r.get("baseline", "B6"),
            "category": r.get("category", "general"),
            "query": r.get("query", ""),
            "graph_retrieval_enabled": r.get("graph_retrieval_enabled", True),
            "graph_retrieval_skip_reason": r.get("graph_retrieval_skip_reason") or "",
            "base_candidates_count": r.get("base_candidates_count", 0),
            "graph_neighbor_paper_ids_count": r.get("graph_neighbor_paper_ids_count", 0),
            "graph_chunk_candidates_count": r.get("graph_chunk_candidates_count", 0),
            "merged_candidates_count_before_reranker": r.get("merged_candidates_count_before_reranker", 0),
            "reranker_input_count_before_limit": r.get("reranker_input_count_before_limit", 0),
            "reranker_input_count_after_limit": r.get("reranker_input_count_after_limit", 0),
            "candidate_count_after_reranker": r.get("candidate_count_after_reranker", 0),
            "graph_chunks_survived_final_context_count": r.get("graph_chunks_survived_final_context_count", 0),
            "graph_chunks_survived_final_context": survived_str,
            "graph_survival_rate": r.get("graph_survival_rate", 0.0),
            "distinct_papers_in_final_context": r.get("distinct_papers_in_final_context", 0),
            "base_candidate_paper_ids_count": base_candidate_paper_ids_count,
            "graph_chunk_candidate_paper_ids_count": graph_chunk_candidate_paper_ids_count,
            "final_context_paper_ids_count": final_context_paper_ids_count
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for crow in csv_rows:
            writer.writerow(crow)

    total_queries = len(csv_rows)
    enabled_count = sum(1 for r in csv_rows if r["graph_retrieval_enabled"])
    queries_with_neighbors = sum(1 for r in csv_rows if r["graph_neighbor_paper_ids_count"] > 0)
    queries_with_chunks = sum(1 for r in csv_rows if r["graph_chunk_candidates_count"] > 0)
    queries_with_survival = sum(1 for r in csv_rows if r["graph_chunks_survived_final_context_count"] > 0)

    avg_base_candidates = statistics.mean([r["base_candidates_count"] for r in csv_rows]) if csv_rows else 0.0
    avg_graph_neighbors = statistics.mean([r["graph_neighbor_paper_ids_count"] for r in csv_rows]) if csv_rows else 0.0
    avg_graph_chunks = statistics.mean([r["graph_chunk_candidates_count"] for r in csv_rows]) if csv_rows else 0.0
    avg_merged_before = statistics.mean([r["merged_candidates_count_before_reranker"] for r in csv_rows]) if csv_rows else 0.0
    avg_survival_rate = statistics.mean([r["graph_survival_rate"] for r in csv_rows]) if csv_rows else 0.0
    avg_distinct_papers = statistics.mean([r["distinct_papers_in_final_context"] for r in csv_rows]) if csv_rows else 0.0

    skip_reasons = {}
    for r in csv_rows:
        reason = r["graph_retrieval_skip_reason"]
        if not r["graph_retrieval_enabled"]:
            reason = "disabled"
        if not reason:
            continue
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    by_category = {}
    categories = set(r["category"] for r in csv_rows)
    for cat in categories:
        cat_rows = [r for r in csv_rows if r["category"] == cat]
        by_category[cat] = {
            "queries": len(cat_rows),
            "queries_with_graph_survival": sum(1 for r in cat_rows if r["graph_chunks_survived_final_context_count"] > 0),
            "avg_graph_chunk_candidates_count": round(statistics.mean([r["graph_chunk_candidates_count"] for r in cat_rows]), 2) if cat_rows else 0.0,
            "avg_graph_survival_rate": round(statistics.mean([r["graph_survival_rate"] for r in cat_rows]), 4) if cat_rows else 0.0
        }

    summary_data = {
        "total_queries": total_queries,
        "queries_with_graph_retrieval_enabled": enabled_count,
        "queries_with_graph_neighbors": queries_with_neighbors,
        "queries_with_graph_chunks": queries_with_chunks,
        "queries_with_graph_survival": queries_with_survival,
        "avg_base_candidates_count": round(avg_base_candidates, 2),
        "avg_graph_neighbor_paper_ids_count": round(avg_graph_neighbors, 2),
        "avg_graph_chunk_candidates_count": round(avg_graph_chunks, 2),
        "avg_merged_candidates_count_before_reranker": round(avg_merged_before, 2),
        "avg_graph_survival_rate": round(avg_survival_rate, 4),
        "avg_distinct_papers_in_final_context": round(avg_distinct_papers, 2),
        "by_category": by_category,
        "skip_reasons": skip_reasons
    }

    with open(parsed_dir / "graph_retrieval_trace.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    return csv_rows, summary_data


def parse_eval_trace(traces_dir: Path, parsed_dir: Path):
    """Parses eval_trace.jsonl to CSV and summary JSON."""
    trace_file = traces_dir / "eval_trace.jsonl"
    if not trace_file.exists():
        trace_file = traces_dir.parent / "eval_trace.jsonl"
    if not trace_file.exists():
        print(f"Warning: eval_trace.jsonl not found in {traces_dir} or {traces_dir.parent}")
        trace_file = traces_dir.parent / "eval_trace.jsonl"
        print(f"Warning: eval_trace.jsonl not found in {traces_dir}")
        return None, None
    rows = []
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        print(f"Error reading eval_trace.jsonl: {e}")
        return None, None

    if not rows:
        return None, None

    csv_file = parsed_dir / "eval_trace.parsed.csv"
    headers = [
        "query_id", "baseline", "category", "judge_model", "latency_sec",
        "retrieval_recall", "context_precision", "faithfulness",
        "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness",
        "ar_sa_f1", "is_answerable"
        "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness"
    ]

    csv_rows = []
    for r in rows:
        is_ans = r.get("is_answerable")
        if is_ans is None:
            is_ans = True
        else:
            is_ans = str(is_ans).lower() == "true"
            
        ar_f1 = r.get("ar_sa_f1")
        if ar_f1 is None and is_ans:
            r_relevance = r.get("answer_relevance")
            s_accuracy = r.get("semantic_accuracy")
            if r_relevance is not None and s_accuracy is not None:
                try:
                    r_val = float(r_relevance)
                    s_val = float(s_accuracy)
                    if r_val + s_val > 0:
                        ar_f1 = round(2.0 * (r_val * s_val) / (r_val + s_val), 4)
                    else:
                        ar_f1 = 0.0
                except (ValueError, TypeError):
                    ar_f1 = 0.0

        csv_rows.append({
            "query_id": r.get("query_id") or r.get("id"),
            "baseline": r.get("baseline"),
            "category": r.get("category", "general"),
            "judge_model": r.get("judge_model", ""),
            "latency_sec": r.get("latency_sec"),
            "retrieval_recall": r.get("retrieval_recall"),
            "context_precision": r.get("context_precision"),
            "faithfulness": r.get("faithfulness"),
            "answer_relevance": r.get("answer_relevance"),
            "citation_fidelity": r.get("citation_fidelity"),
            "semantic_accuracy": r.get("semantic_accuracy"),
            "context_fillness": r.get("context_fillness"),
            "ar_sa_f1": ar_f1,
            "is_answerable": is_ans,
            "context_fillness": r.get("context_fillness")
        })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for crow in csv_rows:
            writer.writerow(crow)

    total_eval_rows = len(csv_rows)
    queries_evaluated = len(set(r["query_id"] for r in csv_rows))
    baselines = sorted(list(set(r["baseline"] for r in csv_rows if r["baseline"])))

    metrics_summary = {}
    metric_names = [
        "retrieval_recall", "context_precision", "faithfulness",
        "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness",
        "ar_sa_f1"
    ]

    for m in metric_names:
        if m == "ar_sa_f1":
            vals = [r[m] for r in csv_rows if r.get("is_answerable") is True and r.get(m) is not None and r[m] != ""]
        else:
            vals = [r[m] for r in csv_rows if r.get(m) is not None and r[m] != ""]
        "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness"
    

    for m in metric_names:
        vals = [r[m] for r in csv_rows if r.get(m) is not None and r[m] != ""]
        if vals:
            metrics_summary[m] = {
                "mean": round(statistics.mean(vals), 4),
                "count": len(vals)
            }

    summary_data = {
        "total_eval_rows": total_eval_rows,
        "queries_evaluated": queries_evaluated,
        "baselines": baselines,
        "metrics": metrics_summary,
        "errors": {
            "count": 0,
            "by_type": {}
        }
    }

    with open(parsed_dir / "eval_trace.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    return csv_rows, summary_data


def print_confusion_matrix_and_metrics_tables(data):
    """
    Computes and prints confusion matrix and classification quality metrics
    for all baselines based on model refusals and semantic accuracy scores.
    """
    if not data or "results" not in data or not data["results"]:
        print("Warning: No results data available to compute confusion matrix.")
        return

    import re
    import math

    results = data["results"]
    baselines = sorted(list(results[0].get("baselines", {}).keys()))

    def is_refusal(text):
        text_lower = text.lower()
        refusal_keywords = [
            "информация отсутствует", "информации отсутствует", "отсутствует информация", "отсутствуют сведения", 
            "не содержит информации", "не содержит сведений", "не содержат информации", "не содержится информации",
            "не содержится сведений", "нет сведений", "нет информации", "нет данных", "нет описания", "нет упоминания",
            "невозможно ответить", "невозможно определить", "не представляется возможным",
            "отказываемся от", "не может быть описано", "не приводятся", "не упоминаются", "не упоминается",
            "не приводится", "не описывается", "не сообщается", "не указано", "не указана", "не указаны",
            "в предоставленном тексте нет", "в предоставленном контексте нет", "в предоставленных материалах нет",
            "в статье нет", "в тексте не содержится", "в контексте не содержится", "в материалах не содержится",
            "не удается найти", "не удалось найти", "не удается определить", "не удалось определить",
            "нельзя сделать вывод", "нельзя ответить", "не представляется возможным ответить",
            "no information", "does not contain", "cannot answer", "not mention", "do not have",
            "not available", "unable to answer", "insufficient information", "not specify", "not specified",
            "not described", "not defined", "not found", "unanswerable"
        ]
        for kw in refusal_keywords:
            if kw in text_lower:
                return True
                
        if re.search(r"не\s+(?:упоминает|упомяну|приводит|описывает|содержит|находится|обнаружено|указано)", text_lower):
            if any(w in text_lower for w in ["текст", "контекст", "источник", "стать", "материал", "информац", "сведен", "данн"]):
                return True
        if "отсутств" in text_lower:
            if any(w in text_lower for w in ["текст", "контекст", "источник", "стать", "материал", "информац", "сведен", "данн"]):
                return True
        if "нет" in text_lower:
            if re.search(r"\bнет\s+(?:упоминания|описания|сведений|информации|данных|деталей|подробностей|сведений|указания)\b", text_lower):
                return True
        if "не упоминается" in text_lower and any(w in text_lower for w in ["предоставлен", "контекст", "текст", "стать"]):
            return True
            
        return False

    classification = {}
    for b in baselines:
        classification[b] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        
    for r in results:
        is_ans = r.get("is_answerable")
        if is_ans is None:
            is_ans = True
        else:
            is_ans = str(is_ans).lower() == "true"
            
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            gen_ans = b_data.get("generated_answer", "")
            eval_metrics = b_data.get("eval_metrics", {})
            sem_acc = eval_metrics.get("semantic_accuracy", 0.0)
            if sem_acc is None:
                sem_acc = 0.0
                
            ref = is_refusal(gen_ans)
            
            if is_ans:
                if ref:
                    classification[b]["FN"] += 1
                else:
                    if sem_acc > 0.0:
                        classification[b]["TP"] += 1
                    else:
                        classification[b]["FN"] += 1
            else:
                if ref:
                    classification[b]["TN"] += 1
                else:
                    classification[b]["FP"] += 1

    # Print Confusion Matrix Table
    print("\nConfusion Matrix:")
    cm_header = "| {:<10} | {:<18} | {:<19} | {:<18} | {:<18} |"
    cm_row = "| {:<10} | {:<18d} | {:<19d} | {:<18d} | {:<18d} |"
    cm_sep = "+" + "-"*12 + "+" + "-"*20 + "+" + "-"*21 + "+" + "-"*20 + "+" + "-"*20 + "+"
    print(cm_sep)
    print(cm_header.format("Baseline", "True Positive (TP)", "False Positive (FP)", "True Negative (TN)", "False Negative (FN)"))
    print(cm_sep)
    for b in baselines:
        stats_b = classification[b]
        print(cm_row.format(b, stats_b["TP"], stats_b["FP"], stats_b["TN"], stats_b["FN"]))
    print(cm_sep)

    # Print Classification Metrics Table
    print("\nClassification Quality Metrics:")
    metrics_header = "| {:<10} | {:<8} | {:<9} | {:<8} | {:<8} | {:<11} | {:<8} | {:<8} | {:<18} | {:<11} | {:<16} |"
    metrics_row = "| {:<10} | {:<8} | {:<9} | {:<8} | {:<8} | {:<11} | {:<8} | {:<8} | {:<18} | {:<11} | {:<16} |"
    metrics_sep = "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*11 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*20 + "+" + "-"*13 + "+" + "-"*18 + "+"
    print(metrics_sep)
    print(metrics_header.format(
        "Baseline", "Accuracy", "Precision", "Recall", "F1 Score", 
        "Specificity", "FPR", "FNR", "Hallucination Rate", "Ans Rate", "Abstention Rate"
    ))
    print(metrics_sep)

    def format_val(val, is_pct=False):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        if is_pct:
            return f"{val * 100:.1f}%"
        return f"{val:.4f}"

    total_q = len(results)
    for b in baselines:
        stats_b = classification[b]
        tp = stats_b["TP"]
        fp = stats_b["FP"]
        tn = stats_b["TN"]
        fn = stats_b["FN"]
        
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        hallucination_rate = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        answer_rate = (tp + fp) / total_q
        abstention_rate = (tn + fn) / total_q

        print(metrics_row.format(
            b,
            format_val(accuracy, True),
            format_val(precision, True),
            format_val(recall, True),
            format_val(f1, False),
            format_val(specificity, True),
            format_val(fpr, True),
            format_val(fnr, True),
            format_val(hallucination_rate, True),
            format_val(answer_rate, True),
            format_val(abstention_rate, True)
        ))
    print(metrics_sep)


def main():
    parser = argparse.ArgumentParser(description="Parse RAG quality metrics and generate reports")
    parser.add_argument(
        "run_dir", type=str, nargs="?", default=None,
        help="Path to benchmark run directory (e.g. graphs/run_XYZ)."
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to result_metrics.yaml file (backward compatibility)."
    )
    parser.add_argument(
        "--output-md", "-m", type=str, default=None,
        help="Path to save summary markdown report."
    )
    parser.add_argument(
        "--csv-summary", type=str, default=None,
        help="Path to save wide-format summary CSV."
    )
    parser.add_argument(
        "--csv-details", type=str, default=None,
        help="Path to save detailed case-by-case CSV."
    )
    parser.add_argument(
        "--traces-only", action="store_true",
        help="Analyze only trace files and skip result_metrics.yaml."
    )
    parser.add_argument(
        "--include-traces", action="store_true",
        help="Included for CLI compatibility (traces are parsed by default if present)."
    )
    parser.add_argument(
        "--confusion", "--confusion-matrix", action="store_true",
        help="Calculate and print confusion matrix and classification quality metrics."
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    # 1. Resolve run directory
    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (project_root / run_dir).resolve()
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = (project_root / file_path).resolve()
        if file_path.is_dir():
            run_dir = file_path
        else:
            run_dir = file_path.parent
    else:
        # Fallback default: project_root / graphs or reports
        run_dir = project_root / "graphs"
        if not run_dir.exists():
            run_dir = project_root / "reports"

    if not run_dir.exists():
        print(f"Error: run directory does not exist: {run_dir}")
        sys.exit(1)

    print(f"Processing run directory: {run_dir}")

    # Set up outputs
    parsed_dir = run_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = run_dir / "traces"

    # Default output files if not specified
    output_md_path = Path(args.output_md) if args.output_md else run_dir / "metrics_summary.md"
    csv_summary_path = Path(args.csv_summary) if args.csv_summary else run_dir / "metrics_summary.csv"
    csv_details_path = Path(args.csv_details) if args.csv_details else run_dir / "metrics_details.csv"

    # Load trace map first for merging
    trace_path = run_dir / "graph_retrieval_trace.jsonl"
    if not trace_path.exists():
        trace_path = run_dir / "traces" / "graph_retrieval_trace.jsonl"
    trace_map = load_graph_retrieval_trace(trace_path)

    # 2. Parse result_metrics.yaml or the best available benchmark report file
    preferred_files = [
        "result_metrics.yaml",
        "result_metrics_judge.yaml",
        "evaluation_results.yaml",
        "evaluation_results_judge.yaml",
        "retrieved_contexts.yaml"
    ]

    input_path = None
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = (project_root / file_path).resolve()
        # If user explicitly passed a yaml report, use it
        if file_path.is_file() and file_path.suffix in (".yaml", ".yml"):
            input_path = file_path

    if not input_path:
        # Search run_dir in preference order
        for filename in preferred_files:
            candidate = run_dir / filename
            if candidate.exists():
                input_path = candidate
                break

    # If still not found, search run_dir for any other yaml files
    if not input_path:
        yaml_files = list(run_dir.glob("*.yaml")) + list(run_dir.glob("*.yml"))
        yaml_files = [f for f in yaml_files if f.name not in ("run_manifest.yaml", "config_snapshot.yaml", "temp_custom_config.yaml")]
        if yaml_files:
            input_path = yaml_files[0]

    # Fallback default
    if not input_path:
        input_path = run_dir / "result_metrics.yaml"

    # Automatically redirect from raw evaluation_results.yaml to evaluated result_metrics.yaml if it exists
    if input_path.name == "evaluation_results.yaml":
        candidate_path = input_path.parent / "result_metrics.yaml"
        if candidate_path.exists():
            print(f"Redirecting from evaluation_results.yaml to result_metrics.yaml (reusing evaluated metrics)")
            input_path = candidate_path

    print(f"Resolved report file to parse: {input_path}")
    # 2. Parse result_metrics.yaml
    input_path = Path(args.file) if args.file else run_dir / "result_metrics.yaml"
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    data = None
    stats = None

    if not args.traces_only and input_path.exists():
        try:
            report = load_report_file(input_path)
            data = report.model_dump()
            stats = analyze_metrics(data, trace_map)
            
            # Print tables to stdout
            print_rich_tables(stats)
            
            # Export reports/CSVs
            generate_markdown_report(stats, output_md_path)
            export_wide_csv(stats, csv_summary_path)
            export_detailed_csv(data, stats, csv_details_path)
            
            # Export metrics_summary.parsed.json
            with open(parsed_dir / "metrics_summary.parsed.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing result_metrics.yaml: {e}")
            sys.exit(1)
    elif not args.traces_only:
        print(f"Warning: result_metrics.yaml not found at {input_path}")

    # 3. Parse Traces
    graph_rows = None
    eval_rows = None
    has_traces = traces_dir.exists() or (run_dir / "graph_retrieval_trace.jsonl").exists() or (run_dir / "eval_trace.jsonl").exists()
    if traces_dir.exists():
        has_traces = traces_dir.exists() or (run_dir / "graph_retrieval_trace.jsonl").exists() or (run_dir / "eval_trace.jsonl").exists()
    if has_traces:
        graph_rows, graph_summary = parse_graph_retrieval_trace(traces_dir, parsed_dir)
        eval_rows, eval_summary = parse_eval_trace(traces_dir, parsed_dir)

    # 4. Detailed Metrics Rows
    metrics_rows = []
    
    # If we have fresh YAML data, always use it to build metrics_rows
    if data and "results" in data:
        baselines = stats["baselines"] if stats else list(data["results"][0].get("baselines", {}).keys())
        for r in data["results"]:
            q_id = r.get("id", "UNKNOWN")
            category = r.get("category", "general")
            for b in baselines:
                b_data = r.get("baselines", {}).get(b, {})
                if not b_data:
                    continue
                eval_metrics = b_data.get("eval_metrics", {})
                is_ans = r.get("is_answerable")
                if is_ans is None:
                    is_ans = True
                else:
                    is_ans = str(is_ans).lower() == "true"
                    
                ar_f1 = eval_metrics.get("ar_sa_f1")
                if ar_f1 is None and is_ans:
                    r_relevance = eval_metrics.get("answer_relevance")
                    s_accuracy = eval_metrics.get("semantic_accuracy")
                    if r_relevance is not None and s_accuracy is not None:
                        try:
                            r_val = float(r_relevance)
                            s_val = float(s_accuracy)
                            if r_val + s_val > 0:
                                ar_f1 = round(2.0 * (r_val * s_val) / (r_val + s_val), 4)
                            else:
                                ar_f1 = 0.0
                        except (ValueError, TypeError):
                            ar_f1 = 0.0

                metrics_rows.append({
                    "query_id": q_id,
                    "category": category,
                    "baseline": b,
                    "status": b_data.get("status", "success"),
                    "latency_sec": b_data.get("latency_sec"),
                    "retrieval_recall": eval_metrics.get("retrieval_recall"),
                    "context_precision": eval_metrics.get("context_precision"),
                    "faithfulness": eval_metrics.get("faithfulness"),
                    "answer_relevance": eval_metrics.get("answer_relevance"),
                    "citation_fidelity": eval_metrics.get("citation_fidelity"),
                    "semantic_accuracy": eval_metrics.get("semantic_accuracy"),
                    "context_fillness": eval_metrics.get("context_fillness"),
                    "ar_sa_f1": ar_f1,
                    "is_answerable": is_ans,
                    "token_output": eval_metrics.get("token_output"),
                    "token_answer": eval_metrics.get("token_answer"),
                    "token_reasoning": eval_metrics.get("token_reasoning")
                })
    # Otherwise (e.g. traces-only mode), fall back to loading from the existing CSV details
    elif csv_details_path.exists():
        try:
            with open(csv_details_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                metrics_rows = list(reader)
        except Exception as e:
            print(f"Warning: Could not read metrics_details.csv: {e}")

    # Write metrics_details.parsed.csv
    if metrics_rows:
        with open(parsed_dir / "metrics_details.parsed.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            writer.writeheader()
            for r in metrics_rows:
                writer.writerow(r)

    # 5. Join per-query data
    joined_data = {}
    if csv_details_path.exists():
        try:
            with open(csv_details_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                metrics_rows = list(reader)
        except Exception as e:
            print(f"Warning: Could not read metrics_details.csv: {e}")

    if not metrics_rows and data and "results" in data:
        baselines = stats["baselines"] if stats else list(data["results"][0].get("baselines", {}).keys())
        for r in data["results"]:
            q_id = r.get("id", "UNKNOWN")
            category = r.get("category", "general")
            for b in baselines:
                b_data = r.get("baselines", {}).get(b, {})
                if not b_data:
                    continue
                eval_metrics = b_data.get("eval_metrics", {})
                metrics_rows.append({
                    "query_id": q_id,
                    "category": category,
                    "baseline": b,
                    "status": b_data.get("status", "success"),
                    "latency_sec": b_data.get("latency_sec"),
                    "retrieval_recall": eval_metrics.get("retrieval_recall"),
                    "context_precision": eval_metrics.get("context_precision"),
                    "faithfulness": eval_metrics.get("faithfulness"),
                    "answer_relevance": eval_metrics.get("answer_relevance"),
                    "citation_fidelity": eval_metrics.get("citation_fidelity"),
                    "semantic_accuracy": eval_metrics.get("semantic_accuracy"),
                    "context_fillness": eval_metrics.get("context_fillness"),
                    "token_output": eval_metrics.get("token_output"),
                    "token_answer": eval_metrics.get("token_answer"),
                    "token_reasoning": eval_metrics.get("token_reasoning")
                })

    # Write metrics_details.parsed.csv
    if metrics_rows:
        with open(parsed_dir / "metrics_details.parsed.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            writer.writeheader()
            for r in metrics_rows:
                writer.writerow(r)

    # 5. Join per-query data
    joined_data = {}
    
    # Standardize and populate metrics rows
    for r in metrics_rows:
        q_id = str(r.get("query_id") or "")
        base = str(r.get("baseline") or "")
        if not q_id or not base:
            continue
        key = (q_id, base)
        is_ans = r.get("is_answerable")
        if is_ans is None or is_ans == "":
            is_ans = True
        else:
            is_ans = str(is_ans).lower() == "true"
            
        ar_f1 = r.get("ar_sa_f1")
        if (ar_f1 is None or ar_f1 == "") and is_ans:
            r_relevance = r.get("answer_relevance")
            s_accuracy = r.get("semantic_accuracy")
            if r_relevance is not None and s_accuracy is not None and r_relevance != "" and s_accuracy != "":
                try:
                    r_val = float(r_relevance)
                    s_val = float(s_accuracy)
                    if r_val + s_val > 0:
                        ar_f1 = round(2.0 * (r_val * s_val) / (r_val + s_val), 4)
                    else:
                        ar_f1 = 0.0
                except (ValueError, TypeError):
                    ar_f1 = 0.0

        joined_data[key] = {
            "query_id": q_id,
            "baseline": base,
            "category": r.get("category", "general"),
            "is_answerable": is_ans,
            "retrieval_recall": r.get("retrieval_recall"),
            "context_precision": r.get("context_precision"),
            "faithfulness": r.get("faithfulness"),
            "answer_relevance": r.get("answer_relevance"),
            "citation_fidelity": r.get("citation_fidelity"),
            "semantic_accuracy": r.get("semantic_accuracy"),
            "context_fillness": r.get("context_fillness"),
            "ar_sa_f1": ar_f1,
            "latency_sec": r.get("latency_sec"),
            "token_output": r.get("token_output"),
            "token_answer": r.get("token_answer"),
            "token_reasoning": r.get("token_reasoning")
        }

    # Merge graph trace rows
    if graph_rows:
        for r in graph_rows:
            q_id = str(r.get("query_id") or "")
            base = str(r.get("baseline") or "")
            key = (q_id, base)
            if key not in joined_data:
                joined_data[key] = {
                    "query_id": q_id,
                    "baseline": base,
                    "category": r.get("category", "general")
                }
            joined_data[key].update({
                "graph_retrieval_enabled": r.get("graph_retrieval_enabled"),
                "graph_retrieval_skip_reason": r.get("graph_retrieval_skip_reason"),
                "base_candidates_count": r.get("base_candidates_count"),
                "graph_neighbor_paper_ids_count": r.get("graph_neighbor_paper_ids_count"),
                "graph_chunk_candidates_count": r.get("graph_chunk_candidates_count"),
                "merged_candidates_count_before_reranker": r.get("merged_candidates_count_before_reranker"),
                "graph_chunks_survived_final_context_count": r.get("graph_chunks_survived_final_context_count"),
                "graph_chunks_survived_final_context": r.get("graph_chunks_survived_final_context"),
                "graph_survival_rate": r.get("graph_survival_rate"),
                "distinct_papers_in_final_context": r.get("distinct_papers_in_final_context")
            })

    # Merge eval trace rows
    if eval_rows:
        for r in eval_rows:
            q_id = str(r.get("query_id") or "")
            base = str(r.get("baseline") or "")
            key = (q_id, base)
            if key not in joined_data:
                joined_data[key] = {
                    "query_id": q_id,
                    "baseline": base,
                    "category": r.get("category", "general")
                }
            for m in ["retrieval_recall", "context_precision", "faithfulness", "answer_relevance", "citation_fidelity", "semantic_accuracy", "context_fillness", "ar_sa_f1", "is_answerable", "latency_sec"]:
                if r.get(m) is not None and (joined_data[key].get(m) is None or joined_data[key].get(m) == ""):
                    joined_data[key][m] = r[m]

    # Write per_query_joined.csv
    joined_csv_path = parsed_dir / "per_query_joined.csv"
    joined_headers = [
        "query_id", "baseline", "category", "is_answerable",
        "retrieval_recall", "context_precision", "faithfulness", "answer_relevance",
        "citation_fidelity", "semantic_accuracy", "context_fillness", "ar_sa_f1", "latency_sec",
        "query_id", "baseline", "category",
        "retrieval_recall", "context_precision", "faithfulness", "answer_relevance",
        "citation_fidelity", "semantic_accuracy", "context_fillness", "latency_sec",
        "token_output", "token_answer", "token_reasoning",
        "graph_retrieval_enabled", "graph_retrieval_skip_reason",
        "base_candidates_count", "graph_neighbor_paper_ids_count",
        "graph_chunk_candidates_count", "merged_candidates_count_before_reranker",
        "graph_chunks_survived_final_context_count", "graph_chunks_survived_final_context",
        "graph_survival_rate", "distinct_papers_in_final_context"
    ]

    with open(joined_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=joined_headers)
        writer.writeheader()
        for key in sorted(joined_data.keys()):
            row = joined_data[key]
            clean_row = {h: row.get(h, "") for h in joined_headers}
            writer.writerow(clean_row)

    # 6. Run summary JSON
    run_summary = {
        "run_id": run_dir.name,
        "baselines": [],
        "query_count": 0,
        "metrics": {},
        "graph_retrieval": {},
        "by_category": {}
    }

    all_baselines = set()
    for row in joined_data.values():
        if row.get("baseline"):
            all_baselines.add(row["baseline"])
    run_summary["baselines"] = sorted(list(all_baselines))
    run_summary["query_count"] = len(set(row["query_id"] for row in joined_data.values()))

    for b in run_summary["baselines"]:
        b_rows = [row for row in joined_data.values() if row["baseline"] == b]
        run_summary["metrics"][b] = {}
        for m in ["semantic_accuracy", "faithfulness", "latency_sec", "retrieval_recall", "context_precision", "answer_relevance", "ar_sa_f1"]:
            vals = []
            for row in b_rows:
                if m == "ar_sa_f1":
                    is_ans = row.get("is_answerable")
                    if is_ans is None or str(is_ans).lower() != "true":
                        continue
        for m in ["semantic_accuracy", "faithfulness", "latency_sec", "retrieval_recall", "context_precision", "answer_relevance"]:
            vals = []
            for row in b_rows:
                val = row.get(m)
                if val is not None and val != "":
                    try:
                        vals.append(float(val))
                    except ValueError:
                        pass
            if vals:
                run_summary["metrics"][b][f"{m}_mean"] = round(statistics.mean(vals), 4)

        g_rows = [row for row in b_rows if row.get("graph_retrieval_enabled") is not None]
        if g_rows:
            queries_with_graph_neighbors = sum(1 for r in g_rows if float(r.get("graph_neighbor_paper_ids_count") or 0) > 0)
            queries_with_graph_chunks = sum(1 for r in g_rows if float(r.get("graph_chunk_candidates_count") or 0) > 0)
            queries_with_graph_survival = sum(1 for r in g_rows if float(r.get("graph_chunks_survived_final_context_count") or 0) > 0)
            
            surv_vals = []
            for r in g_rows:
                s_rate = r.get("graph_survival_rate")
                if s_rate is not None and s_rate != "":
                    try:
                        surv_vals.append(float(s_rate))
                    except ValueError:
                        pass
            avg_survival_rate = statistics.mean(surv_vals) if surv_vals else 0.0

            run_summary["graph_retrieval"][b] = {
                "queries_with_graph_neighbors": queries_with_graph_neighbors,
                "queries_with_graph_chunks": queries_with_graph_chunks,
                "queries_with_graph_survival": queries_with_graph_survival,
                "avg_graph_survival_rate": round(avg_survival_rate, 4)
            }

    categories = set(row["category"] for row in joined_data.values() if row.get("category"))
    for cat in categories:
        run_summary["by_category"][cat] = {}
        cat_rows = [row for row in joined_data.values() if row["category"] == cat]
        for b in run_summary["baselines"]:
            b_cat_rows = [row for row in cat_rows if row["baseline"] == b]
            if not b_cat_rows:
                continue
            run_summary["by_category"][cat][b] = {}
            
            sem_vals = []
            ar_sa_vals = []
            for row in b_cat_rows:
                s_acc = row.get("semantic_accuracy")
                if s_acc is not None and s_acc != "":
                    try:
                        sem_vals.append(float(s_acc))
                    except ValueError:
                        pass
                is_ans = row.get("is_answerable")
                if is_ans is not None and str(is_ans).lower() == "true":
                    val = row.get("ar_sa_f1")
                    if val is not None and val != "":
                        try:
                            ar_sa_vals.append(float(val))
                        except ValueError:
                            pass
            if sem_vals:
                run_summary["by_category"][cat][b]["semantic_accuracy_mean"] = round(statistics.mean(sem_vals), 4)
            if ar_sa_vals:
                run_summary["by_category"][cat][b]["ar_sa_f1_mean"] = round(statistics.mean(ar_sa_vals), 4)
            if sem_vals:
                run_summary["by_category"][cat][b]["semantic_accuracy_mean"] = round(statistics.mean(sem_vals), 4)
            g_cat_rows = [row for row in b_cat_rows if row.get("graph_retrieval_enabled") is not None]
            if g_cat_rows:
                queries_with_graph_survival = sum(1 for r in g_cat_rows if float(r.get("graph_chunks_survived_final_context_count") or 0) > 0)
                run_summary["by_category"][cat][b]["queries_with_graph_survival"] = queries_with_graph_survival

    # 6. Run summary JSON and YAML with highlights
    # Extract Interesting Highlights
    highlights = {
        "low_faithfulness": [],
        "high_latency": [],
        "graph_successes": []
    }

    # Helper to safely convert value to float
    def safe_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    # Find low faithfulness cases (faithfulness < 0.8)
    for key, row in joined_data.items():
        faith = safe_float(row.get("faithfulness"))
        if faith is not None and faith < 0.8:
            highlights["low_faithfulness"].append({
                "query_id": row["query_id"],
                "baseline": row["baseline"],
                "faithfulness": faith,
                "category": row.get("category", "general")
            })

    # Find high latency cases (latency > 4.0 sec)
    for key, row in joined_data.items():
        lat = safe_float(row.get("latency_sec"))
        if lat is not None and lat > 4.0:
            highlights["high_latency"].append({
                "query_id": row["query_id"],
                "baseline": row["baseline"],
                "latency_sec": lat,
                "category": row.get("category", "general")
            })

    # Find graph successes (survival_rate >= 0.5)
    for key, row in joined_data.items():
        surv = safe_float(row.get("graph_survival_rate"))
        if surv is not None and surv >= 0.5:
            highlights["graph_successes"].append({
                "query_id": row["query_id"],
                "baseline": row["baseline"],
                "survival_rate": surv,
                "survived_count": int(safe_float(row.get("graph_chunks_survived_final_context_count")) or 0),
                "category": row.get("category", "general")
            })

    # Sort and slice highlights
    highlights["low_faithfulness"].sort(key=lambda x: x["faithfulness"])
    highlights["low_faithfulness"] = highlights["low_faithfulness"][:3]

    highlights["high_latency"].sort(key=lambda x: x["latency_sec"], reverse=True)
    highlights["high_latency"] = highlights["high_latency"][:3]

    highlights["graph_successes"].sort(key=lambda x: (x["survival_rate"], x["survived_count"]), reverse=True)
    highlights["graph_successes"] = highlights["graph_successes"][:3]

    run_summary["highlights"] = highlights

    with open(parsed_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    with open(parsed_dir / "run_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run_summary, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 7. Print Console Report
    print(f"\nParsed run: {run_dir}")
    print(f"Queries: {run_summary['query_count']}")
    print(f"Baselines: {', '.join(run_summary['baselines'])}")

    if run_summary["metrics"]:
        print("\nQuality Metrics:")
        header_fmt = "| {:<10} | {:<12} | {:<14} | {:<12} | {:<12} | {:<17} | {:<12} | {:<12} |"
        row_fmt = "| {:<10} | {:<12.4f} | {:<14.4f} | {:<12.4f} | {:<12.4f} | {:<17.4f} | {:<12.4f} | {:<11.3f}s |"
        sep = "+" + "-"*12 + "+" + "-"*14 + "+" + "-"*16 + "+" + "-"*14 + "+" + "-"*14 + "+" + "-"*19 + "+" + "-"*14 + "+" + "-"*14 + "+"
        print(sep)
        print(header_fmt.format("Baseline", "Mean Recall", "Mean Precision", "Faithfulness", "Relevance", "Semantic Accuracy", "AR-SA F1", "Mean Latency"))
        print(sep)
        for b in run_summary["baselines"]:
            b_metrics = run_summary["metrics"].get(b, {})
            recall = b_metrics.get("retrieval_recall_mean", 0.0)
            precision = b_metrics.get("context_precision_mean", 0.0)
            faithfulness = b_metrics.get("faithfulness_mean", 0.0)
            relevance = b_metrics.get("answer_relevance_mean", 0.0)
            semantic = b_metrics.get("semantic_accuracy_mean", 0.0)
            ar_sa_f1 = b_metrics.get("ar_sa_f1_mean", 0.0)
            latency = b_metrics.get("latency_sec_mean", 0.0)
            print(row_fmt.format(b, recall, precision, faithfulness, relevance, semantic, ar_sa_f1, latency))
        print(sep)

    if run_summary["graph_retrieval"]:
        print("\nGraph retrieval:")
        for b, g_stats in run_summary["graph_retrieval"].items():
            print(f"Baseline {b}:")
            print(f"  - queries with neighbors: {g_stats['queries_with_graph_neighbors']}/{run_summary['query_count']}")
            print(f"  - queries with graph chunks: {g_stats['queries_with_graph_chunks']}/{run_summary['query_count']}")
            print(f"  - queries with graph chunks survived: {g_stats['queries_with_graph_survival']}/{run_summary['query_count']}")
            print(f"  - avg graph survival rate: {g_stats['avg_graph_survival_rate']:.4f}")

    if highlights["low_faithfulness"] or highlights["high_latency"] or highlights["graph_successes"]:
        print("\nInteresting Query Highlights:")
        
        if highlights["low_faithfulness"]:
            print("  - Low Faithfulness Cases (potential hallucinations):")
            for h in highlights["low_faithfulness"]:
                print(f"    * Query {h['query_id']} ({h['baseline']}): Faithfulness = {h['faithfulness']:.2f} [Category: {h['category']}]")
                
        if highlights["high_latency"]:
            print("  - High Latency Cases (performance bottlenecks):")
            for h in highlights["high_latency"]:
                print(f"    * Query {h['query_id']} ({h['baseline']}): Latency = {h['latency_sec']:.2f}s [Category: {h['category']}]")
                
        if highlights["graph_successes"]:
            print("  - High Graph Survival Cases (graph retrieval impact):")
            for h in highlights["graph_successes"]:
                print(f"    * Query {h['query_id']} ({h['baseline']}): Survival Rate = {h['survival_rate']:.1f} ({h['survived_count']} chunks) [Category: {h['category']}]")

    print(f"\nJoined file:\n  - {joined_csv_path}")
    print(f"\nSummaries:\n  - {parsed_dir / 'run_summary.json'}")
    print(f"  - {parsed_dir / 'run_summary.yaml'}")
    if (parsed_dir / "graph_retrieval_trace.summary.json").exists():
        print(f"  - {parsed_dir / 'graph_retrieval_trace.summary.json'}")

    if args.confusion:
        print_confusion_matrix_and_metrics_tables(data)


if __name__ == "__main__":
    main()
