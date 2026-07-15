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
from core.statistics import StatsConfig
from metrics_stats_connector import export_stats_json, run_statistical_pipeline


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
            
            # Check for outcome/predicted_abstained first
            outcome = b_data.get("answerability_outcome")
            if not outcome:
                # check eval_metrics too
                eval_metrics = b_data.get("eval_metrics", {})
                outcome = eval_metrics.get("answerability_outcome") if isinstance(eval_metrics, dict) else None
                
            if outcome in ("TP", "FP", "TN", "FN"):
                classification[b][outcome] += 1
            else:
                pred_abst = b_data.get("predicted_abstained")
                if pred_abst is None and isinstance(b_data.get("eval_metrics"), dict):
                    pred_abst = b_data.get("eval_metrics", {}).get("predicted_abstained")
                
                if pred_abst is None:
                    try:
                        from core.metrics import detect_abstention
                        pred_abst = detect_abstention(gen_ans)
                    except Exception:
                        pred_abst = is_refusal(gen_ans)
                
                if is_ans:
                    if pred_abst:
                        classification[b]["FN"] += 1
                    else:
                        classification[b]["TP"] += 1
                else:
                    if pred_abst:
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


class MetricsParser:
    """Configurable parser for RAG benchmark metrics, traces, and statistical analysis."""

    def __init__(
        self,
        run_dir: Path,
        *,
        input_path: Path | None = None,
        output_md_path: Path | None = None,
        csv_summary_path: Path | None = None,
        csv_details_path: Path | None = None,
        traces_only: bool = False,
        confusion: bool = False,
        enable_stats: bool = True,
        n_bootstraps: int = 10000,
        alpha: float = 0.05,
        ci_method: str = "percentile",
        correction_method: str = "holm",
        enable_plots: bool = False,
        random_seed: int = 42,
    ):
        self.run_dir = Path(run_dir)
        self.input_path = input_path
        self.output_md_path = output_md_path or self.run_dir / "metrics_summary.md"
        self.csv_summary_path = csv_summary_path or self.run_dir / "metrics_summary.csv"
        self.csv_details_path = csv_details_path or self.run_dir / "metrics_details.csv"
        self.traces_only = traces_only
        self.confusion = confusion
        self.stats_config = StatsConfig(
            enable_stats=enable_stats,
            n_bootstraps=n_bootstraps,
            alpha=alpha,
            ci_method=ci_method,
            correction_method=correction_method,
            enable_plots=enable_plots,
            random_seed=random_seed,
            plots_dir=str(self.run_dir / "parsed" / "stats_plots"),
        )
        self.parsed_dir = self.run_dir / "parsed"
        self.traces_dir = self.run_dir / "traces"
        self.data = None
        self.stats = None
        self.stats_analysis = None

    def run(self) -> dict:
        """Execute the full parse → aggregate → report → statistics pipeline."""
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        trace_path = self.run_dir / "graph_retrieval_trace.jsonl"
        if not trace_path.exists():
            trace_path = self.run_dir / "traces" / "graph_retrieval_trace.jsonl"
        trace_map = load_graph_retrieval_trace(trace_path)

        input_path = self._resolve_input_path()
        if not self.traces_only and input_path.exists():
            try:
                report = load_report_file(input_path)
                self.data = report.model_dump()
                self.stats = analyze_metrics(self.data, trace_map)
                print_rich_tables(self.stats)
            except Exception as e:
                raise RuntimeError(f"Error processing {input_path}: {e}") from e
        elif not self.traces_only:
            print(f"Warning: result_metrics.yaml not found at {input_path}")

        if self.stats and self.data:
            generate_markdown_report(self.stats, self.output_md_path)
            export_wide_csv(self.stats, self.csv_summary_path)
            export_detailed_csv(self.data, self.stats, self.csv_details_path)
            with open(self.parsed_dir / "metrics_summary.parsed.json", "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)

        graph_rows, _ = parse_graph_retrieval_trace(self.traces_dir, self.parsed_dir)
        eval_rows, _ = parse_eval_trace(self.traces_dir, self.parsed_dir)

        joined_data = self._build_joined_data(graph_rows, eval_rows)
        run_summary = self._build_run_summary(joined_data)
        self._write_joined_csv(joined_data)
        self._write_run_summary(run_summary)
        self._print_console_report(run_summary, joined_data)

        if self.confusion and self.data:
            print_confusion_matrix_and_metrics_tables(self.data)

        if self.stats_config.enable_stats and self.data:
            joined_rows = list(joined_data.values())
            plots_dir = Path(self.stats_config.plots_dir) if self.stats_config.plots_dir else self.parsed_dir
            self.stats_analysis = run_statistical_pipeline(
                self.data,
                config=self.stats_config,
                joined_rows=joined_rows,
                output_dir=plots_dir,
            )
            export_stats_json(self.stats_analysis, self.parsed_dir / "statistical_analysis.json")
            if self.stats:
                generate_markdown_report(
                    self.stats,
                    self.output_md_path,
                    stats_analysis=self.stats_analysis,
                )
            if self.stats_analysis.get("markdown_sections"):
                print("\n[+] Statistical analysis appended to markdown report.")

        return {
            "data": self.data,
            "stats": self.stats,
            "stats_analysis": self.stats_analysis,
            "run_summary": run_summary,
        }

    def _resolve_input_path(self) -> Path:
        if self.input_path:
            return Path(self.input_path)
        preferred = [
            "result_metrics.yaml",
            "result_metrics_judge.yaml",
            "evaluation_results.yaml",
            "evaluation_results_judge.yaml",
        ]
        for name in preferred:
            candidate = self.run_dir / name
            if candidate.exists():
                if name == "evaluation_results.yaml":
                    alt = self.run_dir / "result_metrics.yaml"
                    if alt.exists():
                        return alt
                return candidate
        return self.run_dir / "result_metrics.yaml"

    def _build_joined_data(self, graph_rows, eval_rows) -> dict:
        """Build per-query joined data from YAML and trace sources."""
        metrics_rows = []
        if self.data and "results" in self.data:
            baselines = self.stats["baselines"] if self.stats else list(self.data["results"][0].get("baselines", {}).keys())
            for r in self.data["results"]:
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
                    metrics_rows.append({
                        "query_id": q_id,
                        "category": category,
                        "baseline": b,
                        "is_answerable": is_ans,
                        "predicted_abstained": eval_metrics.get("predicted_abstained", False),
                        "answerability_outcome": eval_metrics.get("answerability_outcome", "TP"),
                        "retrieval_recall": eval_metrics.get("retrieval_recall"),
                        "context_precision": eval_metrics.get("context_precision"),
                        "faithfulness": eval_metrics.get("faithfulness"),
                        "answer_relevance": eval_metrics.get("answer_relevance"),
                        "citation_fidelity": eval_metrics.get("citation_fidelity"),
                        "semantic_accuracy": eval_metrics.get("semantic_accuracy"),
                        "context_fillness": eval_metrics.get("context_fillness"),
                        "ar_sa_f1": eval_metrics.get("ar_sa_f1"),
                        "latency_sec": b_data.get("latency_sec"),
                    })
        elif self.csv_details_path.exists():
            with open(self.csv_details_path, "r", encoding="utf-8") as f:
                metrics_rows = list(csv.DictReader(f))

        joined_data = {}
        for r in metrics_rows:
            q_id = str(r.get("query_id") or "")
            base = str(r.get("baseline") or "")
            if not q_id or not base:
                continue
            is_ans = r.get("is_answerable")
            if is_ans is None or is_ans == "":
                is_ans = True
            else:
                is_ans = str(is_ans).lower() == "true"
            joined_data[(q_id, base)] = {
                "query_id": q_id,
                "baseline": base,
                "category": r.get("category", "general"),
                "is_answerable": is_ans,
                "predicted_abstained": r.get("predicted_abstained", False),
                "answerability_outcome": r.get("answerability_outcome", "TP"),
                "retrieval_recall": r.get("retrieval_recall"),
                "context_precision": r.get("context_precision"),
                "faithfulness": r.get("faithfulness"),
                "answer_relevance": r.get("answer_relevance"),
                "citation_fidelity": r.get("citation_fidelity"),
                "semantic_accuracy": r.get("semantic_accuracy"),
                "context_fillness": r.get("context_fillness"),
                "ar_sa_f1": r.get("ar_sa_f1"),
                "latency_sec": r.get("latency_sec"),
            }

        if graph_rows:
            for r in graph_rows:
                key = (str(r.get("query_id") or ""), str(r.get("baseline") or ""))
                if not key[0] or not key[1]:
                    continue
                joined_data.setdefault(key, {"query_id": key[0], "baseline": key[1]})
                joined_data[key].update({
                    "graph_retrieval_enabled": r.get("graph_retrieval_enabled"),
                    "graph_survival_rate": r.get("graph_survival_rate"),
                })

        if eval_rows:
            for r in eval_rows:
                key = (str(r.get("query_id") or ""), str(r.get("baseline") or ""))
                if not key[0] or not key[1]:
                    continue
                joined_data.setdefault(key, {"query_id": key[0], "baseline": key[1]})
                for m in ["retrieval_recall", "faithfulness", "semantic_accuracy", "answerability_outcome"]:
                    if r.get(m) is not None:
                        joined_data[key][m] = r[m]

        return joined_data

    def _write_joined_csv(self, joined_data: dict) -> None:
        if not joined_data:
            return
        headers = list(next(iter(joined_data.values())).keys())
        with open(self.parsed_dir / "per_query_joined.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for key in sorted(joined_data.keys()):
                writer.writerow(joined_data[key])

    def _build_run_summary(self, joined_data: dict) -> dict:
        baselines = sorted({row["baseline"] for row in joined_data.values() if row.get("baseline")})
        run_summary = {
            "run_id": self.run_dir.name,
            "baselines": baselines,
            "query_count": len({row["query_id"] for row in joined_data.values()}),
            "metrics": {},
        }
        for b in baselines:
            b_rows = [row for row in joined_data.values() if row["baseline"] == b]
            run_summary["metrics"][b] = {}
            for m in ["semantic_accuracy", "faithfulness", "ar_sa_f1"]:
                vals = []
                for row in b_rows:
                    outcome = row.get("answerability_outcome", "TP")
                    if outcome not in ("TP", "FP"):
                        continue
                    val = row.get(m)
                    if val is not None and val != "":
                        try:
                            vals.append(float(val))
                        except ValueError:
                            pass
                run_summary["metrics"][b][f"{m}_mean"] = round(statistics.mean(vals), 4) if vals else 0.0
        return run_summary

    def _write_run_summary(self, run_summary: dict) -> None:
        with open(self.parsed_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        with open(self.parsed_dir / "run_summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(run_summary, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _print_console_report(self, run_summary: dict, joined_data: dict) -> None:
        print(f"\nParsed run: {self.run_dir}")
        print(f"Queries: {run_summary['query_count']}")
        print(f"Baselines: {', '.join(run_summary['baselines'])}")
        joined_csv_path = self.parsed_dir / "per_query_joined.csv"
        if joined_csv_path.exists():
            print(f"\nJoined file:\n  - {joined_csv_path}")


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
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Disable statistical analysis (enabled by default)."
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=10000,
        help="Number of bootstrap resamples (default: 10000)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)."
    )
    parser.add_argument(
        "--ci-method", choices=["percentile", "bca"], default="percentile",
        help="Bootstrap CI method: percentile or BCa (default: percentile)."
    )
    parser.add_argument(
        "--correction-method", choices=["holm", "bonferroni", "none"], default="holm",
        help="Multiple-comparison correction (default: holm)."
    )
    parser.add_argument(
        "--stats-plots", action="store_true",
        help="Generate optional statistical plots (boxplots, p-value heatmaps)."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducible bootstrap resampling (default: 42)."
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        for base in (Path.cwd(), script_dir, project_root):
            candidate = (base / path).resolve()
            if candidate.exists():
                return candidate
        return (script_dir / path).resolve()

    # 1. Resolve run directory
    run_dir = None
    if args.run_dir:
        run_dir = _resolve_path(args.run_dir)
        if run_dir.is_file():
            run_dir = run_dir.parent
    elif args.file:
        file_path = _resolve_path(args.file)
        run_dir = file_path if file_path.is_dir() else file_path.parent
    else:
        # Fallback default: project_root / graphs or reports
        run_dir = project_root / "graphs"
        if not run_dir.exists():
            run_dir = project_root / "reports"

    if not run_dir.exists():
        print(f"Error: run directory does not exist: {run_dir}")
        sys.exit(1)

    print(f"Processing run directory: {run_dir}")

    input_path = None
    if args.file:
        file_path = _resolve_path(args.file)
        if file_path.is_file():
            input_path = file_path
    elif args.run_dir:
        run_candidate = _resolve_path(args.run_dir)
        if run_candidate.is_file() and run_candidate.suffix in (".yaml", ".yml"):
            input_path = run_candidate

    output_md_path = Path(args.output_md) if args.output_md else run_dir / "metrics_summary.md"
    csv_summary_path = Path(args.csv_summary) if args.csv_summary else run_dir / "metrics_summary.csv"
    csv_details_path = Path(args.csv_details) if args.csv_details else run_dir / "metrics_details.csv"

    metrics_parser = MetricsParser(
        run_dir,
        input_path=input_path,
        output_md_path=output_md_path,
        csv_summary_path=csv_summary_path,
        csv_details_path=csv_details_path,
        traces_only=args.traces_only,
        confusion=args.confusion,
        enable_stats=not args.no_stats,
        n_bootstraps=args.n_bootstraps,
        alpha=args.alpha,
        ci_method=args.ci_method,
        correction_method=args.correction_method,
        enable_plots=args.stats_plots,
        random_seed=args.random_seed,
    )

    try:
        metrics_parser.run()
    except Exception as e:
        print(f"Error processing benchmark run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
