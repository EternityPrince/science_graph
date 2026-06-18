#!/usr/bin/env python3
"""
Science Graph — RAG Quality Metrics Parser & Aggregator.
Parses result_metrics.yaml, computes summary statistics (mean, min, max, median, stdev),
performs category-wise breakdown, query difficulty analysis, and pairwise win-rates.
Outputs beautiful terminal tables and generates a comprehensive Markdown report.
"""

import sys
import argparse
import yaml
from pathlib import Path
import statistics
from typing import Dict, List, Any, Tuple

# Rich imports for terminal formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.box import ROUNDED, HEAVY_HEAD
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Baseline descriptions copied from run_benchmarks.py
BASELINES_INFO = {
    "B0": "Zero-Shot (Чистая генерация) — базовая LLM без контекста.",
    "B1": "Pure Lexical (Только лексика) — SQLite FTS5.",
    "B2": "Pure Dense (Только векторы) — семантический поиск.",
    "B3": "Dense + HyDE (Векторы + HyDE) — семантический поиск с гипотетическим ответом.",
    "B4": "Standard Hybrid (Базовый гибрид) — FTS5 + Векторы через RRF.",
    "B5": "Hybrid + Graph (Базовый Граф-RAG) — гибридный поиск + статический обход графа.",
    "B6": "Full Pipeline (Максимальный запуск) — все 12 компонентов (граф, реранкер, LLM-расширение и др. без HyDE)."
}

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


def load_yaml(file_path: Path) -> dict:
    """Loads and returns the YAML evaluation file."""
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except Exception as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)


def analyze_metrics(data: dict) -> dict:
    """
    Computes all advanced statistics and aggregations from the raw YAML data.
    """
    results = data.get("results", [])
    if not results:
        print("Error: No results found in the YAML file.")
        sys.exit(1)
        
    # Find all baselines present in the first result
    first_result = results[0]
    baselines = list(first_result.get("baselines", {}).keys())
    if not baselines:
        # Fall back to checking all results if first is empty
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
            sem = eval_metrics.get("semantic_accuracy") if isinstance(eval_metrics, dict) else None
            if sem is None:
                sem = b_data.get("semantic_accuracy")
            if sem is None:
                gold = r.get("golden_answer", "").strip()
                gen = b_data.get("generated_answer", "").strip()
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

    metadata = data.get("metadata", {})
    original_metadata = metadata.get("original_metadata", metadata)
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
                    rec = calculate_retrieval_recall(r.get("expected_papers", []), b_data.get("retrieved_papers", []))
                eval_metrics["retrieval_recall"] = rec
                
            # 2. context_precision
            if eval_metrics.get("context_precision") is None:
                prec = b_data.get("context_precision")
                if prec is None:
                    prec = calculate_context_precision(r.get("expected_papers", []), b_data.get("retrieved_chunks", []))
                eval_metrics["context_precision"] = prec
                
            # 3. context_fillness
            if eval_metrics.get("context_fillness") is None:
                fillness = b_data.get("context_fillness")
                if fillness is None:
                    context_token = b_data.get("context_token")
                    max_input_token_val = b_data.get("max_input_token")
                    if context_token is None:
                        context_token = estimate_prompt_tokens(r.get("query", ""), b_data.get("retrieved_chunks", []), b)
                    if max_input_token_val is None:
                        max_input_token_val = max_input_token
                    fillness = round(context_token / max_input_token_val, 4) if max_input_token_val > 0 else 0.0
                    fillness = min(max(fillness, 0.0), 1.0)
                eval_metrics["context_fillness"] = fillness

    # 1. Collect raw values per baseline and metric
    raw_values = {b: {m: [] for m in ALL_METRICS} for b in baselines}
    for b in baselines:
        raw_values[b]["status"] = []

    
    # Category-based values: raw_categories[category][baseline][metric] = []
    categories = set()
    category_values = {}
    
    # Query difficulty metrics
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
    
    # 2. Compute summary statistics
    summary_stats = {}
    for b in baselines:
        summary_stats[b] = {}
        # Success Rate
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
            
    # 3. Compute category statistics
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
                    
    # 4. Pairwise Win Rate (how often baseline X > baseline Y for a given metric)
    # Let's compute this for semantic_accuracy and overall quality average
    pairwise_win_rates = {}
    for metric in QUALITY_METRICS + ["latency_sec"]:
        pairwise_win_rates[metric] = {b1: {b2: 0.0 for b2 in baselines} for b1 in baselines}
        
        # Count matchups
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
                            # For latency, lower is better
                            if val1 < val2:
                                wins += 1
                        else:
                            # For quality, higher is better
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


def print_rich_tables(stats: dict):
    """Prints beautiful summary tables to the terminal using Rich."""
    console = Console()
    
    console.print(Panel(
        "[bold green]Science Graph RAG Benchmarks Analyzer[/bold green]\n"
        f"Aggregated statistics across [bold cyan]{stats['total_queries']}[/bold cyan] test queries.",
        title="RAG Evaluation Report",
        expand=False
    ))
    
    # 1. Main Summary Table (Average values)
    table = Table(title="[bold]Сводная таблица (Средние значения / Averages)[/bold]", box=ROUNDED, header_style="bold magenta")
    table.add_column("Baseline", style="cyan", no_wrap=True)
    table.add_column("Success Rate", justify="right")
    for m in ALL_METRICS:
        table.add_column(METRIC_LABELS[m], justify="right")
        
    for b in stats["baselines"]:
        row = [f"[bold]{b}[/bold]"]
        # Success rate
        sr = stats["summary"][b]["success_rate"]
        row.append(f"{sr:.1f}%")
        
        # Metrics
        for m in ALL_METRICS:
            val = stats["summary"][b][m]["mean"]
            if m == "latency_sec":
                row.append(f"{val:.2f}s")
            else:
                row.append(f"{val:.3f}")
        table.add_row(*row)
        
    console.print(table)
    console.print()
    
    # 2. Detailed statistics per baseline with Min/Max/Stdev
    for b in stats["baselines"]:
        desc = BASELINES_INFO.get(b, "")
        table_det = Table(title=f"[bold]Детальная статистика: {b}[/bold] ({desc})", box=ROUNDED, header_style="bold yellow")
        table_det.add_column("Metric", style="cyan")
        table_det.add_column("Mean", justify="right")
        table_det.add_column("Min", justify="right")
        table_det.add_column("Max", justify="right")
        table_det.add_column("Median", justify="right")
        table_det.add_column("Std Dev", justify="right")
        
        for m in ALL_METRICS:
            m_stats = stats["summary"][b][m]
            mean_val = m_stats["mean"]
            min_val = m_stats["min"]
            max_val = m_stats["max"]
            med_val = m_stats["median"]
            std_val = m_stats["stdev"]
            
            if m == "latency_sec":
                table_det.add_row(
                    METRIC_LABELS[m],
                    f"{mean_val:.2f}s",
                    f"{min_val:.2f}s",
                    f"{max_val:.2f}s",
                    f"{med_val:.2f}s",
                    f"{std_val:.2f}s"
                )
            else:
                table_det.add_row(
                    METRIC_LABELS[m],
                    f"{mean_val:.3f}",
                    f"{min_val:.3f}",
                    f"{max_val:.3f}",
                    f"{med_val:.3f}",
                    f"{std_val:.3f}"
                )
        console.print(table_det)
        console.print()

    # 3. Category Breakdown (Semantic Accuracy)
    table_cat = Table(title="[bold]Разбивка по категориям (Средняя Semantic Accuracy)[/bold]", box=ROUNDED, header_style="bold blue")
    table_cat.add_column("Категория (Category)", style="cyan")
    for b in stats["baselines"]:
        table_cat.add_column(b, justify="right")
        
    for cat in stats["categories"]:
        row = [cat]
        for b in stats["baselines"]:
            val = stats["category_stats"][cat][b]["semantic_accuracy"]
            row.append(f"{val:.3f}")
        table_cat.add_row(*row)
        
    console.print(table_cat)
    console.print()
    
    # 4. Pairwise Win-rate Matrix (Semantic Accuracy)
    table_win = Table(title="[bold]Матрица побед (Win-Rate Matrix: Semantic Accuracy)[/bold]\nПоказывает как часто строка обыгрывает столбец (Row beats Column %)", box=ROUNDED, header_style="bold green")
    table_win.add_column("Baseline", style="cyan")
    for b in stats["baselines"]:
        table_win.add_column(b, justify="right")
        
    for b1 in stats["baselines"]:
        row = [b1]
        for b2 in stats["baselines"]:
            if b1 == b2:
                row.append("-")
            else:
                rate = stats["pairwise_win_rates"]["semantic_accuracy"][b1][b2]
                row.append(f"{rate:.1f}%")
        table_win.add_row(*row)
        
    console.print(table_win)
    console.print()
    
    # 5. Top 5 hardest queries
    table_hard = Table(title="[bold]Топ-5 самых сложных вопросов (Top 5 Hardest Queries)[/bold]\n(По средней оценке всех baseline)", box=ROUNDED, header_style="bold red")
    table_hard.add_column("ID", style="cyan", no_wrap=True)
    table_hard.add_column("Category", style="magenta")
    table_hard.add_column("Query (Вопрос)")
    table_hard.add_column("Avg Score", justify="right")
    
    for q in stats["query_difficulty"][:5]:
        table_hard.add_row(q["id"], q["category"], q["query"][:80] + "..." if len(q["query"]) > 80 else q["query"], f"{q['avg_score']:.3f}")
        
    console.print(table_hard)


def print_plain_tables(stats: dict):
    """Fallback plain text printer if Rich is not available."""
    print("=== Science Graph RAG Benchmarks Report ===")
    print(f"Total Queries: {stats['total_queries']}\n")
    
    print("--- Сводная таблица (Средние значения) ---")
    headers = ["Baseline", "Success"] + [METRIC_LABELS[m] for m in ALL_METRICS]
    print("\t".join(headers))
    for b in stats["baselines"]:
        sr = stats["summary"][b]["success_rate"]
        row = [b, f"{sr:.1f}%"]
        for m in ALL_METRICS:
            val = stats["summary"][b][m]["mean"]
            suffix = "s" if m == "latency_sec" else ""
            row.append(f"{val:.3f}{suffix}")
        print("\t".join(row))
    print()


def generate_markdown_report(stats: dict, output_path: Path):
    """Generates a beautiful self-contained Markdown report with highlights, tables, and emojis."""
    lines = []
    lines.append("# 📊 Отчет по качеству RAG-системы (RAG Benchmarking Report)")
    lines.append("")
    lines.append(f"**Дата анализа:** 2026-06-13")
    lines.append(f"**Количество тестовых вопросов:** {stats['total_queries']}")
    lines.append("")
    
    lines.append("## 🏷️ Описание протестированных конфигураций (Baselines)")
    lines.append("| Baseline | Описание конфигурации |")
    lines.append("| :--- | :--- |")
    for b, desc in BASELINES_INFO.items():
        if b in stats["baselines"]:
            lines.append(f"| **{b}** | {desc} |")
    lines.append("")
    
    lines.append("## 📈 Сводные результаты (Averages Summary)")
    lines.append("> [!NOTE]")
    lines.append("> Качество оценивалось LLM-судьей по шкале от 0.0 до 1.0 (за исключением Latency).")
    lines.append("")
    
    # Generate main table headers
    headers = ["Baseline", "Success Rate"] + [METRIC_LABELS[m] for m in ALL_METRICS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- | " + " | ".join(["---:"] * (len(headers) - 1)) + " |")
    
    # Find the best baseline per metric (excluding latency where lower is better, others higher is better)
    best_per_metric = {}
    for m in ALL_METRICS:
        if m == "latency_sec":
            best_val = min(stats["summary"][b][m]["mean"] for b in stats["baselines"])
            best_b = [b for b in stats["baselines"] if stats["summary"][b][m]["mean"] == best_val][0]
        else:
            best_val = max(stats["summary"][b][m]["mean"] for b in stats["baselines"])
            best_b = [b for b in stats["baselines"] if stats["summary"][b][m]["mean"] == best_val][0]
        best_per_metric[m] = best_b

    # Add data rows
    for b in stats["baselines"]:
        sr = stats["summary"][b]["success_rate"]
        row = [f"**{b}**", f"{sr:.1f}%"]
        for m in ALL_METRICS:
            val = stats["summary"][b][m]["mean"]
            cell_str = f"{val:.3f}" if m != "latency_sec" else f"{val:.2f}s"
            
            # Bold the best value
            if best_per_metric[m] == b:
                cell_str = f"🏆 **{cell_str}**"
            row.append(cell_str)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    
    lines.append("## 🔍 Детальный анализ стабильности (Min / Max / Median / StdDev)")
    lines.append("Позволяет оценить стабильность работы системы на различных запросах.")
    lines.append("")
    
    for b in stats["baselines"]:
        lines.append(f"### ⚙️ {b} — {BASELINES_INFO.get(b, b)}")
        lines.append("| Метрика | Среднее (Mean) | Минимум (Min) | Максимум (Max) | Медиана (Median) | Отклонение (Std Dev) |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        
        for m in ALL_METRICS:
            m_stats = stats["summary"][b][m]
            mean_val = m_stats["mean"]
            min_val = m_stats["min"]
            max_val = m_stats["max"]
            med_val = m_stats["median"]
            std_val = m_stats["stdev"]
            
            suffix = "s" if m == "latency_sec" else ""
            lines.append(
                f"| {METRIC_LABELS[m]} | "
                f"{mean_val:.3f}{suffix} | "
                f"{min_val:.3f}{suffix} | "
                f"{max_val:.3f}{suffix} | "
                f"{med_val:.3f}{suffix} | "
                f"{std_val:.3f}{suffix} |"
            )
        lines.append("")
        
    lines.append("## 📁 Разбивка по категориям запросов (Category Breakdown)")
    lines.append("Средняя семантическая точность (Semantic Accuracy) в разрезе типов документов/запросов:")
    lines.append("")
    
    cat_headers = ["Категория"] + stats["baselines"]
    lines.append("| " + " | ".join(cat_headers) + " |")
    lines.append("| :--- | " + " | ".join(["---:"] * len(stats["baselines"])) + " |")
    
    for cat in stats["categories"]:
        row = [f"`{cat}`"]
        # Find best baseline in this category
        best_val = max(stats["category_stats"][cat][b]["semantic_accuracy"] for b in stats["baselines"])
        for b in stats["baselines"]:
            val = stats["category_stats"][cat][b]["semantic_accuracy"]
            cell_str = f"{val:.3f}"
            if val == best_val and val > 0:
                cell_str = f"🥇 **{cell_str}**"
            row.append(cell_str)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    
    lines.append("## 🥊 Матрица попарных побед (Pairwise Win Rate Matrix: Semantic Accuracy)")
    lines.append("Процент запросов, на которых конфигурация в строке показала результат **строго выше** конфигурации в столбце:")
    lines.append("")
    
    win_headers = ["Baseline"] + stats["baselines"]
    lines.append("| " + " | ".join(win_headers) + " |")
    lines.append("| :--- | " + " | ".join(["---:"] * len(stats["baselines"])) + " |")
    
    for b1 in stats["baselines"]:
        row = [f"**{b1}**"]
        for b2 in stats["baselines"]:
            if b1 == b2:
                row.append("-")
            else:
                rate = stats["pairwise_win_rates"]["semantic_accuracy"][b1][b2]
                row.append(f"{rate:.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    
    lines.append("## 🧗‍♂️ Топ-10 самых сложных запросов (Top 10 Hardest Queries)")
    lines.append("Запросы, вызвавшие наибольшие затруднения у всех конфигураций (ранжировано по средней оценке):")
    lines.append("")
    lines.append("| ID | Категория | Вопрос | Средняя оценка (Avg Score) |")
    lines.append("| :--- | :--- | :--- | ---: |")
    for q in stats["query_difficulty"][:10]:
        lines.append(f"| `{q['id']}` | `{q['category']}` | {q['query']} | **{q['avg_score']:.3f}** |")
    lines.append("")
    
    lines.append("## 🏆 Главные выводы (Research Summary)")
    
    # Automated generation of main insights
    best_accuracy_b = best_per_metric["semantic_accuracy"]
    best_recall_b = best_per_metric["retrieval_recall"]
    fastest_b = best_per_metric["latency_sec"]
    
    lines.append(f"1. **Абсолютный лидер по качеству ответов:** `{best_accuracy_b}` с Semantic Accuracy **{stats['summary'][best_accuracy_b]['semantic_accuracy']['mean']:.3f}**.")
    lines.append(f"2. **Лучшая глубина поиска (Retrieval Recall):** `{best_recall_b}` с показателем **{stats['summary'][best_recall_b]['retrieval_recall']['mean']:.3f}**.")
    lines.append(f"3. **Самый быстрый отклик (Latency):** `{fastest_b}` со средним временем **{stats['summary'][fastest_b]['latency_sec']['mean']:.2f} сек**.")
    
    # Calculate trade-off score: semantic_accuracy / log10(latency + 1) or similar.
    # Let's just calculate a simple efficiency ratio: Semantic Accuracy / Mean Latency * 10
    lines.append("4. **Эффективность (Качество/Скорость - Semantic Accuracy per 10s latency):**")
    efficiency_list = []
    for b in stats["baselines"]:
        lat = stats["summary"][b]["latency_sec"]["mean"]
        acc = stats["summary"][b]["semantic_accuracy"]["mean"]
        eff = (acc / lat * 10) if lat > 0 else 0
        efficiency_list.append((b, eff))
    efficiency_list.sort(key=lambda x: x[1], reverse=True)
    for rank, (b, eff) in enumerate(efficiency_list, 1):
        lines.append(f"   - #{rank} `{b}`: **{eff:.2f}** pts/10s")
        
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n[+] Created beautiful Markdown report at: {output_path}")


def export_wide_csv(stats: dict, csv_path: Path):
    """Saves wide-format aggregated summary of metrics per baseline."""
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Baseline", "Success Rate", "Recall", "Precision", 
            "Faithfulness", "Relevance", "Citations", "Semantic Accuracy", "Latency (sec)"
        ])
        for b in stats["baselines"]:
            sr = f"{stats['summary'][b]['success_rate']:.1f}%"
            
            def get_val(metric_name):
                # For B0, context metrics are N/A
                if b == "B0" and metric_name in ["retrieval_recall", "context_precision", "faithfulness", "citation_fidelity"]:
                    return "N/A"
                if stats["summary"][b][metric_name]["count"] == 0:
                    return "N/A"
                val = stats["summary"][b][metric_name]["mean"]
                if metric_name == "latency_sec":
                    return f"{val:.2f}"
                else:
                    return f"{val:.4f}"
            
            writer.writerow([
                b,
                sr,
                get_val("retrieval_recall"),
                get_val("context_precision"),
                get_val("faithfulness"),
                get_val("answer_relevance"),
                get_val("citation_fidelity"),
                get_val("semantic_accuracy"),
                get_val("latency_sec")
            ])


def export_detailed_csv(data: dict, stats: dict, csv_path: Path):
    """Saves detailed case-by-case metrics for each baseline and query."""
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results = data.get("results", [])
    if not results:
        return
        
    baselines = stats["baselines"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "category", "baseline", "status", "latency_sec",
            "retrieval_recall", "context_precision", "faithfulness",
            "answer_relevance", "citation_fidelity", "semantic_accuracy"
        ])
        
        for r in results:
            q_id = r.get("id", "UNKNOWN")
            category = r.get("category", "default")
            
            for b in baselines:
                b_data = r.get("baselines", {}).get(b, {})
                if not b_data:
                    continue
                
                status = b_data.get("status", "failed")
                latency = b_data.get("latency_sec")
                eval_metrics = b_data.get("eval_metrics", {})
                
                def get_metric(m_name):
                    if b == "B0" and m_name in ["retrieval_recall", "context_precision", "faithfulness", "citation_fidelity"]:
                        return ""
                    val = eval_metrics.get(m_name)
                    return val if val is not None else ""
                
                writer.writerow([
                    q_id,
                    category,
                    b,
                    status,
                    latency if latency is not None else "",
                    get_metric("retrieval_recall"),
                    get_metric("context_precision"),
                    get_metric("faithfulness"),
                    get_metric("answer_relevance"),
                    get_metric("citation_fidelity"),
                    get_metric("semantic_accuracy")
                ])


import numpy as np
import glob
import tiktoken

embedding_engine = None

def get_embedding_engine():
    global embedding_engine
    if embedding_engine is None:
        try:
            # Set up python path to resolve src imports correctly
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from src.vector_search import EmbeddingEngine
            embedding_engine = EmbeddingEngine()
        except Exception as e:
            print(f"Error initializing EmbeddingEngine: {e}")
            sys.exit(1)
    return embedding_engine


def compute_cosine_similarity(v1, v2):
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    norm1 = np.linalg.norm(v1_arr)
    norm2 = np.linalg.norm(v2_arr)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1_arr, v2_arr) / (norm1 * norm2))


def calculate_semantic_accuracy(golden_answers: List[str], generated_answers: List[str]) -> List[float]:
    if not golden_answers or not generated_answers:
        return []
    
    engine = get_embedding_engine()
    gold_embs = engine.get_embeddings(golden_answers, is_query=False)
    gen_embs = engine.get_embeddings(generated_answers, is_query=False)
    
    similarities = []
    for v1, v2 in zip(gold_embs, gen_embs):
        similarities.append(compute_cosine_similarity(v1, v2))
    return similarities


def calculate_retrieval_recall(expected_papers: List[str], retrieved_papers: List[str]) -> float:
    if not expected_papers:
        return 1.0
    expected_set = {p.strip().lower() for p in expected_papers if p.strip()}
    retrieved_set = {p.strip().lower() for p in retrieved_papers if p.strip()}
    if not expected_set:
        return 1.0
    intersection = expected_set.intersection(retrieved_set)
    return round(len(intersection) / len(expected_set), 4)


def calculate_context_precision(expected_papers: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    if not expected_papers:
        return 1.0
    expected_set = {p.strip().lower() for p in expected_papers if p.strip()}
    if not expected_set:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    precision_sum = 0.0
    relevant_hits = 0
    for idx, chunk in enumerate(retrieved_chunks):
        paper_id = chunk.get("paper_id", "")
        if paper_id and paper_id.strip().lower() in expected_set:
            relevant_hits += 1
            precision_sum += relevant_hits / (idx + 1)
            
    if relevant_hits == 0:
        return 0.0
    return round(precision_sum / relevant_hits, 4)


def estimate_prompt_tokens(query: str, retrieved_chunks: List[dict], baseline: str) -> int:
    if baseline == "B0":
        prompt = f"Вопрос: {query}\nОтветь на основе своих общих знаний."
    else:
        system_prompt = (
            "<|im_start|>system\n"
            "You are a research assistant. Synthesize an answer to the user's question using the retrieved text blocks and the knowledge graph connections.\n"
            "Always mention the titles of the papers, years, authors, and page numbers when citation is needed.\n"
            "If the graph contains citing relationships, use them to explain the context (e.g., \"A cited B\").\n\n"
            "Here is the retrieved context:\n\n"
            "### RELEVANT TEXT FRAGMENTS:\n"
        )
        text_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            text_content = chunk.get("text_content", "").strip()
            paper_id = chunk.get("paper_id", "")
            page = chunk.get("page_number", "")
            text_blocks.append(
                f"Block {idx} (Score: 1.000) | Paper: {paper_id} (Page {page}):\n"
                f"\"\"\"\n{text_content}\n\"\"\""
            )
        context_text = "\n\n".join(text_blocks)
        context_graph = "No direct graph relations found."
        prompt = (
            f"{system_prompt}{context_text}\n\n"
            f"### KNOWLEDGE GRAPH CONNECTIONS:\n{context_graph}\n"
            f"<|im_end|>\n<|im_start|>user\nQuestion: {query}\nAnswer in Russian:\n<|im_end|>\n<|im_start|>assistant\n"
        )
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(prompt))
    except Exception:
        return len(prompt) // 4


def analyze_run_directory(run_dir: Path) -> dict:
    eval_file = run_dir / "result_metrics.yaml"
    if not eval_file.exists():
        eval_file = run_dir / "evaluation_results.yaml"
    if not eval_file.exists():
        return None
    
    data = load_yaml(eval_file)
    results = data.get("results", [])
    if not results:
        return None
        
    metadata = data.get("metadata", {})
    original_metadata = metadata.get("original_metadata", metadata)
    llm_info = original_metadata.get("llm", {})
    model_path = llm_info.get("model_name", "")
    model_name = Path(model_path).name if model_path else run_dir.name
    
    # Identify baselines present in results
    baselines = set()
    for r in results:
        baselines.update(r.get("baselines", {}).keys())
    baselines = sorted(list(baselines))
    
    # Pre-calculate missing semantic accuracies for all results/baselines in this directory
    missing_semantics = []
    golden_list = []
    generated_list = []
    for r_idx, r in enumerate(results):
        for b in baselines:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            eval_metrics = b_data.get("eval_metrics", {})
            sem = eval_metrics.get("semantic_accuracy") if isinstance(eval_metrics, dict) else None
            if sem is None:
                sem = b_data.get("semantic_accuracy")
            if sem is None:
                gold = r.get("golden_answer", "").strip()
                gen = b_data.get("generated_answer", "").strip()
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

    max_input_token = original_metadata.get("llm", {}).get("max_tokens", 10000)
    has_llm_metrics = False
    
    baselines_summary = {}
    for b in baselines:
        recalls = []
        precisions = []
        faithfulnesses = []
        relevances = []
        citations = []
        semantic_accuracies = []
        fillnesses = []
        latencies = []
        
        for r in results:
            b_data = r.get("baselines", {}).get(b, {})
            if not b_data:
                continue
            
            # Ensure eval_metrics is a dict
            if "eval_metrics" not in b_data or not isinstance(b_data["eval_metrics"], dict):
                b_data["eval_metrics"] = {}
            eval_metrics = b_data["eval_metrics"]
            
            # Extract or calculate recall
            recall = eval_metrics.get("retrieval_recall")
            if recall is None:
                recall = b_data.get("retrieval_recall")
            if recall is None:
                recall = calculate_retrieval_recall(r.get("expected_papers", []), b_data.get("retrieved_papers", []))
            eval_metrics["retrieval_recall"] = recall
            recalls.append(recall)
            
            # Extract or calculate precision
            precision = eval_metrics.get("context_precision")
            if precision is None:
                precision = b_data.get("context_precision")
            if precision is None:
                precision = calculate_context_precision(r.get("expected_papers", []), b_data.get("retrieved_chunks", []))
            eval_metrics["context_precision"] = precision
            precisions.append(precision)
            
            # Semantic accuracy (already filled in pre-computation step if missing)
            sem = eval_metrics.get("semantic_accuracy", 0.0)
            semantic_accuracies.append(sem)
            
            # Context fillness
            fillness = eval_metrics.get("context_fillness")
            if fillness is None:
                fillness = b_data.get("context_fillness")
            if fillness is None:
                context_token = b_data.get("context_token")
                max_input_token_val = b_data.get("max_input_token")
                if context_token is None:
                    context_token = estimate_prompt_tokens(r.get("query", ""), b_data.get("retrieved_chunks", []), b)
                if max_input_token_val is None:
                    max_input_token_val = max_input_token
                fillness = round(context_token / max_input_token_val, 4) if max_input_token_val > 0 else 0.0
                fillness = min(max(fillness, 0.0), 1.0)
            eval_metrics["context_fillness"] = fillness
            fillnesses.append(fillness)
            
            # Latency
            lat = b_data.get("latency_sec")
            if lat is not None:
                latencies.append(lat)
                
            # LLM-judge metrics
            faith = eval_metrics.get("faithfulness")
            if faith is not None:
                faithfulnesses.append(faith)
                has_llm_metrics = True
                
            relev = eval_metrics.get("answer_relevance")
            if relev is not None:
                relevances.append(relev)
                has_llm_metrics = True
                
            cite = eval_metrics.get("citation_fidelity")
            if cite is not None:
                citations.append(cite)
                has_llm_metrics = True
                
        avg_recall = statistics.mean(recalls) if recalls else 0.0
        avg_precision = statistics.mean(precisions) if precisions else 0.0
        avg_semantic_accuracy = statistics.mean(semantic_accuracies) if semantic_accuracies else 0.0
        avg_faithfulness = statistics.mean(faithfulnesses) if faithfulnesses else 0.0
        avg_relevance = statistics.mean(relevances) if relevances else 0.0
        avg_citation = statistics.mean(citations) if citations else 0.0
        avg_fillness = statistics.mean(fillnesses) if fillnesses else 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        
        baselines_summary[b] = {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_semantic_accuracy": avg_semantic_accuracy,
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevance": avg_relevance,
            "avg_citation_fidelity": avg_citation,
            "avg_context_fillness": avg_fillness,
            "avg_latency_sec": avg_latency,
            "count": len(recalls)
        }
        
    return {
        "run_dir_name": run_dir.name,
        "model_name": model_name,
        "baselines": baselines_summary,
        "has_llm_metrics": has_llm_metrics
    }


def print_runs_comparison(run_summaries: List[dict]):
    """Prints a beautiful summary table comparing multiple runs using Rich or plain text."""
    if not run_summaries:
        print("No evaluation data found to compare.")
        return
        
    has_llm = any(s.get("has_llm_metrics", False) for s in run_summaries)
    
    if HAS_RICH:
        console = Console()
        console.print(Panel(
            "[bold green]Science Graph RAG Runs Comparison[/bold green]\n"
            f"Comparing [bold cyan]{len(run_summaries)}[/bold cyan] benchmark runs.",
            title="Runs Summary Table",
            expand=False
        ))
        
        table = Table(box=ROUNDED, header_style="bold magenta")
        table.add_column("Run Directory", style="cyan", no_wrap=True)
        table.add_column("Model / LLM", style="green")
        table.add_column("Baseline", style="yellow", justify="center")
        table.add_column("Recall", justify="right")
        table.add_column("Precision", justify="right")
        if has_llm:
            table.add_column("Faithfulness", justify="right")
            table.add_column("Relevance", justify="right")
            table.add_column("Citations", justify="right")
        table.add_column("Semantic", justify="right")
        table.add_column("Fillness", justify="right")
        table.add_column("Latency", justify="right")
        
        # Sort by run name for consistency
        for summary in sorted(run_summaries, key=lambda x: x["run_dir_name"]):
            run_name = summary["run_dir_name"]
            model_name = summary["model_name"]
            run_has_llm = summary.get("has_llm_metrics", False)
            
            for b, b_data in sorted(summary["baselines"].items()):
                prec = b_data["avg_precision"]
                rec = b_data["avg_recall"]
                sem = b_data["avg_semantic_accuracy"]
                fill = b_data.get("avg_context_fillness", 0.0)
                lat = b_data.get("avg_latency_sec", 0.0)
                
                row = [
                    run_name,
                    model_name,
                    b,
                    f"{rec:.2%}" if b != "B0" else "N/A",
                    f"{prec:.2%}" if b != "B0" else "N/A",
                ]
                
                if has_llm:
                    if run_has_llm and b != "B0":
                        faith = b_data.get("avg_faithfulness", 0.0)
                        row.append(f"{faith:.2%}")
                    else:
                        row.append("N/A")
                        
                    if run_has_llm:
                        relev = b_data.get("avg_answer_relevance", 0.0)
                        row.append(f"{relev:.2%}")
                    else:
                        row.append("N/A")
                        
                    if run_has_llm and b != "B0":
                        cite = b_data.get("avg_citation_fidelity", 0.0)
                        row.append(f"{cite:.2%}")
                    else:
                        row.append("N/A")
                
                row.extend([
                    f"{sem:.4f}",
                    f"{fill:.2%}",
                    f"{lat:.2f}s"
                ])
                
                table.add_row(*row)
        console.print(table)
    else:
        print("\n=== Science Graph RAG Runs Comparison ===")
        header_line = f"{'Run Directory':<40} | {'Model':<35} | {'Base':<4} | {'Recall':<6} | {'Precision':<9}"
        if has_llm:
            header_line += " | Faith  | Relev  | Cite  "
        header_line += " | Semantic | Fillness | Latency"
        print(header_line)
        print("-" * (155 if has_llm else 125))
        for summary in sorted(run_summaries, key=lambda x: x["run_dir_name"]):
            run_name = summary["run_dir_name"]
            model_name = summary["model_name"]
            run_has_llm = summary.get("has_llm_metrics", False)
            for b, b_data in sorted(summary["baselines"].items()):
                prec = b_data["avg_precision"]
                rec = b_data["avg_recall"]
                sem = b_data["avg_semantic_accuracy"]
                fill = b_data.get("avg_context_fillness", 0.0)
                lat = b_data.get("avg_latency_sec", 0.0)
                
                rec_str = f"{rec:.1%}" if b != "B0" else "N/A"
                prec_str = f"{prec:.1%}" if b != "B0" else "N/A"
                
                row_str = f"{run_name[:40]:<40} | {model_name[:35]:<35} | {b:<4} | {rec_str:<6} | {prec_str:<9}"
                
                if has_llm:
                    if run_has_llm and b != "B0":
                        faith = b_data.get("avg_faithfulness", 0.0)
                        faith_str = f"{faith:.1%}"
                    else:
                        faith_str = "N/A"
                        
                    if run_has_llm:
                        relev = b_data.get("avg_answer_relevance", 0.0)
                        relev_str = f"{relev:.1%}"
                    else:
                        relev_str = "N/A"
                        
                    if run_has_llm and b != "B0":
                        cite = b_data.get("avg_citation_fidelity", 0.0)
                        cite_str = f"{cite:.1%}"
                    else:
                        cite_str = "N/A"
                        
                    row_str += f" | {faith_str:<6} | {relev_str:<6} | {cite_str:<6}"
                
                row_str += f" | {sem:.4f} | {fill:.1%} | {lat:.2f}s"
                print(row_str)
        print()


def generate_comparison_markdown_report(run_summaries: List[dict], output_path: Path):
    """Generates a markdown report comparing multiple runs."""
    lines = []
    lines.append("# 📊 Сравнительный анализ RAG-запусков (RAG Runs Comparison Report)")
    lines.append("")
    lines.append(f"**Количество проанализированных запусков:** {len(run_summaries)}")
    lines.append("")
    lines.append("## 📈 Сводная таблица результатов (Summary Table)")
    lines.append("")
    
    has_llm = any(s.get("has_llm_metrics", False) for s in run_summaries)
    
    headers = ["Запуск (Run)", "Модель (Model)", "Baseline", "Recall", "Precision"]
    alignments = [":---", ":---", ":---:", "---:", "---:"]
    if has_llm:
        headers.extend(["Faithfulness", "Relevance", "Citations"])
        alignments.extend(["---:", "---:", "---:"])
    headers.extend(["Semantic Accuracy (Cos Sim)", "Context Fillness", "Latency"])
    alignments.extend(["---:", "---:", "---:"])
    
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(alignments) + " |")
    
    for summary in sorted(run_summaries, key=lambda x: x["run_dir_name"]):
        run_name = summary["run_dir_name"]
        model_name = summary["model_name"]
        run_has_llm = summary.get("has_llm_metrics", False)
        for b, b_data in sorted(summary["baselines"].items()):
            prec = b_data["avg_precision"]
            rec = b_data["avg_recall"]
            sem = b_data["avg_semantic_accuracy"]
            fill = b_data.get("avg_context_fillness", 0.0)
            lat = b_data.get("avg_latency_sec", 0.0)
            
            prec_str = f"{prec:.1%}" if b != "B0" else "N/A"
            rec_str = f"{rec:.1%}" if b != "B0" else "N/A"
            
            row = [f"`{run_name}`", model_name, f"**{b}**", rec_str, prec_str]
            
            if has_llm:
                if run_has_llm and b != "B0":
                    faith = b_data.get("avg_faithfulness", 0.0)
                    row.append(f"{faith:.1%}")
                else:
                    row.append("N/A")
                    
                if run_has_llm:
                    relev = b_data.get("avg_answer_relevance", 0.0)
                    row.append(f"{relev:.1%}")
                else:
                    row.append("N/A")
                    
                if run_has_llm and b != "B0":
                    cite = b_data.get("avg_citation_fidelity", 0.0)
                    row.append(f"{cite:.1%}")
                else:
                    row.append("N/A")
            
            row.extend([
                f"**{sem:.4f}**",
                f"{fill:.1%}",
                f"{lat:.2f}s"
            ])
            lines.append("| " + " | ".join(row) + " |")
            
    lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n[+] Created comparison Markdown report at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Science Graph RAG Benchmarking Metrics Parser")
    parser.add_argument(
        "--file", "-f",
        default="reports/result_metrics.yaml",
        help="Path to the result_metrics.yaml file (default: reports/result_metrics.yaml)"
    )
    parser.add_argument(
        "--output-md", "-o",
        default="reports/metrics_summary.md",
        help="Path where to save the Markdown summary report (default: reports/metrics_summary.md)"
    )
    parser.add_argument(
        "--csv",
        help="Path where to save raw summary stats as CSV (optional)"
    )
    parser.add_argument(
        "--csv-summary",
        help="Path where to save wide-format summary stats as CSV (optional)"
    )
    parser.add_argument(
        "--csv-details",
        help="Path where to save raw case-by-case metrics as CSV (optional)"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    output_md_path = Path(args.output_md)
    
    # Check if the input file path is a wildcard or folder pattern
    file_arg = args.file
    if file_arg.startswith("@"):
        file_arg = file_arg[1:]
        
    matched_paths = [Path(p) for p in glob.glob(file_arg)]
    if not matched_paths:
        script_dir = Path(__file__).resolve().parent
        matched_paths = [Path(p) for p in glob.glob(str(script_dir / file_arg))]
        
    run_dirs = []
    for p in matched_paths:
        if p.is_dir():
            if (p / "evaluation_results.yaml").exists() or (p / "result_metrics.yaml").exists():
                run_dirs.append(p)
        elif p.is_file() and p.name in ["evaluation_results.yaml", "result_metrics.yaml"]:
            run_dirs.append(p.parent)
            
    if run_dirs:
        print(f"[*] Found {len(run_dirs)} run directories matching the pattern.")
        run_summaries = []
        for run_dir in run_dirs:
            print(f"[*] Analyzing run directory: {run_dir.name}...")
            summary = analyze_run_directory(run_dir)
            if summary:
                run_summaries.append(summary)
        
        print_runs_comparison(run_summaries)
        generate_comparison_markdown_report(run_summaries, output_md_path)
        sys.exit(0)
        
    print(f"[*] Reading and parsing {file_path}...")
    data = load_yaml(file_path)
    
    stats = analyze_metrics(data)
    
    if HAS_RICH:
        print_rich_tables(stats)
    else:
        print_plain_tables(stats)
        
    generate_markdown_report(stats, output_md_path)
    
    # Export CSV if requested
    if args.csv:
        csv_path = Path(args.csv)
        print(f"[*] Exporting summary stats to {csv_path}...")
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Baseline", "Metric", "Mean", "Min", "Max", "Median", "StdDev"])
            for b in stats["baselines"]:
                for m in ALL_METRICS:
                    m_stats = stats["summary"][b][m]
                    writer.writerow([
                        b, m,
                        m_stats["mean"], m_stats["min"], m_stats["max"],
                        m_stats["median"], m_stats["stdev"]
                    ])
        print(f"[+] Exported CSV successfully.")

    if args.csv_summary:
        csv_summary_path = Path(args.csv_summary)
        print(f"[*] Exporting wide summary stats to {csv_summary_path}...")
        export_wide_csv(stats, csv_summary_path)
        print(f"[+] Exported summary CSV successfully.")

    if args.csv_details:
        csv_details_path = Path(args.csv_details)
        print(f"[*] Exporting detailed case metrics to {csv_details_path}...")
        export_detailed_csv(data, stats, csv_details_path)
        print(f"[+] Exported detailed CSV successfully.")


if __name__ == "__main__":
    main()
