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
    "B6": "Full Pipeline (Максимальный запуск) — все 13 компонентов (граф, реранкер, LLM-расширение, HyDE и др.)."
}

QUALITY_METRICS = [
    "retrieval_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
    "citation_fidelity",
    "semantic_accuracy"
]

ALL_METRICS = QUALITY_METRICS + ["latency_sec"]

METRIC_LABELS = {
    "retrieval_recall": "Retrieval Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "citation_fidelity": "Citation Fidelity",
    "semantic_accuracy": "Semantic Accuracy",
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
