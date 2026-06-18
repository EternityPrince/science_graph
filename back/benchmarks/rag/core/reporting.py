import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Rich imports for terminal formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from core.config import BASELINES_INFO
from core.analytics import ALL_METRICS, QUALITY_METRICS, METRIC_LABELS


def print_rich_tables(stats: dict) -> None:
    """Prints beautiful summary tables to the terminal using Rich."""
    if not HAS_RICH:
        print_plain_tables(stats)
        return
        
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
        sr = stats["summary"][b]["success_rate"]
        row.append(f"{sr:.1f}%")
        
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


def print_plain_tables(stats: dict) -> None:
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


def generate_markdown_report(stats: dict, output_path: Path) -> None:
    """Generates a beautiful self-contained Markdown report with highlights, tables, and emojis."""
    lines = []
    lines.append("# 📊 Отчет по качеству RAG-системы (RAG Benchmarking Report)")
    lines.append("")
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
    
    headers = ["Baseline", "Success Rate"] + [METRIC_LABELS[m] for m in ALL_METRICS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| :--- | " + " | ".join(["---:"] * (len(headers) - 1)) + " |")
    
    best_per_metric = {}
    for m in ALL_METRICS:
        if m == "latency_sec":
            best_val = min(stats["summary"][b][m]["mean"] for b in stats["baselines"])
            best_b = [b for b in stats["baselines"] if stats["summary"][b][m]["mean"] == best_val][0]
        else:
            best_val = max(stats["summary"][b][m]["mean"] for b in stats["baselines"])
            best_b = [b for b in stats["baselines"] if stats["summary"][b][m]["mean"] == best_val][0]
        best_per_metric[m] = best_b

    for b in stats["baselines"]:
        sr = stats["summary"][b]["success_rate"]
        row = [f"**{b}**", f"{sr:.1f}%"]
        for m in ALL_METRICS:
            val = stats["summary"][b][m]["mean"]
            cell_str = f"{val:.3f}" if m != "latency_sec" else f"{val:.2f}s"
            
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
    
    best_accuracy_b = best_per_metric["semantic_accuracy"]
    best_recall_b = best_per_metric["retrieval_recall"]
    fastest_b = best_per_metric["latency_sec"]
    
    lines.append(f"1. **Абсолютный лидер по качеству ответов:** `{best_accuracy_b}` с Semantic Accuracy **{stats['summary'][best_accuracy_b]['semantic_accuracy']['mean']:.3f}**.")
    lines.append(f"2. **Лучшая глубина поиска (Retrieval Recall):** `{best_recall_b}` с показателем **{stats['summary'][best_recall_b]['retrieval_recall']['mean']:.3f}**.")
    lines.append(f"3. **Самый быстрый отклик (Latency):** `{fastest_b}` со средним временем **{stats['summary'][fastest_b]['latency_sec']['mean']:.2f} сек**.")
    
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
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n[+] Created beautiful Markdown report at: {output_path}")


def export_wide_csv(stats: dict, csv_path: Path) -> None:
    """Saves wide-format aggregated summary of metrics per baseline."""
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


def export_detailed_csv(data: dict, stats: dict, csv_path: Path) -> None:
    """Saves detailed case-by-case metrics for each baseline and query."""
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


def save_judge_report(human_data: dict, judge_output_path: Path) -> None:
    """Creates and saves a simplified evaluation report for the LLM judge."""
    judge_results = []
    for case in human_data.get("results", []):
        judge_case = {
            "id": case.get("id"),
            "query": case.get("query"),
            "golden_answer": case.get("golden_answer"),
            "baselines": {}
        }
        for baseline, data in case.get("baselines", {}).items():
            judge_case["baselines"][baseline] = {
                "generated_answer": data.get("generated_answer", "")
            }
        judge_results.append(judge_case)
        
    judge_data = {
        "results": judge_results
    }
    
    with open(judge_output_path, "w", encoding="utf-8") as f:
        yaml.dump(judge_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def save_individual_judge_reports(human_data: dict, output_dir: Path, output_stem: str, output_suffix: str) -> None:
    """Creates and saves individual simplified evaluation reports for each baseline."""
    baselines_dir = output_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    
    all_baselines = set()
    for case in human_data.get("results", []):
        for baseline in case.get("baselines", {}).keys():
            all_baselines.add(baseline)
            
    for baseline in sorted(all_baselines):
        judge_results = []
        for case in human_data.get("results", []):
            baseline_data = case.get("baselines", {}).get(baseline)
            if baseline_data:
                judge_case = {
                    "id": case.get("id"),
                    "query": case.get("query"),
                    "golden_answer": case.get("golden_answer"),
                    "baselines": {
                        baseline: {
                            "generated_answer": baseline_data.get("generated_answer", "")
                        }
                    }
                }
                judge_results.append(judge_case)
                
        judge_data = {
            "results": judge_results
        }
        
        baseline_lower = baseline.lower()
        baseline_file_name = f"{output_stem}_judge_{baseline_lower}{output_suffix}"
        baseline_output_path = baselines_dir / baseline_file_name
        
        with open(baseline_output_path, "w", encoding="utf-8") as f:
            yaml.dump(judge_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
