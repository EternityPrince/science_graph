import csv
import yaml
import json
from pathlib import Path
from typing import Any
from core.models import parse_report, ReportOutput

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
from core.analytics import ALL_METRICS, METRIC_LABELS


NEW_CSV_FIELDS = [
    "graph_retrieval_enabled",
    "graph_retrieval_skip_reason",
    "query_concepts_all_count",
    "query_concepts_strong_count",
    "query_concepts_dropped_count",
    "query_concepts_all",
    "query_concepts_strong",
    "query_concepts_dropped",
    "graph_neighbor_nodes_total",
    "graph_neighbor_paper_nodes_count",
    "graph_neighbor_local_papers_count",
    "graph_neighbor_papers_with_chunks_count",
    "graph_neighbor_placeholder_or_external_count",
    "graph_neighbor_non_paper_nodes_count",
    "graph_neighbor_chunks_retrieved_count",
    "graph_concept_candidate_papers_count",
    "graph_bridge_candidate_papers_count",
    "graph_chunks_before_rerank_count",
    "graph_chunk_candidates_count",
    "graph_candidate_source_breakdown",
    "base_candidates_count",
    "merged_candidates_count_before_reranker",
    "reranker_input_count_before_limit",
    "reranker_input_count_after_limit",
    "candidate_count_after_reranker",
    "graph_candidate_rerank_positions",
    "best_graph_candidate_rank_after_rerank",
    "graph_chunks_survived_final_context_count",
    "graph_survival_rate",
    "graph_chunks_survived_final_context",
    "distinct_papers_in_final_context",
    "graph_neighbor_resolution_sample",
    "graph_concept_candidate_papers",
    "graph_bridge_candidate_papers",
    "graph_chunks_before_rerank"
]

NEW_SUMMARY_HEADERS = [
    "Graph Retrieval Enabled Rate",
    "Graph Retrieval Skipped Rate",
    "Avg Query Concepts",
    "Avg Strong Query Concepts",
    "Avg Dropped Query Concepts",
    "Avg Graph Neighbor Nodes",
    "Avg Graph Neighbor Paper Nodes",
    "Avg Graph Neighbor Local Papers",
    "Avg Graph Neighbor Papers With Chunks",
    "Avg Graph Neighbor Chunks Retrieved",
    "Avg Base Candidates",
    "Avg Graph Chunk Candidates",
    "Avg Merged Candidates Before Reranker",
    "Avg Reranker Input Before Limit",
    "Avg Reranker Input After Limit",
    "Avg Candidates After Reranker",
    "Graph Survival Rate",
    "Queries With Graph Chunks",
    "Queries With Graph Chunks Survived",
    "Avg Best Graph Candidate Rank",
    "Avg Graph Neighbor Candidates",
    "Avg Graph Concept Candidates",
    "Avg Graph Bridge Candidates"
]


NEW_CSV_FIELDS = [
    "graph_retrieval_enabled",
    "graph_retrieval_skip_reason",
    "query_concepts_all_count",
    "query_concepts_strong_count",
    "query_concepts_dropped_count",
    "query_concepts_all",
    "query_concepts_strong",
    "query_concepts_dropped",
    "graph_neighbor_nodes_total",
    "graph_neighbor_paper_nodes_count",
    "graph_neighbor_local_papers_count",
    "graph_neighbor_papers_with_chunks_count",
    "graph_neighbor_placeholder_or_external_count",
    "graph_neighbor_non_paper_nodes_count",
    "graph_neighbor_chunks_retrieved_count",
    "graph_concept_candidate_papers_count",
    "graph_bridge_candidate_papers_count",
    "graph_chunks_before_rerank_count",
    "graph_chunk_candidates_count",
    "graph_candidate_source_breakdown",
    "base_candidates_count",
    "merged_candidates_count_before_reranker",
    "reranker_input_count_before_limit",
    "reranker_input_count_after_limit",
    "candidate_count_after_reranker",
    "graph_candidate_rerank_positions",
    "best_graph_candidate_rank_after_rerank",
    "graph_chunks_survived_final_context_count",
    "graph_survival_rate",
    "graph_chunks_survived_final_context",
    "distinct_papers_in_final_context",
    "graph_neighbor_resolution_sample",
    "graph_concept_candidate_papers",
    "graph_bridge_candidate_papers",
    "graph_chunks_before_rerank"
]

NEW_SUMMARY_HEADERS = [
    "Graph Retrieval Enabled Rate",
    "Graph Retrieval Skipped Rate",
    "Avg Query Concepts",
    "Avg Strong Query Concepts",
    "Avg Dropped Query Concepts",
    "Avg Graph Neighbor Nodes",
    "Avg Graph Neighbor Paper Nodes",
    "Avg Graph Neighbor Local Papers",
    "Avg Graph Neighbor Papers With Chunks",
    "Avg Graph Neighbor Chunks Retrieved",
    "Avg Base Candidates",
    "Avg Graph Chunk Candidates",
    "Avg Merged Candidates Before Reranker",
    "Avg Reranker Input Before Limit",
    "Avg Reranker Input After Limit",
    "Avg Candidates After Reranker",
    "Graph Survival Rate",
    "Queries With Graph Chunks",
    "Queries With Graph Chunks Survived",
    "Avg Best Graph Candidate Rank",
    "Avg Graph Neighbor Candidates",
    "Avg Graph Concept Candidates",
    "Avg Graph Bridge Candidates"
]


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
            if m not in stats["summary"][b]:
                val = 0.0
            else:
                val = stats["summary"][b][m]["mean"]
            if m == "latency_sec":
                row.append(f"{val:.2f}s")
            elif m in ("token_output", "token_answer", "token_reasoning"):
                row.append(f"{val:.1f}")
            else:
                row.append(f"{val:.3f}")
        table.add_row(*row)
        
    console.print(table)
    console.print()
    
    # 1.5. Graph Retrieval Diagnostics Table (if trace exists)
    if stats.get("has_graph_trace"):
        table_graph = Table(title="[bold]Диагностика Graph Retrieval (Graph Retrieval Diagnostics)[/bold]", box=ROUNDED, header_style="bold green")
        table_graph.add_column("Baseline", style="cyan", no_wrap=True)
        table_graph.add_column("Enabled", justify="right")
        table_graph.add_column("Skipped", justify="right")
        table_graph.add_column("Avg Graph Chunks", justify="right")
        table_graph.add_column("Survival", justify="right")
        table_graph.add_column("Queries Survived", justify="right")
        table_graph.add_column("Avg Best Rank", justify="right")
        
        for b in stats["baselines"]:
            gd = stats["summary"][b].get("graph_diagnostics", {})
            enabled_pct = f"{gd.get('enabled_rate', 0.0) * 100:.1f}%"
            skipped_pct = f"{gd.get('skipped_rate', 0.0) * 100:.1f}%"
            avg_chunks = f"{gd.get('avg_graph_chunk_candidates', 0.0):.2f}"
            survival_pct = f"{gd.get('survival_rate', 0.0) * 100:.1f}%"
            queries_survived_pct = f"{gd.get('queries_with_chunks_survived', 0.0) * 100:.1f}%"
            avg_best_rank = f"{gd.get('avg_best_rank'):.2f}" if gd.get("avg_best_rank") is not None else "—"
            
            table_graph.add_row(
                b,
                enabled_pct,
                skipped_pct,
                avg_chunks,
                survival_pct,
                queries_survived_pct,
                avg_best_rank
            )
        console.print(table_graph)
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
            if m not in stats["summary"][b]:
                m_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0}
            else:
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
            elif m in ("token_output", "token_answer", "token_reasoning"):
                table_det.add_row(
                    METRIC_LABELS[m],
                    f"{mean_val:.1f}",
                    f"{min_val:.0f}",
                    f"{max_val:.0f}",
                    f"{med_val:.1f}",
                    f"{std_val:.1f}"
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
            if m not in stats["summary"][b]:
                val = 0.0
            else:
                val = stats["summary"][b][m]["mean"]
            if m == "latency_sec":
                row.append(f"{val:.2f}s")
            elif m in ("token_output", "token_answer", "token_reasoning"):
                row.append(f"{val:.1f}")
            else:
                row.append(f"{val:.3f}")
        print("\t".join(row))
    print()

    if stats.get("has_graph_trace"):
        print("--- Диагностика Graph Retrieval (Graph Retrieval Diagnostics) ---")
        print("\t".join(["Baseline", "Enabled", "Skipped", "Avg Chunks", "Survival", "Queries Survived", "Avg Best Rank"]))
        for b in stats["baselines"]:
            gd = stats["summary"][b].get("graph_diagnostics", {})
            enabled_pct = f"{gd.get('enabled_rate', 0.0) * 100:.1f}%"
            skipped_pct = f"{gd.get('skipped_rate', 0.0) * 100:.1f}%"
            avg_chunks = f"{gd.get('avg_graph_chunk_candidates', 0.0):.2f}"
            survival_pct = f"{gd.get('survival_rate', 0.0) * 100:.1f}%"
            queries_survived_pct = f"{gd.get('queries_with_chunks_survived', 0.0) * 100:.1f}%"
            avg_best_rank = f"{gd.get('avg_best_rank'):.2f}" if gd.get("avg_best_rank") is not None else "—"
            print("\t".join([
                b, enabled_pct, skipped_pct, avg_chunks, survival_pct, queries_survived_pct, avg_best_rank
            ]))
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
        if m in ("token_output", "token_answer", "token_reasoning"):
            continue
        if any(m not in stats["summary"][b] for b in stats["baselines"]):
            continue
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
            if m not in stats["summary"][b]:
                val = 0.0
            else:
                val = stats["summary"][b][m]["mean"]
            if m == "latency_sec":
                cell_str = f"{val:.2f}s"
            elif m in ("token_output", "token_answer", "token_reasoning"):
                cell_str = f"{val:.1f}"
            else:
                cell_str = f"{val:.3f}"
            
            if m in best_per_metric and best_per_metric[m] == b:
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
            if m not in stats["summary"][b]:
                m_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "stdev": 0.0}
            else:
                m_stats = stats["summary"][b][m]
            mean_val = m_stats["mean"]
            min_val = m_stats["min"]
            max_val = m_stats["max"]
            med_val = m_stats["median"]
            std_val = m_stats["stdev"]
            
            if m == "latency_sec":
                lines.append(
                    f"| {METRIC_LABELS[m]} | "
                    f"{mean_val:.3f}s | "
                    f"{min_val:.3f}s | "
                    f"{max_val:.3f}s | "
                    f"{med_val:.3f}s | "
                    f"{std_val:.3f}s |"
                )
            elif m in ("token_output", "token_answer", "token_reasoning"):
                lines.append(
                    f"| {METRIC_LABELS[m]} | "
                    f"{mean_val:.1f} | "
                    f"{min_val:.0f} | "
                    f"{max_val:.0f} | "
                    f"{med_val:.1f} | "
                    f"{std_val:.1f} |"
                )
            else:
                lines.append(
                    f"| {METRIC_LABELS[m]} | "
                    f"{mean_val:.3f} | "
                    f"{min_val:.3f} | "
                    f"{max_val:.3f} | "
                    f"{med_val:.3f} | "
                    f"{std_val:.3f} |"
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

    # Graph Retrieval Diagnostics Section
    if stats.get("has_graph_trace"):
        lines.append("## 📊 Graph Retrieval Diagnostics")
        lines.append("")
        
        def format_pct(val):
            if val is None:
                return "—"
            return f"{val * 100:.1f}%"

        def format_avg(val):
            if val is None:
                return "—"
            return f"{val:.2f}"

        # Table 1: Core diagnostics
        lines.append("| Baseline | Enabled Rate | Skipped Rate | Avg Neighbor Nodes | Avg Local Papers | Avg Papers w/ Chunks | Avg Graph Chunks | Graph Survival Rate | Queries w/ Survived Graph Chunks | Avg Best Graph Rank |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for b in stats["baselines"]:
            gd = stats["summary"][b].get("graph_diagnostics", {})
            row = [
                f"**{b}**",
                format_pct(gd.get("enabled_rate")),
                format_pct(gd.get("skipped_rate")),
                format_avg(gd.get("avg_neighbor_nodes")),
                format_avg(gd.get("avg_neighbor_local_papers")),
                format_avg(gd.get("avg_neighbor_papers_with_chunks")),
                format_avg(gd.get("avg_graph_chunk_candidates")),
                format_pct(gd.get("survival_rate")),
                format_pct(gd.get("queries_with_chunks_survived")),
                format_avg(gd.get("avg_best_rank"))
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Table 2: Source Breakdown
        lines.append("### Graph Candidate Source Breakdown")
        lines.append("")
        lines.append("| Baseline | Avg Neighbor Candidates | Avg Concept Candidates | Avg Bridge Candidates |")
        lines.append("|---|---:|---:|---:|")
        for b in stats["baselines"]:
            gd = stats["summary"][b].get("graph_diagnostics", {})
            row = [
                f"**{b}**",
                format_avg(gd.get("avg_neighbor_cand")),
                format_avg(gd.get("avg_concept_cand")),
                format_avg(gd.get("avg_bridge_cand"))
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Table 3: Query Concept Filtering
        lines.append("### Query Concept Filtering")
        lines.append("")
        lines.append("| Baseline | Avg Query Concepts | Avg Strong Concepts | Avg Dropped Concepts |")
        lines.append("|---|---:|---:|---:|")
        for b in stats["baselines"]:
            gd = stats["summary"][b].get("graph_diagnostics", {})
            row = [
                f"**{b}**",
                format_avg(gd.get("avg_concepts")),
                format_avg(gd.get("avg_strong")),
                format_avg(gd.get("avg_dropped"))
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Table 4: Category Breakdown
        if stats.get("category_graph_stats"):
            lines.append("### Graph Retrieval by Category")
            lines.append("")
            lines.append("| Category | Baseline | Avg Graph Chunks | Graph Survival Rate | Queries w/ Survived Graph Chunks | Avg Distinct Papers |")
            lines.append("|---|---|---:|---:|---:|---:|")
            cat_stats = stats["category_graph_stats"]
            for cat in sorted(cat_stats.keys()):
                for b in stats["baselines"]:
                    b_cat = cat_stats[cat].get(b, {})
                    row = [
                        f"`{cat}`",
                        f"**{b}**",
                        format_avg(b_cat.get("avg_graph_chunk_candidates")),
                        format_pct(b_cat.get("graph_survival_rate")),
                        format_pct(b_cat.get("queries_with_graph_chunks_survived")),
                        format_avg(b_cat.get("avg_distinct_papers_in_final_context"))
                    ]
                    lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        # Table 5: Failure examples
        if stats.get("top_failures"):
            lines.append("### Graph Retrieval Failure Examples")
            lines.append("")
            lines.append("| Query ID | Baseline | Category | Skip Reason | Neighbor Nodes | Papers w/ Chunks | Graph Chunks | Survived |")
            lines.append("|---|---|---|---|---:|---:|---:|---:|")
            for item in stats["top_failures"]:
                row = [
                    f"`{item['query_id']}`",
                    f"**{item['baseline']}**",
                    f"`{item['category']}`",
                    item["skip_reason"] if item["skip_reason"] else "—",
                    str(item["neighbor_nodes"]),
                    str(item["papers_with_chunks"]),
                    str(item["chunks_before"]),
                    str(item["chunks_survived"])
                ]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    else:
        lines.append("## Graph Retrieval Diagnostics")
        lines.append("")
        lines.append("Graph retrieval trace was not found for this run.")
        lines.append("")
    
    lines.append("## 🏆 Главные выводы (Research Summary)")
    lines.append("")
    
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
            "Faithfulness", "Relevance", "Citations", "Semantic Accuracy", "Latency (sec)",
            "Token Output", "Token Answer", "Token Reasoning"
        ] + NEW_SUMMARY_HEADERS)
        for b in stats["baselines"]:
            sr = f"{stats['summary'][b]['success_rate']:.1f}%"
            
            def get_val(metric_name):
                if b == "B0" and metric_name in ["retrieval_recall", "context_precision", "faithfulness", "citation_fidelity"]:
                    return "N/A"
                if metric_name not in stats["summary"][b]:
                    return "N/A"
                if stats["summary"][b][metric_name]["count"] == 0:
                    return "N/A"
                val = stats["summary"][b][metric_name]["mean"]
                if metric_name == "latency_sec":
                    return f"{val:.2f}"
                elif metric_name in ("token_output", "token_answer", "token_reasoning"):
                    return f"{val:.1f}"
                else:
                    return f"{val:.4f}"
            
            row_vals = [
                b,
                sr,
                get_val("retrieval_recall"),
                get_val("context_precision"),
                get_val("faithfulness"),
                get_val("answer_relevance"),
                get_val("citation_fidelity"),
                get_val("semantic_accuracy"),
                get_val("latency_sec"),
                get_val("token_output"),
                get_val("token_answer"),
                get_val("token_reasoning")
            ]

            gd = stats["summary"][b].get("graph_diagnostics", {})
            
            def get_gd_val(key, default=0.0):
                val = gd.get(key)
                if val is None:
                    return default
                return val

            row_vals += [
                get_gd_val("enabled_rate"),
                get_gd_val("skipped_rate"),
                get_gd_val("avg_concepts"),
                get_gd_val("avg_strong"),
                get_gd_val("avg_dropped"),
                get_gd_val("avg_neighbor_nodes"),
                get_gd_val("avg_neighbor_paper_nodes"),
                get_gd_val("avg_neighbor_local_papers"),
                get_gd_val("avg_neighbor_papers_with_chunks"),
                get_gd_val("avg_neighbor_chunks_retrieved"),
                get_gd_val("avg_base_candidates"),
                get_gd_val("avg_graph_chunk_candidates"),
                get_gd_val("avg_merged_before"),
                get_gd_val("avg_reranker_before"),
                get_gd_val("avg_reranker_after"),
                get_gd_val("avg_candidates_after"),
                get_gd_val("survival_rate"),
                get_gd_val("queries_with_chunks"),
                get_gd_val("queries_with_chunks_survived"),
                gd.get("avg_best_rank") if gd.get("avg_best_rank") is not None else "",
                get_gd_val("avg_neighbor_cand"),
                get_gd_val("avg_concept_cand"),
                get_gd_val("avg_bridge_cand")
            ]
            
            writer.writerow(row_vals)


def export_detailed_csv(data: Any, stats: dict, csv_path: Path) -> None:
    """Saves detailed case-by-case metrics for each baseline and query."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, ReportOutput):
        data_dict = data.model_dump()
    else:
        data_dict = parse_report(data).model_dump()
        
    results = data_dict.get("results", [])
    if not results:
        return
        
    baselines = stats["baselines"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "category", "baseline", "status", "latency_sec",
            "is_answerable",
            "retrieval_recall", "context_precision", "faithfulness",
            "answer_relevance", "citation_fidelity", "semantic_accuracy",
            "ar_sa_f1",
            "token_output", "token_answer", "token_reasoning",
            "seed_chunks_from_lexical_dense", "seed_paper_id_list",
            "graph_neighbor_paper_id_list", "candidate_count_before_reranker",
            "candidate_count_after_reranker", "final_context_paper_id_list",
            "final_context_token_count", "whether_graph_neighbor_chunk_survived_into_final_context",
            "answer_token_count"
        ] + NEW_CSV_FIELDS)
        
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
                
                trace = b_data.get("trace", {}) if b_data else {}
                seed_chunks = trace.get("seed_chunks_from_lexical_dense") if isinstance(trace, dict) else None
                if isinstance(seed_chunks, dict):
                    lex_chunks = seed_chunks.get("lexical", [])
                    dense_chunks = seed_chunks.get("dense", [])
                    seed_chunks_str = f"lexical:{','.join(map(str, lex_chunks))}|dense:{','.join(map(str, dense_chunks))}"
                else:
                    seed_chunks_str = ""
                    
                if isinstance(trace, dict):
                    seed_papers = ", ".join(map(str, trace.get("seed_paper_id_list", [])))
                    neighbor_papers = ", ".join(map(str, trace.get("graph_neighbor_paper_id_list", [])))
                    cand_before = trace.get("candidate_count_before_reranker", "")
                    cand_after = trace.get("candidate_count_after_reranker", "")
                    final_papers = ", ".join(map(str, trace.get("final_context_paper_id_list", [])))
                    final_tokens = trace.get("final_context_token_count", "")
                    neighbor_survived = trace.get("whether_graph_neighbor_chunk_survived_into_final_context", "")
                    ans_tokens = trace.get("answer_token_count", "")
                else:
                    seed_papers = ""
                    neighbor_papers = ""
                    cand_before = ""
                    cand_after = ""
                    final_papers = ""
                    final_tokens = ""
                    neighbor_survived = ""
                    ans_tokens = ""

                def format_csv_cell(val):
                    if val is None:
                        return ""
                    if isinstance(val, bool):
                        return str(val)
                    if isinstance(val, (list, dict)):
                        return json.dumps(val, ensure_ascii=False)
                    return str(val)

                is_ans = r.get("is_answerable")
                if is_ans is None:
                    is_ans = True
                else:
                    is_ans = bool(is_ans)
                
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

                row_vals = [
                    q_id,
                    category,
                    b,
                    status,
                    latency if latency is not None else "",
                    is_ans,
                    get_metric("retrieval_recall"),
                    get_metric("context_precision"),
                    get_metric("faithfulness"),
                    get_metric("answer_relevance"),
                    get_metric("citation_fidelity"),
                    get_metric("semantic_accuracy"),
                    ar_f1 if ar_f1 is not None else "",
                    get_metric("token_output"),
                    get_metric("token_answer"),
                    get_metric("token_reasoning"),
                    seed_chunks_str,
                    seed_papers,
                    neighbor_papers,
                    cand_before,
                    cand_after,
                    final_papers,
                    final_tokens,
                    neighbor_survived,
                    ans_tokens
                ]

                # Append new graph retrieval diagnostic fields
                for field in NEW_CSV_FIELDS:
                    val = b_data.get(field)
                    row_vals.append(format_csv_cell(val))

                writer.writerow(row_vals)


def save_judge_report(human_data: dict, judge_output_path: Path) -> None:
    """Creates and saves a simplified evaluation report for the LLM judge."""
    from core.evaluator import get_clean_judge_answer
    judge_results = []
    for case in human_data.get("results", []):
        judge_case = {
            "id": case.get("id"),
            "query": case.get("query"),
            "golden_answer": case.get("golden_answer"),
            "baselines": {}
        }
        for baseline, data in case.get("baselines", {}).items():
            raw_ans = data.get("generated_answer", "")
            baseline_entry = {
                "generated_answer": get_clean_judge_answer(raw_ans)
            }
            if "eval_metrics" in data:
                baseline_entry["eval_metrics"] = data["eval_metrics"]
            if "eval_details" in data:
                baseline_entry["eval_details"] = data["eval_details"]
            judge_case["baselines"][baseline] = baseline_entry
        judge_results.append(judge_case)
        
    judge_data = {
        "results": judge_results
    }
    
    with open(judge_output_path, "w", encoding="utf-8") as f:
        yaml.dump(judge_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def save_individual_judge_reports(human_data: dict, output_dir: Path, output_stem: str, output_suffix: str) -> None:
    """Creates and saves individual simplified evaluation reports for each baseline."""
    from core.evaluator import get_clean_judge_answer
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
                raw_ans = baseline_data.get("generated_answer", "")
                baseline_entry = {
                    "generated_answer": get_clean_judge_answer(raw_ans)
                }
                if "eval_metrics" in baseline_data:
                    baseline_entry["eval_metrics"] = baseline_data["eval_metrics"]
                if "eval_details" in baseline_data:
                    baseline_entry["eval_details"] = baseline_data["eval_details"]
                judge_case = {
                    "id": case.get("id"),
                    "query": case.get("query"),
                    "golden_answer": case.get("golden_answer"),
                    "baselines": {
                        baseline: baseline_entry
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
