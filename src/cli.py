"""
Science Graph — CLI entry point.
All user-facing output uses src.console for consistent rich formatting.
"""

import json
import os
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table
from rich import box

from src import console as con
from src.config import config
from src.indexer import Indexer
from src.llm_engine import LLMEngine
from src.rag import RAGPipeline
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine

app = typer.Typer(
    help="Science Graph — local AI-powered knowledge base for papers, notes & books",
    no_args_is_help=True,
)


# ── Service factory ───────────────────────────────────────────────────────────

def get_services(load_llm: bool = True, load_embeddings: bool = True):
    """Initializes and returns database repositories and engines."""
    graph_repo = SQLiteGraphRepository(config.db_path)
    vector_repo = SQLiteVectorRepository(config.db_path)

    embedding_engine = None
    if load_embeddings:
        embedding_engine = EmbeddingEngine()

    llm_engine = None
    if load_llm:
        try:
            llm_engine = LLMEngine()
        except Exception as e:
            con.error(f"Could not load LLM engine: {e}")

    return graph_repo, vector_repo, embedding_engine, llm_engine


# ── index ─────────────────────────────────────────────────────────────────────

def print_trace_table(source_name: str, trace_info: dict) -> None:
    """Prints a premium, visually satisfying trace table for the ingestion stages."""
    table = Table(
        title=f"Ingestion Trace: {source_name}",
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        show_footer=True,
        box=box.ROUNDED,
    )
    table.add_column("Stage", footer="[accent]Total Duration[/accent]")
    table.add_column("Duration", justify="right", style="yellow")
    table.add_column("LLM Tokens", justify="right", style="bold magenta")

    stages = trace_info.get("stages", {})
    tokens = trace_info.get("tokens", {})

    total_duration = sum(stages.values())
    total_tokens = sum(tokens.values())

    chronological_stages = [
        "Document Parsing",
        "NER Author Fallback",
        "Metadata Enrichment",
        "Concept & Tag Extraction",
        "Graph Persistence",
        "Chunking & Embedding",
        "Archiving",
        "Summary Generation",
    ]

    seen_stages = set()

    for stage_name in chronological_stages:
        if stage_name in stages or stage_name in tokens:
            seen_stages.add(stage_name)
            duration_val = stages.get(stage_name, 0.0)
            duration_str = f"{duration_val:.3f}s" if stage_name in stages else "-"
            token_val = tokens.get(stage_name, 0)
            token_str = f"{token_val:,}" if token_val > 0 else "-"
            table.add_row(stage_name, duration_str, token_str)

    # Any other timing log
    for stage_name, duration_val in stages.items():
        if stage_name not in seen_stages:
            seen_stages.add(stage_name)
            token_val = tokens.get(stage_name, 0)
            token_str = f"{token_val:,}" if token_val > 0 else "-"
            table.add_row(stage_name, f"{duration_val:.3f}s", token_str)

    # Any other token log
    for token_stage, token_val in tokens.items():
        if token_stage not in seen_stages:
            table.add_row(token_stage, "-", f"{token_val:,}")

    # Set footers for totals
    table.columns[1].footer = f"[bold yellow]{total_duration:.3f}s[/bold yellow]"
    table.columns[2].footer = f"[bold magenta]{total_tokens:,}[/bold magenta]" if total_tokens > 0 else "-"

    con.blank()
    con.console.print(table)
    con.blank()


# ── index ─────────────────────────────────────────────────────────────────────

@app.command("index")
def index(
    target: str = typer.Argument(..., help="Path to file, directory, or URL to index"),
    use_llm: bool = typer.Option(True, "--use-llm/--no-llm", help="Use LLM to extract concepts (slower)"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Show detailed execution trace with timing and token count"),
):
    """Index PDF papers, Markdown notes (.md), EPUB books, or URLs into the knowledge graph."""
    if trace:
        con.SHOW_TIME = True
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=use_llm)
    if use_llm and not llm_engine:
        con.warning("Proceeding with regex fallback extraction because LLM engine failed to load.")
    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    def _index_file(path: Path) -> bool:
        t = path.suffix.lower().lstrip(".")
        try:
            if trace:
                trace_info = {"stages": {}, "tokens": {}}
                if t == "pdf":
                    indexer.index_pdf(str(path), trace_info=trace_info)
                elif t == "md":
                    indexer.index_markdown(str(path), trace_info=trace_info)
                elif t == "epub":
                    indexer.index_epub(str(path), trace_info=trace_info)
                else:
                    con.warning(f"Unknown file type '{t}' for {path.name}, skipping.")
                    return False
                print_trace_table(path.name, trace_info)
            else:
                if t == "pdf":
                    indexer.index_pdf(str(path))
                elif t == "md":
                    indexer.index_markdown(str(path))
                elif t == "epub":
                    indexer.index_epub(str(path))
                else:
                    con.warning(f"Unknown file type '{t}' for {path.name}, skipping.")
                    return False
            return True
        except Exception as e:
            con.error(f"Failed to index {path.name}: {e}")
            return False

    if target.startswith("http://") or target.startswith("https://"):
        try:
            if trace:
                trace_info = {"stages": {}, "tokens": {}}
                indexer.index_url(target, trace_info=trace_info)
                print_trace_table(target, trace_info)
            else:
                indexer.index_url(target)
        except Exception as e:
            con.error(f"Failed to index url {target}: {e}")
        return

    path = Path(target).resolve()
    if not path.exists():
        con.error(f"Path not found: {path}")
        raise typer.Exit(1)

    if path.is_file():
        _index_file(path)
    elif path.is_dir():
        allowed = {".pdf", ".md", ".epub"}
        files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in allowed]
        if not files:
            con.warning(f"No supported files found in {path}")
            return

        con.info(f"Found [bold]{len(files)}[/bold] files — starting indexing …")
        ok = sum(1 for f in files if _index_file(f))
        if ok == len(files):
            con.success(f"All {ok} files indexed successfully")
        else:
            con.warning(f"{ok}/{len(files)} files indexed ({len(files)-ok} failed)")


# ── reindex ───────────────────────────────────────────────────────────────────

reindex_app = typer.Typer(help="Re-index paper metadata or everything.")
app.add_typer(reindex_app, name="reindex")

@reindex_app.command("meta")
def reindex_meta(
    missing_authors: bool = typer.Option(False, "--missing-authors", help="Reindex only papers without authors"),
    missing_tags: bool = typer.Option(False, "--missing-tags", help="Reindex only papers without topic tags"),
    all_metadata: bool = typer.Option(False, "--all-metadata", help="Reindex metadata for all papers"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of papers to reindex"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for extracting concepts/tags (slower)"),
):
    """Partially re-index paper metadata (authors, year, tags, citations) without regenerating embeddings."""
    if not all_metadata and not missing_authors and not missing_tags:
        con.warning("Please specify a filter: --missing-authors, --missing-tags, or --all-metadata")
        raise typer.Exit(0)

    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=use_llm)
    if use_llm and not llm_engine:
        con.warning("Proceeding with regex fallback extraction because LLM engine failed to load.")
    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    # Find candidate paper IDs
    non_placeholders = graph_repo.get_non_placeholder_paper_ids()

    candidates = []
    for pid in non_placeholders:
        paper = graph_repo.get_paper(pid)
        if not paper:
            continue
        props = paper.properties
        if missing_authors:
            if not paper.authors:
                candidates.append(pid)
        elif missing_tags:
            tags = props.get("tags", [])
            if not tags:
                candidates.append(pid)
        else:
            candidates.append(pid)

    if limit:
        candidates = candidates[:limit]

    if not candidates:
        con.success("No papers found matching the re-indexing criteria.")
        return

    con.info(f"Starting metadata re-indexing for [bold]{len(candidates)}[/bold] papers …")
    
    success_count = 0
    for paper_id in candidates:
        try:
            if indexer.reindex_metadata(paper_id, use_llm=use_llm):
                success_count += 1
        except Exception as e:
            con.error(f"Failed to re-index {paper_id}: {e}")

    con.blank()
    con.success(f"Re-indexed {success_count}/{len(candidates)} papers successfully.")


@reindex_app.command("full")
def reindex_full(
    all_papers: bool = typer.Option(False, "--all", help="Reindex all papers"),
    paper_id: Optional[str] = typer.Option(None, "--id", help="Reindex a single paper by ID"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of papers to reindex"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for extracting concepts/tags (slower)"),
):
    """Fully re-index papers (re-chunk and recreate embeddings) by re-ingesting original files/URLs."""
    if not all_papers and not paper_id:
        con.warning("Please specify either --all or --id <paper_id>")
        raise typer.Exit(0)

    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=use_llm)
    if use_llm and not llm_engine:
        con.warning("Proceeding with regex fallback extraction because LLM engine failed to load.")
    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    if paper_id:
        # Check if exists
        paper = graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found: {paper_id}")
            raise typer.Exit(1)
        candidates = [paper_id]
    else:
        candidates = graph_repo.get_non_placeholder_paper_ids()

    if limit:
        candidates = candidates[:limit]

    if not candidates:
        con.success("No papers found matching the re-indexing criteria.")
        return

    con.info(f"Starting full re-indexing for [bold]{len(candidates)}[/bold] papers …")
    
    success_count = 0
    for pid in candidates:
        try:
            if indexer.reindex_full(pid):
                success_count += 1
        except Exception as e:
            con.error(f"Failed to fully re-index {pid}: {e}")

    con.blank()
    con.success(f"Fully re-indexed {success_count}/{len(candidates)} papers successfully.")



# ── query ─────────────────────────────────────────────────────────────────────

@app.command("query")
def query(
    text: str = typer.Argument(..., help="Your question about the indexed documents"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of context chunks to retrieve"),
):
    """Answer a question using hybrid RAG over all indexed documents."""
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True)
    if not llm_engine:
        con.error("LLM engine is required for query. Check your model path with: graph config")
        raise typer.Exit(1)

    pipeline = RAGPipeline(graph_repo, vector_repo, embedding_engine, llm_engine)

    con.blank()
    con.search_msg(f"{text}")
    con.blank()

    response = pipeline.ask(text, limit=limit)

    con.blank()
    con.console.print(Panel(
        response,
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))
    con.blank()


# ── stats ─────────────────────────────────────────────────────────────────────

@app.command("stats")
def stats():
    """Show knowledge base statistics and disk storage details."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    db_stats = graph_repo.get_stats()

    # DB file size
    db_path = Path(config.db_path)
    db_size = ""
    if db_path.exists():
        sz = db_path.stat().st_size
        db_size = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.0f} KB"

    table = Table(
        title="📊 Knowledge Base Statistics",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Category", style="bold white", min_width=24)
    table.add_column("Count", justify="right", style="bold green")

    table.add_row("Papers / Books / Notes", str(db_stats["papers"]))
    table.add_row("Authors", str(db_stats["authors"]))
    table.add_row("Concepts", str(db_stats["concepts"]))
    table.add_row("Graph Edges", str(db_stats["edges"]))
    table.add_row("─" * 24, "─" * 8)
    table.add_row("Database Size", db_size or "—")

    con.blank()
    con.console.print(table)
    con.blank()

    # Storage stats
    storage_stats = config.get_storage_stats()
    
    def format_size(bytes_val: int) -> str:
        if bytes_val >= 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"
        elif bytes_val >= 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.2f} MB"
        elif bytes_val >= 1024:
            return f"{bytes_val / 1024:.2f} KB"
        else:
            return f"{bytes_val} B"

    con.console.print(Panel(
        f"[bold white]📂 Storage Location:[/bold white] [cyan]{storage_stats['storage_dir']}[/cyan]\n"
        f"[bold white]📦 Total Storage Size:[/bold white] [green]{format_size(storage_stats['total_size'])}[/green]",
        title="💾 Disk Storage Information",
        border_style="blue",
        expand=False
    ))
    con.blank()

    # Breakdown by Extensions
    if storage_stats["extensions"]:
        ext_table = Table(
            title="🗂️ Storage Breakdown by File Type (Extensions)",
            box=box.MINIMAL_DOUBLE_HEAD,
            border_style="blue",
            show_header=True,
            header_style="bold blue",
        )
        ext_table.add_column("Extension", style="bold white")
        ext_table.add_column("Files Count", justify="right", style="magenta")
        ext_table.add_column("Size", justify="right", style="green")
        ext_table.add_column("Percentage", justify="right", style="yellow")

        tot = storage_stats["total_size"]
        for item in storage_stats["extensions"]:
            pct = (item["size"] / tot * 100) if tot > 0 else 0
            ext_table.add_row(
                item["extension"],
                str(item["count"]),
                format_size(item["size"]),
                f"{pct:.1f}%"
            )
        con.console.print(ext_table)
        con.blank()

    # Breakdown by Sources
    if storage_stats["sources"]:
        src_table = Table(
            title="📥 Archive Breakdown by Source Type",
            box=box.MINIMAL_DOUBLE_HEAD,
            border_style="blue",
            show_header=True,
            header_style="bold blue",
        )
        src_table.add_column("Source Type", style="bold white")
        src_table.add_column("Files Count", justify="right", style="magenta")
        src_table.add_column("Size", justify="right", style="green")

        source_labels = {
            "paper": "📚 Papers / Articles",
            "note": "📝 Notes (Markdown)",
            "book": "📖 Books (EPUB)",
            "webpage": "🌐 Webpages (Scraped)",
            "other": "❓ Other / Uncategorized"
        }

        for item in storage_stats["sources"]:
            lbl = source_labels.get(item["source"], f"❓ {item['source']}")
            src_table.add_row(
                lbl,
                str(item["count"]),
                format_size(item["size"])
            )
        con.console.print(src_table)
        con.blank()


# ── storage ───────────────────────────────────────────────────────────────────

@app.command("storage")
def storage(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of items to display per page"),
):
    """Display indexed data with interactive pagination, deletion and editing."""
    import click
    import math

    TABLES = ["documents", "authors", "concepts"]
    TABLE_LABELS = {"documents": "📚 Documents", "authors": "👥 Authors", "concepts": "🧠 Concepts"}

    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)

    page = 1
    active_table = "documents"   # currently focused table
    selected_idx = None          # 1-based row number within the current page
    status_msg = ""              # feedback line after actions
    search_query = None          # search term
    llm_engine = None            # lazy LLM engine reference

    def _get_rows(tbl: str, pg: int):
        return graph_repo.get_browse_rows(tbl, pg, limit, search_query)

    def _count(tbl: str) -> int:
        return graph_repo.get_browse_count(tbl, search_query)

    def _delete_node(node_id: str) -> str:
        try:
            graph_repo.delete_node(node_id)
            return f"[bold red]✗  Deleted[/bold red] node [dim]{node_id[:40]}[/dim]"
        except Exception as exc:
            return f"[red]Error deleting: {exc}[/red]"

    def _edit_node(node_id: str, tbl: str) -> str:
        """Interactive field editor — prompts for new values for key fields."""
        node = graph_repo.get_node_by_id(node_id)
        if not node:
            return "[red]Record not found.[/red]"
        label, props_str = node
        props = json.loads(props_str)
        con.console.print("\n[bold yellow]— Edit Record —[/bold yellow] (Enter = keep current, Ctrl+C = cancel)\n")
        try:
            if tbl == "documents":
                new_title = input(f"  Title [{props.get('title', '')}]: ").strip()
                if new_title:
                    props["title"] = new_title
                raw_authors = props.get("authors", [])
                new_authors = input(f"  Authors [{', '.join(raw_authors)}]: ").strip()
                if new_authors:
                    props["authors"] = [a.strip() for a in new_authors.split(",") if a.strip()]
            elif tbl == "authors":
                new_name = input(f"  Name [{props.get('name', '')}]: ").strip()
                if new_name:
                    props["name"] = new_name
            else:  # concepts
                new_name = input(f"  Name [{props.get('name', '')}]: ").strip()
                if new_name:
                    props["name"] = new_name
            graph_repo.update_node_properties(node_id, props)
            return "[bold green]✓  Record updated[/bold green]"
        except KeyboardInterrupt:
            return "[yellow]Edit cancelled.[/yellow]"

    while True:
        con.console.clear()

        total = _count(active_table)
        total_pages = max(1, math.ceil(total / limit))
        page = max(1, min(page, total_pages))

        rows = _get_rows(active_table, page)

        # ── Build the active table ───────────────────────────────────────────
        tab_title = TABLE_LABELS[active_table]
        border = {"documents": "blue", "authors": "yellow", "concepts": "magenta"}[active_table]
        search_suffix = f" (filtered: '{search_query}')" if search_query else ""
        table = Table(
            title=f"{tab_title}  [dim](page {page}/{total_pages}, {total} total){search_suffix}[/dim]",
            box=box.ROUNDED, border_style=border, expand=True, header_style="bold cyan",
        )
        table.add_column("#", style="dim", max_width=4, justify="right")

        if active_table == "documents":
            table.add_column("Type", style="cyan", max_width=10)
            table.add_column("Title", style="bold white")
            table.add_column("Authors", style="green", max_width=28)
            table.add_column("Path / URL", style="dim")
        elif active_table == "authors":
            table.add_column("Author Name", style="bold yellow")
            table.add_column("Papers", justify="right", style="cyan", max_width=8)
        else:
            table.add_column("Concept", style="bold magenta")
            table.add_column("Mentions", justify="right", style="cyan", max_width=8)

        for i, r in enumerate(rows, start=1):
            num = f"[bold cyan]{i}[/bold cyan]" if selected_idx == i else str(i)
            props = json.loads(r["properties"])

            if active_table == "documents":
                stype  = props.get("source_type", "paper")
                title  = props.get("title", r["id"])
                if len(title) > 60: title = title[:57] + "…"
                authors = ", ".join(props.get("authors", [])) or "—"
                path   = props.get("file_path") or props.get("url") or "—"
                if len(path) > 55: path = "…" + path[-52:]
                row_style = "on grey15" if selected_idx == i else ""
                table.add_row(num, stype, title, authors, path, style=row_style)
            elif active_table == "authors":
                name = props.get("name", r["id"]).replace("_", " ").title()
                cnt  = str(r["papers_count"])
                row_style = "on grey15" if selected_idx == i else ""
                table.add_row(num, name, cnt, style=row_style)
            else:
                raw  = props.get("name", r["id"])
                name = raw.replace("_", " ").title()
                if len(name) > 65: name = name[:62] + "…"
                cnt  = str(r["degree"])
                row_style = "on grey15" if selected_idx == i else ""
                table.add_row(num, name, cnt, style=row_style)

        con.console.print(table)

        # ── Tab bar ──────────────────────────────────────────────────────────
        tab_bar_parts = []
        for t in TABLES:
            label = TABLE_LABELS[t]
            if t == active_table:
                tab_bar_parts.append(f"[bold white on blue] {label} [/bold white on blue]")
            else:
                tab_bar_parts.append(f"[dim] {label} [/dim]")
        con.console.print("  " + "  ".join(tab_bar_parts))
        con.console.print()

        # ── Status / help line ───────────────────────────────────────────────
        if status_msg:
            con.console.print(f"  {status_msg}")
            status_msg = ""
        else:
            if selected_idx is not None:
                row = rows[selected_idx - 1]
                props = json.loads(row["properties"])
                full_path = props.get("file_path") or props.get("url") or "—"
                con.console.print(f"  [bold cyan]Full Path/URL:[/bold cyan] [dim]{full_path}[/dim]")
                
                doc_actions = ""
                if active_table == "documents":
                    doc_actions = "  [bold]A[/bold] Annotation  [bold]O[/bold] Open  [bold]S[/bold] Summary  "
                con.console.print(
                    f"  Row [bold cyan]{selected_idx}[/bold cyan] selected  │{doc_actions}"
                    "[bold]E[/bold] Edit  [bold]X[/bold] Delete  [bold]Esc[/bold] Deselect"
                )
            else:
                con.console.print(
                    f"  [bold]←/A[/bold] Prev  [bold]→/D[/bold] Next  "
                    "[bold]Tab[/bold] Switch table  "
                    "[bold]/[/bold] Search  "
                    "[bold]Esc[/bold] Clear filter  "
                    "[bold]1-9…[/bold] Select row  [bold]Q[/bold] Quit"
                )

        # ── Input ────────────────────────────────────────────────────────────
        try:
            c = click.getchar()
        except Exception:
            break

        if c in ('q', 'Q', '\x03', '\x04'):
            break
        elif c in ('a', 'A', '\x1b[D'):           # left / previous page or Annotation if doc selected
            if selected_idx is not None and active_table == "documents":
                row = rows[selected_idx - 1]
                props = json.loads(row["properties"])
                abstract = props.get("abstract") or "No abstract/annotation available."
                title = props.get("title") or row["id"]
                con.console.clear()
                con.console.print(Panel(
                    abstract,
                    title=f"[bold green]Annotation: {title[:60]}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                ))
                con.console.print("\n  Press any key to return...")
                click.getchar()
            else:
                page = max(1, page - 1)
                selected_idx = None
        elif c in ('d', 'D', '\x1b[C'):           # right / next page
            page = min(total_pages, page + 1)
            selected_idx = None
        elif c == '\t':                            # Tab — cycle tables
            idx = TABLES.index(active_table)
            active_table = TABLES[(idx + 1) % len(TABLES)]
            page = 1
            selected_idx = None
        elif c == '\x1b':                          # Escape — deselect / clear search
            if selected_idx is not None:
                selected_idx = None
            else:
                search_query = None
                page = 1
        elif c == '/':                             # Slash — search query prompt
            con.console.print("\n  [bold yellow]Search query:[/bold yellow] ", end="")
            try:
                # Get user input
                query_str = input().strip()
                if query_str:
                    search_query = query_str
                else:
                    search_query = None
                page = 1
                selected_idx = None
            except KeyboardInterrupt:
                pass
        elif c in ('o', 'O') and selected_idx is not None and active_table == "documents":
            row = rows[selected_idx - 1]
            props = json.loads(row["properties"])
            path = props.get("file_path") or props.get("url")
            if not path:
                status_msg = "[red]No file path or URL associated with this document.[/red]"
            else:
                con.console.print(f"\n[green]Opening: {path}[/green]")
                try:
                    if path.startswith("http://") or path.startswith("https://"):
                        webbrowser.open(path)
                        status_msg = f"[green]Opened URL in browser: {path[:50]}[/green]"
                    else:
                        import subprocess
                        import sys
                        expanded_path = os.path.expanduser(path)
                        if not os.path.exists(expanded_path):
                            expanded_path = str(Path(path).resolve())
                        
                        if sys.platform == "win32":
                            os.startfile(expanded_path)
                        elif sys.platform == "darwin":
                            subprocess.run(["open", expanded_path])
                        else:
                            subprocess.run(["xdg-open", expanded_path])
                        status_msg = f"[green]Opened file locally: {os.path.basename(path)}[/green]"
                except Exception as e:
                    status_msg = f"[red]Failed to open file: {e}[/red]"
        elif c in ('s', 'S') and selected_idx is not None and active_table == "documents":
            row = rows[selected_idx - 1]
            paper_id = row["id"]
            props = json.loads(row["properties"])
            summary = props.get("summary")
            title = props.get("title") or paper_id
            
            if not summary:
                con.console.print("\n[yellow]Generating LLM Summary... This might take a few seconds.[/yellow]")
                try:
                    # Retrieve chunks to construct content sample using vector repo
                    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True, load_embeddings=False)
                    chunks = vector_repo.get_chunks_for_paper(paper_id)
                    sample_text = "\n\n".join([ch.text_content for ch in chunks[:5]]) if chunks else ""
                    abstract = props.get("abstract") or ""
                    
                    if not llm_engine:
                        raise ValueError("LLM Engine could not be initialized. Please check your model path/provider config.")
                    
                    prompt = (
                        f"Summarize the following document. Focus on key contributions, methodologies, and findings.\n\n"
                        f"Title: {title}\n"
                        f"Abstract: {abstract}\n\n"
                        f"Content snippet:\n{sample_text[:3000]}\n\n"
                        f"Provide a concise, professional markdown summary."
                    )
                    summary = llm_engine.generate_response(prompt)
                    
                    # Save to DB
                    props["summary"] = summary
                    graph_repo.update_node_properties(paper_id, props)
                    con.success("Summary generated and saved to database.")
                except Exception as e:
                    summary = f"Error generating summary: {e}"
            
            con.console.clear()
            con.console.print(Panel(
                summary,
                title=f"[bold cyan]LLM Summary: {title[:60]}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
                expand=True,
            ))
            con.console.print("\n  Press any key to return...")
            click.getchar()
        elif c.isdigit():                          # digit — start building row number
            val = int(c)
            total_rows = len(rows)
            if val == 0:
                status_msg = "[yellow]Row 0 is invalid.[/yellow]"
            elif val * 10 > total_rows:
                # Instant selection when unambiguous
                if 1 <= val <= total_rows:
                    selected_idx = val
                else:
                    status_msg = f"[yellow]Row {val} not on this page (1–{total_rows})[/yellow]"
            else:
                # Ambiguous (could be val or val*10 + next_digit <= total_rows)
                con.console.print(f"\n  Row #: [bold cyan]{val}[/bold cyan]  (Enter to select {val}, or press second digit)", end="")
                nc = click.getchar()
                if nc in ('\r', '\n'):
                    selected_idx = val
                elif nc.isdigit():
                    new_val = val * 10 + int(nc)
                    if 1 <= new_val <= total_rows:
                        selected_idx = new_val
                    else:
                        status_msg = f"[yellow]Row {new_val} not on this page (1–{total_rows})[/yellow]"
                else:
                    status_msg = "[yellow]Selection cancelled.[/yellow]"
        elif c in ('e', 'E') and selected_idx is not None:
            row = rows[selected_idx - 1]
            status_msg = _edit_node(row["id"], active_table)
        elif c in ('x', 'X') and selected_idx is not None:
            row = rows[selected_idx - 1]
            props = json.loads(row["properties"])
            name = props.get("title") or props.get("name") or row["id"]
            # Confirm
            con.console.print(f"\n  [bold red]Delete[/bold red] [white]{name[:60]}[/white]?  Y/n  ", end="")
            confirm = click.getchar()
            if confirm in ('y', 'Y', '\r', '\n'):
                status_msg = _delete_node(row["id"])
                selected_idx = None
            else:
                status_msg = "[dim]Deletion cancelled.[/dim]"


# ── config ────────────────────────────────────────────────────────────────────

@app.command("config")
def show_config():
    """Show current configuration: model paths, database location, and settings."""
    from rich.table import Table
    from rich import box as rbox

    def _check(path: str) -> str:
        """Return ✓ or ✗ depending on whether path exists."""
        return "[green]✓  exists[/green]" if Path(path).exists() else "[red]✗  not found[/red]"

    def _model_info(path: str) -> str:
        """Return a human-readable model name guess from the path."""
        name = Path(path).name.lower()
        if "gemma" in name:
            return "Google Gemma (MLX)"
        if "qwen" in name:
            return "Alibaba Qwen (MLX)"
        if "llama" in name:
            return "Meta LLaMA (MLX)"
        if "mistral" in name:
            return "Mistral (MLX)"
        if "minilm" in name:
            return "MiniLM sentence embedder"
        if "all-mpnet" in name:
            return "MPNet sentence embedder"
        if "mxbai" in name:
            return "MixedBread cross-encoder"
        return "Unknown"

    con.blank()
    con.console.print(Panel(
        "[bold]Science Graph[/bold] — Configuration",
        border_style="cyan",
        padding=(0, 2),
    ))

    # ── Paths ──────────────────────────────────────────────────────────────────
    paths_table = Table(
        title="📂 Paths",
        box=rbox.ROUNDED, border_style="dim",
        show_header=True, header_style="bold cyan",
        expand=True,
    )
    paths_table.add_column("Key", style="bold white", min_width=18)
    paths_table.add_column("Path", style="dim white")
    paths_table.add_column("Status", min_width=14)

    paths_table.add_row("Database",    config.db_path,      _check(config.db_path))
    paths_table.add_row("Archive dir", config.archive_dir,  _check(config.archive_dir))
    paths_table.add_row(
        "Config file",
        str(Path.home() / ".config" / "pdf-graph-analyzer" / "config.yaml"),
        _check(str(Path.home() / ".config" / "pdf-graph-analyzer" / "config.yaml")),
    )
    con.console.print(paths_table)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_table = Table(
        title="⚡ LLM Model (text generation)",
        box=rbox.ROUNDED, border_style="magenta",
        show_header=True, header_style="bold magenta",
        expand=True,
    )
    llm_table.add_column("Setting", style="bold white", min_width=18)
    llm_table.add_column("Value")
    llm_table.add_column("Info", style="dim")

    llm_table.add_row("Provider",    config.llm_provider,         "mlx or openai")
    if config.llm_provider.lower() == "openai":
        llm_table.add_row("Base URL",    config.llm_base_url or "default", "OpenAI compatible API base")
        api_key_masked = "configured" if config.llm_api_key else "[yellow]missing[/yellow]"
        llm_table.add_row("API Key",     api_key_masked, "OpenAI / OpenRouter API Key")
        llm_table.add_row("Model Name",  config.llm_model_path,       "Model name in API")
    else:
        llm_table.add_row("Model path",  config.llm_model_path,       _check(config.llm_model_path))
        llm_table.add_row("Model type",  _model_info(config.llm_model_path), "local MLX inference")
        
    llm_table.add_row("Max tokens",  str(config.llm_max_tokens),  "max output length")
    llm_table.add_row("Temperature", str(config.llm_temp),        "0=deterministic, 1=creative")
    con.console.print(llm_table)

    # ── Embedding ─────────────────────────────────────────────────────────────
    emb_table = Table(
        title="🔢 Embedding Model (vector search)",
        box=rbox.ROUNDED, border_style="blue",
        show_header=True, header_style="bold blue",
        expand=True,
    )
    emb_table.add_column("Setting", style="bold white", min_width=18)
    emb_table.add_column("Value")
    emb_table.add_column("Info", style="dim")

    emb_table.add_row("Model name",    config.embedding_model_name, _model_info(config.embedding_model_name))
    emb_table.add_row("Reranker",      "mixedbread-ai/mxbai-rerank-xsmall-v1", "cross-encoder reranking")
    emb_table.add_row("Chunk size",    str(config.chunk_size),   "chars per text chunk")
    emb_table.add_row("Chunk overlap", str(config.chunk_overlap), "overlap between chunks")
    con.console.print(emb_table)

    # ── NLP & Extraction ──────────────────────────────────────────────────────
    nlp_table = Table(
        title="🧠 NLP & Extraction Models (spacy & ner)",
        box=rbox.ROUNDED, border_style="cyan",
        show_header=True, header_style="bold cyan",
        expand=True,
    )
    nlp_table.add_column("Model Task", style="bold white", min_width=18)
    nlp_table.add_column("Model Name / Path")
    nlp_table.add_column("Info", style="dim")

    nlp_table.add_row("spaCy model", config.spacy_model_name, "lemmatization of concepts")
    nlp_table.add_row("NER model", config.ner_model_name, "author name extraction")
    con.console.print(nlp_table)

    # ── Environment ────────────────────────────────────────────────────────────
    env_table = Table(
        title="🌐 Environment",
        box=rbox.ROUNDED, border_style="dim",
        show_header=True, header_style="bold white",
        expand=True,
    )
    env_table.add_column("Variable", style="bold white", min_width=18)
    env_table.add_column("Value")
    env_table.add_column("Effect", style="dim")

    hf_token = os.environ.get("HF_TOKEN", "")
    hf_status = "[green]set[/green]" if hf_token else "[yellow]not set (rate-limited)[/yellow]"
    env_table.add_row("HF_TOKEN",    hf_status,                           "HuggingFace auth")
    env_table.add_row("HF_HUB_VERBOSITY",
                      os.environ.get("HF_HUB_VERBOSITY", "default"),      "HF Hub log level")
    env_table.add_row("TOKENIZERS_PARALLELISM",
                      os.environ.get("TOKENIZERS_PARALLELISM", "default"), "parallelism for tokenizers")
    con.console.print(env_table)

    con.blank()
    con.dim("To change settings, edit: ~/.config/pdf-graph-analyzer/config.yaml")
    con.dim("To set HF_TOKEN: export HF_TOKEN=hf_xxxx")
    con.blank()


# ── visualize ─────────────────────────────────────────────────────────────────

@app.command("visualize")
def visualize(
    output_path: Path = typer.Option(Path.cwd() / "graph.html", "--output", "-o", help="Output HTML file path"),
):
    """Generate an interactive HTML knowledge graph and open it in the browser."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)

    # Get node degrees
    degrees = {}
    edges_rows = graph_repo.get_all_edges()
    for e in edges_rows:
        src_id, tgt_id, etype, _ = e
        degrees[src_id] = degrees.get(src_id, 0) + 1
        degrees[tgt_id] = degrees.get(tgt_id, 0) + 1

    nodes_rows = graph_repo.get_all_nodes()

    if not nodes_rows:
        con.warning("Knowledge graph is empty. Index some documents first.")
        return

    # Process nodes
    vis_nodes = []
    for r in nodes_rows:
        node_id, label, props_str = r
        props = json.loads(props_str)
        source_type = props.get("source_type", "paper")
        degree = degrees.get(node_id, 0)

        if label == "Paper":
            title = props.get("title", node_id)
            node_label = title if len(title) < 25 else title[:22] + "..."
            color_map = {"note": "#f03e3e", "book": "#7950f2", "paper": "#4c6ef5", "webpage": "#20c997"}
            color = color_map.get(source_type, "#4c6ef5")
            size = 25
            shape = "dot"
        elif label == "Author":
            node_label = props.get("name", node_id).title()
            color = "#fab005"
            size = 20
            shape = "dot"
        elif label == "Concept":
            raw_name = props.get("name", node_id)
            node_label = raw_name.replace("_", " ").title()
            is_tag = props.get("is_tag", False)
            color = "#da77f2" if is_tag else "#12b886"
            size = 18 if is_tag else 20
            shape = "dot"
        else:
            node_label = node_id
            color = "#868e96"
            size = 15
            shape = "dot"

        vis_nodes.append({
            "id": node_id,
            "label": node_label,
            "title": f"<b>{label}</b>: {props.get('title', props.get('name', node_id))}<br>ID: {node_id}<br>Degree: {degree}",
            "color": color,
            "size": size,
            "shape": shape,
            "group": label,
            "degree": degree,
            "created_at": props.get("created_at"),
            "year": props.get("year"),
        })

    vis_edges = []
    for e in edges_rows:
        src_id, tgt_id, edge_type, _ = e
        color = "#adb5bd"
        dashes = False
        width = 1

        if edge_type == "AUTHORED":
            color = "#ffd43b"
            width = 2
        elif edge_type == "MENTIONS_CONCEPT":
            color = "#69db7c"
            dashes = True
        elif edge_type == "CITES":
            color = "#748ffc"
            width = 2
        elif edge_type == "HAS_TAG":
            color = "#da77f2"
            dashes = True
            width = 1


        vis_edges.append({
            "from": src_id,
            "to": tgt_id,
            "label": edge_type,
            "arrows": "to",
            "font": {"size": 8, "align": "top"},
            "color": {"color": color, "highlight": "#495057"},
            "dashes": dashes,
            "width": width,
        })

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Science Graph — Knowledge Network</title>
    <script type="text/javascript">
        // Dynamic loader fallback for vis-network.min.js
        (function() {{
            var urls = [
                "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js",
                "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js",
                "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"
            ];
            var index = 0;
            function tryLoad() {{
                if (index >= urls.length) {{
                    console.error("Failed to load vis-network from all sources.");
                    return;
                }}
                var script = document.createElement("script");
                script.type = "text/javascript";
                script.src = urls[index];
                script.onload = function() {{
                    console.log("Successfully loaded vis-network from: " + urls[index]);
                    if (window.initGraph) {{
                        window.initGraph();
                    }}
                }};
                script.onerror = function() {{
                    console.warn("Failed to load vis-network from: " + urls[index] + ". Trying next...");
                    index++;
                    tryLoad();
                }};
                document.head.appendChild(script);
            }}
            // Start loading when document is ready
            if (document.readyState === "loading") {{
                document.addEventListener("DOMContentLoaded", tryLoad);
            }} else {{
                tryLoad();
            }}
        }})();
    </script>
    <style type="text/css">
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               margin: 0; background-color: #1a1b1e; color: #c1c2c5; display: flex; flex-direction: column; height: 100vh; }}
        #header {{ padding: 15px 20px; background-color: #25262b; border-bottom: 1px solid #2c2e33; display: flex; justify-content: space-between; align-items: center; }}
        h2 {{ margin: 0; color: #fff; font-size: 1.2rem; }}
        #mynetwork {{ flex: 1; width: 100%; background-color: #1a1b1e; }}
        .legend {{ display: inline-block; margin-right: 15px; font-size: 14px; }}
        .legend-color {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                         margin-right: 5px; vertical-align: middle; }}
        .controls {{ display: flex; align-items: center; gap: 15px; }}
        .slider-container {{ display: flex; align-items: center; gap: 10px; background: #2c2e33; padding: 5px 15px; border-radius: 6px; }}
        input[type=range] {{ cursor: pointer; }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h2>🔬 Science Graph — Knowledge Network</h2>
            <div style="margin-top: 8px;">
                <span class="legend"><span class="legend-color" style="background:#4c6ef5"></span>Paper</span>
                <span class="legend"><span class="legend-color" style="background:#f03e3e"></span>Note</span>
                <span class="legend"><span class="legend-color" style="background:#7950f2"></span>Book</span>
                <span class="legend"><span class="legend-color" style="background:#20c997"></span>Webpage</span>
                <span class="legend"><span class="legend-color" style="background:#fab005"></span>Author</span>
                <span class="legend"><span class="legend-color" style="background:#12b886"></span>Concept</span>
                <span class="legend"><span class="legend-color" style="background:#da77f2"></span>Tag</span>
            </div>
        </div>
        <div class="controls">
            <div class="slider-container">
                <label for="yearFilter">Year:</label>
                <select id="yearFilter" onchange="applyFilters()" style="background:#1a1b1e; color:#c1c2c5; border:1px solid #2c2e33; border-radius:4px; padding:3px 8px; outline:none; cursor:pointer;">
                    <option value="all">All Years</option>
                </select>
            </div>
            <div class="slider-container">
                <label for="degreeSlider">Min Connections:</label>
                <input type="range" id="degreeSlider" min="1" max="20" value="3" oninput="applyFilters()">
                <span id="sliderValue" style="font-weight: bold; width: 20px;">3</span>
            </div>
        </div>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var allNodes = {json.dumps(vis_nodes, ensure_ascii=False)};
        var allEdges = {json.dumps(vis_edges, ensure_ascii=False)};
        var nodesView, edgesView, network;

        function initGraph() {{
            nodesView = new vis.DataSet(allNodes);
            edgesView = new vis.DataSet(allEdges);

            network = new vis.Network(
                document.getElementById('mynetwork'),
                {{ nodes: nodesView, edges: edgesView }},
                {{
                    nodes: {{ font: {{ color: '#c1c2c5', size: 12 }} }},
                    edges: {{ smooth: {{ type: 'continuous' }} }},
                    physics: {{ 
                        barnesHut: {{ 
                            gravitationalConstant: -12000,
                            centralGravity: 0.2,
                            springLength: 250,
                            springConstant: 0.04,
                            damping: 0.09,
                            avoidOverlap: 0.3
                        }},
                        stabilization: {{ iterations: 200 }}
                    }}
                }}
            );

            // Populate year options dynamically on init
            var years = new Set();
            allNodes.forEach(n => {{
                if (n.year) {{
                    years.add(n.year);
                }} else if (n.created_at) {{
                    var y = new Date(n.created_at).getFullYear();
                    if (!isNaN(y)) {{
                        years.add(y);
                    }}
                }}
            }});
            var sortedYears = Array.from(years).sort((a,b) => b-a);
            var select = document.getElementById('yearFilter');
            sortedYears.forEach(y => {{
                var opt = document.createElement('option');
                opt.value = y;
                opt.innerText = y;
                select.appendChild(opt);
            }});

            // Initial filter
            var defaultFilter = allNodes.length < 150 ? 1 : 3;
            document.getElementById('degreeSlider').value = defaultFilter;
            applyFilters();
        }}

        function applyFilters() {{
            var val = document.getElementById('degreeSlider').value;
            document.getElementById('sliderValue').innerText = val;
            var minDegree = parseInt(val, 10);
            var yearVal = document.getElementById('yearFilter').value;
            
            // Filter nodes
            var filteredNodes = allNodes.filter(n => {{
                // Check degree
                var degreeMatch = n.degree >= minDegree || n.group === 'Paper';
                if (!degreeMatch) return false;
                
                // Check year
                if (yearVal !== 'all') {{
                    var targetYear = parseInt(yearVal, 10);
                    var nodeYear = n.year;
                    if (!nodeYear && n.created_at) {{
                        var d = new Date(n.created_at);
                        if (!isNaN(d)) nodeYear = d.getFullYear();
                    }}
                    if (nodeYear !== targetYear) return false;
                }}
                return true;
            }});
            
            if (nodesView) {{
                nodesView.clear();
                nodesView.add(filteredNodes);
            }}
            
            var validIds = new Set(filteredNodes.map(n => n.id));
            var filteredEdges = allEdges.filter(e => validIds.has(e.from) && validIds.has(e.to));
            if (edgesView) {{
                edgesView.clear();
                edgesView.add(filteredEdges);
            }}
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    con.success(f"Graph saved to [bold]{output_path}[/bold]")
    try:
        webbrowser.open(Path(output_path).as_uri())
        con.info("Opening in browser …")
    except Exception as e:
        con.warning(f"Could not open browser: {e}")


# ── chat ──────────────────────────────────────────────────────────────────────

@app.command("chat")
def chat():
    """Start an interactive TUI chat session with RAG memory."""
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True)
    if not llm_engine:
        con.error("LLM engine is not available. Run: graph config")
        raise typer.Exit(1)

    rag_pipeline = RAGPipeline(graph_repo, vector_repo, embedding_engine, llm_engine)

    from src.tui import run_tui_chat
    run_tui_chat(rag_pipeline)


# ── review ────────────────────────────────────────────────────────────────────

@app.command("review")
def review(
    topic: str = typer.Argument(..., help="Research topic to review"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max number of chunks to retrieve"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the Markdown report"),
    fast: bool = typer.Option(False, "--fast", help="Skip LLM clustering (single-section mode)"),
):
    """Generate a full Markdown literature review on a topic using the indexed knowledge base."""
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True)
    if not llm_engine:
        con.error("LLM engine is required for review generation. Run: graph config")
        raise typer.Exit(1)

    from src.review_agent import ReviewAgent

    agent = ReviewAgent(graph_repo, vector_repo, embedding_engine, llm_engine)

    if output is None:
        safe_name = topic.lower().replace(" ", "_")[:40]
        output = Path.cwd() / f"review_{safe_name}.md"

    report = agent.run(topic=topic, limit=limit, output_path=output, fast=fast)

    con.blank()
    con.console.print(Panel(
        report[:600] + ("…" if len(report) > 600 else ""),
        title="[bold green]Preview[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
):
    """Start the Science Graph Web UI (FastAPI + interactive vis-network graph)."""
    try:
        import uvicorn
    except ImportError:
        con.error("uvicorn is not installed. Run: uv add uvicorn")
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    con.blank()
    con.console.print(Panel(
        f"[bold]Science Graph Web UI[/bold]\n\n"
        f"  [cyan]URL:[/cyan]    {url}\n"
        f"  [cyan]Reload:[/cyan] {'yes' if reload else 'no'}\n\n"
        f"  Press [bold]Ctrl+C[/bold] to stop.",
        border_style="cyan",
        padding=(1, 2),
    ))
    con.blank()

    if open_browser:
        import threading, time
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "src.web_app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",   # suppress uvicorn INFO noise
    )


@app.command("extract-file")
def extract_file(
    target: str = typer.Argument(..., help="Path to text document"),
    use_llm: bool = typer.Option(True, "--use-llm/--no-llm", help="Use LLM to extract concepts"),
):
    """Extract authors, concepts, and tags from a text document and output as JSON graph."""
    import sys
    import re
    # Redirect con output to stderr to keep stdout clean for JSON
    old_file = con.console._file
    con.console._file = sys.stderr
    try:
        path = Path(target)
        if not path.exists():
            con.error(f"File not found: {target}")
            raise typer.Exit(1)
            
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            con.error(f"Failed to read file: {e}")
            raise typer.Exit(1)
            
        # Extract title, abstract, and body text
        first_line = content.split('\n')[0].strip() if content else ""
        if first_line.startswith("# "):
            title = first_line.lstrip("# ").strip()
            full_text = content[len(first_line):].strip()
        else:
            title = path.stem
            full_text = content

        paragraphs = [p.strip() for p in re.split(r'\n\n+', full_text) if p.strip()]
        abstract = paragraphs[0][:800] if paragraphs else ""

        # Load LLM engine if use_llm is True
        llm_engine = None
        if use_llm:
            try:
                llm_engine = LLMEngine()
            except Exception as e:
                con.warning(f"Could not load LLM engine: {e}. Falling back to regex extraction.")

        from src.services.extraction_service import ExtractionService
        extractor = ExtractionService(llm_engine=llm_engine)
        
        try:
            result = extractor.extract(title, abstract, full_text, use_llm=use_llm)
        except Exception as e:
            con.error(f"Extraction failed: {e}")
            raise typer.Exit(1)

        output_dict = {
            "authors": result.authors,
            "concepts": result.concepts,
            "tags": result.tags,
        }
        
        sys.stdout.write(json.dumps(output_dict, indent=2, ensure_ascii=False) + "\n")
    finally:
        con.console._file = old_file


@app.command("cleanup")
def cleanup():
    """Remove orphaned Concept nodes with degree 0 (no connected papers/notes)."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    con.dim("Starting database cleanup...")
    deleted_count = graph_repo.cleanup_orphaned_concepts()
    if deleted_count > 0:
        con.success(f"Successfully cleaned up [bold]{deleted_count}[/bold] orphaned concept nodes.")
    else:
        con.info("No orphaned concept nodes found in the database.")


@app.command("reset")
def reset():
    """Completely reset the database, vector index, and local archives."""
    con.warning("This will delete all papers, notes, concepts, embeddings, and cached files!")
    
    # First confirmation
    first_confirm = typer.confirm("Are you sure you want to completely reset the database?", default=False)
    if not first_confirm:
        con.info("Reset cancelled.")
        raise typer.Exit(0)
        
    # Second confirmation
    second_confirm = typer.confirm("WARNING: This action is irreversible. Are you REALLY sure?", default=False)
    if not second_confirm:
        con.info("Reset cancelled.")
        raise typer.Exit(0)

    # Proceed with deletion
    con.dim("Starting database reset...")
    
    # 1. Database path
    db_path = Path(config.db_path)
    if db_path.exists():
        try:
            db_path.unlink()
            con.success(f"Deleted SQLite database: [dim]{db_path}[/dim]")
        except Exception as e:
            con.error(f"Failed to delete SQLite database: {e}")
            
    # Also delete WAL and shared memory files if they exist (graph.db-wal, graph.db-shm)
    for suffix in ["-wal", "-shm"]:
        side_file = Path(str(db_path) + suffix)
        if side_file.exists():
            try:
                side_file.unlink()
            except Exception:
                pass
                
    # 2. Vector index path
    usearch_path = Path(str(db_path).replace(".db", ".usearch"))
    if usearch_path.exists():
        try:
            usearch_path.unlink()
            con.success(f"Deleted vector index: [dim]{usearch_path}[/dim]")
        except Exception as e:
            con.error(f"Failed to delete vector index: {e}")
            
    # 3. Archive directory
    archive_dir = Path(config.archive_dir)
    if archive_dir.exists():
        try:
            import shutil
            for child in archive_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            con.success(f"Cleared archive directory: [dim]{archive_dir}[/dim]")
        except Exception as e:
            con.error(f"Failed to clear archive directory: {e}")
            
    con.success("Database and environment successfully reset to a pristine state.")


@app.command("doctor")
def doctor(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Actually apply changes to sanitize and clean LLM artifacts and formatters",
    ),
):
    """
    Scan database through the repository layer, detecting and fixing LLM output artifacts,
    unapplied formatting, and incorrect identifiers due to formatting anomalies.
    """
    graph_repo, vector_repo, _, _ = get_services(load_llm=False, load_embeddings=False)
    
    from src.services.doctor_service import DoctorService
    
    con.info("🩺 Starting Science Graph Database Doctor Diagnostics...")
    if fix:
        con.warning("🔧 Running in [bold yellow]FIX[/bold yellow] mode. Anomalies will be corrected in place.")
    else:
        con.info("🔍 Running in [bold cyan]CHECK-ONLY[/bold cyan] mode. No writes will be made. Run with [bold]--fix[/bold] to repair.")
        
    con.blank()
    
    doctor_service = DoctorService(graph_repo, vector_repo)
    report = doctor_service.run_diagnostics(fix=fix)
    
    # 1. Print Stats Table
    table = Table(
        title="📊 Diagnostics Scan Statistics",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Entity Type", style="bold white", min_width=24)
    table.add_column("Checked", justify="right", style="bold white")
    table.add_column("Fixed/Corrected", justify="right", style="bold green")
    table.add_column("Migrated", justify="right", style="bold yellow")
    table.add_column("Merged", justify="right", style="bold magenta")
    
    stats = report["stats"]
    table.add_row("Papers", str(stats["papers_checked"]), str(stats["papers_fixed"]), "—", "—")
    table.add_row("Authors", str(stats["authors_checked"]), str(stats["authors_fixed"]), str(stats["authors_migrated"]), str(stats["authors_merged"]))
    table.add_row("Concepts / Tags", str(stats["concepts_checked"]), str(stats["concepts_fixed"]), str(stats["concepts_migrated"]), str(stats["concepts_merged"]))
    table.add_row("Chunks", str(stats["chunks_checked"]), str(stats["chunks_fixed"]), "—", "—")
    
    con.console.print(table)
    con.blank()
    
    # 2. Print Detailed Anomalies
    anomalies = report["anomalies"]
    total_issues = (
        len(anomalies["papers"]) +
        len(anomalies["authors"]) +
        len(anomalies["concepts"]) +
        len(anomalies["chunks"])
    )
    
    if total_issues == 0:
        con.success("🎉 No anomalies found! Database texts are completely sanitized and formatted.")
        return
        
    # Detail Papers
    if anomalies["papers"]:
        con.console.print("[bold cyan]📄 Paper Anomalies:[/bold cyan]")
        for paper in anomalies["papers"]:
            con.console.print(f"  • [bold]{paper['id']}[/bold]")
            if paper["old_title"] != paper["new_title"]:
                con.console.print(f"    - Title: [red]\"{paper['old_title']}\"[/red] -> [green]\"{paper['new_title']}\"[/green]")
            if paper["old_abstract"] != paper["new_abstract"]:
                con.console.print("    - Abstract updated")
            if paper["old_authors"] != paper["new_authors"]:
                con.console.print(f"    - Authors: [red]{paper['old_authors']}[/red] -> [green]{paper['new_authors']}[/green]")
        con.blank()
        
    # Detail Authors
    if anomalies["authors"]:
        con.console.print("[bold magenta]👤 Author Anomalies:[/bold magenta]")
        for author in anomalies["authors"]:
            con.console.print(f"  • ID: [bold]{author['id']}[/bold]")
            con.console.print(f"    - Name: [red]\"{author['old_name']}\"[/red] -> [green]\"{author['new_name']}\"[/green]")
            con.console.print(f"    - Action: [bold yellow]{author['action']}[/bold yellow]")
        con.blank()
        
    # Detail Concepts
    if anomalies["concepts"]:
        con.console.print("[bold yellow]💡 Concept/Tag Anomalies:[/bold yellow]")
        for concept in anomalies["concepts"]:
            con.console.print(f"  • ID: [bold]{concept['id']}[/bold]")
            if concept["old_name"] != concept["new_name"]:
                con.console.print(f"    - Name: [red]\"{concept['old_name']}\"[/red] -> [green]\"{concept['new_name']}\"[/green]")
            if concept["old_description"] != concept["new_description"]:
                con.console.print(f"    - Description: [red]\"{concept['old_description']}\"[/red] -> [green]\"{concept['new_description']}\"[/green]")
            con.console.print(f"    - Action: [bold yellow]{concept['action']}[/bold yellow]")
        con.blank()
        
    # Detail Chunks
    if anomalies["chunks"]:
        con.console.print("[bold blue]🧩 Text Chunk Anomalies:[/bold blue]")
        con.console.print(f"  • Found [bold]{len(anomalies['chunks'])}[/bold] chunk text content anomalies containing LLM artifacts or spacing issues.")
        con.blank()
        
    if fix:
        con.success(f"✔️ Successfully corrected [bold]{total_issues}[/bold] anomalies across all tables!")
    else:
        con.warning(f"⚠️ Found [bold]{total_issues}[/bold] anomalies. Run with [bold]--fix[/bold] to repair them.")


if __name__ == "__main__":
    app()

