"""
Science Graph — CLI entry point.
All user-facing output uses src.console for consistent rich formatting.
"""

import json
import os
import sqlite3
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

@app.command("index")
def index(
    target: str = typer.Argument(..., help="Path to file, directory, or URL to index"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM to extract concepts (slower)"),
):
    """Index PDF papers, Markdown notes (.md), EPUB books, or URLs into the knowledge graph."""
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=use_llm)
    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    def _index_file(path: Path) -> bool:
        t = path.suffix.lower().lstrip(".")
        try:
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
    """Show knowledge base statistics."""
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
    table.add_column("Category", style="bold white", min_width=16)
    table.add_column("Count", justify="right", style="bold green")

    table.add_row("Papers / Books / Notes", str(db_stats["papers"]))
    table.add_row("Authors", str(db_stats["authors"]))
    table.add_row("Concepts", str(db_stats["concepts"]))
    table.add_row("Graph Edges", str(db_stats["edges"]))
    table.add_row("─" * 20, "─" * 8)
    table.add_row("Database Size", db_size or "—")

    con.blank()
    con.console.print(table)
    con.blank()


# ── storage ───────────────────────────────────────────────────────────────────

@app.command("storage")
def storage(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of items to display per page"),
):
    """Display indexed data in beautiful tables with interactive pagination."""
    import click
    import math
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    conn = sqlite3.connect(graph_repo.db_path)
    conn.row_factory = sqlite3.Row

    page = 1

    while True:
        con.console.clear()
        
        offset = (page - 1) * limit
        
        total_docs = conn.execute("SELECT count(*) FROM nodes WHERE label='Paper'").fetchone()[0]
        total_authors = conn.execute("SELECT count(*) FROM nodes WHERE label='Author'").fetchone()[0]
        total_concepts = conn.execute("SELECT count(*) FROM nodes WHERE label='Concept'").fetchone()[0]
        
        max_total = max(total_docs, total_authors, total_concepts)
        total_pages = max(1, math.ceil(max_total / limit))
        page = min(page, total_pages)
        page = max(1, page)
        
        # Fetch Documents
        papers = conn.execute("SELECT id, properties FROM nodes WHERE label='Paper' LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        if papers:
            table = Table(title=f"📚 Documents (Page {page}/{math.ceil(total_docs/limit) or 1})", box=box.ROUNDED, border_style="blue", expand=True, header_style="bold cyan")
            table.add_column("Type", style="cyan", max_width=12)
            table.add_column("Title", style="bold white")
            table.add_column("Authors", style="green", max_width=30)
            table.add_column("Path / URL", style="dim")
            
            for p in papers:
                props = json.loads(p["properties"])
                source_type = props.get("source_type", "paper")
                title = props.get("title", p["id"])
                authors = ", ".join(props.get("authors", [])) or "—"
                path = props.get("file_path", props.get("url", "—"))
                table.add_row(source_type, title, authors, path)
            con.console.print(table)
            con.blank()
            
        # Fetch Authors
        authors_query = """
            SELECT id, properties, 
                   (SELECT count(*) FROM edges WHERE source_id=nodes.id AND type='AUTHORED') as papers_count 
            FROM nodes 
            WHERE label='Author' 
            ORDER BY papers_count DESC 
            LIMIT ? OFFSET ?
        """
        authors = conn.execute(authors_query, (limit, offset)).fetchall()
        if authors:
            table = Table(title=f"👥 Authors (Page {page}/{math.ceil(total_authors/limit) or 1})", box=box.ROUNDED, border_style="yellow", expand=True, header_style="bold cyan")
            table.add_column("Author Name", style="bold yellow")
            table.add_column("Papers Authored", justify="right", style="cyan")
            for a in authors:
                props = json.loads(a["properties"])
                name = props.get("name", a["id"]).title()
                papers_count = str(a["papers_count"])
                table.add_row(name, papers_count)
            con.console.print(table)
            con.blank()
            
        # Fetch Concepts
        concepts_query = """
            SELECT id, properties, 
                   (SELECT count(*) FROM edges WHERE target_id=nodes.id AND type='MENTIONS_CONCEPT') as degree 
            FROM nodes 
            WHERE label='Concept' 
            ORDER BY degree DESC 
            LIMIT ? OFFSET ?
        """
        concepts = conn.execute(concepts_query, (limit, offset)).fetchall()
        if concepts:
            table = Table(title=f"🧠 Concepts (Page {page}/{math.ceil(total_concepts/limit) or 1})", box=box.ROUNDED, border_style="magenta", expand=True, header_style="bold cyan")
            table.add_column("Concept", style="bold magenta")
            table.add_column("Mentions", justify="right", style="cyan")
            
            for c in concepts:
                props = json.loads(c["properties"])
                raw_name = props.get("name", c["id"])
                name = raw_name.replace("_", " ").title()
                if len(name) > 75:
                    name = name[:72] + "..."
                degree = str(c["degree"])
                table.add_row(name, degree)
            con.console.print(table)
            con.blank()
            
        con.console.print(f"[bold cyan]Page {page} of {total_pages}[/bold cyan] | Use [bold]← / A[/bold] for previous, [bold]→ / D[/bold] for next. Press [bold]Q[/bold] to quit.")
        
        # Handle input
        try:
            c = click.getchar()
        except Exception:
            break
            
        if c in ('q', 'Q', '\x03', '\x04'):
            break
        elif c in ('a', 'A', '\x1b[D'):
            page = max(1, page - 1)
        elif c in ('d', 'D', '\x1b[C'):
            page = min(total_pages, page + 1)
            
    conn.close()


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

    conn = sqlite3.connect(graph_repo.db_path)
    conn.row_factory = sqlite3.Row

    # Get node degrees
    degrees = {}
    edges_rows = conn.execute("SELECT source_id, target_id, type FROM edges").fetchall()
    for e in edges_rows:
        degrees[e["source_id"]] = degrees.get(e["source_id"], 0) + 1
        degrees[e["target_id"]] = degrees.get(e["target_id"], 0) + 1

    nodes_rows = conn.execute("SELECT id, label, properties FROM nodes").fetchall()
    conn.close()

    if not nodes_rows:
        con.warning("Knowledge graph is empty. Index some documents first.")
        return

    # Process nodes
    vis_nodes = []
    for r in nodes_rows:
        node_id = r["id"]
        label = r["label"]
        props = json.loads(r["properties"])
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
            color = "#12b886"
            size = 20
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
        })

    vis_edges = []
    for r in edges_rows:
        edge_type = r["type"]
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

        vis_edges.append({
            "from": r["source_id"],
            "to": r["target_id"],
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
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
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
            </div>
        </div>
        <div class="controls">
            <div class="slider-container">
                <label for="degreeSlider">Min Connections:</label>
                <input type="range" id="degreeSlider" min="1" max="20" value="3" oninput="updateFilter(this.value)">
                <span id="sliderValue" style="font-weight: bold; width: 20px;">3</span>
            </div>
        </div>
    </div>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var allNodes = {json.dumps(vis_nodes, ensure_ascii=False)};
        var allEdges = {json.dumps(vis_edges, ensure_ascii=False)};
        
        var nodesView = new vis.DataSet(allNodes);
        var edgesView = new vis.DataSet(allEdges);

        var network = new vis.Network(
            document.getElementById('mynetwork'),
            {{ nodes: nodesView, edges: edgesView }},
            {{
                nodes: {{ font: {{ color: '#c1c2c5', size: 12 }} }},
                edges: {{ smooth: {{ type: 'continuous' }} }},
                physics: {{ 
                    barnesHut: {{ 
                        gravitationalConstant: -10000, 
                        springLength: 300, 
                        nodeDistance: 150 
                    }},
                    stabilization: {{ iterations: 150 }}
                }}
            }}
        );

        function updateFilter(val) {{
            document.getElementById('sliderValue').innerText = val;
            var minDegree = parseInt(val, 10);
            
            // Filter nodes
            var filteredNodes = allNodes.filter(n => n.degree >= minDegree || n.group === 'Paper');
            nodesView.clear();
            nodesView.add(filteredNodes);
            
            var validIds = new Set(filteredNodes.map(n => n.id));
            var filteredEdges = allEdges.filter(e => validIds.has(e.from) && validIds.has(e.to));
            edgesView.clear();
            edgesView.add(filteredEdges);
        }}
        // Initial filter
        var defaultFilter = allNodes.length < 150 ? 1 : 3;
        document.getElementById('degreeSlider').value = defaultFilter;
        updateFilter(defaultFilter);
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


if __name__ == "__main__":
    app()
