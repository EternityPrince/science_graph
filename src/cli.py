import os
import sys
import typer
import webbrowser
import json
from pathlib import Path
from typing import Optional
from src.config import config
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine
from src.indexer import Indexer
from src.llm_engine import LLMEngine
from src.rag import RAGPipeline

app = typer.Typer(help="Graph-based PDF analysis CLI tool")

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
            typer.echo(f"[!] Could not load LLM engine: {e}. LLM commands will not work.", err=True)
            
    return graph_repo, vector_repo, embedding_engine, llm_engine

@app.command("index")
def index(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to a single PDF file"),
    directory: Optional[Path] = typer.Option(None, "--dir", "-d", help="Path to a directory containing PDF files")
):
    """Indexes a single PDF or all PDFs in a directory."""
    if not file and not directory:
        typer.echo("Error: Please specify either --file or --dir", err=True)
        raise typer.Exit(1)
        
    graph_repo, vector_repo, embedding_engine, _ = get_services(load_llm=False)
    indexer = Indexer(graph_repo, vector_repo, embedding_engine)
    
    if file:
        if not file.exists() or file.suffix.lower() != ".pdf":
            typer.echo(f"Error: {file} is not a valid PDF file", err=True)
            raise typer.Exit(1)
        try:
            indexer.index_pdf(str(file))
            typer.echo(f"[+] Successfully indexed {file.name}")
        except Exception as e:
            typer.echo(f"[!] Error indexing {file.name}: {e}", err=True)
            raise typer.Exit(1)
            
    if directory:
        if not directory.is_dir():
            typer.echo(f"Error: {directory} is not a directory", err=True)
            raise typer.Exit(1)
            
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            typer.echo(f"No PDF files found in {directory}")
            return
            
        typer.echo(f"[*] Found {len(pdf_files)} PDF files. Starting indexing...")
        success_count = 0
        for pdf_file in pdf_files:
            try:
                indexer.index_pdf(str(pdf_file))
                success_count += 1
            except Exception as e:
                typer.echo(f"[!] Error indexing {pdf_file.name}: {e}", err=True)
                
        typer.echo(f"[+] Indexing completed. Indexed {success_count}/{len(pdf_files)} papers successfully.")

@app.command("query")
def query(
    text: str = typer.Argument(..., help="Your question about the indexed papers"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of context text blocks to retrieve")
):
    """Answers questions on indexed papers using vector + graph hybrid RAG."""
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True)
    if not llm_engine:
        typer.echo("Error: LLM model could not be loaded. Check your config.yaml or local paths.", err=True)
        raise typer.Exit(1)
        
    pipeline = RAGPipeline(graph_repo, vector_repo, embedding_engine, llm_engine)
    
    typer.echo(f"[*] Query: {text}")
    response = pipeline.ask(text, limit=limit)
    
    typer.echo("\n=== ANSWER ===")
    typer.echo(response)
    typer.echo("==============\n")

@app.command("stats")
def stats():
    """Prints index statistics (number of nodes and edges)."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    db_stats = graph_repo.get_stats()
    
    typer.echo("=== Database Statistics ===")
    typer.echo(f"Papers:  {db_stats['papers']}")
    typer.echo(f"Authors: {db_stats['authors']}")
    typer.echo(f"Concepts:{db_stats['concepts']}")
    typer.echo(f"Edges:   {db_stats['edges']}")
    typer.echo("===========================")

@app.command("visualize")
def visualize(
    output_path: Path = typer.Option(Path.cwd() / "graph.html", "--output", "-o", help="Path to save visual HTML file")
):
    """Generates an interactive HTML graph visualization using vis-network and opens it in browser."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    
    # Retrieve all nodes and edges from SQLite
    import sqlite3
    conn = sqlite3.connect(graph_repo.db_path)
    conn.row_factory = sqlite3.Row
    
    nodes_rows = conn.execute("SELECT id, label, properties FROM nodes").fetchall()
    edges_rows = conn.execute("SELECT source_id, target_id, type FROM edges").fetchall()
    conn.close()
    
    if not nodes_rows:
        typer.echo("Graph database is empty. Index some papers first.")
        return

    # Process nodes
    vis_nodes = []
    for r in nodes_rows:
        node_id = r["id"]
        label = r["label"]
        props = json.loads(r["properties"])
        
        # Determine node display name and color
        if label == "Paper":
            title = props.get("title", node_id)
            # Truncate title for node label
            node_label = title if len(title) < 25 else title[:22] + "..."
            color = "#4c6ef5" # Blue
            size = 25
        elif label == "Author":
            node_label = props.get("name", node_id)
            color = "#fab005" # Yellow
            size = 20
        elif label == "Concept":
            node_label = props.get("name", node_id)
            color = "#12b886" # Teal
            size = 20
        else:
            node_label = node_id
            color = "#868e96" # Grey
            size = 15
            
        vis_nodes.append({
            "id": node_id,
            "label": node_label,
            "title": f"<b>{label}</b>: {props.get('title', props.get('name', node_id))}<br>ID: {node_id}",
            "color": color,
            "size": size,
            "shape": "dot"
        })
        
    # Process edges
    vis_edges = []
    for r in edges_rows:
        vis_edges.append({
            "from": r["source_id"],
            "to": r["target_id"],
            "label": r["type"],
            "arrows": "to",
            "font": {"size": 8, "align": "top"},
            "color": {"color": "#adb5bd", "highlight": "#495057"}
        })

    # Read/Create the template HTML
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>PDF Bibliography & Concept Network</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            background-color: #1a1b1e;
            color: #c1c2c5;
        }}
        #header {{
            padding: 15px 20px;
            background-color: #25262b;
            border-bottom: 1px solid #2c2e33;
        }}
        h2 {{ margin: 0 0 5px 0; color: #fff; }}
        #mynetwork {{
            width: 100vw;
            height: calc(100vh - 75px);
            background-color: #1a1b1e;
        }}
        .legend {{
            display: inline-block;
            margin-right: 15px;
            font-size: 14px;
        }}
        .legend-color {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h2>Локальный граф научных публикаций и концептов</h2>
        <div>
            <span class="legend"><span class="legend-color" style="background-color: #4c6ef5;"></span>Статья</span>
            <span class="legend"><span class="legend-color" style="background-color: #fab005;"></span>Автор</span>
            <span class="legend"><span class="legend-color" style="background-color: #12b886;"></span>Концепт</span>
        </div>
    </div>
    <div id="mynetwork"></div>

    <script type="text/javascript">
        // Data generated from SQLite
        var nodes = new vis.DataSet({json.dumps(vis_nodes, ensure_ascii=False)});
        var edges = new vis.DataSet({json.dumps(vis_edges, ensure_ascii=False)});

        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                font: {{ color: '#c1c2c5', size: 12 }}
            }},
            edges: {{
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springLength: 200
                }}
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    typer.echo(f"[+] Graph visualization saved to {output_path}")
    
    # Auto-open in browser
    try:
        webbrowser.open(output_path.as_uri())
        typer.echo("[+] Opening visualization in browser...")
    except Exception as e:
        typer.echo(f"[!] Could not open browser automatically: {e}", err=True)

if __name__ == "__main__":
    app()
