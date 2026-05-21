"""
FastAPI Web Application for Science Graph.

Endpoints:
  GET  /              → serves web/index.html
  GET  /api/graph     → all nodes + edges for vis-network
  GET  /api/stats     → database statistics
  POST /api/query     → SSE-streamed RAG answer
  GET  /api/paper/{id}→ paper metadata + concepts/authors
  GET  /api/search    → quick title search
  POST /api/upload    → index uploaded PDF/MD/EPUB
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional, List

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import config
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import LLMEngine
from src.services.rag_service import RAGService
from src.services.note_service import NoteService
from src.schemas import (
    QueryRequest,
    NoteCreate,
    NoteResponse,
    OpenFileRequest,
    GraphResponse,
    SearchResponse,
)

# ── Dependency Injection Providers (with caching for performance) ──

_graph_repo_inst: Optional[SQLiteGraphRepository] = None
_vector_repo_inst: Optional[SQLiteVectorRepository] = None
_embedding_engine_inst: Optional[EmbeddingEngine] = None
_llm_engine_inst: Optional[LLMEngine] = None
_rag_service_inst: Optional[RAGService] = None
_note_service_inst: Optional[NoteService] = None


def get_graph_repo() -> SQLiteGraphRepository:
    global _graph_repo_inst
    if _graph_repo_inst is None:
        _graph_repo_inst = SQLiteGraphRepository(config.db_path)
    return _graph_repo_inst


def get_vector_repo() -> SQLiteVectorRepository:
    global _vector_repo_inst
    if _vector_repo_inst is None:
        _vector_repo_inst = SQLiteVectorRepository(config.db_path)
    return _vector_repo_inst


def get_embedding_engine() -> EmbeddingEngine:
    global _embedding_engine_inst
    if _embedding_engine_inst is None:
        _embedding_engine_inst = EmbeddingEngine()
    return _embedding_engine_inst


def get_llm_engine() -> Optional[LLMEngine]:
    global _llm_engine_inst
    if _llm_engine_inst is None:
        try:
            _llm_engine_inst = LLMEngine()
        except Exception as e:
            print(f"[!] LLM engine unavailable: {e}")
    return _llm_engine_inst


def get_rag_service(
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    llm_engine: Optional[LLMEngine] = Depends(get_llm_engine)
) -> Optional[RAGService]:
    global _rag_service_inst
    if _rag_service_inst is None:
        if llm_engine is None:
            return None
        _rag_service_inst = RAGService(graph_repo, vector_repo, embedding_engine, llm_engine)
    return _rag_service_inst


def get_note_service(
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    llm_engine: Optional[LLMEngine] = Depends(get_llm_engine)
) -> NoteService:
    global _note_service_inst
    if _note_service_inst is None:
        _note_service_inst = NoteService(graph_repo, vector_repo, embedding_engine, llm_engine)
    return _note_service_inst


# ── App Setup ──

app = FastAPI(title="Science Graph", version="0.1.0")

# Serve static files (frontend/ directory next to this file)
_WEB_DIR = Path(__file__).parent / "frontend"

# Mount static assets if directory exists (JS, CSS, images)
if (_WEB_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(_WEB_DIR / "js")), name="js")
if (_WEB_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(_WEB_DIR / "css")), name="css")
if (_WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index_file = _WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(str(index_file))


@app.get("/vis-network.min.js", include_in_schema=False)
async def get_vis_network():
    vis_file = _WEB_DIR / "vis-network.min.js"
    if vis_file.exists():
        return FileResponse(str(vis_file))
    raise HTTPException(status_code=404, detail="vis-network.min.js not found.")


@app.get("/favicon.png", include_in_schema=False)
async def get_favicon():
    fav_file = _WEB_DIR / "favicon.png"
    if fav_file.exists():
        return FileResponse(str(fav_file))
    raise HTTPException(status_code=404, detail="favicon.png not found.")


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon_ico():
    fav_file = _WEB_DIR / "favicon.png"
    if fav_file.exists():
        return FileResponse(str(fav_file))
    raise HTTPException(status_code=404, detail="favicon.ico not found.")


# ── /api/stats ──

@app.get("/api/stats")
async def get_stats(graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)):
    stats = await asyncio.to_thread(graph_repo.get_stats)
    try:
        storage_stats = await asyncio.to_thread(config.get_storage_stats)
        stats["storage"] = storage_stats
    except Exception:
        pass
    return JSONResponse(stats)


# ── /api/graph ──

@app.get("/api/graph", response_model=GraphResponse)
async def get_graph(graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)):
    """Returns all nodes and edges formatted for vis-network."""
    nodes_rows = await asyncio.to_thread(graph_repo.get_all_nodes)
    edges_rows = await asyncio.to_thread(graph_repo.get_all_edges)

    vis_nodes = []
    for node_id, label, properties_json in nodes_rows:
        props = json.loads(properties_json or "{}")
        source_type = props.get("source_type", "paper")

        if label == "Paper":
            title = props.get("title", node_id)
            display = title if len(title) < 28 else title[:25] + "…"
            color_map = {"note": "#f03e3e", "book": "#7950f2", "paper": "#4c6ef5"}
            color = color_map.get(source_type, "#4c6ef5")
            size = 25
            group = source_type
        elif label == "Author":
            display = props.get("name", node_id)
            color = "#fab005"
            size = 18
            group = "author"
        elif label == "Concept":
            display = props.get("name", node_id)
            if props.get("is_tag"):
                color = "#e64980"
                size = 15
                group = "tag"
            else:
                color = "#12b886"
                size = 16
                group = "concept"
        else:
            display = node_id
            color = "#868e96"
            size = 14
            group = "other"

        tooltip = f"<b>{label}</b>: {props.get('title', props.get('name', node_id))}"
        if props.get("year"):
            tooltip += f"<br>Year: {props['year']}"
        if props.get("authors"):
            tooltip += f"<br>Authors: {', '.join(props['authors'][:3])}"

        vis_nodes.append({
            "id": node_id,
            "label": display,
            "title": tooltip,
            "color": color,
            "size": size,
            "group": group,
            "shape": "dot",
            "created_at": props.get("created_at"),
            "source_type": source_type,
            "full_title": props.get("title", props.get("name", node_id)),
        })

    vis_edges = []
    for source_id, target_id, edge_type, edge_properties in edges_rows:
        vis_edges.append({
            "from": source_id,
            "to": target_id,
            "label": edge_type,
            "arrows": "to",
            "font": {"size": 8, "align": "top"},
            "color": {"color": "#adb5bd", "highlight": "#74c0fc"},
        })

    return {"nodes": vis_nodes, "edges": vis_edges}


# ── /api/paper/{id} ──

@app.get("/api/paper/{paper_id:path}")
async def get_paper(
    paper_id: str,
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    node = await asyncio.to_thread(graph_repo.get_node_by_id, paper_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    label, properties_json = node
    props = json.loads(properties_json or "{}")

    if label == "Paper":
        paper = await asyncio.to_thread(graph_repo.get_paper, paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        neighbors = await asyncio.to_thread(graph_repo.get_neighbors, paper_id, max_depth=1)

        # ── Collect IDs for batch fetching ──
        concept_ids: list[str] = []
        tag_ids: list[str] = []
        author_ids: list[str] = []
        citation_ids: list[str] = []
        cited_by_ids: list[str] = []

        for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props_json in neighbors:
            if edge_type == "MENTIONS_CONCEPT":
                concept_ids.append(tgt_id)
            elif edge_type == "HAS_TAG":
                tag_ids.append(tgt_id)
            elif edge_type == "AUTHORED":
                author_ids.append(src_id)
            elif edge_type == "CITES" and src_id == paper_id:
                citation_ids.append(tgt_id)
            elif edge_type == "CITES" and tgt_id == paper_id:
                cited_by_ids.append(src_id)

        # ── Batch fetch citations and cited_by papers ──
        citations_map = await asyncio.to_thread(
            graph_repo.get_papers_batch, citation_ids
        ) if citation_ids else {}
        cited_by_map = await asyncio.to_thread(
            graph_repo.get_papers_batch, cited_by_ids
        ) if cited_by_ids else {}

        # ── Batch fetch concept and tag node info ──
        async def _get_node(nid: str):
            return await asyncio.to_thread(graph_repo.get_node_by_id, nid)

        import asyncio as _aio
        concept_nodes = await _aio.gather(*[_get_node(cid) for cid in concept_ids])
        tag_nodes = await _aio.gather(*[_get_node(tid) for tid in tag_ids])

        concepts = []
        for cid, node in zip(concept_ids, concept_nodes):
            if node:
                c_props = json.loads(node[1] or "{}")
                concepts.append({"id": cid, "name": c_props.get("name", cid)})
            else:
                concepts.append({"id": cid, "name": cid})

        tags = []
        for tid, node in zip(tag_ids, tag_nodes):
            if node:
                t_props = json.loads(node[1] or "{}")
                tags.append({"id": tid, "name": t_props.get("name", tid)})
            else:
                tags.append({"id": tid, "name": tid})

        # Resolve author names (typically a small set, so individual calls are acceptable)
        authors = paper.authors or []
        if not authors and author_ids:
            author_objs = await _aio.gather(
                *[asyncio.to_thread(graph_repo.get_author, aid) for aid in author_ids]
            )
            authors = [a.name for a in author_objs if a]

        citations = [
            {"id": pid, "title": p.title}
            for pid in citation_ids
            if (p := citations_map.get(pid)) and p.title
        ][:10]

        cited_by = [
            {"id": pid, "title": p.title}
            for pid in cited_by_ids
            if (p := cited_by_map.get(pid)) and p.title
        ][:10]

        return JSONResponse({
            "type": "paper",
            "id": paper.id,
            "title": paper.title,
            "authors": authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "source_type": paper.properties.get("source_type", "paper"),
            "concepts": concepts,
            "tags": tags,
            "citations": citations,
            "cited_by": cited_by,
            "file_path": paper.file_path,
            "summary": paper.properties.get("summary"),
            "created_at": paper.created_at,
        })

    elif label == "Author":
        papers_list = await asyncio.to_thread(graph_repo.get_papers_by_author, paper_id)
        papers = []
        for p in papers_list:
            papers.append({
                "id": p.id,
                "title": p.title,
                "source_type": p.properties.get("source_type", "paper")
            })

        return JSONResponse({
            "type": "author",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "papers": papers,
            "papers_count": len(papers)
        })

    elif label == "Concept":
        is_tag = props.get("is_tag", False)
        edge_type = "HAS_TAG" if is_tag else "MENTIONS_CONCEPT"
        papers_list = await asyncio.to_thread(graph_repo.get_papers_by_entity, paper_id, edge_type)

        papers = []
        for p in papers_list:
            papers.append({
                "id": p.id,
                "title": p.title,
                "source_type": p.properties.get("source_type", "paper")
            })

        related_entities = []
        if papers:
            paper_ids = [p["id"] for p in papers]
            related_edge_type = "HAS_TAG" if not is_tag else "MENTIONS_CONCEPT"
            targets = await asyncio.to_thread(graph_repo.get_distinct_targets, paper_ids, related_edge_type)
            for t_id, t_props_json in targets:
                t_props = json.loads(t_props_json or "{}")
                related_entities.append({
                    "id": t_id,
                    "name": t_props.get("name", t_id)
                })

        return JSONResponse({
            "type": "tag" if is_tag else "concept",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "description": props.get("description", f"No description available for '{props.get('name', paper_id)}'."),
            "papers": papers,
            "related": related_entities
        })


# ── /api/search ──

@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo)
):
    """Quick keyword search over paper/note/book titles."""
    papers = await asyncio.to_thread(graph_repo.search_papers_by_title, q, 20)

    results = []
    for paper in papers:
        results.append({
            "id": paper.id,
            "title": paper.title,
            "year": paper.year,
            "source_type": paper.properties.get("source_type", "paper")
        })

    return {"results": results}


# ── /api/query (SSE streaming) ──

@app.post("/api/query")
async def query_rag(
    body: QueryRequest,
    rag_service: Optional[RAGService] = Depends(get_rag_service)
):
    """
    SSE-streamed RAG answer.
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="LLM engine is not available.")

    async def event_stream() -> AsyncGenerator[dict, None]:
        async for event in rag_service.stream_rag_response(body.question, body.limit):
            yield {"data": json.dumps(event)}

    return EventSourceResponse(event_stream())


# ── /api/open-file ──

@app.post("/api/open-file")
async def open_file(body: OpenFileRequest):
    file_path = body.file_path
    expanded = os.path.expanduser(file_path)
    if not os.path.exists(expanded):
        expanded = str(Path(file_path).resolve())
        if not os.path.exists(expanded):
            raise HTTPException(status_code=404, detail=f"File not found on host: {file_path}")

    import subprocess
    import sys

    def _open():
        if sys.platform == "win32":
            os.startfile(expanded)
        elif sys.platform == "darwin":
            subprocess.run(["open", expanded], check=True)
        else:
            subprocess.run(["xdg-open", expanded], check=True)

    try:
        await asyncio.to_thread(_open)
        return {"status": "ok", "message": f"Opened {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/notes ──

@app.get("/api/notes", response_model=List[NoteResponse])
async def get_notes(note_service: NoteService = Depends(get_note_service)):
    notes = await asyncio.to_thread(note_service.get_notes)
    return notes


@app.post("/api/notes")
async def create_note(
    body: NoteCreate,
    note_service: NoteService = Depends(get_note_service)
):
    try:
        paper_id, note_path = await asyncio.to_thread(
            note_service.create_note,
            body.title,
            body.content,
            body.authors,
            body.tags
        )
        return JSONResponse({"status": "ok", "id": paper_id, "file_path": note_path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/upload ──

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    graph_repo: SQLiteGraphRepository = Depends(get_graph_repo),
    vector_repo: SQLiteVectorRepository = Depends(get_vector_repo),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    llm_engine: Optional[LLMEngine] = Depends(get_llm_engine)
):
    """Accepts a PDF, .md or .epub file and indexes it."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".epub"}:
        raise HTTPException(status_code=400, detail="Only PDF, Markdown (.md), and EPUB files are supported.")
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine is not available for indexing.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from src.indexer import Indexer
        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

        def _index():
            if suffix == ".pdf":
                return indexer.index_pdf(tmp_path)
            elif suffix == ".md":
                return indexer.index_markdown(tmp_path)
            else:
                return indexer.index_epub(tmp_path)

        paper_id = await asyncio.to_thread(_index)
        return JSONResponse({"status": "ok", "id": paper_id, "filename": file.filename})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
