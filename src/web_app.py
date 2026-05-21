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
import sqlite3
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import config
from src.indexer import Indexer
from src.llm_engine import LLMEngine
from src.rag import RAGPipeline
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine

# ── Singletons (loaded once at startup) ──────────────────────────────────────

_graph_repo: Optional[SQLiteGraphRepository] = None
_vector_repo: Optional[SQLiteVectorRepository] = None
_embedding_engine: Optional[EmbeddingEngine] = None
_llm_engine: Optional[LLMEngine] = None
_rag_pipeline: Optional[RAGPipeline] = None


def _get_repos():
    global _graph_repo, _vector_repo
    if _graph_repo is None:
        _graph_repo = SQLiteGraphRepository(config.db_path)
        _vector_repo = SQLiteVectorRepository(config.db_path)
    return _graph_repo, _vector_repo


def _get_embedding_engine() -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine


def _get_llm_engine() -> Optional[LLMEngine]:
    global _llm_engine
    if _llm_engine is None:
        try:
            _llm_engine = LLMEngine()
        except Exception as e:
            print(f"[!] LLM engine unavailable: {e}")
    return _llm_engine


def _get_rag() -> Optional[RAGPipeline]:
    global _rag_pipeline
    if _rag_pipeline is None:
        gr, vr = _get_repos()
        emb = _get_embedding_engine()
        llm = _get_llm_engine()
        if llm is None:
            return None
        _rag_pipeline = RAGPipeline(gr, vr, emb, llm)
    return _rag_pipeline


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Science Graph", version="0.1.0")

# Serve static files (web/ directory next to this file)
_WEB_DIR = Path(__file__).parent / "web"

# Mount static assets if directory exists (JS, CSS, images)
if (_WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index_file = _WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(str(index_file))


# ── /api/stats ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    gr, _ = _get_repos()
    stats = gr.get_stats()
    return JSONResponse(stats)


# ── /api/graph ────────────────────────────────────────────────────────────────

@app.get("/api/graph")
async def get_graph():
    """Returns all nodes and edges formatted for vis-network."""
    gr, _ = _get_repos()

    conn = sqlite3.connect(gr.db_path)
    conn.row_factory = sqlite3.Row

    nodes_rows = conn.execute("SELECT id, label, properties FROM nodes").fetchall()
    edges_rows = conn.execute("SELECT source_id, target_id, type FROM edges").fetchall()
    conn.close()

    vis_nodes = []
    for r in nodes_rows:
        props = json.loads(r["properties"] or "{}")
        source_type = props.get("source_type", "paper")
        label = r["label"]

        if label == "Paper":
            title = props.get("title", r["id"])
            display = title if len(title) < 28 else title[:25] + "…"
            # Colour by source_type
            color_map = {"note": "#f03e3e", "book": "#7950f2", "paper": "#4c6ef5"}
            color = color_map.get(source_type, "#4c6ef5")
            size = 25
            group = source_type
        elif label == "Author":
            display = props.get("name", r["id"])
            color = "#fab005"
            size = 18
            group = "author"
        elif label == "Concept":
            display = props.get("name", r["id"])
            color = "#12b886"
            size = 16
            group = "concept"
        else:
            display = r["id"]
            color = "#868e96"
            size = 14
            group = "other"

        tooltip = f"<b>{label}</b>: {props.get('title', props.get('name', r['id']))}"
        if props.get("year"):
            tooltip += f"<br>Year: {props['year']}"
        if props.get("authors"):
            tooltip += f"<br>Authors: {', '.join(props['authors'][:3])}"

        vis_nodes.append({
            "id": r["id"],
            "label": display,
            "title": tooltip,
            "color": color,
            "size": size,
            "group": group,
            "shape": "dot",
        })

    vis_edges = []
    for r in edges_rows:
        vis_edges.append({
            "from": r["source_id"],
            "to": r["target_id"],
            "label": r["type"],
            "arrows": "to",
            "font": {"size": 8, "align": "top"},
            "color": {"color": "#adb5bd", "highlight": "#74c0fc"},
        })

    return JSONResponse({"nodes": vis_nodes, "edges": vis_edges})


# ── /api/paper/{id} ──────────────────────────────────────────────────────────

@app.get("/api/paper/{paper_id:path}")
async def get_paper(paper_id: str):
    gr, _ = _get_repos()
    paper = gr.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    neighbors = gr.get_neighbors(paper_id, max_depth=1)
    concepts = []
    authors = []
    citations = []
    cited_by = []

    for src_id, src_label, edge_type, tgt_id, tgt_label, _ in neighbors:
        if edge_type == "MENTIONS_CONCEPT":
            c = gr.get_concept(tgt_id)
            if c:
                concepts.append(c.name)
        elif edge_type == "AUTHORED":
            a = gr.get_author(src_id)
            if a:
                authors.append(a.name)
        elif edge_type == "CITES" and src_id == paper_id:
            ref = gr.get_paper(tgt_id)
            if ref and ref.title:
                citations.append(ref.title)
        elif edge_type == "CITES" and tgt_id == paper_id:
            citer = gr.get_paper(src_id)
            if citer and citer.title:
                cited_by.append(citer.title)

    return JSONResponse({
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors or authors,
        "year": paper.year,
        "doi": paper.doi,
        "abstract": paper.abstract,
        "source_type": paper.properties.get("source_type", "paper"),
        "concepts": concepts,
        "citations": citations[:10],
        "cited_by": cited_by[:10],
    })


# ── /api/search ──────────────────────────────────────────────────────────────

@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):
    """Quick keyword search over paper/note/book titles."""
    gr, _ = _get_repos()

    conn = sqlite3.connect(gr.db_path)
    conn.row_factory = sqlite3.Row
    q_like = f"%{q.lower()}%"
    rows = conn.execute(
        "SELECT id, label, properties FROM nodes WHERE label = 'Paper' AND LOWER(properties) LIKE ? LIMIT 20",
        (q_like,),
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        props = json.loads(r["properties"] or "{}")
        title = props.get("title", r["id"])
        if q.lower() in title.lower():
            results.append({
                "id": r["id"],
                "title": title,
                "year": props.get("year"),
                "source_type": props.get("source_type", "paper"),
            })

    return JSONResponse({"results": results})


# ── /api/query (SSE streaming) ────────────────────────────────────────────────

@app.post("/api/query")
async def query_rag(body: dict):
    """
    SSE-streamed RAG answer.
    Body: {"question": "...", "limit": 5}
    Streams events:
      - data: {"type": "token", "text": "..."}
      - data: {"type": "done"}
      - data: {"type": "error", "text": "..."}
    """
    question = (body.get("question") or "").strip()
    limit = int(body.get("limit") or 5)

    if not question:
        raise HTTPException(status_code=422, detail="'question' field is required")

    rag = _get_rag()

    async def event_stream() -> AsyncGenerator[dict, None]:
        if rag is None:
            yield {"data": json.dumps({"type": "error", "text": "LLM engine is not available."})}
            return

        # Run blocking RAG in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()

        # Build context first (fast, no LLM)
        try:
            query_emb = await loop.run_in_executor(None, rag.emb_engine.get_embedding, question)
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "text": f"Embedding failed: {e}"})}
            return

        # Hybrid retrieval
        try:
            all_chunks = await loop.run_in_executor(None, rag.vector_repo.get_all_chunks)
            if not all_chunks:
                yield {"data": json.dumps({"type": "error", "text": "No documents indexed yet."})}
                return

            from src.vector_search import BM25
            dense = await loop.run_in_executor(
                None, rag.vector_repo.search_similar_chunks, query_emb, limit * 2
            )
            corpus = [(c.id, c.text_content) for c in all_chunks]
            bm25 = BM25(corpus)
            bm25_results = bm25.score(question)[: limit * 2]

            id_to_chunk = {c.id: c for c in all_chunks}
            rrf: dict = {}
            for rank, (chunk, _) in enumerate(dense, start=1):
                rrf[chunk.id] = rrf.get(chunk.id, 0.0) + 1.0 / (60.0 + rank)
            for rank, (cid, _) in enumerate(bm25_results, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60.0 + rank)

            sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[: limit * 2]
            candidates = [id_to_chunk[cid] for cid in sorted_ids if cid in id_to_chunk]

            # Rerank
            try:
                reranker = await loop.run_in_executor(None, rag._get_reranker)
                pairs = [(question, c.text_content) for c in candidates]
                scores = await loop.run_in_executor(None, reranker.predict, pairs)
                scored = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
                final_chunks = [(c, s) for c, s in scored[:limit]]
            except Exception:
                final_chunks = [(id_to_chunk[cid], rrf[cid]) for cid in sorted_ids[:limit] if cid in id_to_chunk]

            context_text, context_graph = rag.build_context(final_chunks)
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "text": f"Retrieval failed: {e}"})}
            return

        # Build prompt
        prompt = (
            "<|im_start|>system\n"
            "You are a research assistant. Synthesize an answer using the retrieved context.\n"
            "Always cite paper titles, years, and authors. Use the graph connections if relevant.\n\n"
            f"### RELEVANT TEXT FRAGMENTS:\n{context_text}\n\n"
            f"### KNOWLEDGE GRAPH CONNECTIONS:\n{context_graph}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {question}\nAnswer in Russian:\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Stream tokens via mlx_lm
        try:
            from mlx_lm import stream_generate

            def _stream():
                return stream_generate(
                    model=rag.llm_engine.model,
                    tokenizer=rag.llm_engine.tokenizer,
                    prompt=prompt,
                    max_tokens=config.llm_max_tokens,
                )

            gen = await loop.run_in_executor(None, _stream)
            # stream_generate returns an iterator of GenerationResponse objects
            for response in gen:
                token_text = response.text if hasattr(response, "text") else str(response)
                if token_text:
                    yield {"data": json.dumps({"type": "token", "text": token_text})}
                    await asyncio.sleep(0)  # yield control to event loop

        except ImportError:
            # Fallback: generate full answer at once, stream word by word
            full_answer = await loop.run_in_executor(
                None, rag.llm_engine.generate_response, prompt
            )
            for word in full_answer.split(" "):
                yield {"data": json.dumps({"type": "token", "text": word + " "})}
                await asyncio.sleep(0.01)

        except Exception as e:
            yield {"data": json.dumps({"type": "error", "text": f"Generation failed: {e}"})}
            return

        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(event_stream())


# ── /api/upload ───────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accepts a PDF, .md or .epub file and indexes it."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".epub"}:
        raise HTTPException(status_code=400, detail="Only PDF, Markdown (.md), and EPUB files are supported.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        gr, vr = _get_repos()
        emb = _get_embedding_engine()
        llm = _get_llm_engine()
        indexer = Indexer(gr, vr, emb, llm)

        loop = asyncio.get_event_loop()
        if suffix == ".pdf":
            paper_id = await loop.run_in_executor(None, indexer.index_pdf, tmp_path)
        elif suffix == ".md":
            paper_id = await loop.run_in_executor(None, indexer.index_markdown, tmp_path)
        else:
            paper_id = await loop.run_in_executor(None, indexer.index_epub, tmp_path)

        return JSONResponse({"status": "ok", "id": paper_id, "filename": file.filename})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
