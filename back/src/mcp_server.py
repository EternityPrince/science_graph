import os
import sys
import json
import logging
from typing import List
from pathlib import Path

from fastmcp import FastMCP

# Ensure we set MCP mode if not already set, so console output is redirected to stderr
os.environ["SCIENCE_GRAPH_MCP_MODE"] = "1"

# Add parent directory of src to path to ensure proper imports if run directly
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config import config
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository
from src.vector_search import EmbeddingEngine
from src.indexer import DuplicateDocumentError, Indexer
from src.llm_engine import LLMEngine
from src.services.rag_service import RAGService
from src.services.note_service import NoteService

# Suppress debug logs from fastmcp/mcp which write to stdout/stderr
logging.basicConfig(level=logging.WARNING)

mcp = FastMCP("Science Graph")

# ── Service Accessors (Singletons) ──
_graph_repo = None
_vector_repo = None
_emb_engine = None

def get_graph_repo():
    global _graph_repo
    if _graph_repo is None:
        _graph_repo = SQLiteGraphRepository(config.db_path)
    return _graph_repo

def get_vector_repo():
    global _vector_repo
    if _vector_repo is None:
        _vector_repo = SQLiteVectorRepository(config.db_path)
    return _vector_repo

def get_embedding_engine():
    global _emb_engine
    if _emb_engine is None:
        _emb_engine = EmbeddingEngine()
    return _emb_engine

def get_llm_engine(use_cloud: bool = False) -> LLMEngine:
    # Use config default if use_cloud is not specified or fallback
    use_cloud_flag = use_cloud or os.environ.get("SCIENCE_GRAPH_USE_CLOUD") == "1"
    llm = LLMEngine(use_cloud=use_cloud_flag)
    return llm

def get_rag_service(use_cloud: bool = False) -> RAGService:
    return RAGService(
        get_graph_repo(),
        get_vector_repo(),
        get_embedding_engine(),
        get_llm_engine(use_cloud)
    )

def get_note_service(use_cloud: bool = False) -> NoteService:
    return NoteService(
        get_graph_repo(),
        get_vector_repo(),
        get_embedding_engine(),
        get_llm_engine(use_cloud)
    )


# ── MCP Tools ──

@mcp.tool
def get_stats() -> dict:
    """Get database statistics including counts of papers, authors, concepts, tags, and database storage details."""
    graph_repo = get_graph_repo()
    stats = graph_repo.get_stats()
    try:
        storage_stats = config.get_storage_stats()
        stats["storage"] = storage_stats
    except Exception:
        pass
    return stats

@mcp.tool
def search_papers(query: str, limit: int = 20) -> list:
    """Quick keyword search over paper/note/book titles. Returns list of matches with ID and title."""
    graph_repo = get_graph_repo()
    papers = graph_repo.search_papers_by_title(query, limit)
    results = []
    for paper in papers:
        results.append({
            "id": paper.id,
            "title": paper.title,
            "year": paper.year,
            "source_type": paper.properties.get("source_type", "paper")
        })
    return results

@mcp.tool
def query_rag(question: str, limit: int = 5, use_cloud: bool = False) -> str:
    """
    Query the RAG pipeline.
    Retrieves relevant text chunks and graph connections, runs them through the local (or cloud) LLM,
    and returns a cited, context-aware answer in Russian.
    """
    try:
        rag_service = get_rag_service(use_cloud=use_cloud)
        response = rag_service.ask(question, limit=limit)
        return response
    except Exception as e:
        return f"Error executing RAG query: {str(e)}"

@mcp.tool
def get_paper_details(paper_id: str) -> dict:
    """Get detailed metadata, abstract, authors, concepts, tags, and citation relationships for a specific paper or node ID."""
    graph_repo = get_graph_repo()
    node = graph_repo.get_node_by_id(paper_id)
    if not node:
        return {"error": f"Node with ID '{paper_id}' not found."}

    label, properties_json = node
    props = json.loads(properties_json or "{}")

    if label == "Paper":
        paper = graph_repo.get_paper(paper_id)
        if not paper:
            return {"error": f"Paper with ID '{paper_id}' not found."}

        neighbors = graph_repo.get_neighbors(paper_id, max_depth=1)

        concept_ids = []
        tag_ids = []
        author_ids = []
        citation_ids = []
        cited_by_ids = []

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

        citations_map = graph_repo.get_papers_batch(citation_ids) if citation_ids else {}
        cited_by_map = graph_repo.get_papers_batch(cited_by_ids) if cited_by_ids else {}

        concepts = []
        for cid in concept_ids:
            c_node = graph_repo.get_node_by_id(cid)
            c_name = json.loads(c_node[1] or "{}").get("name", cid) if c_node else cid
            concepts.append({"id": cid, "name": c_name})

        tags = []
        for tid in tag_ids:
            t_node = graph_repo.get_node_by_id(tid)
            t_name = json.loads(t_node[1] or "{}").get("name", tid) if t_node else tid
            tags.append({"id": tid, "name": t_name})

        authors = paper.authors or []
        if not authors and author_ids:
            author_objs = [graph_repo.get_author(aid) for aid in author_ids]
            authors = [a.name for a in author_objs if a]

        citations = [{"id": pid, "title": p.title} for pid in citation_ids if (p := citations_map.get(pid)) and p.title]
        cited_by = [{"id": pid, "title": p.title} for pid in cited_by_ids if (p := cited_by_map.get(pid)) and p.title]

        return {
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
            "properties": paper.properties,
        }
    elif label == "Author":
        papers_list = graph_repo.get_papers_by_author(paper_id)
        papers = [{"id": p.id, "title": p.title, "source_type": p.properties.get("source_type", "paper")} for p in papers_list]
        return {
            "type": "author",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "papers": papers,
            "papers_count": len(papers)
        }
    elif label == "Concept":
        is_tag = props.get("is_tag", False)
        edge_type = "HAS_TAG" if is_tag else "MENTIONS_CONCEPT"
        papers_list = graph_repo.get_papers_by_entity(paper_id, edge_type)
        papers = [{"id": p.id, "title": p.title, "source_type": p.properties.get("source_type", "paper")} for p in papers_list]
        return {
            "type": "tag" if is_tag else "concept",
            "id": paper_id,
            "name": props.get("name", paper_id),
            "description": props.get("description", f"No description available for '{props.get('name', paper_id)}'."),
            "papers": papers,
        }
    return {"id": paper_id, "label": label, "properties": props}

@mcp.tool
def index_file(file_path: str, use_cloud: bool = False) -> dict:
    """Index a local document file (PDF, EPUB, or Markdown) by its path on the host. Extracted concepts and metadata will be added to the graph."""
    expanded_path = os.path.expanduser(file_path)
    if not os.path.exists(expanded_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    suffix = Path(expanded_path).suffix.lower()
    if suffix not in {".pdf", ".md", ".epub"}:
        return {"status": "error", "message": "Only PDF, Markdown (.md), and EPUB files are supported."}

    try:
        graph_repo = get_graph_repo()
        vector_repo = get_vector_repo()
        embedding_engine = get_embedding_engine()
        llm_engine = get_llm_engine(use_cloud=use_cloud)

        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)
        if suffix == ".pdf":
            paper_id = indexer.index_pdf(expanded_path)
        elif suffix == ".md":
            paper_id = indexer.index_markdown(expanded_path)
        else:
            paper_id = indexer.index_epub(expanded_path)

        return {"status": "success", "id": paper_id, "file_path": expanded_path}
    except DuplicateDocumentError as e:
        return {"status": "error", "message": f"Duplicate document: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Indexing failed: {str(e)}"}

@mcp.tool
def index_url(url: str, use_cloud: bool = False) -> dict:
    """Index a webpage or YouTube video URL. Transcribes YouTube audio locally using Whisper, extracts text and entities, and adds them to the graph."""
    try:
        graph_repo = get_graph_repo()
        vector_repo = get_vector_repo()
        embedding_engine = get_embedding_engine()
        llm_engine = get_llm_engine(use_cloud=use_cloud)

        indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)
        
        import re
        urls = [u.strip() for u in re.split(r'[,;]', url) if u.strip()]
        if not urls:
            return {"status": "error", "message": "No URLs provided."}
            
        paper_ids = []
        titles = []
        
        for u in urls:
            paper_id = indexer.index_url(u)
            paper = graph_repo.get_paper(paper_id)
            title = paper.title if paper else u
            paper_ids.append(paper_id)
            titles.append(title)
            
        return {
            "status": "success", 
            "id": ", ".join(paper_ids), 
            "title": ", ".join(titles),
            "url": url
        }
    except DuplicateDocumentError as e:
        return {"status": "error", "message": f"Duplicate document: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Indexing URL failed: {str(e)}"}

@mcp.tool
def get_notes() -> list:
    """Retrieve list of all research notes."""
    note_service = get_note_service()
    notes = note_service.get_notes()
    return notes

@mcp.tool
def create_note(title: str, content: str, authors: List[str] = None, tags: List[str] = None, use_cloud: bool = False) -> dict:
    """Create a new research note with title, markdown content, optional authors, and tags. Synthesizes concepts and adds the note to the graph."""
    try:
        note_service = get_note_service(use_cloud=use_cloud)
        paper_id, note_path = note_service.create_note(
            title,
            content,
            authors or [],
            tags or []
        )
        return {"status": "success", "id": paper_id, "file_path": note_path}
    except Exception as e:
        return {"status": "error", "message": f"Failed to create note: {str(e)}"}

if __name__ == "__main__":
    mcp.run()
