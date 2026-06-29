import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.models import Paper, Concept, Chunk
from src.repository.sqlite_impl import SQLiteGraphRepository, SQLiteVectorRepository

def make_chunk(chunk_id: str, paper_id: str, chunk_index: int, text: str, embedding: Optional[List[float]] = None) -> Chunk:
    """Helper to create a Chunk object."""
    return Chunk(
        id=chunk_id,
        paper_id=paper_id,
        text_content=text,
        page_number=1,
        embedding=embedding or [0.1] * 384
    )

def seed_minimal_graph(graph_repo: SQLiteGraphRepository, vector_repo: SQLiteVectorRepository):
    """
    Seeds a SQLite database with the minimal test graph.
    P1 seed: mentions c_memory, cites P3
    P2 seed: mentions c_agents
    P3 candidate: mentions c_memory, citation neighbor of P1
    P4 bridge: mentions c_memory, mentions c_agents (connects P1 & P2)
    P5 unrelated: mentions c_unrelated
    P6 high-frequency concept paper: mentions c_general
    """
    # 1. Clear database tables first to be completely clean
    with graph_repo._get_connection() as conn:
        conn.execute("DELETE FROM edges;")
        conn.execute("DELETE FROM nodes;")
        conn.execute("DELETE FROM chunks;")
        conn.commit()

    # 2. Save Papers
    papers = [
        Paper(id="P1", title="Paper One", authors=["Author One"], year=2020, doi="doi_1", abstract="Seed paper one abstract"),
        Paper(id="P2", title="Paper Two", authors=["Author Two"], year=2021, doi="doi_2", abstract="Seed paper two abstract"),
        Paper(id="P3", title="Paper Three", authors=["Author Three"], year=2022, doi="doi_3", abstract="Candidate paper three abstract"),
        Paper(id="P4", title="Paper Four", authors=["Author Four"], year=2023, doi="doi_4", abstract="Bridge paper four abstract"),
        Paper(id="P5", title="Paper Five", authors=["Author Five"], year=2024, doi="doi_5", abstract="Unrelated paper five abstract"),
        Paper(id="P6", title="Paper Six", authors=["Author Six"], year=2025, doi="doi_6", abstract="General paper six abstract"),
    ]
    for p in papers:
        graph_repo.save_paper(p)

    # 3. Save Concepts
    concepts = [
        Concept(id="c_memory", name="memory", properties={"name": "memory", "aliases": ["ltm", "working memory"]}),
        Concept(id="c_agents", name="agents", properties={"name": "agents", "aliases": ["agentic"]}),
        Concept(id="c_general", name="general", properties={"name": "general", "aliases": ["common"]}),
        Concept(id="c_unrelated", name="nonexistent_concept", properties={"name": "nonexistent_concept", "aliases": []}),
    ]
    for c in concepts:
        graph_repo.save_concept(c)

    # 4. Save Edges
    edges = [
        ("P1", "c_memory", "MENTIONS_CONCEPT"),
        ("P3", "c_memory", "MENTIONS_CONCEPT"),
        ("P4", "c_memory", "MENTIONS_CONCEPT"),
        ("P2", "c_agents", "MENTIONS_CONCEPT"),
        ("P4", "c_agents", "MENTIONS_CONCEPT"),
        ("P6", "c_general", "MENTIONS_CONCEPT"),
        ("P5", "c_unrelated", "MENTIONS_CONCEPT"),
        ("P1", "P3", "CITES"),
    ]
    with graph_repo.transaction():
        for src, tgt, etype in edges:
            graph_repo.add_edge(src, tgt, etype)

    # 5. Save Chunks (2 chunks per paper for scoped limit checks)
    chunk_embeddings = {
        "p1#0": [1.0] + [0.0] * 383,
        "p1#1": [0.8, 0.6] + [0.0] * 382,
        "p2#0": [1.0] + [0.0] * 383,
        "p2#1": [0.8, 0.6] + [0.0] * 382,
        "p3#0": [1.0] + [0.0] * 383,
        "p3#1": [0.8, 0.6] + [0.0] * 382,
        "p4#0": [0.9, 0.1] + [0.0] * 382,
        "p4#1": [0.7, 0.7] + [0.0] * 382,
        "p5#0": [1.0] + [0.0] * 383,
        "p5#1": [0.8, 0.6] + [0.0] * 382,
        "p6#0": [1.0] + [0.0] * 383,
        "p6#1": [0.8, 0.6] + [0.0] * 382,
    }

    chunks = []
    for pid in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        for idx in range(2):
            cid = f"{pid.lower()}#{idx}"
            chunks.append(Chunk(
                id=cid,
                paper_id=pid,
                text_content=f"This is chunk {idx} of paper {pid}.",
                page_number=1,
                embedding=chunk_embeddings[cid]
            ))
    vector_repo.save_chunks(chunks)


class FakeGraphRepository:
    def __init__(self):
        self.papers = {
            "P1": {"id": "P1", "label": "Paper", "title": "Paper One", "properties": {"title": "Paper One", "year": 2020}},
            "P2": {"id": "P2", "label": "Paper", "title": "Paper Two", "properties": {"title": "Paper Two", "year": 2021}},
            "P3": {"id": "P3", "label": "Paper", "title": "Paper Three", "properties": {"title": "Paper Three", "year": 2022}},
            "P4": {"id": "P4", "label": "Paper", "title": "Paper Four", "properties": {"title": "Paper Four", "year": 2023}},
            "P5": {"id": "P5", "label": "Paper", "title": "Paper Five", "properties": {"title": "Paper Five", "year": 2024}},
            "P6": {"id": "P6", "label": "Paper", "title": "Paper Six", "properties": {"title": "Paper Six", "year": 2025}},
        }
        self.concepts = {
            "c_memory": {"id": "c_memory", "label": "Concept", "title": "memory", "properties": {"name": "memory", "aliases": ["ltm", "working memory"]}},
            "c_agents": {"id": "c_agents", "label": "Concept", "title": "agents", "properties": {"name": "agents", "aliases": ["agentic"]}},
            "c_general": {"id": "c_general", "label": "Concept", "title": "general", "properties": {"name": "general", "aliases": ["common"]}},
            "c_unrelated": {"id": "c_unrelated", "label": "Concept", "title": "nonexistent_concept", "properties": {"name": "nonexistent_concept", "aliases": []}},
        }
        self.edges = [
            ("P1", "c_memory", "MENTIONS_CONCEPT"),
            ("P3", "c_memory", "MENTIONS_CONCEPT"),
            ("P4", "c_memory", "MENTIONS_CONCEPT"),
            ("P2", "c_agents", "MENTIONS_CONCEPT"),
            ("P4", "c_agents", "MENTIONS_CONCEPT"),
            ("P6", "c_general", "MENTIONS_CONCEPT"),
            ("P5", "c_unrelated", "MENTIONS_CONCEPT"),
            ("P1", "P3", "CITES"),
        ]
        self.chunks = {
            "P1": [
                Chunk(id="p1#0", paper_id="P1", text_content="This is chunk 0 of paper P1.", page_number=1, embedding=[1.0] + [0.0] * 383),
                Chunk(id="p1#1", paper_id="P1", text_content="This is chunk 1 of paper P1.", page_number=1, embedding=[0.8, 0.6] + [0.0] * 382),
            ],
            "P2": [
                Chunk(id="p2#0", paper_id="P2", text_content="This is chunk 0 of paper P2.", page_number=1, embedding=[1.0] + [0.0] * 383),
                Chunk(id="p2#1", paper_id="P2", text_content="This is chunk 1 of paper P2.", page_number=1, embedding=[0.8, 0.6] + [0.0] * 382),
            ],
            "P3": [
                Chunk(id="p3#0", paper_id="P3", text_content="This is chunk 0 of paper P3.", page_number=1, embedding=[1.0] + [0.0] * 383),
                Chunk(id="p3#1", paper_id="P3", text_content="This is chunk 1 of paper P3.", page_number=1, embedding=[0.8, 0.6] + [0.0] * 382),
            ],
            "P4": [
                Chunk(id="p4#0", paper_id="P4", text_content="This is chunk 0 of paper P4.", page_number=1, embedding=[0.9, 0.1] + [0.0] * 382),
                Chunk(id="p4#1", paper_id="P4", text_content="This is chunk 1 of paper P4.", page_number=1, embedding=[0.7, 0.7] + [0.0] * 382),
            ],
            "P5": [
                Chunk(id="p5#0", paper_id="P5", text_content="This is chunk 0 of paper P5.", page_number=1, embedding=[1.0] + [0.0] * 383),
                Chunk(id="p5#1", paper_id="P5", text_content="This is chunk 1 of paper P5.", page_number=1, embedding=[0.8, 0.6] + [0.0] * 382),
            ],
            "P6": [
                Chunk(id="p6#0", paper_id="P6", text_content="This is chunk 0 of paper P6.", page_number=1, embedding=[1.0] + [0.0] * 383),
                Chunk(id="p6#1", paper_id="P6", text_content="This is chunk 1 of paper P6.", page_number=1, embedding=[0.8, 0.6] + [0.0] * 382),
            ]
        }

    def get_papers_mentioning_concepts(self, concept_ids: List[str]) -> List[Tuple[str, str]]:
        res = []
        for src, tgt, etype in self.edges:
            if etype == "MENTIONS_CONCEPT" and tgt in concept_ids:
                paper = self.papers.get(src)
                if paper:
                    res.append((src, paper["title"]))
        return sorted(list(set(res)))

    def get_concepts_for_papers(self, paper_ids: List[str]) -> List[Tuple[str, str, str]]:
        res = []
        for src, tgt, etype in self.edges:
            if etype == "MENTIONS_CONCEPT" and src in paper_ids:
                concept = self.concepts.get(tgt)
                if concept:
                    res.append((src, tgt, concept["properties"].get("name", "")))
        return sorted(list(set(res)))

    def get_concept_document_frequencies(self, concept_ids: List[str]) -> Dict[str, int]:
        freqs = {cid: 0 for cid in concept_ids}
        for src, tgt, etype in self.edges:
            if etype == "MENTIONS_CONCEPT" and tgt in concept_ids:
                freqs[tgt] += 1
        return freqs

    def get_total_paper_count(self) -> int:
        return len(self.papers)

    def get_citation_neighbors(self, paper_ids: List[str]) -> List[Tuple[str, str, str, str]]:
        res = []
        for src, tgt, etype in self.edges:
            if etype == "CITES":
                if src in paper_ids:
                    cand = self.papers.get(tgt)
                    title = cand["title"] if cand else ""
                    res.append((src, tgt, "seed_cites_candidate", title))
                if tgt in paper_ids:
                    cand = self.papers.get(src)
                    title = cand["title"] if cand else ""
                    res.append((tgt, src, "candidate_cites_seed", title))
        return res

    def search_chunks_within_papers(self, query_embedding: List[float], paper_ids: List[str], limit_per_paper: int = 1) -> List[Tuple[Chunk, float]]:
        res = []
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        for pid in paper_ids:
            if pid not in self.chunks:
                continue
            scored = []
            for chunk in self.chunks[pid]:
                chunk_copy = Chunk(
                    id=chunk.id,
                    paper_id=chunk.paper_id,
                    text_content=chunk.text_content,
                    page_number=chunk.page_number,
                    embedding=chunk.embedding,
                    parent_id=chunk.parent_id,
                    parent_text=chunk.parent_text
                )
                c_vec = np.array(chunk.embedding, dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                if q_norm > 0 and c_norm > 0:
                    sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                else:
                    sim = 0.0
                scored.append((chunk_copy, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            res.extend(scored[:limit_per_paper])
        return res

    def get_concept_aliases(self) -> Dict[str, str]:
        aliases = {}
        for cid, node in self.concepts.items():
            for alias in node["properties"].get("aliases", []):
                aliases[alias] = cid
        return aliases

    def get_nodes_by_label(self, label: str) -> List[Tuple[str, Dict[str, Any]]]:
        if label == "Concept":
            return [(cid, node["properties"]) for cid, node in self.concepts.items()]
        return []

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        p = self.papers.get(paper_id)
        if p:
            return Paper(id=p["id"], title=p["title"], authors=[], year=p["properties"].get("year", 2000), doi="", abstract="")
        return None

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        c = self.concepts.get(concept_id)
        if c:
            return Concept(id=c["id"], name=c["properties"].get("name", ""))
        return None

    def get_author(self, author_id: str) -> Optional[Any]:
        return None

    def get_papers_batch(self, paper_ids: List[str]) -> Dict[str, Paper]:
        res = {}
        for pid in paper_ids:
            p = self.get_paper(pid)
            if p:
                res[pid] = p
        return res

    def get_neighbors(self, paper_id: str, max_depth: int = 1) -> List[Tuple[str, str, str, str, str, Dict[str, Any]]]:
        res = []
        for src, tgt, etype in self.edges:
            if src == paper_id:
                src_label = "Paper" if src.startswith("P") else "Concept"
                tgt_label = "Concept" if tgt.startswith("c_") or tgt.startswith("C_") else "Paper"
                res.append((src, src_label, etype, tgt, tgt_label, {}))
            if tgt == paper_id:
                src_label = "Paper" if src.startswith("P") else "Concept"
                tgt_label = "Concept" if tgt.startswith("c_") or tgt.startswith("C_") else "Paper"
                res.append((src, src_label, etype, tgt, tgt_label, {}))
        return res


class FakeReranker:
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores = []
        for query, text in pairs:
            if "P4" in text or "Four" in text or "p4" in text:
                scores.append(0.9)
            elif "P3" in text or "Three" in text or "p3" in text:
                scores.append(0.8)
            elif "P1" in text or "One" in text or "p1" in text:
                scores.append(0.7)
            else:
                scores.append(0.5)
        return scores

    def rerank_chunks(self, query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
        res = []
        for chunk in chunks:
            if chunk.paper_id == "P4":
                score = 0.9
            elif chunk.paper_id == "P3":
                score = 0.8
            elif chunk.paper_id == "P1":
                score = 0.7
            else:
                score = 0.5
            res.append((chunk, score))
        return sorted(res, key=lambda x: x[1], reverse=True)
