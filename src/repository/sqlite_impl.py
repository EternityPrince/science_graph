import sqlite3
import json
import os
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from src.repository.base import GraphRepository, VectorRepository
from src.models import Paper, Author, Concept, Chunk, Edge

def stable_hash(text: str) -> int:
    """Returns a stable 60-bit integer hash of a string ID."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:15], 16)

class SQLiteGraphRepository(GraphRepository):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            # Create nodes table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                properties TEXT NOT NULL
            );
            """)
            
            # Create edges table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, type),
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);")
            conn.commit()

    def save_paper(self, paper: Paper) -> None:
        props = {**paper.properties, "title": paper.title, "authors": paper.authors, "year": paper.year, "doi": paper.doi, "abstract": paper.abstract, "file_path": paper.file_path}
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                (paper.id, "Paper", json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label = 'Paper'", (paper_id,)).fetchone()
            if not row:
                return None
            props = json.loads(row["properties"])
            return Paper(
                id=row["id"],
                title=props.get("title", ""),
                authors=props.get("authors", []),
                year=props.get("year"),
                doi=props.get("doi"),
                abstract=props.get("abstract"),
                file_path=props.get("file_path"),
                properties=props
            )

    def save_author(self, author: Author) -> None:
        props = {**author.properties, "name": author.name}
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                (author.id, "Author", json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def get_author(self, author_id: str) -> Optional[Author]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label = 'Author'", (author_id,)).fetchone()
            if not row:
                return None
            props = json.loads(row["properties"])
            return Author(
                id=row["id"],
                name=props.get("name", row["id"]),
                properties=props
            )

    def save_concept(self, concept: Concept) -> None:
        props = {**concept.properties, "name": concept.name}
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                (concept.id, "Concept", json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT id, properties FROM nodes WHERE id = ? AND label = 'Concept'", (concept_id,)).fetchone()
            if not row:
                return None
            props = json.loads(row["properties"])
            return Concept(
                id=row["id"],
                name=props.get("name", row["id"]),
                properties=props
            )

    def add_edge(self, source_id: str, target_id: str, edge_type: str, properties: Dict[str, Any] = None) -> None:
        props = properties or {}
        with self._get_connection() as conn:
            # First, check if source and target nodes exist, otherwise insert placeholder nodes
            for node_id in (source_id, target_id):
                exists = conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone()
                if not exists:
                    # Insert placeholder node
                    label = "Paper" if ":" in node_id or "/" in node_id else "Concept"
                    conn.execute(
                        "INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                        (node_id, label, json.dumps({"title": node_id, "placeholder": True}, ensure_ascii=False))
                    )
            
            conn.execute(
                "INSERT OR REPLACE INTO edges (source_id, target_id, type, properties) VALUES (?, ?, ?, ?)",
                (source_id, target_id, edge_type, json.dumps(props, ensure_ascii=False))
            )
            conn.commit()

    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List[tuple[str, str, str, str, str, str]]:
        if max_depth < 1:
            return []
        
        # Simple BFS / traversal using raw SQLite
        # Let's perform a query for 1-hop first
        query = """
        SELECT 
            e.source_id as src_id, n1.label as src_label,
            e.type as edge_type,
            e.target_id as tgt_id, n2.label as tgt_label,
            e.properties as edge_props
        FROM edges e
        JOIN nodes n1 ON e.source_id = n1.id
        JOIN nodes n2 ON e.target_id = n2.id
        WHERE e.source_id = ? OR e.target_id = ?
        """
        
        visited_edges = set()
        results = []
        
        with self._get_connection() as conn:
            # We fetch starting node neighbors
            rows = conn.execute(query, (node_id, node_id)).fetchall()
            for r in rows:
                edge_key = (r["src_id"], r["tgt_id"], r["edge_type"])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    results.append((
                        r["src_id"], r["src_label"],
                        r["edge_type"],
                        r["tgt_id"], r["tgt_label"],
                        r["edge_props"]
                    ))
                    
            # If max_depth > 1, we traverse further
            current_nodes = {node_id}
            for _ in range(1, max_depth):
                next_nodes = set()
                for edge in list(results):
                    next_nodes.add(edge[0]) # src
                    next_nodes.add(edge[3]) # tgt
                
                # Exclude starting nodes to avoid re-querying
                query_nodes = next_nodes - current_nodes
                if not query_nodes:
                    break
                
                current_nodes.update(query_nodes)
                
                for q_node in query_nodes:
                    rows = conn.execute(query, (q_node, q_node)).fetchall()
                    for r in rows:
                        edge_key = (r["src_id"], r["tgt_id"], r["edge_type"])
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            results.append((
                                r["src_id"], r["src_label"],
                                r["edge_type"],
                                r["tgt_id"], r["tgt_label"],
                                r["edge_props"]
                            ))
                            
        return results

    def get_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            paper_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Paper'").fetchone()[0]
            author_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Author'").fetchone()[0]
            concept_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE label = 'Concept'").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            return {
                "papers": paper_count,
                "authors": author_count,
                "concepts": concept_count,
                "edges": edge_count
            }


class SQLiteVectorRepository(VectorRepository):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        self._usearch_index = None

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_index(self, ndim: int = 384):
        if self._usearch_index is not None:
            return self._usearch_index
        
        from usearch.index import Index
        self._usearch_index = Index(ndim=ndim, metric="cos")
        self.index_path = None if self.db_path == ":memory:" else self.db_path.replace(".db", ".usearch")
        
        if self.index_path and os.path.exists(self.index_path):
            try:
                self._usearch_index.load(self.index_path)
            except Exception:
                pass
                
        # Self-healing: if SQLite has chunks, but USearch is empty, rebuild the index
        if len(self._usearch_index) == 0:
            try:
                with self._get_connection() as conn:
                    rows = conn.execute("SELECT id, embedding FROM chunks").fetchall()
                if rows:
                    for r in rows:
                        key = stable_hash(r["id"])
                        if key not in self._usearch_index:
                            emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
                            self._usearch_index.add(key, emb_array)
                    if self.index_path:
                        self._usearch_index.save(self.index_path)
            except Exception:
                pass
                
        return self._usearch_index

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                text_content TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            """)
            # Migration check: add id_hash column if not present
            cursor = conn.execute("PRAGMA table_info(chunks);")
            columns = [row["name"] for row in cursor.fetchall()]
            if "id_hash" not in columns:
                try:
                    conn.execute("ALTER TABLE chunks ADD COLUMN id_hash INTEGER;")
                    rows = conn.execute("SELECT id FROM chunks").fetchall()
                    for r in rows:
                        h = stable_hash(r["id"])
                        conn.execute("UPDATE chunks SET id_hash = ? WHERE id = ?", (h, r["id"]))
                    conn.commit()
                except Exception:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(id_hash);")
            conn.commit()

    def save_chunks(self, chunks: List[Chunk]) -> None:
        with self._get_connection() as conn:
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                # Ensure parent paper exists
                exists = conn.execute("SELECT 1 FROM nodes WHERE id = ?", (chunk.paper_id,)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO nodes (id, label, properties) VALUES (?, ?, ?)",
                        (chunk.paper_id, "Paper", json.dumps({"title": chunk.paper_id, "placeholder": True}, ensure_ascii=False))
                    )
                emb_array = np.array(chunk.embedding, dtype=np.float32)
                emb_blob = emb_array.tobytes()
                conn.execute(
                    "INSERT OR REPLACE INTO chunks (id, paper_id, text_content, page_number, embedding, id_hash) VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk.id, chunk.paper_id, chunk.text_content, chunk.page_number, emb_blob, stable_hash(chunk.id))
                )
            conn.commit()

        # Update USearch index
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if valid_chunks:
            ndim = len(valid_chunks[0].embedding)
            index = self._get_index(ndim)
            for chunk in valid_chunks:
                key = stable_hash(chunk.id)
                if key not in index:
                    index.add(key, np.array(chunk.embedding, dtype=np.float32))
            
            self.index_path = None if self.db_path == ":memory:" else self.db_path.replace(".db", ".usearch")
            if self.index_path:
                index.save(self.index_path)

    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, paper_id, text_content, page_number, embedding FROM chunks WHERE paper_id = ?",
                (paper_id,)
            ).fetchall()
            
            chunks = []
            for r in rows:
                emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
                chunks.append(Chunk(
                    id=r["id"],
                    paper_id=r["paper_id"],
                    text_content=r["text_content"],
                    page_number=r["page_number"],
                    embedding=emb_array.tolist()
                ))
            return chunks

    def get_all_chunks(self) -> List[Chunk]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, paper_id, text_content, page_number, embedding FROM chunks"
            ).fetchall()
            
            chunks = []
            for r in rows:
                emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
                chunks.append(Chunk(
                    id=r["id"],
                    paper_id=r["paper_id"],
                    text_content=r["text_content"],
                    page_number=r["page_number"],
                    embedding=emb_array.tolist()
                ))
            return chunks

    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[tuple[Chunk, float]]:
        ndim = len(query_embedding)
        index = self._get_index(ndim)
        
        if len(index) == 0:
            return []
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        matches = index.search(q_vec, limit)
        
        if len(matches) == 0:
            return []
            
        keys_list = [int(k) for k in matches.keys]
        placeholders = ",".join("?" for _ in keys_list)
        
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT id, paper_id, text_content, page_number, embedding, id_hash FROM chunks WHERE id_hash IN ({placeholders})",
                keys_list
            ).fetchall()
            
        key_to_dist = {int(k): float(d) for k, d in zip(matches.keys, matches.distances)}
        
        results = []
        for r in rows:
            emb_array = np.frombuffer(r["embedding"], dtype=np.float32)
            chunk = Chunk(
                id=r["id"],
                paper_id=r["paper_id"],
                text_content=r["text_content"],
                page_number=r["page_number"],
                embedding=emb_array.tolist()
            )
            dist = key_to_dist.get(r["id_hash"], 1.0)
            similarity = 1.0 - dist
            results.append((chunk, similarity))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
