import time
import logging
import threading
import numpy as np
import difflib
from typing import Optional, Dict, List, Tuple
from src.models import slugify
from src.repository.base import GraphRepository
from src.vector_search import EmbeddingEngine

class EntityResolver:
    def __init__(self, graph_repo: GraphRepository, emb_engine: EmbeddingEngine):
        self.graph_repo = graph_repo
        self.emb_engine = emb_engine
        self._aliases_cache: Optional[Dict[str, str]] = None
        self._entity_cache: Dict[str, List[Tuple[str, Dict]]] = {}
        self._lock = threading.Lock()

    def invalidate_concept_cache(self) -> None:
        with self._lock:
            self._aliases_cache = None
            if "Concept" in self._entity_cache:
                del self._entity_cache["Concept"]

    def resolve_entity(self, label: str, name: str) -> str:
        if not name:
            return ""
        
        name_clean = name.strip()
        slug = slugify(name_clean)
        t0 = time.perf_counter()

        if label == "Concept":
            if self._aliases_cache is None:
                with self._lock:
                    if self._aliases_cache is None:
                        try:
                            self._aliases_cache = self.graph_repo.get_concept_aliases()
                        except Exception:
                            self._aliases_cache = {}
            aliases_map = self._aliases_cache
            if name_clean.lower() in aliases_map:
                canonical = aliases_map[name_clean.lower()]
                logging.debug(f"resolve_entity '{name}' resolved from aliases_map in {time.perf_counter() - t0:.6f}s")
                return slugify(canonical)

        with self._lock:
            if label not in self._entity_cache:
                try:
                    self._entity_cache[label] = self.graph_repo.get_nodes_by_label(label)
                except Exception:
                    self._entity_cache[label] = []
            existing_nodes = self._entity_cache[label]

        for eid, props in existing_nodes:
            if eid == slug:
                return eid
            node_name = props.get("name", "")
            if node_name and slugify(node_name) == slug:
                return eid
                
        valid_candidates = []
        for eid, props in existing_nodes:
            node_emb = props.get("embedding")
            if node_emb and (isinstance(node_emb, list) or isinstance(node_emb, np.ndarray)):
                valid_candidates.append((eid, node_emb))

        if valid_candidates:
            try:
                candidate_emb = self.emb_engine.get_embedding(name_clean)
            except Exception:
                candidate_emb = None

            if candidate_emb is not None and (isinstance(candidate_emb, list) or isinstance(candidate_emb, np.ndarray)):
                try:
                    query_vec = np.array(candidate_emb, dtype=np.float32)
                    query_norm = np.linalg.norm(query_vec)
                    if query_norm > 0:
                        node_embs = np.array([emb for _, emb in valid_candidates], dtype=np.float32)
                        node_norms = np.linalg.norm(node_embs, axis=1)
                        dots = np.dot(node_embs, query_vec)
                        norms_product = query_norm * node_norms
                        
                        sims = np.zeros_like(dots)
                        valid_mask = norms_product > 0
                        sims[valid_mask] = dots[valid_mask] / norms_product[valid_mask]
                        
                        matching_indices = np.where(sims > 0.95)[0]
                        if len(matching_indices) > 0:
                            return valid_candidates[matching_indices[0]][0]
                except ValueError as ve:
                    logging.warning(f"Embedding dimension mismatch or calculation error in EntityResolver: {ve}")

        for eid, props in existing_nodes:
            node_name = props.get("name", "")
            if node_name:
                ratio = difflib.SequenceMatcher(None, name_clean.lower(), node_name.lower()).ratio()
                if ratio > 0.95:
                    return eid
                    
        return slug

    def add_resolved_entity_to_cache(self, label: str, entity_id: str, name: str, embedding: Optional[List[float]] = None) -> None:
        with self._lock:
            if label not in self._entity_cache:
                try:
                    self._entity_cache[label] = self.graph_repo.get_nodes_by_label(label)
                except Exception:
                    self._entity_cache[label] = []
            
            exists = any(eid == entity_id for eid, _ in self._entity_cache[label])
            if not exists:
                new_list = list(self._entity_cache[label])
                new_list.append((entity_id, {"name": name, "embedding": embedding}))
                self._entity_cache[label] = new_list
                if label == "Concept":
                    self._aliases_cache = None

