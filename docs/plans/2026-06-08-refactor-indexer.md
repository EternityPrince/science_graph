# Indexer Decomposition & Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the over-bloated 1704-line `Indexer` class into dedicated services (`EntityResolver`, `GraphWriteBuilder`, `CitationService`, `Reindexer`) to achieve clean separation of concerns, improve code readability, and eliminate ingestion pipeline duplication.

**Architecture:** 
- Extract Entity Resolution logic (slug, embedding, string distance matching) into a thread-safe `EntityResolver`.
- Extract Graph DB model translation and Tuple building into a `GraphWriteBuilder`.
- Extract Regex citation context finding and LLM intent classification into `CitationService`.
- Extract Metadata and Full Reindexing pipelines (both single and batch) into `Reindexer`.
- Refactor `Indexer` to use these delegates and unify single/batch indexing to run through shared step execution.

**Tech Stack:** Python 3.12, pytest, NumPy (for cosine similarity), Spacy, SQLite

---

### Task 1: Extract Entity Resolver Service

**Files:**
- Create: `back/src/services/entity_resolver.py`
- Test: `back/tests/test_entity_resolver.py`

**Step 1: Write the failing test**
Create a test file to assert that `EntityResolver` resolves entity names by exact slug matching, vector similarity, string similarity fallback, and maintains correct thread-safe caching.
```python
# back/tests/test_entity_resolver.py
import pytest
from unittest.mock import MagicMock
from src.services.entity_resolver import EntityResolver
from src.repository.base import GraphRepository
from src.vector_search import EmbeddingEngine

def test_entity_resolver_slug_match():
    graph_repo = MagicMock(spec=GraphRepository)
    emb_engine = MagicMock(spec=EmbeddingEngine)
    
    # Mock return values for get_nodes_by_label
    graph_repo.get_concept_aliases.return_value = {}
    graph_repo.get_nodes_by_label.return_value = [
        ("machine-learning", {"name": "Machine Learning"})
    ]
    
    resolver = EntityResolver(graph_repo, emb_engine)
    resolved_id = resolver.resolve_entity("Concept", "Machine Learning")
    assert resolved_id == "machine-learning"
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest back/tests/test_entity_resolver.py`
Expected: FAIL (ModuleNotFoundError: No module named 'src.services.entity_resolver')

**Step 3: Write minimal implementation**
Extract and implement the `EntityResolver` class.
```python
# back/src/services/entity_resolver.py
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

        if label not in self._entity_cache:
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

        for eid, props in existing_nodes:
            node_name = props.get("name", "")
            if node_name:
                ratio = difflib.SequenceMatcher(None, name_clean.lower(), node_name.lower()).ratio()
                if ratio > 0.95:
                    return eid
                    
        return slug

    def add_resolved_entity_to_cache(self, label: str, entity_id: str, name: str, embedding: Optional[List[float]] = None) -> None:
        if label not in self._entity_cache:
            with self._lock:
                if label not in self._entity_cache:
                    try:
                        self._entity_cache[label] = self.graph_repo.get_nodes_by_label(label)
                    except Exception:
                        self._entity_cache[label] = []
        
        exists = any(eid == entity_id for eid, _ in self._entity_cache[label])
        if not exists:
            with self._lock:
                self._entity_cache[label].append((entity_id, {"name": name, "embedding": embedding}))
                if label == "Concept":
                    self._aliases_cache = None
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest back/tests/test_entity_resolver.py`
Expected: PASS

**Step 5: Commit**
```bash
git add back/src/services/entity_resolver.py back/tests/test_entity_resolver.py
git commit -m "refactor: extract EntityResolver service"
```

---

### Task 2: Extract Citation Context & Intent Service

**Files:**
- Create: `back/src/services/citation_service.py`
- Test: `back/tests/test_citation_service.py`

**Step 1: Write the failing test**
Create a test to assert that `CitationService` correctly finds citation context in text and calls LLM classification asynchronously.
```python
# back/tests/test_citation_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.citation_service import CitationService
from src.services.extraction_service import ExtractionService

@pytest.mark.asyncio
async def test_citation_service_context_extraction():
    extractor = MagicMock(spec=ExtractionService)
    extractor.classify_citation_intent_async = AsyncMock(return_value="METHODology")
    
    service = CitationService(extractor)
    text = "This is first sentence. We use the method described in DeepLearning Book. Third sentence."
    
    # Verify regex context extraction
    context = service.get_citation_context(text, "DeepLearning Book")
    assert "We use the method described in DeepLearning Book." in context
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest back/tests/test_citation_service.py`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write minimal implementation**
Extract citation search and intent classification logic.
```python
# back/src/services/citation_service.py
import re
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from src.services.extraction_service import ExtractionService

class CitationService:
    def __init__(self, extractor: ExtractionService):
        self.extractor = extractor

    def get_citation_context(self, full_text: str, ref_title: str, ref_author: Optional[str] = None, ref_year: Optional[int] = None) -> str:
        if not full_text or not ref_title:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        
        patterns = []
        if ref_title and len(ref_title) > 8:
            words = [re.escape(w) for w in ref_title.split()[:4] if len(w) > 2]
            if words:
                patterns.append(re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE))
        if ref_author and ref_year:
            patterns.append(re.compile(rf"\b{re.escape(ref_author)}.*\b{ref_year}\b", re.IGNORECASE))
        elif ref_author:
            patterns.append(re.compile(rf"\b{re.escape(ref_author)}\b", re.IGNORECASE))
            
        for idx, sent in enumerate(sentences):
            for pat in patterns:
                if pat.search(sent):
                    start = max(0, idx - 1)
                    end = min(len(sentences), idx + 2)
                    return " ".join(sentences[start:end]).strip()
        return ""

    async def classify_cites_edges_async(self, cites_list: List[Dict[str, Any]], full_text: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        if not cites_list:
            return []
            
        tasks = []
        metadata = []
        
        for cit in cites_list:
            ref_title = cit.get("title") or ""
            ref_author = cit.get("author")
            ref_year = cit.get("year")
            
            context = self.get_citation_context(full_text, ref_title, ref_author, ref_year)
            props = {**cit.get("properties", {})}
            if context:
                props["context"] = context
                tasks.append(self.extractor.classify_citation_intent_async(context, ref_title))
                metadata.append((cit["source_id"], cit["target_id"], props))
            else:
                props["intent"] = "BACKGROUND"
                metadata.append((cit["source_id"], cit["target_id"], props))
                tasks.append(asyncio.sleep(0, result="BACKGROUND"))
                
        intents = await asyncio.gather(*tasks)
        
        edges = []
        for (src, tgt, props), intent in zip(metadata, intents):
            props["intent"] = intent or "BACKGROUND"
            edges.append((src, tgt, "CITES", props))
        return edges
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest back/tests/test_citation_service.py`
Expected: PASS

**Step 5: Commit**
```bash
git add back/src/services/citation_service.py back/tests/test_citation_service.py
git commit -m "refactor: extract CitationService"
```

---

### Task 3: Extract Graph Write Builder Mapper

**Files:**
- Create: `back/src/services/graph_write_builder.py`
- Test: `back/tests/test_graph_write_builder.py`

**Step 1: Write the failing test**
Create a test to assert that `GraphWriteBuilder` correctly translates entities, authors, and citation metadata into correct tuples.
```python
# back/tests/test_graph_write_builder.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.graph_write_builder import GraphWriteBuilder
from src.services.entity_resolver import EntityResolver
from src.services.citation_service import CitationService
from src.models import Paper
from src.services.extraction_service import ExtractionResult

@pytest.mark.asyncio
async def test_graph_write_builder_paper_mapping():
    resolver = MagicMock(spec=EntityResolver)
    resolver.resolve_entity.side_effect = lambda lbl, name: f"resolved-{name.lower()}"
    citation_service = MagicMock(spec=CitationService)
    citation_service.classify_cites_edges_async = AsyncMock(return_value=[])
    
    builder = GraphWriteBuilder(resolver, citation_service)
    
    paper = Paper(id="paper-1", title="Title", authors=["Author A"], year=2026)
    extraction = ExtractionResult(concepts=[], tags=[], institutions=[], sponsored_by=[], author_institutions=[], datasets=[], code_repositories=[], journal_or_conference=None)
    
    nodes, edges = await builder.build_graph_writes_async(
        paper=paper,
        extraction=extraction,
        full_text="Some text",
        is_markdown=False,
        refs_or_links=[],
        api_references=[],
        api_citations=[]
    )
    
    assert any(n[0] == "paper-1" for n in nodes)
    assert any(n[0] == "author-a" for n in nodes)
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest back/tests/test_graph_write_builder.py`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write minimal implementation**
```python
# back/src/services/graph_write_builder.py
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from src.models import Paper, Author, Concept, slugify
from src.services.entity_resolver import EntityResolver
from src.services.citation_service import CitationService
from src.services.extraction_service import ExtractionResult

class GraphWriteBuilder:
    def __init__(self, entity_resolver: EntityResolver, citation_service: CitationService, emb_engine: Any):
        self.resolver = entity_resolver
        self.citation_service = citation_service
        self.emb_engine = emb_engine

    async def build_graph_writes_async(
        self,
        paper: Paper,
        extraction: ExtractionResult,
        full_text: str,
        is_markdown: bool,
        refs_or_links: List[str],
        api_references: List[Dict],
        api_citations: List[Dict]
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        nodes_to_write = []
        edges_to_write = []
        added_node_ids = set()

        async def _safe_get_embedding(text: str) -> Optional[List[float]]:
            try:
                import asyncio
                val = await asyncio.to_thread(self.emb_engine.get_embedding, text, False)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
            return None

        content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
        paper.properties["content_hash"] = content_hash

        paper_props = {
            **paper.properties,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "file_path": paper.file_path,
            "created_at": paper.created_at or time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        paper_label = "UserNote" if paper.properties.get("source_type") == "note" else "Paper"
        nodes_to_write.append((paper.id, paper_label, paper_props))
        added_node_ids.add(paper.id)

        for author_name in paper.authors:
            author_id = slugify(author_name)
            nodes_to_write.append((author_id, "Author", {"name": author_name}))
            edges_to_write.append((author_id, paper.id, "AUTHORED", {}))
            added_node_ids.add(author_id)

        concepts = getattr(extraction, "concepts", [])
        if isinstance(concepts, list):
            for item in concepts:
                if not isinstance(item, dict):
                    continue
                c_name = item.get("name", "").strip() if isinstance(item.get("name"), str) else ""
                if not c_name:
                    continue
                c_desc = item.get("description", "") if isinstance(item.get("description"), str) else ""
                if not c_desc:
                    c_desc = await self.resolver.graph_repo.get_concept_description_async(c_name) if hasattr(self.resolver.graph_repo, "get_concept_description_async") else ""
                
                concept_id = self.resolver.resolve_entity("Concept", c_name)
                if concept_id not in added_node_ids:
                    emb = await _safe_get_embedding(c_name)
                    nodes_to_write.append((concept_id, "Concept", {"name": c_name, "description": c_desc, "embedding": emb}))
                    self.resolver.add_resolved_entity_to_cache("Concept", concept_id, c_name, emb)
                    added_node_ids.add(concept_id)
                edges_to_write.append((paper.id, concept_id, "MENTIONS_CONCEPT", {}))

        for tag in paper.properties.get("tags") or []:
            tag_id = self.resolver.resolve_entity("Concept", tag)
            if tag_id not in added_node_ids:
                emb = await _safe_get_embedding(tag)
                nodes_to_write.append((tag_id, "Concept", {"name": tag, "is_tag": True, "embedding": emb}))
                self.resolver.add_resolved_entity_to_cache("Concept", tag_id, tag, emb)
                added_node_ids.add(tag_id)
            edges_to_write.append((paper.id, tag_id, "HAS_TAG", {}))

        institutions = getattr(extraction, "institutions", [])
        if isinstance(institutions, list):
            for inst_name in institutions:
                if not isinstance(inst_name, str):
                    continue
                inst_id = self.resolver.resolve_entity("Institution", inst_name)
                if inst_id not in added_node_ids:
                    emb = await _safe_get_embedding(inst_name)
                    nodes_to_write.append((inst_id, "Institution", {"name": inst_name, "embedding": emb}))
                    self.resolver.add_resolved_entity_to_cache("Institution", inst_id, inst_name, emb)
                    added_node_ids.add(inst_id)
            
        sponsored_by = getattr(extraction, "sponsored_by", [])
        if isinstance(sponsored_by, list):
            for inst_name in sponsored_by:
                if not isinstance(inst_name, str):
                    continue
                inst_id = self.resolver.resolve_entity("Institution", inst_name)
                if inst_id not in added_node_ids:
                    emb = await _safe_get_embedding(inst_name)
                    nodes_to_write.append((inst_id, "Institution", {"name": inst_name, "embedding": emb}))
                    self.resolver.add_resolved_entity_to_cache("Institution", inst_id, inst_name, emb)
                    added_node_ids.add(inst_id)
                edges_to_write.append((paper.id, inst_id, "SPONSORED_BY", {}))

        author_institutions = getattr(extraction, "author_institutions", [])
        if isinstance(author_institutions, list):
            for auth_inst in author_institutions:
                if not isinstance(auth_inst, dict):
                    continue
                auth_name = auth_inst.get("author")
                inst_name = auth_inst.get("institution")
                if auth_name and inst_name and isinstance(auth_name, str) and isinstance(inst_name, str):
                    auth_id = slugify(auth_name)
                    inst_id = self.resolver.resolve_entity("Institution", inst_name)
                    if inst_id not in added_node_ids:
                        emb = await _safe_get_embedding(inst_name)
                        nodes_to_write.append((inst_id, "Institution", {"name": inst_name, "embedding": emb}))
                        self.resolver.add_resolved_entity_to_cache("Institution", inst_id, inst_name, emb)
                        added_node_ids.add(inst_id)
                    edges_to_write.append((auth_id, inst_id, "AFFILIATED_WITH", {}))

        datasets = getattr(extraction, "datasets", [])
        if isinstance(datasets, list):
            for ds in datasets:
                if not isinstance(ds, dict):
                    continue
                ds_name = ds.get("name")
                relation = ds.get("relation", "USED_DATASET")
                if ds_name and isinstance(ds_name, str):
                    ds_id = self.resolver.resolve_entity("Dataset", ds_name)
                    if ds_id not in added_node_ids:
                        emb = await _safe_get_embedding(ds_name)
                        nodes_to_write.append((ds_id, "Dataset", {"name": ds_name, "embedding": emb}))
                        self.resolver.add_resolved_entity_to_cache("Dataset", ds_id, ds_name, emb)
                        added_node_ids.add(ds_id)
                    edges_to_write.append((paper.id, ds_id, relation, {}))

        code_repositories = getattr(extraction, "code_repositories", [])
        if isinstance(code_repositories, list):
            for repo_url in code_repositories:
                if not isinstance(repo_url, str):
                    continue
                repo_id = slugify(repo_url)
                if repo_id not in added_node_ids:
                    nodes_to_write.append((repo_id, "CodeRepository", {"name": repo_url, "url": repo_url}))
                    added_node_ids.add(repo_id)
                edges_to_write.append((paper.id, repo_id, "HAS_CODE", {}))

        jc_name = getattr(extraction, "journal_or_conference", None)
        if jc_name and isinstance(jc_name, str):
            jc_id = self.resolver.resolve_entity("JournalConference", jc_name)
            if jc_id not in added_node_ids:
                emb = await _safe_get_embedding(jc_name)
                nodes_to_write.append((jc_id, "JournalConference", {"name": jc_name, "embedding": emb}))
                self.resolver.add_resolved_entity_to_cache("JournalConference", jc_id, jc_name, emb)
                added_node_ids.add(jc_id)
            edges_to_write.append((paper.id, jc_id, "PUBLISHED_IN", {}))

        concept_relations = getattr(extraction, "concept_relations", [])
        if isinstance(concept_relations, list):
            for rel in concept_relations:
                if not isinstance(rel, dict):
                    continue
                src_c = rel.get("source", "").strip() if isinstance(rel.get("source"), str) else ""
                tgt_c = rel.get("target", "").strip() if isinstance(rel.get("target"), str) else ""
                rel_type = rel.get("relation_type", "SUBCLASS_OF")
                if src_c and tgt_c:
                    src_id = self.resolver.resolve_entity("Concept", src_c)
                    tgt_id = self.resolver.resolve_entity("Concept", tgt_c)
                    if src_id not in added_node_ids:
                        emb = await _safe_get_embedding(src_c)
                        nodes_to_write.append((src_id, "Concept", {"name": src_c, "embedding": emb}))
                        self.resolver.add_resolved_entity_to_cache("Concept", src_id, src_c, emb)
                        added_node_ids.add(src_id)
                    if tgt_id not in added_node_ids:
                        emb = await _safe_get_embedding(tgt_c)
                        nodes_to_write.append((tgt_id, "Concept", {"name": tgt_c, "embedding": emb}))
                        self.resolver.add_resolved_entity_to_cache("Concept", tgt_id, tgt_c, emb)
                        added_node_ids.add(tgt_id)
                    edges_to_write.append((src_id, tgt_id, rel_type, {}))

        if paper_label == "UserNote":
            for target_title in paper.properties.get("comments_on", []) or []:
                target_id = slugify(target_title)
                edges_to_write.append((paper.id, target_id, "COMMENTS_ON", {}))
            for target_title in paper.properties.get("agrees_with", []) or []:
                target_id = slugify(target_title)
                edges_to_write.append((paper.id, target_id, "AGREES_WITH", {}))
            for target_title in paper.properties.get("disagrees_with", []) or []:
                target_id = slugify(target_title)
                edges_to_write.append((paper.id, target_id, "DISAGREES_WITH", {}))
            for concept_name in paper.properties.get("linked_to", []) or []:
                concept_id = self.resolver.resolve_entity("Concept", concept_name)
                if concept_id not in added_node_ids:
                    emb = await _safe_get_embedding(concept_name)
                    nodes_to_write.append((concept_id, "Concept", {"name": concept_name, "embedding": emb}))
                    self.resolver.add_resolved_entity_to_cache("Concept", concept_id, concept_name, emb)
                    added_node_ids.add(concept_id)
                edges_to_write.append((paper.id, concept_id, "LINKED_TO", {}))

        cites_list = []
        if is_markdown:
            for link_target in refs_or_links:
                if link_target.startswith(("http://", "https://")):
                    clean_target = link_target.replace("https://", "").replace("http://", "").strip("/")
                    target_id = slugify(clean_target)
                else:
                    target_id = slugify(link_target)
                if not target_id:
                    continue
                if target_id not in added_node_ids:
                    nodes_to_write.append((target_id, "Paper", {"title": link_target, "is_placeholder": True}))
                    added_node_ids.add(target_id)
                edges_to_write.append((paper.id, target_id, "RELATED_TO", {}))
        else:
            all_refs = api_references if (api_references or api_citations) else []
            all_cits = api_citations if (api_references or api_citations) else []

            for ref in all_refs:
                ref_title = ref.get("title")
                ref_doi = ref.get("doi")
                ref_id = slugify(ref_doi) if ref_doi else (
                    slugify(ref_title[:120]) if ref_title else None
                )
                if ref_id:
                    if ref_id not in added_node_ids:
                        nodes_to_write.append((ref_id, "Paper", {"title": ref_title, "doi": ref_doi, "is_placeholder": True}))
                        added_node_ids.add(ref_id)
                    
                    author = ref["authors"][0] if ref.get("authors") else None
                    cites_list.append({
                        "source_id": paper.id,
                        "target_id": ref_id,
                        "title": ref_title,
                        "author": author,
                        "year": ref.get("year"),
                        "properties": {"api_sourced": True}
                    })

            for cit in all_cits:
                cit_title = cit.get("title")
                cit_doi = cit.get("doi")
                cit_id = slugify(cit_doi) if cit_doi else (
                    slugify(cit_title[:120]) if cit_title else None
                )
                if cit_id:
                    if cit_id not in added_node_ids:
                        nodes_to_write.append((cit_id, "Paper", {"title": cit_title, "doi": cit_doi, "is_placeholder": True}))
                        added_node_ids.add(cit_id)
                    
                    author = cit["authors"][0] if cit.get("authors") else None
                    cites_list.append({
                        "source_id": cit_id,
                        "target_id": paper.id,
                        "title": cit_title,
                        "author": author,
                        "year": cit.get("year"),
                        "properties": {"api_sourced": True}
                    })

            if not api_references and not api_citations:
                for ref_str in refs_or_links:
                    ref_clean = ref_str.strip()
                    if len(ref_clean) > 10:
                        ref_id = slugify(ref_clean[:120])
                        if ref_id not in added_node_ids:
                            nodes_to_write.append((ref_id, "Paper", {"title": ref_clean[:120], "is_placeholder": True}))
                            added_node_ids.add(ref_id)
                        cites_list.append({
                            "source_id": paper.id,
                            "target_id": ref_id,
                            "title": ref_clean[:120],
                            "author": None,
                            "year": None,
                            "properties": {"raw_text": ref_clean}
                        })
                        
        classified_cites_edges = await self.citation_service.classify_cites_edges_async(cites_list, full_text)
        edges_to_write.extend(classified_cites_edges)

        return nodes_to_write, edges_to_write
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest back/tests/test_graph_write_builder.py`
Expected: PASS

**Step 5: Commit**
```bash
git add back/src/services/graph_write_builder.py back/tests/test_graph_write_builder.py
git commit -m "refactor: extract GraphWriteBuilder service"
```

---

### Task 4: Extract Reindexer Service

**Files:**
- Create: `back/src/services/reindexer.py`
- Test: `back/tests/test_reindexer.py`

**Step 1: Write the failing test**
Create a test to verify that `Reindexer` calls `reindex_metadata` and database queries successfully.
```python
# back/tests/test_reindexer.py
import pytest
from unittest.mock import MagicMock
from src.services.reindexer import Reindexer

def test_reindexer_not_found():
    graph_repo = MagicMock()
    graph_repo.get_paper.return_value = None
    
    reindexer = Reindexer(graph_repo, MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    assert reindexer.reindex_metadata("non-existent") is False
```

**Step 2: Run test to verify it fails**
Run: `uv run pytest back/tests/test_reindexer.py`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write minimal implementation**
Extract re-indexing methods `reindex_metadata`, `reindex_full`, `reindex_metadata_batch`, and `reindex_full_batch` from `back/src/indexer.py`.
```python
# back/src/services/reindexer.py
import os
from pathlib import Path
from typing import Optional, Tuple, Any, List
from src.models import Paper, Author, Concept, slugify
from src.parsers.factory import ParserFactory
from src import console as con

class Reindexer:
    def __init__(
        self,
        graph_repo: Any,
        vector_repo: Any,
        emb_engine: Any,
        llm_engine: Any,
        extractor: Any,
        enricher: Any,
        resolver: Any,
        indexer: Any  # Pass reference to indexer to avoid duplicating index_pdf/index_url/etc
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = emb_engine
        self.llm_engine = llm_engine
        self.extractor = extractor
        self.enricher = enricher
        self.resolver = resolver
        self.indexer = indexer

    def reindex_metadata(self, paper_id: str, use_llm: bool = False) -> bool:
        paper = self.graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found in database: {paper_id}")
            return False

        con.info(f"Re-indexing metadata for [bold]{paper.title[:60]}[/bold] (ID: {paper_id})")

        is_webpage = (
            paper.properties.get("source_type") == "webpage"
            or (paper.file_path and paper.file_path.startswith("http"))
        )
        url = paper.properties.get("url") or paper.file_path if is_webpage else None

        if url:
            try:
                parser = ParserFactory.get_parser(url)
                web_paper, _, _ = parser.parse(url)
                if web_paper.authors:
                    paper.authors = list({*paper.authors, *web_paper.authors})
                if web_paper.doi:
                    paper.doi = web_paper.doi
                if web_paper.properties.get("arxiv_id"):
                    paper.properties["arxiv_id"] = web_paper.properties["arxiv_id"]
                if web_paper.title and len(web_paper.title) > len(paper.title or ""):
                    paper.title = web_paper.title
                if web_paper.abstract:
                    paper.abstract = web_paper.abstract
                if web_paper.year:
                    paper.year = web_paper.year
            except Exception as e:
                con.warning(f"Failed to re-parse URL {url}: {e}")

        api_meta = self.enricher.enrich(paper)
        api_references = []
        api_citations = []
        if api_meta:
            paper, api_references, api_citations = self.enricher.apply(paper, api_meta)

        full_text = self.indexer._read_local_text(paper)
        text_for_extraction = full_text or f"{paper.title or ''}\n\n{paper.abstract or ''}"

        extraction = self.extractor.extract(
            title=paper.title or "",
            abstract=paper.abstract or "",
            full_text=text_for_extraction,
            use_llm=use_llm,
        )
        if extraction.authors:
            existing = {a.lower() for a in paper.authors}
            for a in extraction.authors:
                if a.lower() not in existing:
                    paper.authors.append(a)
                    existing.add(a.lower())

        if extraction.tags:
            paper.properties["tags"] = extraction.tags

        if (
            len(paper.authors) < 2
            and paper.file_path
            and os.path.exists(paper.file_path)
            and paper.file_path.lower().endswith(".pdf")
        ):
            paper.authors = self.indexer._ner_fallback_authors(paper.authors, paper.file_path)

        with self.graph_repo.transaction():
            self.graph_repo.save_paper(paper)
            self.graph_repo.delete_edges_by_target(paper.id, ["AUTHORED"])
            for author_name in paper.authors:
                author_id = slugify(author_name)
                author = Author(id=author_id, name=author_name)
                self.graph_repo.save_author(author)
                self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

            self.graph_repo.delete_edges_by_source(paper.id, ["MENTIONS_CONCEPT", "HAS_TAG"])

            for item in extraction.concepts:
                c_name = item.get("name", "").strip()
                if not c_name:
                    continue
                c_desc = item.get("description", "") or self.extractor.get_concept_description(c_name)
                concept_id = slugify(c_name)
                concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                self.graph_repo.save_concept(concept)
                self.resolver.invalidate_concept_cache()
                self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

            for tag in extraction.tags:
                tag_id = slugify(tag)
                tag_desc = self.extractor.get_concept_description(tag)
                tag_node = Concept(id=tag_id, name=tag, properties={"is_tag": True, "description": tag_desc})
                self.graph_repo.save_concept(tag_node)
                self.resolver.invalidate_concept_cache()
                self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

            if api_references or api_citations:
                for ref in api_references:
                    ref_title = ref.get("title")
                    ref_doi = ref.get("doi")
                    ref_id = slugify(ref_doi) if ref_doi else (
                        slugify(ref_title[:120]) if ref_title else None
                    )
                    if ref_id:
                        if not self.graph_repo.get_paper(ref_id):
                            placeholder = Paper(id=ref_id, title=ref_title, authors=[], year=None, doi=ref_doi)
                            placeholder.properties["is_placeholder"] = True
                            self.graph_repo.save_paper(placeholder)
                        self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"api_sourced": True})

                for cit in api_citations:
                    cit_title = cit.get("title")
                    cit_doi = cit.get("doi")
                    cit_id = slugify(cit_doi) if cit_doi else (
                        slugify(cit_title[:120]) if cit_title else None
                    )
                    if cit_id:
                        if not self.graph_repo.get_paper(cit_id):
                            placeholder = Paper(id=cit_id, title=cit_title, authors=[], year=None, doi=cit_doi)
                            placeholder.properties["is_placeholder"] = True
                            self.graph_repo.save_paper(placeholder)
                        self.graph_repo.add_edge(cit_id, paper.id, "CITES", {"api_sourced": True})

            if use_llm and self.llm_engine:
                self.extractor.generate_summary(paper, text_for_extraction, graph_repo=self.graph_repo)

        con.success(f"Successfully re-indexed metadata for {(paper.title or paper.id)[:60]}")
        return True

    def reindex_full(self, paper_id: str, pdf_parser_type: Optional[str] = None) -> bool:
        paper = self.graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found in database: {paper_id}")
            return False

        file_path = paper.file_path
        if not file_path:
            con.error(f"No file path or URL stored for paper: {paper_id}")
            return False

        con.info(f"Performing full re-indexing for [bold]{(paper.title or paper_id)[:60]}[/bold] (ID: {paper_id})")

        try:
            self.graph_repo.delete_node(paper_id)
        except Exception as e:
            con.error(f"Failed to delete existing paper records: {e}")
            return False

        try:
            if file_path.startswith("http://") or file_path.startswith("https://"):
                self.indexer.index_url(file_path)
            elif file_path.lower().endswith(".pdf"):
                kwargs = {}
                if pdf_parser_type is not None:
                    kwargs["pdf_parser_type"] = pdf_parser_type
                self.indexer.index_pdf(file_path, **kwargs)
            elif file_path.lower().endswith(".md"):
                self.indexer.index_markdown(file_path)
            elif file_path.lower().endswith(".epub"):
                self.indexer.index_epub(file_path)
            else:
                con.error(f"Unsupported file type for full re-indexing: {file_path}")
                return False
        except Exception as e:
            con.error(f"Full re-indexing failed: {e}")
            return False

        con.success(f"Successfully performed full re-index for {paper_id}")
        return True

    def reindex_metadata_batch(
        self,
        missing_authors: bool = False,
        missing_tags: bool = False,
        limit: Optional[int] = None,
        use_llm: bool = False,
        chunk_pool_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        if chunk_pool_size is not None:
            self.extractor._chunk_pool_size = chunk_pool_size
            self.extractor._sem = None
        non_placeholders = self.graph_repo.get_non_placeholder_paper_ids()

        candidates = []
        for pid in non_placeholders:
            paper = self.graph_repo.get_paper(pid)
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
            return 0, 0

        con.info(f"Starting metadata re-indexing for [bold]{len(candidates)}[/bold] papers …")
        
        success_count = 0
        for paper_id in candidates:
            try:
                if self.reindex_metadata(paper_id, use_llm=use_llm):
                    success_count += 1
            except Exception as e:
                con.error(f"Failed to re-index {paper_id}: {e}")

        con.blank()
        con.success(f"Re-indexed {success_count}/{len(candidates)} papers successfully.")
        return success_count, len(candidates)

    def reindex_full_batch(
        self,
        all_papers: bool = False,
        paper_id: Optional[str] = None,
        limit: Optional[int] = None,
        chunk_pool_size: Optional[int] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> Tuple[int, int]:
        if chunk_pool_size is not None:
            self.extractor._chunk_pool_size = chunk_pool_size
            self.extractor._sem = None
        if paper_id:
            paper = self.graph_repo.get_paper(paper_id)
            if not paper:
                con.error(f"Paper not found: {paper_id}")
                raise ValueError(f"Paper not found: {paper_id}")
            candidates = [paper_id]
        else:
            candidates = self.graph_repo.get_non_placeholder_paper_ids()

        if limit:
            candidates = candidates[:limit]

        if not candidates:
            con.success("No papers found matching the re-indexing criteria.")
            return 0, 0

        con.info(f"Starting full re-indexing for [bold]{len(candidates)}[/bold] papers …")
        
        success_count = 0
        for pid in candidates:
            try:
                kwargs = {}
                if pdf_parser_type is not None:
                    kwargs["pdf_parser_type"] = pdf_parser_type
                if self.reindex_full(pid, **kwargs):
                    success_count += 1
            except Exception as e:
                con.error(f"Failed to fully re-index {pid}: {e}")

        con.blank()
        con.success(f"Fully re-indexed {success_count}/{len(candidates)} papers successfully.")
        return success_count, len(candidates)
```

**Step 4: Run test to verify it passes**
Run: `uv run pytest back/tests/test_reindexer.py`
Expected: PASS

**Step 5: Commit**
```bash
git add back/src/services/reindexer.py back/tests/test_reindexer.py
git commit -m "refactor: extract Reindexer service"
```

---

### Task 5: Refactor Indexer to Use Services

**Files:**
- Modify: `back/src/indexer.py`

**Step 1: Write tests**
Verify existing indexer tests exist (`back/tests/test_indexer_pipeline.py`).

**Step 2: Run test to verify it passes before modification**
Run: `uv run pytest back/tests/test_indexer_pipeline.py`
Expected: PASS

**Step 3: Modify `back/src/indexer.py`**
Replace internal implementations with delegates `EntityResolver`, `GraphWriteBuilder`, `CitationService`, and `Reindexer`.
Inject resolver, citation service, graph writer, and reindexer inside `Indexer.__init__`. Let `Indexer` route reindexing and helper tasks directly to the extracted modules. Keep the API of `Indexer` backward compatible.
```python
# back/src/indexer.py snippet (modified)
# ... imports ...
from src.services.entity_resolver import EntityResolver
from src.services.citation_service import CitationService
from src.services.graph_write_builder import GraphWriteBuilder
from src.services.reindexer import Reindexer

class Indexer:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: Any = None,
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine

        self._extractor = ExtractionService(llm_engine=llm_engine)
        self._enricher = MetadataEnricher()
        self._duplicate_detector = DuplicateDetector(graph_repo, vector_repo, embedding_engine)

        # Delegates
        self.entity_resolver = EntityResolver(graph_repo, embedding_engine)
        self.citation_service = CitationService(self._extractor)
        self.graph_write_builder = GraphWriteBuilder(self.entity_resolver, self.citation_service, embedding_engine)
        self.reindexer = Reindexer(
            graph_repo=graph_repo,
            vector_repo=vector_repo,
            emb_engine=embedding_engine,
            llm_engine=llm_engine,
            extractor=self._extractor,
            enricher=self._enricher,
            resolver=self.entity_resolver,
            indexer=self
        )

    # Delegation methods
    def invalidate_concept_cache(self) -> None:
        self.entity_resolver.invalidate_concept_cache()

    def resolve_entity(self, label: str, name: str) -> str:
        return self.entity_resolver.resolve_entity(label, name)

    def _add_resolved_entity_to_cache(self, label: str, entity_id: str, name: str, embedding: Optional[List[float]] = None) -> None:
        self.entity_resolver.add_resolved_entity_to_cache(label, entity_id, name, embedding)

    def reindex_metadata(self, paper_id: str, use_llm: bool = False) -> bool:
        return self.reindexer.reindex_metadata(paper_id, use_llm)

    def reindex_full(self, paper_id: str, pdf_parser_type: Optional[str] = None) -> bool:
        return self.reindexer.reindex_full(paper_id, pdf_parser_type)

    def reindex_metadata_batch(
        self,
        missing_authors: bool = False,
        missing_tags: bool = False,
        limit: Optional[int] = None,
        use_llm: bool = False,
        chunk_pool_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        return self.reindexer.reindex_metadata_batch(missing_authors, missing_tags, limit, use_llm, chunk_pool_size)

    def reindex_full_batch(
        self,
        all_papers: bool = False,
        paper_id: Optional[str] = None,
        limit: Optional[int] = None,
        chunk_pool_size: Optional[int] = None,
        pdf_parser_type: Optional[str] = None,
    ) -> Tuple[int, int]:
        return self.reindexer.reindex_full_batch(all_papers, paper_id, limit, chunk_pool_size, pdf_parser_type)

    # Use graph_write_builder
    async def _build_graph_writes_async(
        self,
        paper: Paper,
        extraction: ExtractionResult,
        full_text: str,
        is_markdown: bool,
        refs_or_links: List[str],
        api_references: List[Dict],
        api_citations: List[Dict]
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        return await self.graph_write_builder.build_graph_writes_async(
            paper, extraction, full_text, is_markdown, refs_or_links, api_references, api_citations
        )
```

**Step 4: Run all tests to verify it passes**
Run: `uv run pytest tests/`
Expected: PASS (All 36 pipeline tests and general project tests should run and pass)

**Step 5: Commit**
```bash
git add back/src/indexer.py
git commit -m "refactor: integrate services into Indexer"
```
