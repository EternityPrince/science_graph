"""
Indexer — Thin Orchestrator for the Ingestion Pipeline.

Workflow for every document type:
    Parse → Enrich Metadata → Extract Concepts → Chunk & Embed → Save Graph → Archive

Parser-specific logic lives in src/parsers/*.
Knowledge extraction lives in ExtractionService.
Metadata enrichment lives in MetadataEnricher.
All graph writes go through GraphRepository / VectorRepository interfaces.
No direct sqlite3 access.
"""

import os
import shutil
import time
import contextlib
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from src.models import Paper, Author, Concept, slugify
from src.vector_search import EmbeddingEngine, split_text_to_chunks
from src.repository.base import GraphRepository, VectorRepository
from src.services.extraction_service import ExtractionService, ExtractionResult
from src.services.metadata_enricher import MetadataEnricher
from src.parsers.factory import ParserFactory
from src.config import config
from src import console as con


from src.services.duplicate_detector import DuplicateDetector, _split_text_to_chunks_raw


class DuplicateDocumentError(ValueError):
    """Exception raised when an ingested document is detected as already existing in the database."""
    def __init__(self, message: str, duplicate_paper_id: str):
        super().__init__(message)
        self.duplicate_paper_id = duplicate_paper_id


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

    def resolve_entity(self, label: str, name: str) -> str:
        """
        Resolves an entity name of a given label to a canonical node ID.
        Checks:
        1. Exact slug match.
        2. Cosine similarity of name embedding > 0.95.
        3. String similarity > 0.95.
        """
        if not name:
            return ""
        
        name_clean = name.strip()
        slug = slugify(name_clean)
        
        # If Concept, check aliases map first
        if label == "Concept":
            try:
                aliases_map = self.graph_repo.get_concept_aliases()
                if name_clean.lower() in aliases_map:
                    canonical = aliases_map[name_clean.lower()]
                    return slugify(canonical)
            except Exception:
                pass

        if not hasattr(self, "_entity_cache"):
            self._entity_cache = {}
            
        if label not in self._entity_cache:
            try:
                self._entity_cache[label] = self.graph_repo.get_nodes_by_label(label)
            except Exception:
                self._entity_cache[label] = []

        existing_nodes = self._entity_cache[label]
        
        # 1. Exact slug match
        for eid, props in existing_nodes:
            if eid == slug:
                return eid
            node_name = props.get("name", "")
            if node_name and slugify(node_name) == slug:
                return eid
                
        # 2. Embedding similarity check
        candidate_emb = None
        for eid, props in existing_nodes:
            node_emb = props.get("embedding")
            if node_emb and isinstance(node_emb, list):
                if candidate_emb is None:
                    try:
                        candidate_emb = self.emb_engine.get_embedding(name_clean)
                        if not isinstance(candidate_emb, list):
                            candidate_emb = None
                    except Exception:
                        break  # Skip embedding similarity if embedding fails
                
                if candidate_emb is not None:
                    import math
                    dot = sum(a * b for a, b in zip(candidate_emb, node_emb))
                    norm1 = math.sqrt(sum(a * a for a in candidate_emb))
                    norm2 = math.sqrt(sum(b * b for b in node_emb))
                    sim = 0.0
                    if norm1 > 0.0 and norm2 > 0.0:
                        sim = dot / (norm1 * norm2)
                    
                    if sim > 0.95:
                        return eid

        # 3. String similarity fallback
        import difflib
        for eid, props in existing_nodes:
            node_name = props.get("name", "")
            if node_name:
                ratio = difflib.SequenceMatcher(None, name_clean.lower(), node_name.lower()).ratio()
                if ratio > 0.95:
                    return eid
                    
        return slug

    def _add_resolved_entity_to_cache(self, label: str, entity_id: str, name: str, embedding: Optional[List[float]] = None):
        if not hasattr(self, "_entity_cache"):
            self._entity_cache = {}
        if label not in self._entity_cache:
            try:
                self._entity_cache[label] = self.graph_repo.get_nodes_by_label(label)
            except Exception:
                self._entity_cache[label] = []
        
        # Check if already in cache
        exists = any(eid == entity_id for eid, _ in self._entity_cache[label])
        if not exists:
            self._entity_cache[label].append((entity_id, {"name": name, "embedding": embedding}))

    def detect_duplicate(self, paper: Paper, full_text: str) -> Optional[Tuple[str, str]]:
        """
        Detects if the given paper/document is already present in the database.
        Returns:
            Optional[Tuple[str, str]]: (duplicate_paper_id, matching_reason) if a duplicate is found,
                                       else None.
        """
        return self._duplicate_detector.detect_duplicate(paper, full_text)

    @contextlib.contextmanager
    def _trace_stage(self, stage_name: str, trace_info: Optional[dict]):
        """Context manager to measure and accumulate time taken by an ingestion stage."""
        if trace_info is None:
            yield
        else:
            t0 = time.perf_counter()
            yield
            dt = time.perf_counter() - t0
            stages = trace_info.setdefault("stages", {})
            stages[stage_name] = stages.get(stage_name, 0.0) + dt

    def index_pdf(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Runs the complete ingestion pipeline for a single PDF. Returns the paper ID."""
        return asyncio.run(self.index_pdf_async(file_path, trace_info))

    async def index_pdf_async(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Runs the complete ingestion pipeline for a single PDF asynchronously."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        con.info(f"Parsing [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, raw_references, full_text = parser.parse(file_path)

        # Determine archive path before enrichment (paper.id is stable)
        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.pdf"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        paper.file_path = str(archive_path)

        # NER fallback for author detection (PDF only, run early to populate paper.authors)
        if len(paper.authors) < 2:
            with self._trace_stage("NER Author Fallback", trace_info):
                paper.authors = self._ner_fallback_authors(paper.authors, file_path)

        def _archive():
            self._archive_pdf(file_path, archive_path)

        return await self._run_pipeline_async(
            paper=paper,
            full_text=full_text,
            refs_or_links=raw_references,
            is_markdown=False,
            needs_enrichment=True,
            archive_fn=_archive,
            source_path=file_path,
            trace_info=trace_info,
        )

    def index_markdown(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a Markdown note (.md) into the knowledge graph. Returns the note ID."""
        return asyncio.run(self.index_markdown_async(file_path, trace_info))

    async def index_markdown_async(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a Markdown note (.md) asynchronously."""
        con.info(f"Parsing note [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, wiki_links, body = parser.parse(file_path)

        return await self._run_pipeline_async(
            paper=paper,
            full_text=body,
            refs_or_links=wiki_links,
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None,
            source_path=file_path,
            trace_info=trace_info,
        )

    def index_epub(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes an EPUB book into the knowledge graph. Returns the book ID."""
        return asyncio.run(self.index_epub_async(file_path, trace_info))

    async def index_epub_async(self, file_path: str, trace_info: Optional[dict] = None) -> str:
        """Indexes an EPUB book asynchronously."""
        con.info(f"Parsing EPUB [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, _, full_text = parser.parse(file_path)

        return await self._run_pipeline_async(
            paper=paper,
            full_text=full_text,
            refs_or_links=[],
            is_markdown=False,
            needs_enrichment=False,
            archive_fn=None,
            source_path=file_path,
            trace_info=trace_info,
        )

    def index_url(self, url: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a webpage URL into the knowledge graph. Returns the page ID."""
        return asyncio.run(self.index_url_async(url, trace_info))

    async def index_url_async(self, url: str, trace_info: Optional[dict] = None) -> str:
        """Indexes a webpage URL asynchronously."""
        con.info(f"Parsing URL [bold]{url}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(url)
            paper, web_links, body = parser.parse(url)

        # Save local archive of the webpage
        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.md"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            archive_path.write_text(body, encoding="utf-8")
            paper.file_path = str(archive_path)
            con.dim(f"Saved local archive of website to {archive_path}")
        except Exception as e:
            con.warning(f"Could not save local archive of website: {e}")

        # URL parser already enriches via Semantic Scholar, so no second enrichment
        return await self._run_pipeline_async(
            paper=paper,
            full_text=body,
            refs_or_links=web_links,
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None,
            source_path=None,
            trace_info=trace_info,
        )

    def _get_citation_context(self, full_text: str, ref_title: str, ref_author: Optional[str] = None, ref_year: Optional[int] = None) -> str:
        if not full_text or not ref_title:
            return ""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        
        # Build patterns
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

    async def _classify_cites_edges_async(self, cites_list: List[Dict[str, Any]], full_text: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Takes a list of citation dicts containing:
          - source_id: str
          - target_id: str
          - title: str
          - author: Optional[str]
          - year: Optional[int]
          - properties: Dict[str, Any]
        Extracts contexts and classifies intents in parallel.
        Returns:
          List[Tuple[source_id, target_id, "CITES", edge_properties]]
        """
        if not cites_list:
            return []
            
        tasks = []
        metadata = []
        
        for cit in cites_list:
            ref_title = cit.get("title") or ""
            ref_author = cit.get("author")
            ref_year = cit.get("year")
            
            context = self._get_citation_context(full_text, ref_title, ref_author, ref_year)
            props = {**cit.get("properties", {})}
            if context:
                props["context"] = context
                
                # Create classification task
                tasks.append(self._extractor.classify_citation_intent_async(context, ref_title))
                metadata.append((cit["source_id"], cit["target_id"], props))
            else:
                props["intent"] = "BACKGROUND"
                metadata.append((cit["source_id"], cit["target_id"], props))
                # Add a dummy task that returns immediately to keep index alignment
                tasks.append(asyncio.sleep(0, result="BACKGROUND"))
                
        intents = await asyncio.gather(*tasks)
        
        edges = []
        for (src, tgt, props), intent in zip(metadata, intents):
            props["intent"] = intent or "BACKGROUND"
            edges.append((src, tgt, "CITES", props))
        return edges

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
        nodes_to_write = []
        edges_to_write = []
        added_node_ids = set()

        async def _safe_get_embedding(text: str) -> Optional[List[float]]:
            try:
                val = await asyncio.to_thread(self.emb_engine.get_embedding, text)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
            return None

        import hashlib
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
        
        # Determine label for paper
        paper_label = "UserNote" if paper.properties.get("source_type") == "note" else "Paper"
        nodes_to_write.append((paper.id, paper_label, paper_props))
        added_node_ids.add(paper.id)

        # 1. Authors
        for author_name in paper.authors:
            author_id = slugify(author_name)
            nodes_to_write.append((author_id, "Author", {"name": author_name}))
            edges_to_write.append((author_id, paper.id, "AUTHORED", {}))
            added_node_ids.add(author_id)

        # 2. Concepts and Tags
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
                    c_desc = await self._extractor.get_concept_description_async(c_name)
                
                concept_id = self.resolve_entity("Concept", c_name)
                if concept_id not in added_node_ids:
                    emb = await _safe_get_embedding(c_name)
                    nodes_to_write.append((concept_id, "Concept", {"name": c_name, "description": c_desc, "embedding": emb}))
                    self._add_resolved_entity_to_cache("Concept", concept_id, c_name, emb)
                    added_node_ids.add(concept_id)
                edges_to_write.append((paper.id, concept_id, "MENTIONS_CONCEPT", {}))

        for tag in paper.properties.get("tags") or []:
            tag_id = self.resolve_entity("Concept", tag)
            if tag_id not in added_node_ids:
                tag_desc = await self._extractor.get_concept_description_async(tag)
                emb = await _safe_get_embedding(tag)
                nodes_to_write.append((tag_id, "Concept", {"name": tag, "is_tag": True, "description": tag_desc, "embedding": emb}))
                self._add_resolved_entity_to_cache("Concept", tag_id, tag, emb)
                added_node_ids.add(tag_id)
            edges_to_write.append((paper.id, tag_id, "HAS_TAG", {}))

        # 3. New Entities: Institution
        # AFFILIATED_WITH (Author -> Institution)
        # SPONSORED_BY (Paper -> Institution)
        institutions = getattr(extraction, "institutions", [])
        if isinstance(institutions, list):
            for inst_name in institutions:
                if not isinstance(inst_name, str):
                    continue
                inst_id = self.resolve_entity("Institution", inst_name)
                if inst_id not in added_node_ids:
                    emb = await _safe_get_embedding(inst_name)
                    nodes_to_write.append((inst_id, "Institution", {"name": inst_name, "embedding": emb}))
                    self._add_resolved_entity_to_cache("Institution", inst_id, inst_name, emb)
                    added_node_ids.add(inst_id)
            
        sponsored_by = getattr(extraction, "sponsored_by", [])
        if isinstance(sponsored_by, list):
            for inst_name in sponsored_by:
                if not isinstance(inst_name, str):
                    continue
                inst_id = self.resolve_entity("Institution", inst_name)
                if inst_id not in added_node_ids:
                    emb = await _safe_get_embedding(inst_name)
                    nodes_to_write.append((inst_id, "Institution", {"name": inst_name, "embedding": emb}))
                    self._add_resolved_entity_to_cache("Institution", inst_id, inst_name, emb)
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
                    inst_id = self.resolve_entity("Institution", inst_name)
                    if inst_id not in added_node_ids:
                        emb = await _safe_get_embedding(inst_name)
                        nodes_to_write.append((inst_id, "Institution", {"name": inst_name, "embedding": emb}))
                        self._add_resolved_entity_to_cache("Institution", inst_id, inst_name, emb)
                        added_node_ids.add(inst_id)
                    edges_to_write.append((auth_id, inst_id, "AFFILIATED_WITH", {}))

        # 4. New Entities: Dataset
        # USED_DATASET (Paper -> Dataset) or INTRODUCED_DATASET (Paper -> Dataset)
        datasets = getattr(extraction, "datasets", [])
        if isinstance(datasets, list):
            for ds in datasets:
                if not isinstance(ds, dict):
                    continue
                ds_name = ds.get("name")
                relation = ds.get("relation", "USED_DATASET")
                if ds_name and isinstance(ds_name, str):
                    ds_id = self.resolve_entity("Dataset", ds_name)
                    if ds_id not in added_node_ids:
                        emb = await _safe_get_embedding(ds_name)
                        nodes_to_write.append((ds_id, "Dataset", {"name": ds_name, "embedding": emb}))
                        self._add_resolved_entity_to_cache("Dataset", ds_id, ds_name, emb)
                        added_node_ids.add(ds_id)
                    edges_to_write.append((paper.id, ds_id, relation, {}))

        # 5. New Entities: CodeRepository
        # HAS_CODE (Paper -> CodeRepository)
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

        # 6. New Entities: JournalConference
        # PUBLISHED_IN (Paper -> JournalConference)
        jc_name = getattr(extraction, "journal_or_conference", None)
        if jc_name and isinstance(jc_name, str):
            jc_id = self.resolve_entity("JournalConference", jc_name)
            if jc_id not in added_node_ids:
                emb = await _safe_get_embedding(jc_name)
                nodes_to_write.append((jc_id, "JournalConference", {"name": jc_name, "embedding": emb}))
                self._add_resolved_entity_to_cache("JournalConference", jc_id, jc_name, emb)
                added_node_ids.add(jc_id)
            edges_to_write.append((paper.id, jc_id, "PUBLISHED_IN", {}))

        # 7. Concept relations (subclass/prerequisite)
        # SUBCLASS_OF, IS_A, PREREQUISITE_FOR
        concept_relations = getattr(extraction, "concept_relations", [])
        if isinstance(concept_relations, list):
            for rel in concept_relations:
                if not isinstance(rel, dict):
                    continue
                src_c = rel.get("source", "").strip() if isinstance(rel.get("source"), str) else ""
                tgt_c = rel.get("target", "").strip() if isinstance(rel.get("target"), str) else ""
                rel_type = rel.get("relation_type", "SUBCLASS_OF")
                if src_c and tgt_c:
                    src_id = self.resolve_entity("Concept", src_c)
                    tgt_id = self.resolve_entity("Concept", tgt_c)
                    if src_id not in added_node_ids:
                        emb = await _safe_get_embedding(src_c)
                        nodes_to_write.append((src_id, "Concept", {"name": src_c, "embedding": emb}))
                        self._add_resolved_entity_to_cache("Concept", src_id, src_c, emb)
                        added_node_ids.add(src_id)
                    if tgt_id not in added_node_ids:
                        emb = await _safe_get_embedding(tgt_c)
                        nodes_to_write.append((tgt_id, "Concept", {"name": tgt_c, "embedding": emb}))
                        self._add_resolved_entity_to_cache("Concept", tgt_id, tgt_c, emb)
                        added_node_ids.add(tgt_id)
                    edges_to_write.append((src_id, tgt_id, rel_type, {}))

        # 8. Note-specific relationships from frontmatter
        # COMMENTS_ON, AGREES_WITH, DISAGREES_WITH, LINKED_TO (to Concept)
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
                concept_id = self.resolve_entity("Concept", concept_name)
                if concept_id not in added_node_ids:
                    emb = await _safe_get_embedding(concept_name)
                    nodes_to_write.append((concept_id, "Concept", {"name": concept_name, "embedding": emb}))
                    self._add_resolved_entity_to_cache("Concept", concept_id, concept_name, emb)
                    added_node_ids.add(concept_id)
                edges_to_write.append((paper.id, concept_id, "LINKED_TO", {}))

        # 9. Citation Intent and Context Classification for CITES edges
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
                    
                    author = None
                    if ref.get("authors"):
                        author = ref["authors"][0]
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
                    
                    author = None
                    if cit.get("authors"):
                        author = cit["authors"][0]
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
                        
        classified_cites_edges = await self._classify_cites_edges_async(cites_list, full_text)
        edges_to_write.extend(classified_cites_edges)

        return nodes_to_write, edges_to_write

    def _run_pipeline(
        self,
        paper: Paper,
        full_text: str,
        refs_or_links: List[str],
        is_markdown: bool,
        needs_enrichment: bool,
        archive_fn: Optional[Callable[[], None]],
        source_path: Optional[str] = None,
        trace_info: Optional[dict] = None,
    ) -> str:
        return asyncio.run(self._run_pipeline_async(
            paper=paper,
            full_text=full_text,
            refs_or_links=refs_or_links,
            is_markdown=is_markdown,
            needs_enrichment=needs_enrichment,
            archive_fn=archive_fn,
            source_path=source_path,
            trace_info=trace_info,
        ))

    async def _run_pipeline_async(
        self,
        paper: Paper,
        full_text: str,
        refs_or_links: List[str],
        is_markdown: bool,
        needs_enrichment: bool,
        archive_fn: Optional[Callable[[], None]],
        source_path: Optional[str] = None,
        trace_info: Optional[dict] = None,
    ) -> str:
        """
        Asynchronous unified ingestion pipeline using a DAG execution flow.
        """
        # Check for duplicates first
        dup_info = await asyncio.to_thread(self.detect_duplicate, paper, full_text)
        if dup_info:
            dup_id, reason = dup_info
            raise DuplicateDocumentError(
                f"Document already exists in database (ID: {dup_id}, match: {reason})",
                duplicate_paper_id=dup_id
            )

        # We will run Paths A, B, C, D concurrently
        api_references: List[Dict] = []
        api_citations: List[Dict] = []

        async def path_a_enrichment():
            nonlocal paper, api_references, api_citations
            if needs_enrichment:
                t0 = time.perf_counter()
                api_meta = await self._enricher.enrich_async(paper)
                if api_meta:
                    paper, api_references, api_citations = self._enricher.apply(paper, api_meta)
                dt = time.perf_counter() - t0
                if trace_info is not None:
                    trace_info.setdefault("stages", {})["Metadata Enrichment"] = dt
            return paper, api_references, api_citations

        async def path_b_extraction():
            t0 = time.perf_counter()
            extraction = await self._extractor.extract_async(
                title=paper.title or "",
                abstract=paper.abstract or "",
                full_text=full_text,
                use_llm=True,
                trace_info=trace_info,
            )
            dt = time.perf_counter() - t0
            if trace_info is not None:
                trace_info.setdefault("stages", {})["Concept & Tag Extraction"] = dt
            return extraction

        orig_size = 0
        if source_path and os.path.exists(source_path):
            orig_size = os.path.getsize(source_path)

        async def path_c_archive():
            if archive_fn is not None:
                t0 = time.perf_counter()
                await asyncio.to_thread(archive_fn)
                dt = time.perf_counter() - t0
                if trace_info is not None:
                    trace_info.setdefault("stages", {})["Archiving"] = dt

        async def path_d_summary():
            t0 = time.perf_counter()
            summary = await self._extractor.generate_summary_async(paper, full_text, graph_repo=None, trace_info=trace_info)
            dt = time.perf_counter() - t0
            if trace_info is not None:
                trace_info.setdefault("stages", {})["Summary Generation"] = dt
            return summary

        # Launch concurrent background tasks
        enrich_fut = asyncio.create_task(path_a_enrichment())
        extract_fut = asyncio.create_task(path_b_extraction())
        archive_fut = asyncio.create_task(path_c_archive())
        summary_fut = asyncio.create_task(path_d_summary())

        # Wait for all tasks to complete
        _, extraction, _, summary_text = await asyncio.gather(
            enrich_fut, extract_fut, archive_fut, summary_fut
        )

        # Sync Point: Merge all results
        if summary_text:
            paper.properties["summary"] = summary_text

        if orig_size > 0 and paper.file_path and os.path.exists(paper.file_path):
            new_size = os.path.getsize(paper.file_path)
            if trace_info is not None:
                trace_info["original_size"] = orig_size
                trace_info["compressed_size"] = new_size

        if extraction.authors:
            existing = {a.lower() for a in paper.authors}
            for a in extraction.authors:
                if a.lower() not in existing:
                    paper.authors.append(a)
                    existing.add(a.lower())

        existing_tags = paper.properties.get("tags") or []
        if extraction.tags:
            seen_tags = {t.lower().strip() for t in existing_tags}
            merged_tags = list(existing_tags)
            for t in extraction.tags:
                t_clean = t.strip()
                if t_clean.lower() not in seen_tags:
                    merged_tags.append(t_clean)
                    seen_tags.add(t_clean.lower())
            paper.properties["tags"] = merged_tags

        # ── Chunk + embed ──
        t0 = time.perf_counter()
        con.dim(f"Chunking and embedding: {(paper.title or paper.id)[:60]}")
        is_pdf = archive_fn is not None and (source_path or paper.file_path) and (source_path or paper.file_path).endswith(".pdf")
        if is_pdf:
            chunks = await asyncio.to_thread(split_text_to_chunks, paper.id, source_path or paper.file_path)
        else:
            chunks = _split_text_to_chunks_raw(paper.id, full_text)

        if chunks and paper.properties.get("source_type") == "video":
            con.dim("Filtering video transcript chunks for database relevance...")
            filtered_chunks = []
            for chunk in chunks:
                if self._extractor.is_chunk_relevant(chunk.text_content, paper.title or paper.id):
                    filtered_chunks.append(chunk)
            chunks = filtered_chunks

        if chunks:
            embeddings = await asyncio.to_thread(self.emb_engine.get_embeddings, [c.text_content for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
        dt = time.perf_counter() - t0
        if trace_info is not None:
            trace_info.setdefault("stages", {})["Chunking & Embedding"] = dt

        # ── Save Graph and Chunks Bulk ──
        t0 = time.perf_counter()
        
        nodes_to_write, edges_to_write = await self._build_graph_writes_async(
            paper=paper,
            extraction=extraction,
            full_text=full_text,
            is_markdown=is_markdown,
            refs_or_links=refs_or_links,
            api_references=api_references,
            api_citations=api_citations
        )

        def _write_bulk_db():
            with self.graph_repo.transaction():
                self.graph_repo.save_nodes_bulk(nodes_to_write)
                self.graph_repo.save_edges_bulk(edges_to_write)
            if chunks:
                self.vector_repo.save_chunks_bulk(chunks)

        await asyncio.to_thread(_write_bulk_db)

        dt = time.perf_counter() - t0
        if trace_info is not None:
            trace_info.setdefault("stages", {})["Graph & Vector Persistence"] = dt
            trace_info["authors_count"] = len(paper.authors)
            trace_info["concepts_count"] = len(extraction.concepts)
            trace_info["tags_count"] = len(paper.properties.get("tags") or [])
            trace_info["references_count"] = len(edges_to_write)

        con.success(f"Indexed [bold]{(paper.title or paper.id)[:70]}[/bold] (ID: {paper.id[:12]}…)")
        return paper.id

    def reindex_metadata(self, paper_id: str, use_llm: bool = False) -> bool:
        """
        Re-indexes metadata for a specific paper without regenerating embeddings.
        """
        paper = self.graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found in database: {paper_id}")
            return False

        con.info(f"Re-indexing metadata for [bold]{paper.title[:60]}[/bold] (ID: {paper_id})")

        # ── Fetch updated metadata ──
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

        api_meta = self._enricher.enrich(paper)
        api_references: List[Dict] = []
        api_citations: List[Dict] = []
        if api_meta:
            paper, api_references, api_citations = self._enricher.apply(paper, api_meta)

        # ── Re-extract concepts/tags ──
        full_text = self._read_local_text(paper)
        text_for_extraction = full_text or f"{paper.title or ''}\n\n{paper.abstract or ''}"

        extraction = self._extractor.extract(
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

        # ── NER fallback for PDF with few authors ──
        if (
            len(paper.authors) < 2
            and paper.file_path
            and os.path.exists(paper.file_path)
            and paper.file_path.lower().endswith(".pdf")
        ):
            paper.authors = self._ner_fallback_authors(paper.authors, paper.file_path)

        # ── Persist updated paper and relationships ──
        with self.graph_repo.transaction():
            # ── Persist updated paper ──
            self.graph_repo.save_paper(paper)

            # ── Refresh AUTHORED edges ──
            self.graph_repo.delete_edges_by_target(paper.id, ["AUTHORED"])
            for author_name in paper.authors:
                author_id = slugify(author_name)
                author = Author(id=author_id, name=author_name)
                self.graph_repo.save_author(author)
                self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

            # ── Refresh MENTIONS_CONCEPT / HAS_TAG edges ──
            self.graph_repo.delete_edges_by_source(paper.id, ["MENTIONS_CONCEPT", "HAS_TAG"])

            for item in extraction.concepts:
                c_name = item.get("name", "").strip()
                if not c_name:
                    continue
                c_desc = item.get("description", "") or self._extractor.get_concept_description(c_name)
                concept_id = slugify(c_name)
                concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                self.graph_repo.save_concept(concept)
                self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

            for tag in extraction.tags:
                tag_id = slugify(tag)
                tag_desc = self._extractor.get_concept_description(tag)
                tag_node = Concept(id=tag_id, name=tag, properties={"is_tag": True, "description": tag_desc})
                self.graph_repo.save_concept(tag_node)
                self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

            # ── Update citation edges ──
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
                self._extractor.generate_summary(paper, text_for_extraction, graph_repo=self.graph_repo)

        con.success(f"Successfully re-indexed metadata for {(paper.title or paper.id)[:60]}")
        return True

    def reindex_full(self, paper_id: str) -> bool:
        """
        Performs full reindexing by deleting the paper node and re-indexing the original file/URL.
        """
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
                self.index_url(file_path)
            elif file_path.lower().endswith(".pdf"):
                self.index_pdf(file_path)
            elif file_path.lower().endswith(".md"):
                self.index_markdown(file_path)
            elif file_path.lower().endswith(".epub"):
                self.index_epub(file_path)
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
    ) -> Tuple[int, int]:
        """
        Batch re-indexes paper metadata (authors, tags, etc.) based on filters.
        Returns:
            Tuple[int, int]: (success_count, total_count)
        """
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
    ) -> Tuple[int, int]:
        """
        Batch fully re-indexes papers by re-ingesting original files/URLs.
        Returns:
            Tuple[int, int]: (success_count, total_count)
        """
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
                if self.reindex_full(pid):
                    success_count += 1
            except Exception as e:
                con.error(f"Failed to fully re-index {pid}: {e}")

        con.blank()
        con.success(f"Fully re-indexed {success_count}/{len(candidates)} papers successfully.")
        return success_count, len(candidates)

    def _ner_fallback_authors(self, existing_authors: List[str], file_path: str) -> List[str]:
        """Runs NER on the first PDF page to extract author names when few are known."""
        try:
            from src.ner_engine import extract_persons_from_text
            import fitz
            doc = fitz.open(file_path)
            first_page_text = doc[0].get_text() if len(doc) > 0 else ""
            doc.close()
            ner_names = extract_persons_from_text(first_page_text[:2000])
            candidates = [n for n in ner_names if 1 < len(n.split()) <= 5]
        except Exception:
            return existing_authors

        seen = {a.lower().strip() for a in existing_authors}
        merged = list(existing_authors)
        for name in candidates:
            key = name.lower().strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(name)
        return merged

    def _archive_pdf(self, source_path: str, archive_path: Path) -> None:
        """Compresses (or copies) a PDF to the archive directory."""
        from src.parsers.pdf_parser import PDFParser
        if config.pdf_compression_enabled:
            con.dim(f"Compressing and saving PDF to archive: {archive_path} ...")
            try:
                orig_size = os.path.getsize(source_path)
                PDFParser.compress_and_save_pdf(
                    input_path=source_path,
                    output_path=str(archive_path),
                    dpi_threshold=config.pdf_compression_dpi_threshold,
                    dpi_target=config.pdf_compression_dpi_target,
                    quality=config.pdf_compression_quality,
                )
                if archive_path.exists():
                     new_size = archive_path.stat().st_size
                     ratio = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
                     con.success(
                         f"PDF compressed: {orig_size / 1024 / 1024:.1f} MB → "
                         f"{new_size / 1024 / 1024:.1f} MB ({ratio:.1f}% saved)"
                     )
                if Path(source_path).resolve() != archive_path.resolve():
                    os.remove(source_path)
            except Exception as e:
                con.warning(f"PDF compression failed: {e}. Falling back to standard copy.")
                if Path(source_path).resolve() != archive_path.resolve():
                    shutil.copy2(source_path, archive_path)
                    os.remove(source_path)
        else:
            if Path(source_path).resolve() != archive_path.resolve():
                shutil.copy2(source_path, archive_path)
                os.remove(source_path)
                con.info(f"PDF moved to archive: {archive_path}")

    def _read_local_text(self, paper: Paper) -> str:
        """Reads the full document text from the local file if available."""
        fp = paper.file_path
        if not fp or not os.path.exists(fp) or fp.startswith("http"):
            return ""
        try:
            if fp.lower().endswith(".pdf"):
                parser = ParserFactory.get_parser(fp)
                _, _, text = parser.parse(fp)
                return text
            elif fp.lower().endswith(".epub"):
                parser = ParserFactory.get_parser(fp)
                _, _, text = parser.parse(fp)
                return text
            else:
                return Path(fp).read_text(encoding="utf-8")
        except Exception as e:
            con.warning(f"Could not read local file {fp} for text extraction: {e}")
            return ""

    def index_batch(
        self,
        targets: List[str],
        use_llm: bool = True,
        trace: bool = False,
        chunk_pool_size: Optional[int] = None
    ) -> List[dict]:
        """Synchronous wrapper for batch indexing."""
        return asyncio.run(self.index_batch_async(
            targets=targets,
            use_llm=use_llm,
            trace=trace,
            chunk_pool_size=chunk_pool_size
        ))

    async def index_batch_async(
        self,
        targets: List[str],
        use_llm: bool = True,
        trace: bool = False,
        chunk_pool_size: Optional[int] = None
    ) -> List[dict]:
        """Unified staged batch ingestion pipeline."""
        if chunk_pool_size is not None:
            self._extractor._chunk_pool_size = chunk_pool_size
            self._extractor._sem = None  # Force semaphore re-creation

        session_traces = []
        resolved_targets = []

        # Resolve all targets (expanding directories and checking extensions)
        for tgt in targets:
            tgt_clean = tgt.strip()
            if not tgt_clean:
                continue
            if tgt_clean.startswith(("http://", "https://")):
                resolved_targets.append({"target": tgt_clean, "type": "url"})
            else:
                path = Path(tgt_clean).resolve()
                if not path.exists():
                    con.error(f"Path not found: {path}")
                    raise FileNotFoundError(f"Path not found: {path}")
                if path.is_file():
                    ext = path.suffix.lower().lstrip(".")
                    if ext in ("pdf", "md", "epub"):
                        resolved_targets.append({"target": str(path), "type": ext})
                    else:
                        con.warning(f"Unsupported file type '{ext}' for {path.name}, skipping.")
                elif path.is_dir():
                    allowed = {".pdf", ".md", ".epub"}
                    files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in allowed]
                    for f in files:
                        resolved_targets.append({"target": str(f), "type": f.suffix.lower().lstrip(".")})

        if not resolved_targets:
            con.warning("No valid targets found to index.")
            return []

        # Stage 1: Parsing
        parsed_items = []
        for item in resolved_targets:
            tgt = item["target"]
            t = item["type"]
            trace_info = {
                "stages": {},
                "tokens": {},
                "success": False,
                "name": os.path.basename(tgt) if t != "url" else tgt
            }

            try:
                if t == "pdf":
                    con.info(f"Parsing [bold]{os.path.basename(tgt)}[/bold]")
                    t0 = time.perf_counter()
                    parser = ParserFactory.get_parser(tgt)
                    paper, raw_references, full_text = await asyncio.to_thread(parser.parse, tgt)
                    trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0

                    archive_dir = Path(config.archive_dir)
                    archive_path = archive_dir / f"{paper.id}.pdf"
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    paper.file_path = str(archive_path)

                    if len(paper.authors) < 2:
                        t0_ner = time.perf_counter()
                        paper.authors = await asyncio.to_thread(self._ner_fallback_authors, paper.authors, tgt)
                        trace_info["stages"]["NER Author Fallback"] = time.perf_counter() - t0_ner

                    def _archive():
                        self._archive_pdf(tgt, archive_path)

                    parsed_items.append({
                        "item": item,
                        "paper": paper,
                        "full_text": full_text,
                        "refs_or_links": raw_references,
                        "is_markdown": False,
                        "needs_enrichment": True,
                        "archive_fn": _archive,
                        "source_path": tgt,
                        "trace_info": trace_info
                    })

                elif t == "md":
                    con.info(f"Parsing note [bold]{os.path.basename(tgt)}[/bold]")
                    t0 = time.perf_counter()
                    parser = ParserFactory.get_parser(tgt)
                    paper, wiki_links, body = await asyncio.to_thread(parser.parse, tgt)
                    trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0

                    parsed_items.append({
                        "item": item,
                        "paper": paper,
                        "full_text": body,
                        "refs_or_links": wiki_links,
                        "is_markdown": True,
                        "needs_enrichment": False,
                        "archive_fn": None,
                        "source_path": tgt,
                        "trace_info": trace_info
                    })

                elif t == "epub":
                    con.info(f"Parsing EPUB [bold]{os.path.basename(tgt)}[/bold]")
                    t0 = time.perf_counter()
                    parser = ParserFactory.get_parser(tgt)
                    paper, _, full_text = await asyncio.to_thread(parser.parse, tgt)
                    trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0

                    parsed_items.append({
                        "item": item,
                        "paper": paper,
                        "full_text": full_text,
                        "refs_or_links": [],
                        "is_markdown": False,
                        "needs_enrichment": False,
                        "archive_fn": None,
                        "source_path": tgt,
                        "trace_info": trace_info
                    })

                elif t == "url":
                    con.info(f"Parsing URL [bold]{tgt}[/bold]")
                    t0 = time.perf_counter()
                    parser = ParserFactory.get_parser(tgt)
                    paper, web_links, body = await asyncio.to_thread(parser.parse, tgt)
                    trace_info["stages"]["Document Parsing"] = time.perf_counter() - t0

                    archive_dir = Path(config.archive_dir)
                    archive_path = archive_dir / f"{paper.id}.md"
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        await asyncio.to_thread(archive_path.write_text, body, encoding="utf-8")
                        paper.file_path = str(archive_path)
                        con.dim(f"Saved local archive of website to {archive_path}")
                    except Exception as e:
                        con.warning(f"Could not save local archive of website: {e}")

                    parsed_items.append({
                        "item": item,
                        "paper": paper,
                        "full_text": body,
                        "refs_or_links": web_links,
                        "is_markdown": True,
                        "needs_enrichment": False,
                        "archive_fn": None,
                        "source_path": None,
                        "trace_info": trace_info
                    })
            except Exception as e:
                con.error(f"Failed to parse target {tgt}: {e}")
                trace_info["success"] = False
                session_traces.append(trace_info)

        # Duplicate checking & initial filtering
        filtered_parsed_items = []
        for p_item in parsed_items:
            paper = p_item["paper"]
            full_text = p_item["full_text"]
            trace_info = p_item["trace_info"]

            dup_info = await asyncio.to_thread(self.detect_duplicate, paper, full_text)
            if dup_info:
                dup_id, reason = dup_info
                con.warning(f"Duplicate detected: Document already exists in database (ID: {dup_id}, match: {reason})")
                trace_info["skipped_duplicate"] = True
                trace_info["success"] = False
                session_traces.append(trace_info)
            else:
                filtered_parsed_items.append(p_item)

        # Archiving tasks
        for p_item in filtered_parsed_items:
            archive_fn = p_item["archive_fn"]
            trace_info = p_item["trace_info"]
            if archive_fn:
                t0_arch = time.perf_counter()
                await asyncio.to_thread(archive_fn)
                trace_info["stages"]["Archiving"] = time.perf_counter() - t0_arch

        # Parallel metadata enrichment
        async def enrich_item(p_item):
            paper = p_item["paper"]
            trace_info = p_item["trace_info"]
            if p_item["needs_enrichment"]:
                t0_enrich = time.perf_counter()
                api_meta = await self._enricher.enrich_async(paper)
                if api_meta:
                    paper, api_refs, api_cits = self._enricher.apply(paper, api_meta)
                    p_item["paper"] = paper
                    p_item["api_references"] = api_refs
                    p_item["api_citations"] = api_cits
                trace_info["stages"]["Metadata Enrichment"] = time.perf_counter() - t0_enrich

        enrich_tasks = [
            asyncio.create_task(enrich_item(p_item))
            for p_item in filtered_parsed_items
        ]
        if enrich_tasks:
            await asyncio.gather(*enrich_tasks)

        # Stage 2: Chunking
        for p_item in filtered_parsed_items:
            paper = p_item["paper"]
            full_text = p_item["full_text"]
            source_path = p_item["source_path"]
            trace_info = p_item["trace_info"]
            p_item["failed"] = False

            try:
                t0_chunk = time.perf_counter()
                con.dim(f"Chunking: {(paper.title or paper.id)[:60]}")

                is_pdf = p_item["archive_fn"] is not None and (source_path or paper.file_path) and (source_path or paper.file_path).endswith(".pdf")
                if is_pdf:
                    chunks = await asyncio.to_thread(split_text_to_chunks, paper.id, source_path or paper.file_path)
                else:
                    chunks = _split_text_to_chunks_raw(paper.id, full_text)

                if chunks and paper.properties.get("source_type") == "video":
                    con.dim("Filtering video transcript chunks for database relevance...")
                    filtered_chunks = []
                    for chunk in chunks:
                        if self._extractor.is_chunk_relevant(chunk.text_content, paper.title or paper.id):
                            filtered_chunks.append(chunk)
                    chunks = filtered_chunks

                p_item["chunks"] = chunks
                p_item["chunk_time"] = time.perf_counter() - t0_chunk
            except Exception as e:
                con.error(f"Failed during chunking for {paper.id}: {e}")
                p_item["failed"] = True
                trace_info["success"] = False
                session_traces.append(trace_info)

        # Stage 2.2: Batch Embedding
        items_to_embed = [p for p in filtered_parsed_items if not p.get("failed", False)]
        all_chunks = []
        for p_item in items_to_embed:
            all_chunks.extend(p_item["chunks"])

        if all_chunks:
            try:
                t0_embed = time.perf_counter()
                embeddings = await asyncio.to_thread(self.emb_engine.get_embeddings, [c.text_content for c in all_chunks])
                total_embed_time = time.perf_counter() - t0_embed

                idx = 0
                for p_item in items_to_embed:
                    doc_chunks = p_item["chunks"]
                    for chunk in doc_chunks:
                        chunk.embedding = embeddings[idx]
                        idx += 1
                    prop_embed_time = total_embed_time * (len(doc_chunks) / len(all_chunks)) if len(all_chunks) > 0 else 0.0
                    p_item["trace_info"]["stages"]["Chunking & Embedding"] = p_item["chunk_time"] + prop_embed_time
            except Exception as e:
                con.error(f"Failed to generate embeddings for batch: {e}")
                for p_item in items_to_embed:
                    p_item["failed"] = True
                    p_item["trace_info"]["success"] = False
                    session_traces.append(p_item["trace_info"])

        # Stage 3: LLM Concept Extraction & Summary Generation
        items_for_llm = [p for p in items_to_embed if not p.get("failed", False)]

        async def process_llm_for_item(p_item):
            paper = p_item["paper"]
            full_text = p_item["full_text"]
            trace_info = p_item["trace_info"]

            try:
                # Concept Extraction
                t0_extract = time.perf_counter()
                extraction = await self._extractor.extract_async(
                    title=paper.title or "",
                    abstract=paper.abstract or "",
                    full_text=full_text,
                    use_llm=use_llm,
                    trace_info=trace_info,
                )
                trace_info["stages"]["Concept & Tag Extraction"] = time.perf_counter() - t0_extract
                p_item["extraction"] = extraction

                # Summary Generation
                t0_summary = time.perf_counter()
                summary_text = await self._extractor.generate_summary_async(paper, full_text, graph_repo=None, trace_info=trace_info)
                trace_info["stages"]["Summary Generation"] = time.perf_counter() - t0_summary
                p_item["summary_text"] = summary_text
            except Exception as e:
                con.error(f"Failed during LLM extraction/summary for {paper.id}: {e}")
                p_item["failed"] = True
                trace_info["success"] = False
                session_traces.append(trace_info)

        llm_tasks = [
            asyncio.create_task(process_llm_for_item(p_item))
            for p_item in items_for_llm
        ]
        if llm_tasks:
            await asyncio.gather(*llm_tasks)

        # Stage 4: Database writes
        items_to_persist = [p for p in items_for_llm if not p.get("failed", False)]
        for p_item in items_to_persist:
            paper = p_item["paper"]
            full_text = p_item["full_text"]
            extraction = p_item["extraction"]
            summary_text = p_item["summary_text"]
            chunks = p_item["chunks"]
            is_markdown = p_item["is_markdown"]
            refs_or_links = p_item["refs_or_links"]
            api_references = p_item.get("api_references") or []
            api_citations = p_item.get("api_citations") or []
            trace_info = p_item["trace_info"]
            source_path = p_item["source_path"]

            try:
                t0_db = time.perf_counter()

                if summary_text:
                    paper.properties["summary"] = summary_text

                orig_size = 0
                if source_path and os.path.exists(source_path):
                    orig_size = os.path.getsize(source_path)
                if orig_size > 0 and paper.file_path and os.path.exists(paper.file_path):
                    new_size = os.path.getsize(paper.file_path)
                    trace_info["original_size"] = orig_size
                    trace_info["compressed_size"] = new_size

                if extraction.authors:
                    existing = {a.lower() for a in paper.authors}
                    for a in extraction.authors:
                        if a.lower() not in existing:
                            paper.authors.append(a)
                            existing.add(a.lower())

                existing_tags = paper.properties.get("tags") or []
                if extraction.tags:
                    seen_tags = {t.lower().strip() for t in existing_tags}
                    merged_tags = list(existing_tags)
                    for t in extraction.tags:
                        t_clean = t.strip()
                        if t_clean.lower() not in seen_tags:
                            merged_tags.append(t_clean)
                            seen_tags.add(t_clean.lower())
                    paper.properties["tags"] = merged_tags

                nodes_to_write, edges_to_write = await self._build_graph_writes_async(
                    paper=paper,
                    extraction=extraction,
                    full_text=full_text,
                    is_markdown=is_markdown,
                    refs_or_links=refs_or_links,
                    api_references=api_references,
                    api_citations=api_citations
                )

                def _write_bulk_db():
                    with self.graph_repo.transaction():
                        self.graph_repo.save_nodes_bulk(nodes_to_write)
                        self.graph_repo.save_edges_bulk(edges_to_write)
                    if chunks:
                        self.vector_repo.save_chunks_bulk(chunks)

                await asyncio.to_thread(_write_bulk_db)

                trace_info["stages"]["Graph & Vector Persistence"] = time.perf_counter() - t0_db
                trace_info["authors_count"] = len(paper.authors)
                trace_info["concepts_count"] = len(extraction.concepts)
                trace_info["tags_count"] = len(paper.properties.get("tags") or [])
                trace_info["references_count"] = len(edges_to_write)
                trace_info["success"] = True

                session_traces.append(trace_info)
                con.success(f"Indexed [bold]{(paper.title or paper.id)[:70]}[/bold] (ID: {paper.id[:12]}…)")
            except Exception as e:
                con.error(f"Failed to write to database for {paper.id}: {e}")
                trace_info["success"] = False
                session_traces.append(trace_info)

        return session_traces
