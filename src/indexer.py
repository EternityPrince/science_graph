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
import re
import shutil
import time
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from src.models import Paper, Author, Concept, Chunk, slugify
from src.vector_search import EmbeddingEngine, split_text_to_chunks
from src.repository.base import GraphRepository, VectorRepository
from src.services.extraction_service import ExtractionService, ExtractionResult
from src.services.metadata_enricher import MetadataEnricher
from src.parsers.factory import ParserFactory
from src.config import config
from src import console as con


def _split_text_to_chunks_raw(
    paper_id: str, text: str, chunk_size: int = None, chunk_overlap: int = None
) -> List[Chunk]:
    """Splits a plain text string (not PDF) into overlapping Chunk objects."""
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap

    paragraphs = re.split(r'\n{2,}', text)
    chunks: List[Chunk] = []
    chunk_idx = 0
    page_num = 1
    buffer = ""

    for para in paragraphs:
        para_clean = re.sub(r'\s+', ' ', para).strip()
        if not para_clean:
            page_num += 1
            continue
        buffer += " " + para_clean
        while len(buffer) >= chunk_size:
            window = buffer[:chunk_size]
            if len(window) > 50:
                chunks.append(Chunk(
                    id=f"{paper_id}#{chunk_idx}",
                    paper_id=paper_id,
                    text_content=window.strip(),
                    page_number=page_num,
                ))
                chunk_idx += 1
            buffer = buffer[chunk_size - chunk_overlap:]

    remainder = buffer.strip()
    if len(remainder) > 50:
        chunks.append(Chunk(
            id=f"{paper_id}#{chunk_idx}",
            paper_id=paper_id,
            text_content=remainder,
            page_number=page_num,
        ))

    return chunks


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

    def detect_duplicate(self, paper: Paper, full_text: str) -> Optional[Tuple[str, str]]:
        """
        Detects if the given paper/document is already present in the database.
        Returns:
            Optional[Tuple[str, str]]: (duplicate_paper_id, matching_reason) if a duplicate is found,
                                       else None.
        """
        import hashlib

        # Helper to check if a paper is a placeholder
        def is_placeholder(p: Paper) -> bool:
            if not p:
                return False
            return bool(p.properties.get("placeholder") or p.properties.get("is_placeholder"))

        # 1. Exact ID check
        existing = self.graph_repo.get_paper(paper.id)
        if existing and not is_placeholder(existing):
            return existing.id, "exact_id"

        # 2. DOI check
        if paper.doi:
            existing_doi = self.graph_repo.find_paper_by_doi(paper.doi)
            if existing_doi and not is_placeholder(existing_doi):
                return existing_doi.id, "doi"

        # 3. Content Hash check (for incoming text)
        content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
        existing_hash = self.graph_repo.find_paper_by_content_hash(content_hash)
        if existing_hash and not is_placeholder(existing_hash):
            return existing_hash.id, "content_hash"

        # Helper for Jaccard similarity of author lists
        def author_jaccard_similarity(authors1: List[str], authors2: List[str]) -> float:
            set1 = {slugify(a) for a in authors1 if slugify(a)}
            set2 = {slugify(a) for a in authors2 if slugify(a)}
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            return len(set1.intersection(set2)) / len(set1.union(set2))

        # Helper for 3-word shingles
        def get_3_shingles(text: str) -> set:
            words = re.findall(r'\b\w+\b', text.lower())
            shingles = set()
            for i in range(len(words) - 2):
                shingles.add((words[i], words[i+1], words[i+2]))
            return shingles

        # Helper for word Jaccard similarity
        def word_jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            return len(words1.intersection(words2)) / len(words1.union(words2))

        # Helper to reconstruct text from chunks
        def reconstruct_text(paper_id: str) -> str:
            chunks = self.vector_repo.get_chunks_for_paper(paper_id)
            def get_idx(chunk):
                try:
                    return int(chunk.id.split('#')[-1])
                except Exception:
                    return chunk.id
            chunks_sorted = sorted(chunks, key=get_idx)
            return " ".join(c.text_content for c in chunks_sorted)

        # Build candidate set of Papers to avoid database-wide scans
        candidates = {}

        def add_candidate(p: Paper):
            if p and p.id != paper.id and not is_placeholder(p):
                candidates[p.id] = p

        # Check exact ID candidate
        # Check DOI candidate
        if paper.doi:
            add_candidate(self.graph_repo.find_paper_by_doi(paper.doi))

        # Check content hash candidate
        add_candidate(self.graph_repo.find_paper_by_content_hash(content_hash))

        # Check case-insensitive title candidate
        if paper.title:
            add_candidate(self.graph_repo.find_paper_by_title(paper.title))

        # Check shared author candidates
        for author_name in paper.authors:
            author_id = slugify(author_name)
            if author_id:
                for p in self.graph_repo.get_papers_by_author(author_id):
                    add_candidate(p)

        # Check vector similarity candidate & embedding similarity check
        chunks = _split_text_to_chunks_raw(paper.id, full_text)
        first_chunk_text = ""
        if chunks:
            first_chunk_text = chunks[0].text_content
        elif full_text.strip():
            first_chunk_text = full_text.strip()
        elif paper.title.strip():
            first_chunk_text = paper.title.strip()

        if first_chunk_text:
            emb = self.emb_engine.get_embeddings([first_chunk_text])[0]
            similar_chunks = self.vector_repo.search_similar_chunks(emb, limit=10)
            for c, sim in similar_chunks:
                if sim >= 0.95:
                    cand_paper = self.graph_repo.get_paper(c.paper_id)
                    if cand_paper and not is_placeholder(cand_paper):
                        add_candidate(cand_paper)
                        if word_jaccard_similarity(first_chunk_text, c.text_content) >= 0.80:
                            return cand_paper.id, "embedding_similarity"

        # Now detailed checking on all candidates
        shingles_new = get_3_shingles(full_text)

        for cand_id, cand in candidates.items():
            # A. Exact Title and Author similarity > 0.3 or both empty
            if paper.title and cand.title:
                if paper.title.strip().lower() == cand.title.strip().lower():
                    author_sim = author_jaccard_similarity(paper.authors, cand.authors)
                    if author_sim > 0.3 or (not paper.authors and not cand.authors):
                        return cand_id, "title_author_similarity"

            # B. Legacy content hash comparison
            cand_hash = cand.properties.get("content_hash")
            cand_text = None
            if not cand_hash:
                cand_text = reconstruct_text(cand_id)
                cand_hash = hashlib.sha256(cand_text.encode('utf-8')).hexdigest()
                updated_props = {**cand.properties, "content_hash": cand_hash}
                self.graph_repo.update_node_properties(cand_id, updated_props)

            if cand_hash == content_hash:
                return cand_id, "content_hash"

            # C. 3-word shingles check (threshold >= 0.70)
            if shingles_new:
                if cand_text is None:
                    cand_text = reconstruct_text(cand_id)
                shingles_cand = get_3_shingles(cand_text)
                if shingles_cand:
                    intersection = len(shingles_new.intersection(shingles_cand))
                    union = len(shingles_new.union(shingles_cand))
                    jaccard = intersection / union if union > 0 else 0.0
                    if jaccard >= 0.70:
                        return cand_id, "shingle_similarity"

        return None

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

        return self._run_pipeline(
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
        con.info(f"Parsing note [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, wiki_links, body = parser.parse(file_path)

        return self._run_pipeline(
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
        con.info(f"Parsing EPUB [bold]{os.path.basename(file_path)}[/bold]")
        with self._trace_stage("Document Parsing", trace_info):
            parser = ParserFactory.get_parser(file_path)
            paper, _, full_text = parser.parse(file_path)

        return self._run_pipeline(
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
        return self._run_pipeline(
            paper=paper,
            full_text=body,
            refs_or_links=web_links,
            is_markdown=True,
            needs_enrichment=False,
            archive_fn=None,
            source_path=None,
            trace_info=trace_info,
        )


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
        """
        Unified ingestion pipeline.
        """
        # Check for duplicates first
        dup_info = self.detect_duplicate(paper, full_text)
        if dup_info:
            dup_id, reason = dup_info
            raise DuplicateDocumentError(
                f"Document already exists in database (ID: {dup_id}, match: {reason})",
                duplicate_paper_id=dup_id
            )
        # ── Step 1: Enrich metadata ──
        api_references: List[Dict] = []
        api_citations: List[Dict] = []
        if needs_enrichment:
            with self._trace_stage("Metadata Enrichment", trace_info):
                api_meta = self._enricher.enrich(paper)
                if api_meta:
                    paper, api_references, api_citations = self._enricher.apply(paper, api_meta)

        # ── Step 2: Extract concepts + tags ──
        with self._trace_stage("Concept & Tag Extraction", trace_info):
            extraction: ExtractionResult = self._extractor.extract(
                title=paper.title or "",
                abstract=paper.abstract or "",
                full_text=full_text,
                use_llm=True,
                trace_info=trace_info,
            )
            # Merge any LLM-discovered authors into paper.authors
            if extraction.authors:
                existing = {a.lower() for a in paper.authors}
                for a in extraction.authors:
                    if a.lower() not in existing:
                        paper.authors.append(a)
                        existing.add(a.lower())

            # Apply extracted tags to paper properties, merging with existing tags
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

        # ── Step 3: Save Paper node & Step 4-7: Persistence ──
        with self._trace_stage("Graph Persistence", trace_info):
            with self.graph_repo.transaction():
                # ── Step 3: Save Paper node ──
                import hashlib
                content_hash = hashlib.sha256(full_text.encode('utf-8')).hexdigest()
                paper.properties["content_hash"] = content_hash
                self.graph_repo.save_paper(paper)

                # ── Step 4: Save Author nodes + AUTHORED edges ──
                for author_name in paper.authors:
                    author_id = slugify(author_name)
                    author = Author(id=author_id, name=author_name)
                    self.graph_repo.save_author(author)
                    self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

                # ── Step 5: Concept nodes + MENTIONS_CONCEPT edges ──
                for item in extraction.concepts:
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = item.get("description", "") or self._extractor.get_concept_description(c_name, trace_info=trace_info)
                    concept_id = slugify(c_name)
                    concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

                # ── Step 6: Tag nodes + HAS_TAG edges ──
                for tag in paper.properties.get("tags") or []:
                    tag_id = slugify(tag)
                    if not tag_id:
                        continue
                    tag_desc = self._extractor.get_concept_description(tag, trace_info=trace_info)
                    tag_node = Concept(id=tag_id, name=tag, properties={"is_tag": True, "description": tag_desc})
                    self.graph_repo.save_concept(tag_node)
                    self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

                # ── Step 7: Wiki-links (is_markdown=True) or References (is_markdown=False) ──
                if is_markdown:
                    for link_target in refs_or_links:
                        if link_target.startswith(("http://", "https://")):
                            clean_target = link_target.replace("https://", "").replace("http://", "").strip("/")
                            target_id = slugify(clean_target)
                        else:
                            target_id = slugify(link_target)
                        if not target_id:
                            continue
                        if not self.graph_repo.get_paper(target_id):
                            placeholder = Paper(
                                id=target_id,
                                title=link_target,
                                authors=[],
                                year=None,
                                doi=None,
                                properties={"is_placeholder": True}
                            )
                            self.graph_repo.save_paper(placeholder)
                        self.graph_repo.add_edge(paper.id, target_id, "RELATED_TO")
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
                            if not self.graph_repo.get_paper(ref_id):
                                placeholder = Paper(id=ref_id, title=ref_title, authors=[], year=None, doi=ref_doi)
                                placeholder.properties["is_placeholder"] = True
                                self.graph_repo.save_paper(placeholder)
                            self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"api_sourced": True})

                    for cit in all_cits:
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

                    # Fallback: raw text references (PDF parsing fallback, no Semantic Scholar)
                    if not api_references and not api_citations:
                        for ref_str in refs_or_links:
                            ref_clean = ref_str.strip()
                            if len(ref_clean) > 10:
                                ref_id = slugify(ref_clean[:120])
                                if not self.graph_repo.get_paper(ref_id):
                                    placeholder = Paper(id=ref_id, title=ref_clean[:120], authors=[], year=None, doi=None)
                                    placeholder.properties["is_placeholder"] = True
                                    self.graph_repo.save_paper(placeholder)
                                self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"raw_text": ref_clean})

        # ── Step 8: Chunk + embed ──
        with self._trace_stage("Chunking & Embedding", trace_info):
            con.dim(f"Chunking and embedding: {(paper.title or paper.id)[:60]}")
            is_pdf = archive_fn is not None and (source_path or paper.file_path) and (source_path or paper.file_path).endswith(".pdf")
            if is_pdf:
                # For PDFs, split_text_to_chunks reads the actual file
                chunks = split_text_to_chunks(paper.id, source_path or paper.file_path)
            else:
                chunks = _split_text_to_chunks_raw(paper.id, full_text)

            # Filter relevant chunks for video documents
            if chunks and paper.properties.get("source_type") == "video":
                con.dim("Filtering video transcript chunks for database relevance...")
                filtered_chunks = []
                for chunk in chunks:
                    if self._extractor.is_chunk_relevant(chunk.text_content, paper.title or paper.id):
                        filtered_chunks.append(chunk)
                chunks = filtered_chunks

            if chunks:
                embeddings = self.emb_engine.get_embeddings([c.text_content for c in chunks])
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                self.vector_repo.save_chunks(chunks)

        # ── Step 9: Archive ──
        if archive_fn is not None:
            with self._trace_stage("Archiving", trace_info):
                archive_fn()

        # ── Step 10: Summary ──
        with self._trace_stage("Summary Generation", trace_info):
            self._extractor.generate_summary(paper, full_text, graph_repo=self.graph_repo, trace_info=trace_info)

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
