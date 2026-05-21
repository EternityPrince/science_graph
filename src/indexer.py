import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any
from src.models import Paper, Author, Concept, Chunk
from src.parser import PDFParser
from src.vector_search import EmbeddingEngine, split_text_to_chunks
from src.repository.base import GraphRepository, VectorRepository
from src.config import config
from src import console as con


def _split_text_to_chunks_raw(
    paper_id: str, text: str, chunk_size: int = None, chunk_overlap: int = None
) -> List[Chunk]:
    """Splits a plain text string (not PDF) into overlapping Chunk objects."""
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap

    # Split into rough «pages» by double newlines first, then window
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
        # Flush when buffer is large enough
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

    # Flush remainder
    remainder = buffer.strip()
    if len(remainder) > 50:
        chunks.append(Chunk(
            id=f"{paper_id}#{chunk_idx}",
            paper_id=paper_id,
            text_content=remainder,
            page_number=page_num,
        ))

    return chunks

# List of key AI concepts to extract from papers
AI_CONCEPTS = {
    "transformer": "Transformer Architecture",
    "self-attention": "Self-Attention",
    "attention mechanism": "Attention Mechanism",
    "reinforcement learning": "Reinforcement Learning",
    "diffusion model": "Diffusion Models",
    "large language model": "Large Language Models",
    "llm": "Large Language Models",
    "fine-tuning": "Fine-Tuning",
    "lora": "LoRA (Low-Rank Adaptation)",
    "rag": "Retrieval-Augmented Generation",
    "retrieval-augmented": "Retrieval-Augmented Generation",
    "mixture of experts": "Mixture of Experts (MoE)",
    "moe": "Mixture of Experts (MoE)",
    "contrastive learning": "Contrastive Learning",
    "prompt engineering": "Prompt Engineering",
    "supervised fine-tuning": "Supervised Fine-Tuning",
    "sft": "Supervised Fine-Tuning",
    "rlhf": "Reinforcement Learning from Human Feedback (RLHF)",
    "dpo": "Direct Preference Optimization (DPO)",
    "quantization": "Model Quantization",
    "mlx": "MLX Framework",
    "llama": "LLaMA Models",
    "gemma": "Gemma Models"
}

# List of high-level topic tags to extract/classify papers
AI_TOPICS = {
    "statistics": "Statistics",
    "probability": "Probability Theory",
    "gradient descent": "Gradient Descent",
    "optimization": "Optimization Methods",
    "deep learning": "Deep Learning",
    "machine learning": "Machine Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision",
    "reinforcement learning": "Reinforcement Learning",
    "generative model": "Generative Models",
    "linear regression": "Linear Regression",
    "neural network": "Neural Networks",
    "convex optimization": "Convex Optimization",
    "bayesian": "Bayesian Methods",
    "diffusion": "Diffusion Models",
    "transformer": "Transformers",
    "llm": "Large Language Models",
    "large language model": "Large Language Models",
}

class Indexer:
    def __init__(self, graph_repo: GraphRepository, vector_repo: VectorRepository, embedding_engine: EmbeddingEngine, llm_engine: Any = None):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine

    def _slugify(self, text: str) -> str:
        """Converts text to a safe alphanumeric lowercase slug."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s-]', '', text)
        return re.sub(r'[\s-]+', '_', text)

    def index_pdf(self, file_path: str) -> str:
        """
        Runs the complete ingestion pipeline for a single PDF.
        Returns the paper ID.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # 1. Parse PDF and extract metadata & references
        con.info(f"Parsing [bold]{os.path.basename(file_path)}[/bold]")
        paper, raw_references, full_text = PDFParser.extract_text_and_metadata(file_path)
        
        # 1b. Fetch external metadata from Semantic Scholar API
        from src.external_api import fetch_paper_metadata
        con.dim("Fetching metadata from Semantic Scholar …")
        api_meta = fetch_paper_metadata(doi=paper.doi, title=paper.title)
        
        api_references = []
        api_citations = []
        
        if api_meta:
            if api_meta.get("title"):
                paper.title = api_meta["title"]
            if api_meta.get("authors"):
                paper.authors = api_meta["authors"]
            if api_meta.get("year"):
                paper.year = api_meta["year"]
            if api_meta.get("abstract"):
                paper.abstract = api_meta["abstract"]
            if api_meta.get("doi"):
                paper.doi = api_meta["doi"]
            api_references = api_meta.get("references", [])
            api_citations = api_meta.get("citations", [])
            con.success(f"Metadata enriched from Semantic Scholar: [bold]{paper.title[:60]}[/bold]")
            
        # 1c. LLM-based concept/entity extraction
        llm_authors = []
        llm_concepts = []
        llm_tags = []
        if self.llm_engine:
            con.dim("Extracting concepts via LLM …")
            # Combine abstract/introduction
            sample_text = (paper.title or "") + "\n\n" + (paper.abstract or "") + "\n\n" + full_text[:4000]
            llm_data = self.llm_engine.extract_concepts_and_metadata(sample_text)
            if llm_data:
                llm_authors = llm_data.get("authors", [])
                llm_concepts = llm_data.get("concepts", [])
                llm_tags = llm_data.get("tags", [])
                con.dim(f"LLM found {len(llm_authors)} authors, {len(llm_concepts)} concepts, {len(llm_tags)} tags")
        
        # Determine tags
        tags_to_save = []
        if llm_tags:
            tags_to_save = llm_tags
        else:
            text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + full_text[:10000]).lower()
            for keyword, tag_name in AI_TOPICS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    if tag_name not in tags_to_save:
                        tags_to_save.append(tag_name)
        paper.properties["tags"] = tags_to_save

        # Determine archival path
        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.pdf"
        paper.file_path = str(archive_path)
        
        # 2. Save Paper Node
        self.graph_repo.save_paper(paper)
        
        # 3. Create Author nodes & AUTHORED edges
        # Combine: PDF-parsed authors + LLM-parsed authors + NER from first page
        all_authors_raw = list(paper.authors) + list(llm_authors)
        
        # NER on first page as additional fallback when we have very few authors
        if len(all_authors_raw) < 2:
            try:
                from src.ner_engine import extract_persons_from_text
                import fitz
                doc = fitz.open(file_path)
                first_page_text = doc[0].get_text() if len(doc) > 0 else ""
                doc.close()
                ner_names = extract_persons_from_text(first_page_text[:2000])
                all_authors_raw += [n for n in ner_names if 1 < len(n.split()) <= 5]
            except Exception:
                pass
        
        # Deduplicate (case-insensitive)
        seen_authors = set()
        all_authors = []
        for a in all_authors_raw:
            key = a.lower().strip()
            if key and key not in seen_authors:
                seen_authors.add(key)
                all_authors.append(a)
        
        for author_name in all_authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")
            
        # 4. Extract concepts and create MENTIONS_CONCEPT edges
        if llm_concepts:
            for item in llm_concepts:
                c_name = item.get("name")
                c_desc = item.get("description", "")
                if c_name:
                    concept_id = self._slugify(c_name)
                    concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")
        else:
            # Fallback to keyword-based concept extraction
            text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + full_text[:10000]).lower()
            for keyword, concept_name in AI_CONCEPTS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    concept_id = self._slugify(concept_name)
                    concept = Concept(id=concept_id, name=concept_name)
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

        # 4b. Save tag nodes and create HAS_TAG edges
        for tag in tags_to_save:
            tag_id = self._slugify(tag)
            concept = Concept(id=tag_id, name=tag, properties={"is_tag": True})
            self.graph_repo.save_concept(concept)
            self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

        # 5. Extract citations and create CITES edges
        if api_references or api_citations:
            for ref in api_references:
                ref_title = ref.get("title")
                ref_doi = ref.get("doi")
                ref_id = self._slugify(ref_doi) if ref_doi else self._slugify(ref_title[:120])
                if ref_id:
                    exists = self.graph_repo.get_paper(ref_id)
                    if not exists:
                        placeholder_paper = Paper(
                            id=ref_id,
                            title=ref_title,
                            authors=[],
                            year=None,
                            doi=ref_doi
                        )
                        self.graph_repo.save_paper(placeholder_paper)
                    self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"api_sourced": True})
                    
            for cit in api_citations:
                cit_title = cit.get("title")
                cit_doi = cit.get("doi")
                cit_id = self._slugify(cit_doi) if cit_doi else self._slugify(cit_title[:120])
                if cit_id:
                    exists = self.graph_repo.get_paper(cit_id)
                    if not exists:
                        placeholder_paper = Paper(
                            id=cit_id,
                            title=cit_title,
                            authors=[],
                            year=None,
                            doi=cit_doi
                        )
                        self.graph_repo.save_paper(placeholder_paper)
                    self.graph_repo.add_edge(cit_id, paper.id, "CITES", {"api_sourced": True})
        else:
            # Fallback to parsed raw references
            for ref_str in raw_references:
                ref_clean = ref_str.strip()
                if len(ref_clean) > 10:
                    ref_id = self._slugify(ref_clean[:120])
                    self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"raw_text": ref_clean})

        # 6. Chunk text and generate embeddings
        con.dim(f"Chunking and embedding: {paper.title[:60]}")
        chunks = split_text_to_chunks(paper.id, file_path)
        
        if chunks:
            chunk_texts = [c.text_content for c in chunks]
            embeddings = self.emb_engine.get_embeddings(chunk_texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            
            # Save chunks to vector store
            self.vector_repo.save_chunks(chunks)

        # 7. Move PDF to Archive
        if Path(file_path).resolve() != archive_path.resolve():
            shutil.copy2(file_path, archive_path)
            os.remove(file_path)
            print(f"[+] PDF moved to archive: {archive_path}")

        con.success(f"Indexed [bold]{paper.title[:70]}[/bold] (ID: {paper.id[:12]}…)")
        return paper.id

    # ──────────────────────────────────────────────────────────────────────────
    # Markdown (Obsidian notes)
    # ──────────────────────────────────────────────────────────────────────────

    def index_markdown(self, file_path: str) -> str:
        """
        Indexes a Markdown note (.md) into the knowledge graph.
        Returns the note ID.
        """
        from src.parsers.md_parser import parse_markdown

        con.info(f"Parsing note [bold]{os.path.basename(file_path)}[/bold]")
        paper, wiki_links, body = parse_markdown(file_path)

        # Tags from properties → concept nodes
        tags = paper.properties.get("tags", [])
        
        # Fallback keyword-based tag extraction if none in frontmatter
        if not tags:
            tags_to_save = []
            text_to_scan = (paper.title + " " + body[:10000]).lower()
            for keyword, tag_name in AI_TOPICS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    if tag_name not in tags_to_save:
                        tags_to_save.append(tag_name)
            paper.properties["tags"] = tags_to_save
            tags = tags_to_save

        # Save note node
        self.graph_repo.save_paper(paper)

        # Author nodes (if front-matter has authors)
        for author_name in paper.authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

        for tag in tags:
            concept_id = self._slugify(tag)
            concept = Concept(id=concept_id, name=tag, properties={"is_tag": True})
            self.graph_repo.save_concept(concept)
            self.graph_repo.add_edge(paper.id, concept_id, "HAS_TAG")

        # Wiki-links [[Target]] → paper/note nodes or fallback concept nodes + RELATED_TO edges
        for link_target in wiki_links:
            matched_paper = self.graph_repo.find_paper_by_title(link_target)
            if matched_paper:
                self.graph_repo.add_edge(paper.id, matched_paper.id, "RELATED_TO")
            else:
                concept_id = self._slugify(link_target)
                concept = Concept(id=concept_id, name=link_target)
                self.graph_repo.save_concept(concept)
                self.graph_repo.add_edge(paper.id, concept_id, "RELATED_TO")

        # Chunk + embed body text
        con.dim(f"Chunking note: {paper.title[:60]}")
        chunks = _split_text_to_chunks_raw(paper.id, body)
        if chunks:
            embeddings = self.emb_engine.get_embeddings([c.text_content for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self.vector_repo.save_chunks(chunks)

        con.success(f"Note indexed: [bold]{paper.title[:70]}[/bold]")
        return paper.id

    # ──────────────────────────────────────────────────────────────────────────
    # Webpages (URLs)
    # ──────────────────────────────────────────────────────────────────────────

    def index_url(self, url: str) -> str:
        """
        Indexes a webpage URL into the knowledge graph.
        Returns the page ID.
        """
        from src.parsers.url_parser import parse_url

        con.info(f"Parsing URL [bold]{url}[/bold]")
        paper, body = parse_url(url)

        # LLM-based concept & tag extraction if enabled
        llm_concepts = []
        llm_tags = []
        if self.llm_engine:
            con.dim("Extracting concepts via LLM …")
            sample_text = (paper.title or "") + "\n\n" + (paper.abstract or "") + "\n\n" + body[:4000]
            llm_data = self.llm_engine.extract_concepts_and_metadata(sample_text)
            if llm_data:
                for a in llm_data.get("authors", []):
                    if a not in paper.authors:
                        paper.authors.append(a)
                llm_concepts = llm_data.get("concepts", [])
                llm_tags = llm_data.get("tags", [])
                con.dim(f"LLM found {len(paper.authors)} total authors, {len(llm_concepts)} concepts, {len(llm_tags)} tags")

        # Fallback to keyword tags
        tags_to_save = []
        if llm_tags:
            tags_to_save = llm_tags
        else:
            text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + body[:10000]).lower()
            for keyword, tag_name in AI_TOPICS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    if tag_name not in tags_to_save:
                        tags_to_save.append(tag_name)
        paper.properties["tags"] = tags_to_save

        # Determine archival path for webpage
        from src.config import config
        from pathlib import Path
        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.md"
        try:
            with open(archive_path, "w", encoding="utf-8") as f:
                f.write(body)
            paper.file_path = str(archive_path)
            con.dim(f"Saved local archive of website to {archive_path}")
        except Exception as e:
            con.warning(f"Could not save local archive of website: {e}")

        # Save webpage node
        self.graph_repo.save_paper(paper)

        # Create Author nodes and AUTHORED edges
        for author_name in paper.authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

        # Save concepts
        if llm_concepts:
            for item in llm_concepts:
                c_name = item.get("name")
                c_desc = item.get("description", "")
                if c_name:
                    concept_id = self._slugify(c_name)
                    concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")
        else:
            # Keyword-based concept extraction
            text_to_scan = (paper.title + " " + body[:10000]).lower()
            for keyword, concept_name in AI_CONCEPTS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    concept_id = self._slugify(concept_name)
                    concept = Concept(id=concept_id, name=concept_name)
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

        # Save tags and create HAS_TAG edges
        for tag in tags_to_save:
            tag_id = self._slugify(tag)
            concept = Concept(id=tag_id, name=tag, properties={"is_tag": True})
            self.graph_repo.save_concept(concept)
            self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

        # Chunk + embed body text
        con.dim(f"Chunking webpage: {paper.title[:60]}")
        chunks = _split_text_to_chunks_raw(paper.id, body)
        if chunks:
            embeddings = self.emb_engine.get_embeddings([c.text_content for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self.vector_repo.save_chunks(chunks)

        con.success(f"Webpage indexed: [bold]{paper.title[:70]}[/bold]")
        return paper.id

    # ──────────────────────────────────────────────────────────────────────────
    # EPUB books
    # ──────────────────────────────────────────────────────────────────────────

    def index_epub(self, file_path: str) -> str:
        """
        Indexes an EPUB book into the knowledge graph.
        Returns the book ID.
        """
        from src.parsers.epub_parser import parse_epub

        con.info(f"Parsing EPUB [bold]{os.path.basename(file_path)}[/bold]")
        paper, _, full_text = parse_epub(file_path)

        # LLM-based concept & tag extraction if enabled
        llm_concepts = []
        llm_tags = []
        if self.llm_engine:
            con.dim("Extracting concepts via LLM …")
            sample_text = (paper.title or "") + "\n\n" + (paper.abstract or "") + "\n\n" + full_text[:4000]
            llm_data = self.llm_engine.extract_concepts_and_metadata(sample_text)
            if llm_data:
                for a in llm_data.get("authors", []):
                    if a not in paper.authors:
                        paper.authors.append(a)
                llm_concepts = llm_data.get("concepts", [])
                llm_tags = llm_data.get("tags", [])
                con.dim(f"LLM found {len(paper.authors)} total authors, {len(llm_concepts)} concepts, {len(llm_tags)} tags")

        # Fallback to keyword tags
        tags_to_save = []
        if llm_tags:
            tags_to_save = llm_tags
        else:
            text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + full_text[:10000]).lower()
            for keyword, tag_name in AI_TOPICS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    if tag_name not in tags_to_save:
                        tags_to_save.append(tag_name)
        paper.properties["tags"] = tags_to_save

        # Save book node
        self.graph_repo.save_paper(paper)

        # Author nodes
        for author_name in paper.authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

        # Save concepts
        if llm_concepts:
            for item in llm_concepts:
                c_name = item.get("name")
                c_desc = item.get("description", "")
                if c_name:
                    concept_id = self._slugify(c_name)
                    concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")
        else:
            # Keyword-based concept extraction (same fallback as PDF)
            text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + full_text[:10000]).lower()
            for keyword, concept_name in AI_CONCEPTS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    concept_id = self._slugify(concept_name)
                    concept = Concept(id=concept_id, name=concept_name)
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

        # Save tags and create HAS_TAG edges
        for tag in tags_to_save:
            tag_id = self._slugify(tag)
            concept = Concept(id=tag_id, name=tag, properties={"is_tag": True})
            self.graph_repo.save_concept(concept)
            self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")

        # Chunk + embed
        con.dim(f"Chunking book: {paper.title[:60]}")
        chunks = _split_text_to_chunks_raw(paper.id, full_text)
        if chunks:
            embeddings = self.emb_engine.get_embeddings([c.text_content for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self.vector_repo.save_chunks(chunks)

        con.success(f"Book indexed: [bold]{paper.title[:70]}[/bold] ({len(chunks)} chunks)")
        return paper.id

    def reindex_metadata(self, paper_id: str, use_llm: bool = False) -> bool:
        """
        Re-indexes metadata for a specific paper without regenerating embeddings.
        Fetches updated metadata from Semantic Scholar using DOI, Title, or arXiv ID,
        runs NER/LLM to extract authors, concepts, and tags,
        and updates the nodes & edges in the graph database.
        """
        paper = self.graph_repo.get_paper(paper_id)
        if not paper:
            con.error(f"Paper not found in database: {paper_id}")
            return False

        con.info(f"Re-indexing metadata for [bold]{paper.title[:60]}[/bold] (ID: {paper_id})")

        # Determine if we can get data from Semantic Scholar
        doi = paper.doi or paper.properties.get("doi")
        arxiv_id = paper.properties.get("arxiv_id")
        title = paper.title
        
        is_webpage = paper.properties.get("source_type") == "webpage" or (paper.file_path and paper.file_path.startswith("http"))
        url = paper.properties.get("url") or paper.file_path if is_webpage else None
        
        # 1. Fetch updated metadata from external source
        if url:
            try:
                from src.parsers.url_parser import parse_url
                # Re-parse webpage to extract metadata
                web_paper, _ = parse_url(url)
                if web_paper.authors:
                    paper.authors = list(set(paper.authors + web_paper.authors))
                if web_paper.doi:
                    doi = web_paper.doi
                if web_paper.properties.get("arxiv_id"):
                    arxiv_id = web_paper.properties.get("arxiv_id")
                if web_paper.title and len(web_paper.title) > len(paper.title or ""):
                    paper.title = web_paper.title
                if web_paper.abstract:
                    paper.abstract = web_paper.abstract
                if web_paper.year:
                    paper.year = web_paper.year
            except Exception as e:
                con.warning(f"Failed to re-parse URL {url}: {e}")

        # Fetch from Semantic Scholar
        from src.external_api import fetch_paper_metadata
        con.dim("Querying Semantic Scholar for updated metadata …")
        api_meta = fetch_paper_metadata(doi=doi, title=title, arxiv_id=arxiv_id)
        
        api_references = []
        api_citations = []
        
        if api_meta:
            if api_meta.get("title"):
                paper.title = api_meta["title"]
            if api_meta.get("authors"):
                paper.authors = api_meta["authors"]
            if api_meta.get("year"):
                paper.year = api_meta["year"]
            if api_meta.get("abstract"):
                paper.abstract = api_meta["abstract"]
            if api_meta.get("doi"):
                paper.doi = api_meta["doi"]
            api_references = api_meta.get("references", [])
            api_citations = api_meta.get("citations", [])
            con.success(f"Metadata enriched from Semantic Scholar: {paper.title[:60]}")
            
        # 2. Extract concepts and tags
        llm_authors = []
        llm_concepts = []
        llm_tags = []
        
        # Read text from file if available to use for extraction
        full_text = ""
        if paper.file_path and os.path.exists(paper.file_path) and not paper.file_path.startswith("http"):
            try:
                if paper.file_path.lower().endswith(".pdf"):
                    from src.parser import PDFParser
                    # Only read text
                    _, _, full_text = PDFParser.extract_text_and_metadata(paper.file_path)
                elif paper.file_path.lower().endswith(".md"):
                    with open(paper.file_path, "r", encoding="utf-8") as f:
                        full_text = f.read()
                elif paper.file_path.lower().endswith(".epub"):
                    from src.parsers.epub_parser import parse_epub
                    _, _, full_text = parse_epub(paper.file_path)
            except Exception as e:
                con.warning(f"Could not read local file {paper.file_path} for concept extraction: {e}")
                
        text_for_extraction = full_text or (paper.title or "") + "\n\n" + (paper.abstract or "")
        
        if use_llm and self.llm_engine:
            con.dim("Extracting concepts/tags via LLM …")
            llm_data = self.llm_engine.extract_concepts_and_metadata(text_for_extraction[:5000])
            if llm_data:
                llm_authors = llm_data.get("authors", [])
                llm_concepts = llm_data.get("concepts", [])
                llm_tags = llm_data.get("tags", [])
                con.dim(f"LLM found {len(llm_authors)} authors, {len(llm_concepts)} concepts, {len(llm_tags)} tags")

        # Combine authors
        all_authors_raw = list(paper.authors) + list(llm_authors)
        
        # NER fallback for PDF papers
        if len(all_authors_raw) < 2 and paper.file_path and os.path.exists(paper.file_path) and paper.file_path.lower().endswith(".pdf"):
            try:
                from src.ner_engine import extract_persons_from_text
                import fitz
                doc = fitz.open(paper.file_path)
                first_page_text = doc[0].get_text() if len(doc) > 0 else ""
                doc.close()
                ner_names = extract_persons_from_text(first_page_text[:2000])
                all_authors_raw += [n for n in ner_names if 1 < len(n.split()) <= 5]
            except Exception:
                pass
                
        seen_authors = set()
        all_authors = []
        for a in all_authors_raw:
            key = a.lower().strip()
            if key and key not in seen_authors:
                seen_authors.add(key)
                all_authors.append(a)
                
        paper.authors = all_authors
        
        # Determine tags
        tags_to_save = []
        if llm_tags:
            tags_to_save = llm_tags
        else:
            text_to_scan = text_for_extraction.lower()
            for keyword, tag_name in AI_TOPICS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    if tag_name not in tags_to_save:
                        tags_to_save.append(tag_name)
        paper.properties["tags"] = tags_to_save
        
        # Save updated Paper Node
        self.graph_repo.save_paper(paper)
        
        # Clean old AUTHORED edges first
        with self.graph_repo._get_connection() as conn:
            conn.execute("DELETE FROM edges WHERE target_id = ? AND type = 'AUTHORED'", (paper.id,))
            conn.commit()

        for author_name in all_authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")
            
        # Clean old MENTIONS_CONCEPT and HAS_TAG edges
        with self.graph_repo._get_connection() as conn:
            conn.execute("DELETE FROM edges WHERE source_id = ? AND type IN ('MENTIONS_CONCEPT', 'HAS_TAG')", (paper.id,))
            conn.commit()
            
        if llm_concepts:
            for item in llm_concepts:
                c_name = item.get("name")
                c_desc = item.get("description", "")
                if c_name:
                    concept_id = self._slugify(c_name)
                    concept = Concept(id=concept_id, name=c_name, properties={"description": c_desc})
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")
        else:
            text_to_scan = text_for_extraction.lower()
            for keyword, concept_name in AI_CONCEPTS.items():
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                    concept_id = self._slugify(concept_name)
                    concept = Concept(id=concept_id, name=concept_name)
                    self.graph_repo.save_concept(concept)
                    self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")
                    
        # Save tags
        for tag in tags_to_save:
            tag_id = self._slugify(tag)
            concept = Concept(id=tag_id, name=tag, properties={"is_tag": True})
            self.graph_repo.save_concept(concept)
            self.graph_repo.add_edge(paper.id, tag_id, "HAS_TAG")
            
        # Update citations
        if api_references or api_citations:
            for ref in api_references:
                ref_title = ref.get("title")
                ref_doi = ref.get("doi")
                ref_id = self._slugify(ref_doi) if ref_doi else self._slugify(ref_title[:120])
                if ref_id:
                    exists = self.graph_repo.get_paper(ref_id)
                    if not exists:
                        placeholder_paper = Paper(id=ref_id, title=ref_title, authors=[], year=None, doi=ref_doi)
                        self.graph_repo.save_paper(placeholder_paper)
                    self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"api_sourced": True})
                    
            for cit in api_citations:
                cit_title = cit.get("title")
                cit_doi = cit.get("doi")
                cit_id = self._slugify(cit_doi) if cit_doi else self._slugify(cit_title[:120])
                if cit_id:
                    exists = self.graph_repo.get_paper(cit_id)
                    if not exists:
                        placeholder_paper = Paper(id=cit_id, title=cit_title, authors=[], year=None, doi=cit_doi)
                        self.graph_repo.save_paper(placeholder_paper)
                    self.graph_repo.add_edge(cit_id, paper.id, "CITES", {"api_sourced": True})

        con.success(f"Successfully re-indexed metadata for {paper.title[:60]}")
        return True

