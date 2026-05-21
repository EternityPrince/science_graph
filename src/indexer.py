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
        if self.llm_engine:
            con.dim("Extracting concepts via LLM …")
            # Combine abstract/introduction
            sample_text = (paper.title or "") + "\n\n" + (paper.abstract or "") + "\n\n" + full_text[:4000]
            llm_data = self.llm_engine.extract_concepts_and_metadata(sample_text)
            if llm_data:
                llm_authors = llm_data.get("authors", [])
                llm_concepts = llm_data.get("concepts", [])
                con.dim(f"LLM found {len(llm_authors)} authors, {len(llm_concepts)} concepts")
        
        # Determine archival path
        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.pdf"
        paper.file_path = str(archive_path)
        
        # 2. Save Paper Node
        self.graph_repo.save_paper(paper)
        
        # 3. Create Author nodes & AUTHORED edges
        all_authors = list(set(paper.authors + llm_authors))
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

        # Save note node
        self.graph_repo.save_paper(paper)

        # Author nodes (if front-matter has authors)
        for author_name in paper.authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

        # Tags from properties → concept nodes
        tags: List[str] = paper.properties.get("tags", [])
        for tag in tags:
            concept_id = self._slugify(tag)
            concept = Concept(id=concept_id, name=tag)
            self.graph_repo.save_concept(concept)
            self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

        # Wiki-links [[Target]] → concept nodes + RELATED_TO edges
        for link_target in wiki_links:
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

        # Save webpage node
        self.graph_repo.save_paper(paper)

        # Keyword-based concept extraction
        text_to_scan = (paper.title + " " + body[:10000]).lower()
        for keyword, concept_name in AI_CONCEPTS.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                concept_id = self._slugify(concept_name)
                concept = Concept(id=concept_id, name=concept_name)
                self.graph_repo.save_concept(concept)
                self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

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

        # Save book node
        self.graph_repo.save_paper(paper)

        # Author nodes
        for author_name in paper.authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")

        # Keyword-based concept extraction (same fallback as PDF)
        text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + full_text[:10000]).lower()
        for keyword, concept_name in AI_CONCEPTS.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                concept_id = self._slugify(concept_name)
                concept = Concept(id=concept_id, name=concept_name)
                self.graph_repo.save_concept(concept)
                self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

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
