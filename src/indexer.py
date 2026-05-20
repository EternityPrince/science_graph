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
    def __init__(self, graph_repo: GraphRepository, vector_repo: VectorRepository, embedding_engine: EmbeddingEngine):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine

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
        print(f"[*] Parsing PDF structure: {os.path.basename(file_path)}")
        paper, raw_references, full_text = PDFParser.extract_text_and_metadata(file_path)
        
        # Determine archival path
        archive_dir = Path(config.archive_dir)
        archive_path = archive_dir / f"{paper.id}.pdf"
        paper.file_path = str(archive_path)
        
        # 2. Save Paper Node
        self.graph_repo.save_paper(paper)
        
        # 3. Create Author nodes & AUTHORED edges
        for author_name in paper.authors:
            author_id = self._slugify(author_name)
            author = Author(id=author_id, name=author_name)
            self.graph_repo.save_author(author)
            self.graph_repo.add_edge(author_id, paper.id, "AUTHORED")
            
        # 4. Extract concepts and create MENTIONS_CONCEPT edges
        text_to_scan = (paper.title + " " + (paper.abstract or "") + " " + full_text[:10000]).lower()
        for keyword, concept_name in AI_CONCEPTS.items():
            # Use word boundaries to search
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_to_scan):
                concept_id = self._slugify(concept_name)
                concept = Concept(id=concept_id, name=concept_name)
                self.graph_repo.save_concept(concept)
                self.graph_repo.add_edge(paper.id, concept_id, "MENTIONS_CONCEPT")

        # 5. Extract citations and create CITES edges
        for ref_str in raw_references:
            # We slugify the reference string to use as node ID.
            # If the reference contains a title or key phrase, we grab it.
            # For simplicity, we hash the first 100 characters of clean reference as placeholder ID
            ref_clean = ref_str.strip()
            if len(ref_clean) > 10:
                # Attempt to extract title/doi from reference string
                ref_id = self._slugify(ref_clean[:120])
                self.graph_repo.add_edge(paper.id, ref_id, "CITES", {"raw_text": ref_clean})

        # 6. Chunk text and generate embeddings
        print(f"[*] Chunking and generating embeddings for: {paper.title}")
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
            # Remove original file (mv instead of rm, as user requested "mv, not rm")
            # Wait, the user said: "original PDF can be deleted or moved (mv, not rm)"
            # So moving is perfect.
            os.remove(file_path)
            print(f"[+] PDF moved to archive: {archive_path}")

        print(f"[+] Successfully indexed paper: {paper.title} (ID: {paper.id})")
        return paper.id
