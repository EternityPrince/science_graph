import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from src.config import config
from src.models import Chunk

class EmbeddingEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model_name
        # SentenceTransformer automatically handles MPS/CUDA/CPU device placement
        self.model = SentenceTransformer(self.model_name)

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding for a single text string."""
        emb = self.model.encode(text, convert_to_numpy=True)
        return emb.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        if not texts:
            return []
        embs = self.model.encode(texts, convert_to_numpy=True)
        return embs.tolist()


def split_text_to_chunks(paper_id: str, file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
    """
    Reads PDF page by page and splits it into overlapping text chunks.
    Preserves page number references for each chunk.
    """
    import fitz  # PyMuPDF
    
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap
    
    doc = fitz.open(file_path)
    chunks = []
    chunk_idx = 0
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        # Basic cleanup: replace multiple spaces/newlines
        text_clean = re.sub(r'\s+', ' ', text).strip()
        
        if not text_clean:
            continue
            
        # If the page text is shorter than chunk_size, keep it as one chunk
        if len(text_clean) <= chunk_size:
            chunks.append(Chunk(
                id=f"{paper_id}#{chunk_idx}",
                paper_id=paper_id,
                text_content=text_clean,
                page_number=page_num
            ))
            chunk_idx += 1
            continue
            
        # Slide window across the page text
        start = 0
        while start < len(text_clean):
            end = start + chunk_size
            chunk_text = text_clean[start:end]
            
            # If the last chunk is too small, we can discard it or merge it if it is very short,
            # but standard windowing is fine
            if len(chunk_text) > 50 or start == 0:
                chunks.append(Chunk(
                    id=f"{paper_id}#{chunk_idx}",
                    paper_id=paper_id,
                    text_content=chunk_text,
                    page_number=page_num
                ))
                chunk_idx += 1
                
            start += (chunk_size - chunk_overlap)
            
    return chunks
