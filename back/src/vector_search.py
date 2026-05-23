import re
import math
from typing import List, Tuple
from src.config import config
from src.models import Chunk

class EmbeddingEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model_name
        self.model = None

    def _ensure_model_loaded(self):
        if self.model is None:
            from src import console as con
            short = self.model_name.split("/")[-1]
            con.model_msg(f"Loading embeddings [bold]{short}[/bold] …")
            with con.suppress_stderr(), con.suppress_stdout():
                import torch
                from sentence_transformers import SentenceTransformer
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                self.model = SentenceTransformer(self.model_name, device=device)
            con.success(f"Embeddings ready: [bold]{short}[/bold] on {device.upper()}")

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding for a single text string."""
        self._ensure_model_loaded()
        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return emb.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        if not texts:
            return []
        self._ensure_model_loaded()
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
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


class BM25:
    """A pure-Python implementation of BM25 search."""
    def __init__(self, corpus: List[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        """
        corpus: List of Tuple[chunk_id, text_content]
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = {}
        self.doc_term_freqs = {}  # doc_id -> term -> freq
        self.doc_ids = []
        self.doc_texts = {}
        
        # Calculate doc lengths and term frequencies
        total_len = 0
        self.doc_freqs = {}  # term -> count of docs containing term
        
        for doc_id, text in corpus:
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text
            # Simple tokenization: lowercase, alphanumeric words
            words = [w for w in re.findall(r'\w+', text.lower()) if w]
            doc_len = len(words)
            self.doc_lens[doc_id] = doc_len
            total_len += doc_len
            
            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            self.doc_term_freqs[doc_id] = tf
            
            # Document frequency (doc count containing term)
            for w in tf.keys():
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        # Precompute IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # BM25 IDF formula with smoothing to avoid negative values
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[str, float]]:
        """Scores all documents in the corpus for the given query."""
        query_terms = [w for w in re.findall(r'\w+', query.lower()) if w]
        scores = []
        
        for doc_id in self.doc_ids:
            score = 0.0
            doc_len = self.doc_lens[doc_id]
            tf = self.doc_term_freqs[doc_id]
            
            for term in query_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                idf = self.idf.get(term, 0.0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (numerator / denominator)
                
            scores.append((doc_id, score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)
