import re
import time
import fitz
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi
from src.config import config
from src.models import Chunk

class EmbeddingEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model_name
        self.model = None
        self._query_cache = {}

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

    def unload_model(self):
        if self.model is not None:
            import gc
            self.model = None
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            from src import console as con
            con.success("EmbeddingEngine model unloaded and GPU cache cleared")

    def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """Generates embedding for a single text string."""
        if is_query and text in self._query_cache:
            return self._query_cache[text]

        self._ensure_model_loaded()
        
        # Prepend query: or passage: prefix for E5 models if not already present
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            if not text.startswith(prefix):
                text = prefix + text
                
        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        emb_list = emb.tolist()
        if is_query:
            self._query_cache[text] = emb_list
        return emb_list

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        if not texts:
            return []

        # If is_query, check which texts are already cached
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for idx, text in enumerate(texts):
            if is_query and text in self._query_cache:
                results[idx] = self._query_cache[text]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            self._ensure_model_loaded()
            
            # Prepend query: or passage: prefix for E5 models if not already present
            processed_texts = []
            if "e5" in self.model_name.lower():
                prefix = "query: " if is_query else "passage: "
                for t in uncached_texts:
                    if not t.startswith(prefix):
                        processed_texts.append(prefix + t)
                    else:
                        processed_texts.append(t)
            else:
                processed_texts = uncached_texts

            embs = self.model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=False)
            embs_list = embs.tolist()
            
            for idx, emb_val in zip(uncached_indices, embs_list):
                results[idx] = emb_val
                if is_query:
                    self._query_cache[texts[idx]] = emb_val
                    
        return results



def split_text_to_chunks(paper_id: str, file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
    """
    Reads PDF page by page and splits it into overlapping text chunks using sentence-aware boundary detection.
    Preserves page number references for each chunk.
    Keeps Markdown tables and LaTeX equations intact using structure-aware block extraction.
    """
    import logging
    t0 = time.perf_counter()
    
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else 50
    
    doc = fitz.open(file_path)
    chunks = []
    chunk_idx = 0
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        
        # 1. Pre-extract non-splittable structures (Markdown tables & LaTeX equations)
        # Markdown tables
        table_pattern = re.compile(r'((?:\n\|[^\n]+\|\s*)+)', re.MULTILINE)
        tables = table_pattern.findall(text)
        
        # LaTeX equations (explicitly marked block/inline: $$, \begin{equation}, \[, \()
        equation_pattern = re.compile(
            r'(\$\$.*?\$\$|\\begin\{equation\}.*?\\end\{equation\}|\\\[.*?\\\]|\\\(.*?\\\))', 
            re.DOTALL
        )
        equations = equation_pattern.findall(text)
        
        # Replace non-splittable blocks with placeholders to protect them from splitting
        non_splittable = []
        text_clean = text
        
        for idx, table in enumerate(tables):
            placeholder = f" __STRUCT_BLOCK_TABLE_{idx}__ "
            non_splittable.append((placeholder.strip(), table))
            text_clean = text_clean.replace(table, placeholder)
            
        for idx, eq in enumerate(equations):
            placeholder = f" __STRUCT_BLOCK_EQ_{idx}__ "
            non_splittable.append((placeholder.strip(), eq))
            text_clean = text_clean.replace(eq, placeholder)

        # Inline dollar equations $...$ with filtering
        inline_dollar_pattern = re.compile(r'\$([^\$\n]+)\$')
        inline_dollar_idx = 0
        
        def is_valid_latex_inline(content: str) -> bool:
            if not content.strip():
                return False
            # Simple number or currency matching (e.g. 100, 10.50, 100,000, 50k, 20m)
            if re.match(r'^\s*\d+[\d\s\.,]*(?:[kKmMbB]|\s*billion|\s*million)?\s*$', content):
                return False
            if ' ' in content:
                # If it has spaces, it must contain mathematical operators, brackets or backslashes
                math_indicators = r'[\+\-\*/=\\_^<>\(\)\[\]\{\}\\\|~]'
                if not re.search(math_indicators, content):
                    return False
            return True

        def dollar_replacer(match):
            nonlocal inline_dollar_idx
            content = match.group(1)
            if is_valid_latex_inline(content):
                placeholder = f" __STRUCT_BLOCK_INLINE_DOLLAR_{inline_dollar_idx}__ "
                non_splittable.append((placeholder.strip(), match.group(0)))
                inline_dollar_idx += 1
                return placeholder
            return match.group(0)

        text_clean = inline_dollar_pattern.sub(dollar_replacer, text_clean)
            
        # Basic cleanup: replace multiple spaces/newlines
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        
        if not text_clean:
            continue
            
        # If the page text is shorter than chunk_size, keep it as one chunk
        page_chunks = []
        
        if len(text_clean) <= chunk_size:
            page_chunks.append(Chunk(
                id=f"{paper_id}#{chunk_idx}",
                paper_id=paper_id,
                text_content=text_clean,
                page_number=page_num
            ))
            chunk_idx += 1
        else:
            # Split on sentence boundaries
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_clean) if s.strip()]
            
            current_chunk = []
            current_len = 0
            for sentence in sentences:
                sentence_len = len(sentence)
                if current_len + sentence_len > chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    page_chunks.append(Chunk(
                        id=f"{paper_id}#{chunk_idx}",
                        paper_id=paper_id,
                        text_content=chunk_text,
                        page_number=page_num
                    ))
                    chunk_idx += 1
                    
                    # Setup overlap sentences
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        s_len = len(s)
                        if overlap_len + s_len + (1 if overlap_sentences else 0) <= chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += s_len + (1 if overlap_sentences else 0)
                        else:
                            break
                    current_chunk = overlap_sentences + [sentence]
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                else:
                    if sentence_len > chunk_size:
                        # Split extremely long sentences into character-based sub-chunks
                        start = 0
                        while start < sentence_len:
                            end = start + chunk_size
                            sub_text = sentence[start:end].strip()
                            if sub_text:
                                page_chunks.append(Chunk(
                                    id=f"{paper_id}#{chunk_idx}",
                                    paper_id=paper_id,
                                    text_content=sub_text,
                                    page_number=page_num
                                ))
                                chunk_idx += 1
                            start += (chunk_size - chunk_overlap)
                        current_chunk = []
                        current_len = 0
                    else:
                        current_chunk.append(sentence)
                        current_len += sentence_len + (1 if len(current_chunk) > 1 else 0)
                    
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                page_chunks.append(Chunk(
                    id=f"{paper_id}#{chunk_idx}",
                    paper_id=paper_id,
                    text_content=chunk_text,
                    page_number=page_num
                ))
                chunk_idx += 1

        # 2. Substitute non-splittable structures back into chunks
        for chunk in page_chunks:
            for placeholder, block in non_splittable:
                if placeholder in chunk.text_content:
                    # Resolve placeholder to original structure block
                    chunk.text_content = chunk.text_content.replace(placeholder, block)
            chunks.append(chunk)

    dt = time.perf_counter() - t0
    logging.debug(f"split_text_to_chunks for paper {paper_id} completed in {dt:.6f}s")
    return chunks


class BM25(BM25Okapi):
    """
    A long-lived, rank_bm25-based BM25 implementation supporting incremental updates.
    """
    def __init__(self, corpus: List[Tuple[str, str]], k1: Optional[float] = None, b: Optional[float] = None):
        # Store original mapping
        self.doc_ids: List[str] = []
        self.doc_texts: Dict[str, str] = {}
        self.nd: Dict[str, int] = {}
        
        k1_val = k1 if k1 is not None else config.bm25_k1
        b_val = b if b is not None else config.bm25_b
        
        tokenized_corpus = []
        for doc_id, text in corpus:
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text
            words = self._tokenize(text)
            tokenized_corpus.append(words)
            
        super().__init__(tokenized_corpus, k1=k1_val, b=b_val)
        
        # Populating self.nd from base class's doc_freqs
        for tf in self.doc_freqs:
            for word in tf.keys():
                self.nd[word] = self.nd.get(word, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in re.findall(r'\w+', text.lower()) if w]

    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Incrementally updates the corpus without full re-tokenization.
        """
        if not documents:
            return
            
        for doc_id, text in documents:
            if doc_id in self.doc_texts:
                continue  # Avoid duplicate document IDs
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text
            
            words = self._tokenize(text)
            self.doc_len.append(len(words))
            self.corpus_size += 1
            
            frequencies = {}
            for word in words:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for word in frequencies.keys():
                self.nd[word] = self.nd.get(word, 0) + 1
                
        total_len = sum(self.doc_len)
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0
        self._calc_idf(self.nd)

    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        Scores all documents in the corpus for the given query.
        Returns a sorted list of (doc_id, score) tuples in descending order.
        """
        import time
        import logging
        t0 = time.perf_counter()
        
        tokenized_query = self._tokenize(query)
        scores = self.get_scores(tokenized_query)
        results = [(self.doc_ids[i], float(scores[i])) for i in range(len(self.doc_ids))]
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        dt = time.perf_counter() - t0
        logging.debug(f"BM25 score completed in {dt:.6f}s for query: {query!r}")
        return sorted_results
