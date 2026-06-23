import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Global cache for embedding engine to prevent redundant loads
_embedding_engine = None


def get_embedding_engine():
    """Lazily loads and returns the EmbeddingEngine from the parent src codebase."""
    global _embedding_engine
    if _embedding_engine is None:
        try:
            # Adjust path to resolve src imports
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from src.vector_search import EmbeddingEngine
            _embedding_engine = EmbeddingEngine()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not load EmbeddingEngine: {e}")
            raise e
    return _embedding_engine


def calculate_retrieval_recall(expected_papers: List[str], retrieved_papers: List[str]) -> float:
    """Computes retrieval recall: proportion of expected papers that were successfully retrieved."""
    if not expected_papers:
        return 1.0
    expected_set = {p.strip().lower() for p in expected_papers if p.strip()}
    retrieved_set = {p.strip().lower() for p in retrieved_papers if p.strip()}
    if not expected_set:
        return 1.0
    intersection = expected_set.intersection(retrieved_set)
    return round(len(intersection) / len(expected_set), 4)


def calculate_context_precision(expected_papers: List[str], retrieved_chunks: List[Dict[str, Any]]) -> float:
    """Computes context precision (Mean Average Precision on paper retrieval relevance at K)."""
    if not expected_papers:
        return 1.0
    expected_set = {p.strip().lower() for p in expected_papers if p.strip()}
    if not expected_set:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    precision_sum = 0.0
    relevant_hits = 0
    for idx, chunk in enumerate(retrieved_chunks):
        paper_id = chunk.get("paper_id", "")
        if paper_id and paper_id.strip().lower() in expected_set:
            relevant_hits += 1
            precision_sum += relevant_hits / (idx + 1)
            
    if relevant_hits == 0:
        return 0.0
    return round(precision_sum / relevant_hits, 4)


def compute_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes the cosine similarity between two vector embeddings."""
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    norm1 = np.linalg.norm(v1_arr)
    norm2 = np.linalg.norm(v2_arr)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1_arr, v2_arr) / (norm1 * norm2))


def calculate_semantic_accuracy(golden_answers: List[str], generated_answers: List[str]) -> List[float]:
    """Generates embeddings for lists of golden and generated answers and calculates cosine similarity."""
    if not golden_answers or not generated_answers:
        return []
    
    engine = get_embedding_engine()
    gold_embs = engine.get_embeddings(golden_answers, is_query=False)
    gen_embs = engine.get_embeddings(generated_answers, is_query=False)
    
    similarities = []
    for v1, v2 in zip(gold_embs, gen_embs):
        similarities.append(compute_cosine_similarity(v1, v2))
    return similarities


def count_text_tokens(text: str) -> int:
    """Counts token count in text using tiktoken or simple character heuristic."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    # Simple heuristic fallback (approx. 4 characters per token for English/Russian mixed)
    return max(1, len(text) // 4)


def estimate_prompt_tokens(query: str, retrieved_chunks: List[Dict[str, Any]], baseline: str) -> int:
    """Estimates prompt token count using tiktoken or simple character heuristic."""
    if baseline == "B0":
        prompt = f"Вопрос: {query}\nОтветь на основе своих общих знаний."
    else:
        system_prompt = (
            "<|im_start|>system\n"
            "You are a research assistant. Synthesize an answer to the user's question using the retrieved text blocks and the knowledge graph connections.\n"
            "Always mention the titles of the papers, years, authors, and page numbers when citation is needed.\n"
            "If the graph contains citing relationships, use them to explain the context (e.g., \"A cited B\").\n\n"
            "Here is the retrieved context:\n\n"
            "### RELEVANT TEXT FRAGMENTS:\n"
        )
        text_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            text_content = chunk.get("text_content", "").strip()
            paper_id = chunk.get("paper_id", "")
            page = chunk.get("page_number", "")
            text_blocks.append(
                f"Block {idx} (Score: 1.000) | Paper: {paper_id} (Page {page}):\n"
                f"\"\"\"\n{text_content}\n\"\"\""
            )
        context_text = "\n\n".join(text_blocks)
        context_graph = "No direct graph relations found."
        prompt = (
            f"{system_prompt}{context_text}\n\n"
            f"### KNOWLEDGE GRAPH CONNECTIONS:\n{context_graph}\n"
            f"<|im_end|>\n<|im_start|>user\nQuestion: {query}\nAnswer in Russian:\n<|im_end|>\n<|im_start|>assistant\n"
        )
    
    return count_text_tokens(prompt)
