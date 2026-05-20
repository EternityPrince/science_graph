import json
from typing import List, Dict, Any, Tuple
from src.models import Chunk, Paper
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import LLMEngine

class RAGPipeline:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: LLMEngine
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from sentence_transformers import CrossEncoder
        print("[*] Loading local Cross-Encoder reranker (mixedbread-ai/mxbai-rerank-xsmall-v1)...")
        # Lightweight and fast reranker model
        self._reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
        return self._reranker

    def _resolve_node_name(self, node_id: str, label: str) -> str:
        """Resolves node ID to its actual name/title using DB lookup."""
        if label == "Paper":
            paper = self.graph_repo.get_paper(node_id)
            return f"'{paper.title}'" if paper else f"'{node_id}'"
        elif label == "Author":
            author = self.graph_repo.get_author(node_id)
            return author.name if author else node_id
        elif label == "Concept":
            concept = self.graph_repo.get_concept(node_id)
            return concept.name if concept else node_id
        return node_id

    def build_context(self, similar_chunks: List[tuple[Chunk, float]]) -> Tuple[str, str]:
        """
        Builds two context blocks:
        1. Semantic text blocks
        2. Knowledge graph relationships
        """
        # 1. Format text chunks
        text_blocks = []
        paper_ids = set()
        
        for idx, (chunk, score) in enumerate(similar_chunks, start=1):
            paper_ids.add(chunk.paper_id)
            paper = self.graph_repo.get_paper(chunk.paper_id)
            
            title = paper.title if paper else chunk.paper_id
            year_str = f", {paper.year}" if paper and paper.year else ""
            authors_str = f" by {', '.join(paper.authors)}" if paper and paper.authors else ""
            
            text_blocks.append(
                f"Block {idx} (Score: {score:.3f}) | Paper: {title}{authors_str}{year_str} (Page {chunk.page_number}):\n"
                f"\"\"\"\n{chunk.text_content.strip()}\n\"\"\""
            )
            
        context_text = "\n\n".join(text_blocks)

        # 2. Format Graph Subgraph around the relevant papers
        graph_lines = []
        seen_edges = set()
        
        for paper_id in paper_ids:
            neighbors = self.graph_repo.get_neighbors(paper_id, max_depth=1)
            for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props in neighbors:
                edge_key = (src_id, tgt_id, edge_type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    
                    # Resolve node display names
                    src_name = self._resolve_node_name(src_id, src_label)
                    tgt_name = self._resolve_node_name(tgt_id, tgt_label)
                    
                    if edge_type == "AUTHORED":
                        graph_lines.append(f"- {src_name} (Author) authored paper {tgt_name}")
                    elif edge_type == "MENTIONS_CONCEPT":
                        graph_lines.append(f"- Paper {src_name} mentions concept/topic '{tgt_name}'")
                    elif edge_type == "CITES":
                        try:
                            props = json.loads(edge_props) if edge_props else {}
                        except Exception:
                            props = {}
                        raw_text = props.get("raw_text")
                        if raw_text:
                            # Truncate raw reference for cleanliness
                            ref_preview = raw_text if len(raw_text) < 100 else raw_text[:100] + "..."
                            graph_lines.append(f"- Paper {src_name} cites: {ref_preview}")
                        else:
                            graph_lines.append(f"- Paper {src_name} cites paper {tgt_name}")
                    else:
                        graph_lines.append(f"- Node '{src_name}' is connected to '{tgt_name}' via {edge_type}")

        context_graph = "\n".join(graph_lines) if graph_lines else "No direct graph relations found."
        
        return context_text, context_graph

    def ask(self, query: str, limit: int = 5, history_str: str = "") -> str:
        """Runs hybrid search + Cross-Encoder reranking + graph retrieval and generates answers from the LLM."""
        # 1. Compute embedding of the query
        query_emb = self.emb_engine.get_embedding(query)
        
        # 2. Get all chunks for BM25
        all_chunks = self.vector_repo.get_all_chunks()
        if not all_chunks:
            return "Не найдено релевантных фрагментов статей в базе данных. Пожалуйста, сначала проиндексируйте документы."
            
        # 2b. Perform Dense Search
        dense_results = self.vector_repo.search_similar_chunks(query_emb, limit=limit * 2)
        
        # 2c. Perform BM25 Search
        from src.vector_search import BM25
        corpus = [(c.id, c.text_content) for c in all_chunks]
        bm25 = BM25(corpus)
        bm25_results = bm25.score(query)[:limit * 2]
        
        # 2d. Merge with Reciprocal Rank Fusion (RRF)
        id_to_chunk = {c.id: c for c in all_chunks}
        rrf_scores = {}
        
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (60.0 + rank))
            
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank))
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        candidate_ids = sorted_ids[:limit * 2]
        candidates = [id_to_chunk[cid] for cid in candidate_ids]
        
        # 2e. Rerank with Cross-Encoder
        final_chunks = []
        if candidates:
            try:
                reranker = self._get_reranker()
                pairs = [(query, c.text_content) for c in candidates]
                scores = reranker.predict(pairs)
                
                scored_candidates = list(zip(candidates, scores))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                final_chunks = [(chunk, float(score)) for chunk, score in scored_candidates[:limit]]
                print(f"[+] Reranked top {len(final_chunks)} chunks using Cross-Encoder.")
            except Exception as e:
                print(f"[!] Cross-Encoder reranking failed ({e}), falling back to RRF candidates.")
                final_chunks = [(id_to_chunk[cid], rrf_scores[cid]) for cid in sorted_ids[:limit]]
        else:
            final_chunks = []

        if not final_chunks:
            return "Не найдено релевантных фрагментов статей в базе данных. Пожалуйста, сначала проиндексируйте документы."

        # 3. Build context
        context_text, context_graph = self.build_context(final_chunks)

        # 4. Construct prompt for Gemma / Qwen
        prompt = f"""<|im_start|>system
You are a research assistant. Synthesize an answer to the user's question using the retrieved text blocks and the knowledge graph connections.
Always mention the titles of the papers, years, authors, and page numbers when citation is needed.
If the graph contains citing relationships, use them to explain the context (e.g., "A cited B").

Here is the retrieved context:

### RELEVANT TEXT FRAGMENTS:
{context_text}

### KNOWLEDGE GRAPH CONNECTIONS:
{context_graph}
<|im_end|>
{history_str}<|im_start|>user
Question: {query}
Answer in Russian:
<|im_end|>
<|im_start|>assistant
"""
        # 5. Generate completion
        print("[*] Generating answer using local LLM...")
        return self.llm_engine.generate_response(prompt)
