import json
import asyncio
from typing import List, Tuple, AsyncGenerator, Any, Optional
from src.models import Chunk
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import BaseLLMEngine
from src.config import config
from src import console as con

class RAGService:
    def __init__(
        self,
        graph_repo: GraphRepository,
        vector_repo: VectorRepository,
        embedding_engine: EmbeddingEngine,
        llm_engine: BaseLLMEngine,
        expander: Optional[Any] = None
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine
        self.expander = expander
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from sentence_transformers import CrossEncoder
        con.model_msg("Loading reranker [bold]mxbai-rerank-xsmall-v1[/bold] …")
        with con.suppress_stderr(), con.suppress_stdout():
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1", device=device)
        con.success(f"Reranker ready on {device.upper()}")
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

        Uses get_papers_batch() to avoid N+1 query patterns.
        """
        # ── Batch-fetch all papers in one query ──
        paper_ids = list({chunk.paper_id for chunk, _ in similar_chunks})
        papers_map = self.graph_repo.get_papers_batch(paper_ids)

        # 1. Format text chunks using the pre-fetched map
        text_blocks = []
        for idx, (chunk, score) in enumerate(similar_chunks, start=1):
            paper = papers_map.get(chunk.paper_id)
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
                            ref_preview = raw_text if len(raw_text) < 100 else raw_text[:100] + "..."
                            graph_lines.append(f"- Paper {src_name} cites: {ref_preview}")
                        else:
                            graph_lines.append(f"- Paper {src_name} cites paper {tgt_name}")
                    else:
                        graph_lines.append(f"- Node '{src_name}' is connected to '{tgt_name}' via {edge_type}")

        context_graph = "\n".join(graph_lines) if graph_lines else "No direct graph relations found."
        return context_text, context_graph

    def retrieve_relevant_chunks(self, query: str, limit: int = 5) -> List[tuple[Chunk, float]]:
        # Dense + FTS5 + RRF + Rerank
        query_emb = self.emb_engine.get_embedding(query)
        dense_results = self.vector_repo.search_similar_chunks(query_emb, limit=limit * 2)
        fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2)
        
        if not dense_results and not fts5_results:
            return []
            
        id_to_chunk = {}
        for chunk, _ in dense_results:
            id_to_chunk[chunk.id] = chunk
        for chunk, _ in fts5_results:
            id_to_chunk[chunk.id] = chunk

        rrf_scores = {}
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (60.0 + rank))
            
        for rank, (chunk, _) in enumerate(fts5_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (60.0 + rank))
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        candidate_ids = sorted_ids[:limit * 2]
        candidates = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]
        
        if not candidates:
            return []
            
        try:
            reranker = self._get_reranker()
            pairs = [(query, c.text_content) for c in candidates]
            scores = reranker.predict(pairs)
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [(chunk, float(score)) for chunk, score in scored_candidates[:limit]]
        except Exception as e:
            con.warning(f"Reranking failed ({e}), falling back to RRF ranking.")
            return [(id_to_chunk[cid], rrf_scores[cid]) for cid in sorted_ids[:limit] if cid in id_to_chunk]

    def ask(self, query: str, limit: int = 5, history_str: str = "") -> str:
        final_chunks = self.retrieve_relevant_chunks(query, limit)
        if not final_chunks:
            return "Не найдено релевантных фрагментов статей в базе данных. Пожалуйста, сначала проиндексируйте документы."  # noqa: E501
            
        if self.expander:
            if self.expander.reranker is None:
                self.expander.reranker = self._get_reranker()
            enrichment_block = self.expander.expand(query, final_chunks)
            prompt = f"""<|im_start|>system
You are a research assistant. Synthesize an answer to the user's question using the retrieved text blocks and the knowledge graph connections.
Always mention the titles of the papers, years, authors, and page numbers when citation is needed.
If the graph contains citing relationships, use them to explain the context (e.g., "A cited B").

Here is the retrieved context:

### KNOWLEDGE GRAPH ENRICHMENT:
{enrichment_block}
<|im_end|>
{history_str}<|im_start|>user
Question: {query}
Answer in Russian:
<|im_end|>
<|im_start|>assistant
"""
        else:
            context_text, context_graph = self.build_context(final_chunks)
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
        con.search_msg("Generating answer …")
        return self.llm_engine.generate_response(prompt)

    async def generate_stream(self, question: str, limit: int = 5) -> AsyncGenerator[dict, None]:
        import queue
        import threading

        try:
            final_chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, question, limit)
        except Exception as e:
            yield {"type": "error", "text": f"Retrieval failed: {e}"}
            return

        if not final_chunks:
            yield {"type": "error", "text": "No documents indexed yet."}
            return

        try:
            if self.expander:
                if self.expander.reranker is None:
                    self.expander.reranker = await asyncio.to_thread(self._get_reranker)
                enrichment_block = await asyncio.to_thread(self.expander.expand, question, final_chunks)
                prompt = (
                    "<|im_start|>system\n"
                    "You are a research assistant. Synthesize an answer using the retrieved context.\n"
                    "Always cite paper titles, years, and authors. Use the graph connections if relevant.\n\n"
                    f"### KNOWLEDGE GRAPH ENRICHMENT:\n{enrichment_block}\n"
                    "<|im_end|>\n"
                    f"<|im_start|>user\nQuestion: {question}\nAnswer in Russian:\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                context_text, context_graph = await asyncio.to_thread(self.build_context, final_chunks)
                prompt = (
                    "<|im_start|>system\n"
                    "You are a research assistant. Synthesize an answer using the retrieved context.\n"
                    "Always cite paper titles, years, and authors. Use the graph connections if relevant.\n\n"
                    f"### RELEVANT TEXT FRAGMENTS:\n{context_text}\n\n"
                    f"### KNOWLEDGE GRAPH CONNECTIONS:\n{context_graph}\n"
                    "<|im_end|>\n"
                    f"<|im_start|>user\nQuestion: {question}\nAnswer in Russian:\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
        except Exception as e:
            yield {"type": "error", "text": f"Context building failed: {e}"}
            return

        token_queue = queue.Queue()

        def run_mlx_stream():
            try:
                if hasattr(self.llm_engine, "model") and hasattr(self.llm_engine, "tokenizer") and self.llm_engine.model is not None:
                    from mlx_lm import stream_generate
                    gen = stream_generate(
                        model=self.llm_engine.model,
                        tokenizer=self.llm_engine.tokenizer,
                        prompt=prompt,
                        max_tokens=config.llm_max_tokens,
                    )
                    for response in gen:
                        token_text = response.text if hasattr(response, "text") else str(response)
                        if token_text:
                            token_queue.put(token_text)
                else:
                    full_answer = self.llm_engine.generate_response(prompt)
                    for word in full_answer.split(" "):
                        token_queue.put(word + " ")
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=run_mlx_stream)
        thread.start()

        while True:
            item = await asyncio.to_thread(token_queue.get)
            if item is None:
                break
            if isinstance(item, Exception):
                yield {"type": "error", "text": f"Generation failed: {item}"}
                return
            yield {"type": "token", "text": item}

        yield {"type": "done"}


