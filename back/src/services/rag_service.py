import json
import asyncio
import re
from typing import List, Tuple, AsyncGenerator, Any, Optional
import tiktoken
from src.models import Chunk
from src.repository.base import GraphRepository, VectorRepository
from src.vector_search import EmbeddingEngine
from src.llm_engine import BaseLLMEngine
from src.config import config
from src import console as con
from src.prompts import prompts

TECHNICAL_TOKEN_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|im_sep\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\|eot_id\|>|<\|eom_id\|>|<\|endoftext\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|<\|end\|>|\[INST\]|\[/INST\]|<s>|</s>|<start_of_turn>|<end_of_turn>|<<SYS>>|<</SYS>>|<pad>|<unk>)",
    re.IGNORECASE
)

def count_prompt_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())

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

    def _get_scored_graph_lines(self, paper_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Retrieves graph neighbor relations for the given paper IDs, formats them,
        and assigns importance scores.
        """
        scored_lines = []
        seen_edges = set()

        for paper_id in paper_ids:
            neighbors = self.graph_repo.get_neighbors(paper_id, max_depth=1)
            for src_id, src_label, edge_type, tgt_id, tgt_label, edge_props in neighbors:
                edge_key = (src_id, tgt_id, edge_type)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)

                    src_name = self._resolve_node_name(src_id, src_label)
                    tgt_name = self._resolve_node_name(tgt_id, tgt_label)

                    try:
                        props = json.loads(edge_props) if edge_props else {}
                    except Exception:
                        props = {}

                    # Try to extract explicit score or weight from properties
                    score_val = props.get("score") or props.get("weight")
                    if score_val is not None:
                        score = float(score_val)
                    else:
                        # Default heuristic scores based on edge type
                        if edge_type == "AUTHORED":
                            score = 0.8
                        elif edge_type == "CITES":
                            score = 0.7
                        elif edge_type == "MENTIONS_CONCEPT":
                            score = 0.6
                        else:
                            score = 0.5

                    if edge_type == "AUTHORED":
                        line = f"- {src_name} (Author) authored paper {tgt_name}"
                    elif edge_type == "MENTIONS_CONCEPT":
                        line = f"- Paper {src_name} mentions concept/topic '{tgt_name}'"
                    elif edge_type == "CITES":
                        raw_text = props.get("raw_text")
                        if raw_text:
                            ref_preview = raw_text if len(raw_text) < 100 else raw_text[:100] + "..."
                            line = f"- Paper {src_name} cites: {ref_preview}"
                        else:
                            line = f"- Paper {src_name} cites paper {tgt_name}"
                    else:
                        line = f"- Node '{src_name}' is connected to '{tgt_name}' via {edge_type}"

                    scored_lines.append((line, score))

        return scored_lines

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

        # 2. Format Graph Subgraph using helper
        scored_lines = self._get_scored_graph_lines(paper_ids)
        graph_lines = [line for line, _ in scored_lines]
        context_graph = "\n".join(graph_lines) if graph_lines else "No direct graph relations found."
        return context_text, context_graph

    def trim_context(
        self,
        context_text: str,
        context_graph: str,
        final_chunks: list,
        query: str,
        history_str: str,
        system_prompt: str,
        model_max_context: int = 4096,
        reserved_tokens: int = 500
    ) -> tuple[str, str, list]:
        """Возвращает (trimmed_context_text, trimmed_context_graph, trimmed_chunks)."""
        def get_total_tokens(ctx_text: str, ctx_graph: str) -> int:
            combined_text = ctx_text + ctx_graph + system_prompt + query + history_str
            return count_prompt_tokens(combined_text)

        tokens_limit = model_max_context - reserved_tokens
        current_chunks = list(final_chunks)
        current_text = context_text
        current_graph = context_graph

        total_tokens = get_total_tokens(current_text, current_graph)

        if total_tokens <= tokens_limit:
            return current_text, current_graph, current_chunks

        # Iteratively trim least relevant chunks (from tail)
        # Note: final_chunks is already sorted by relevance descending after reranker
        while len(current_chunks) > 1 and total_tokens > tokens_limit:
            current_chunks.pop()
            current_text, current_graph = self.build_context(current_chunks)
            total_tokens = get_total_tokens(current_text, current_graph)

        # Log warning if chunks were trimmed
        num_trimmed = len(final_chunks) - len(current_chunks)
        if num_trimmed > 0:
            con.warning(f"Trimmed to {len(current_chunks)} chunks ({total_tokens} tokens)")

        # If even 1 chunk is left but still exceeds limit, trim context_graph
        if len(current_chunks) == 1 and total_tokens > tokens_limit:
            paper_ids = [current_chunks[0][0].paper_id]
            scored_lines = self._get_scored_graph_lines(paper_ids)
            
            # Sort by score descending (highest score first)
            scored_lines.sort(key=lambda x: x[1], reverse=True)
            
            # Iteratively remove lowest score edges from the tail
            while len(scored_lines) > 0 and total_tokens > tokens_limit:
                scored_lines.pop()
                current_graph = "\n".join([line for line, _ in scored_lines]) if scored_lines else "No direct graph relations found."
                total_tokens = get_total_tokens(current_text, current_graph)
                
            con.warning(f"Trimmed context_graph to {len(scored_lines)} edges ({total_tokens} tokens)")

        return current_text, current_graph, current_chunks

    def _expand_query(self, query: str) -> List[str]:
        try:
            max_expanded = config.max_expanded_queries
        except Exception:
            max_expanded = 3

        if max_expanded <= 1 or len(query) > 200:
            return [query]

        try:
            prompt = (
                "<|im_start|>system\n"
                "You are a search query expansion assistant for a scientific knowledge base.\n"
                "Given a user query, generate 2-3 alternative phrasings that capture the same\n"
                "research intent but use different terminology. Output as a JSON list of strings.\n"
                "Keep each variant under 15 words. Do NOT output anything else.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"Query: {query}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant"
            )
            max_tokens = min(config.llm_max_tokens, 200)
            response = self.llm_engine.generate_response(prompt, max_tokens=max_tokens)
            
            clean_json = self.llm_engine.extract_json(response)
            parsed = json.loads(clean_json)
            
            if isinstance(parsed, list):
                variants = []
                seen = {query.lower()}
                for v in parsed:
                    if isinstance(v, str):
                        v_clean = v.strip()
                        if v_clean and v_clean.lower() not in seen:
                            variants.append(v_clean)
                            seen.add(v_clean.lower())
                
                limit_variants = max_expanded - 1
                return [query] + variants[:limit_variants]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Query expansion failed: {e}")
            
        return [query]

    def retrieve_relevant_chunks(self, query: str, limit: int = 5, paper_id: Optional[str] = None) -> List[tuple[Chunk, float]]:
        # Focused document RAG: cosine similarity search directly over document chunks in Python
        if paper_id:
            import numpy as np
            chunks = self.vector_repo.get_chunks_for_paper(paper_id)
            if not chunks:
                return []
            query_emb = np.array(self.emb_engine.get_embedding(query), dtype=np.float32)
            scored = []
            for c in chunks:
                c_emb = np.array(c.embedding, dtype=np.float32)
                sim = float(np.dot(c_emb, query_emb)) if len(c_emb) == len(query_emb) else 0.0
                scored.append((c, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Apply reranking on candidate chunks if reranker is available
            candidates = [s[0] for s in scored[:limit * 2]]
            try:
                reranker = self._get_reranker()
                pairs = [(query, c.text_content) for c in candidates]
                scores = reranker.predict(pairs)
                scored_candidates = list(zip(candidates, scores))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return [(chunk, float(score)) for chunk, score in scored_candidates[:limit]]
            except Exception as e:
                return scored[:limit]

        # Dense + FTS5 + RRF + Rerank
        expanded_queries = self._expand_query(query)
        
        # Determine dense retrieval limit per query.
        # If query expansion is disabled or not used (i.e. only 1 query), use limit * 2 for backward compatibility.
        dense_limit = limit * 2 if len(expanded_queries) == 1 else limit
        
        all_dense_results = {}
        for variant in expanded_queries:
            variant_emb = self.emb_engine.get_embedding(variant)
            dense_res = self.vector_repo.search_similar_chunks(variant_emb, limit=dense_limit)
            for chunk, score in dense_res:
                if chunk.id not in all_dense_results:
                    all_dense_results[chunk.id] = (chunk, score)
                else:
                    existing_chunk, existing_score = all_dense_results[chunk.id]
                    if score > existing_score:
                        all_dense_results[chunk.id] = (chunk, score)
                        
        dense_results = list(all_dense_results.values())
        # Sort dense results by score descending to assign proper ranks for RRF
        dense_results.sort(key=lambda x: x[1], reverse=True)

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

    def ask(self, query: str, limit: int = 5, history_str: str = "", paper_id: Optional[str] = None) -> str:
        final_chunks = self.retrieve_relevant_chunks(query, limit, paper_id=paper_id)
        if not final_chunks:
            return "Не найдено релевантных фрагментов статей в базе данных. Пожалуйста, сначала проиндексируйте документы."  # noqa: E501

        # Build initial context
        context_text, context_graph = self.build_context(final_chunks)

        # Get system prompt for token counting
        if self.expander:
            system_prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block="", history_str=history_str, query=query)
        else:
            system_prompt = prompts.get_prompt("rag", "ask_no_expander", context_text="", context_graph="", history_str=history_str, query=query)

        model_max_context = getattr(config, "llm_model_max_context", 4096)

        trimmed_text, trimmed_graph, trimmed_chunks = self.trim_context(
            context_text=context_text,
            context_graph=context_graph,
            final_chunks=final_chunks,
            query=query,
            history_str=history_str,
            system_prompt=system_prompt,
            model_max_context=model_max_context,
            reserved_tokens=500
        )

        if self.expander:
            if self.expander.reranker is None:
                self.expander.reranker = self._get_reranker()
            enrichment_block = self.expander.expand(query, trimmed_chunks)
            prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block=enrichment_block, history_str=history_str, query=query)
        else:
            prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str=history_str, query=query)

        con.search_msg("Generating answer …")
        return self.llm_engine.generate_response(prompt)

    async def generate_stream(self, question: str, limit: int = 5, paper_id: Optional[str] = None) -> AsyncGenerator[dict, None]:
        import queue
        import threading

        try:
            final_chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, question, limit, paper_id)
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
                prompt = prompts.get_prompt("rag", "stream_expander", enrichment_block=enrichment_block, question=question)
            else:
                context_text, context_graph = await asyncio.to_thread(self.build_context, final_chunks)
                prompt = prompts.get_prompt("rag", "stream_no_expander", context_text=context_text, context_graph=context_graph, question=question)

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
            cleaned_item = TECHNICAL_TOKEN_RE.sub("", item)
            if not cleaned_item and item:
                continue
            yield {"type": "token", "text": cleaned_item}

        yield {"type": "done"}


