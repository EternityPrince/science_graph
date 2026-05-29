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
        expander: Optional[Any] = None,
        warmup: bool = False
    ):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.emb_engine = embedding_engine
        self.llm_engine = llm_engine
        self.expander = expander
        self._reranker = None
        if warmup:
            try:
                embedding_engine.get_embedding("warmup query")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Eager embedding engine warmup failed: {e}")
            try:
                self._get_reranker()
                self._reranker.predict([("warmup query", "warmup context")])
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Eager reranker warmup failed: {e}")

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

        # Group chunks by paper_id to keep document context unified
        paper_chunks = {}
        for chunk, score in final_chunks:
            paper_chunks.setdefault(chunk.paper_id, []).append((chunk, score))

        # Sort chunks within each paper by page_number (restores chronological flow)
        for pid in paper_chunks:
            paper_chunks[pid].sort(key=lambda x: x[0].page_number)

        # Order papers by their highest chunk score
        sorted_papers = sorted(
            paper_chunks.items(),
            key=lambda x: max(score for _, score in x[1]),
            reverse=True
        )

        from copy import copy
        current_papers = [
            [pid, [(copy(c), s) for c, s in chs]]
            for pid, chs in sorted_papers
        ]

        # Get papers map to resolve names when rebuilding
        paper_ids = list({c.paper_id for c, _ in final_chunks})
        papers_map = self.graph_repo.get_papers_batch(paper_ids)

        while total_tokens > tokens_limit:
            # Try to prune from the least relevant paper (last in sorted list)
            pruned = False
            for i in range(len(current_papers) - 1, -1, -1):
                chunks_list = current_papers[i][1]
                if len(chunks_list) > 0:
                    # Prune a sentence from the last chunk.
                    # First and last sentences are preserved; starting with the middle.
                    last_chunk, score = chunks_list[-1]
                    sentences = last_chunk.text_content.strip().split(". ")
                    sentences = [s.strip() for s in sentences if s.strip()]
                    if len(sentences) > 2:
                        # Soft trim: remove the second-to-last sentence (from the middle)
                        sentences.pop(-2)
                        new_text = ". ".join(sentences)
                        if not any(new_text.endswith(p) for p in [".", "?", "!"]):
                            new_text += "."
                        last_chunk.text_content = new_text
                    elif len(sentences) == 2:
                        # Soft trim: remove the last sentence (since only 2 sentences left, no middle exists)
                        sentences.pop()
                        new_text = sentences[0]
                        if not any(new_text.endswith(p) for p in [".", "?", "!"]):
                            new_text += "."
                        last_chunk.text_content = new_text
                    else:
                        # Hard trim: remove the entire chunk if only 1 sentence left
                        chunks_list.pop()
                    pruned = True
                    break

            if not pruned:
                break

            # Rebuild context and recalculate tokens
            flat_active = []
            for _, chs in current_papers:
                flat_active.extend(chs)

            if not flat_active:
                break

            current_text, current_graph = self.build_context(flat_active)
            current_chunks = flat_active
            total_tokens = get_total_tokens(current_text, current_graph)

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

        variants = []
        seen = {query.lower()}

        # 1. Short Query Concept/Synonym Lookup from Graph Ontology
        if len(query.strip().split()) <= 2:
            try:
                aliases_map = self.graph_repo.get_concept_aliases()
                canonical = aliases_map.get(query.strip().lower())
                if canonical:
                    concept_node = self.graph_repo.get_concept(canonical)
                    if concept_node:
                        props = concept_node.properties if hasattr(concept_node, "properties") else {}
                        aliases = props.get("aliases") or []
                        names = [props.get("name_en"), props.get("name_ru"), props.get("name")]
                        extra_variants = [canonical] + aliases + names
                        for ev in extra_variants:
                            if isinstance(ev, str) and ev.strip():
                                ev_clean = ev.strip()
                                if ev_clean.lower() not in seen:
                                    variants.append(ev_clean)
                                    seen.add(ev_clean.lower())
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Short query concept enrichment failed: {e}")

        # 2. If we still need more variants, call the LLM
        if len(variants) < max_expanded - 1:
            try:
                # Detect language & generate variants on the same language
                is_cyrillic = bool(re.search('[а-яА-ЯёЁ]', query))
                language = "Russian" if is_cyrillic else "English"
                prompt = prompts.get_prompt("rag", "query_expander", query=query, language=language)
                
                max_tokens = min(config.llm_max_tokens, 200)
                response = self.llm_engine.generate_response(prompt, max_tokens=max_tokens)
                
                clean_json = self.llm_engine.extract_json(response)
                parsed = json.loads(clean_json)
                
                if isinstance(parsed, list):
                    for v in parsed:
                        if isinstance(v, str):
                            v_clean = v.strip()
                            if v_clean and v_clean.lower() not in seen:
                                variants.append(v_clean)
                                seen.add(v_clean.lower())
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Query expansion failed: {e}")
                
        limit_variants = max_expanded - 1
        return [query] + variants[:limit_variants]

    def _classify_intent_and_extract_filters(self, query: str) -> Tuple[str, Optional[dict]]:
        import re
        import datetime
        
        time_keywords = ["год", "лет", "last", "year", "recent", "новые", "newest", "старые", "oldest"]
        author_keywords = ["автор", "by ", "author", "написал", "wrote"]
        venue_keywords = ["journal", "conference", "журнал", "конференция", "venue"]
        
        query_lower = query.lower()
        has_time = any(w in query_lower for w in time_keywords) or bool(re.search(r'\b\d{4}\b', query))
        has_author = any(w in query_lower for w in author_keywords)
        has_venue = any(w in query_lower for w in venue_keywords)
        
        if not (has_time or has_author or has_venue):
            return query, None
            
        current_year = datetime.datetime.now().year
        
        prompt = (
            f"You are a scientific database query analyzer. The current year is {current_year}.\n"
            "Given a user query, extract metadata filters and return a JSON object with fields:\n"
            "- search_query: clean search query without relative/filter terms\n"
            "- year_start: start year (integer or null)\n"
            "- year_end: end year (integer or null)\n"
            "- author: author name (string or null)\n"
            "- venue: journal/conference name (string or null)\n"
            "Example: 'статьи за последние 2 года о сверточных сетях'\n"
            f"Output: {{\"search_query\": \"сверточные сети\", \"year_start\": {current_year - 2}, \"year_end\": {current_year}, \"author\": null, \"venue\": null}}\n"
            f"Query: {query}\n"
            "Return ONLY JSON."
        )
        
        try:
            response = self.llm_engine.generate_response(prompt)
            clean_json = self.llm_engine.extract_json(response)
            parsed = json.loads(clean_json)
            
            filters = {}
            for k in ["year_start", "year_end", "author", "venue"]:
                if parsed.get(k) is not None:
                    filters[k] = parsed[k]
                    
            clean_q = parsed.get("search_query", query)
            con.success(f"Extracted filters: {filters} | Clean query: '{clean_q}'")
            return clean_q, filters if filters else None
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Query intent classification failed: {e}")
            return query, None

    def _validate_and_repair_citations(self, response: str, retrieved_chunks: list) -> str:
        """
        Parses LLM response for bracketed numeric citations (e.g., [1], [2])
        and cross-checks them against the available metadata. Removes hallucinated
        citations or maps them to correct document indexes.
        """
        import re
        max_idx = len(retrieved_chunks)
        citation_regex = re.compile(r"\[(?:Block\s+)?(\d+)\]|Block\s+(\d+)", re.IGNORECASE)
        
        def replace_citation(match):
            val = match.group(1) or match.group(2)
            cit_idx = int(val)
            if 1 <= cit_idx <= max_idx:
                return match.group(0)
            return ""
            
        repaired = citation_regex.sub(replace_citation, response)
        repaired = re.sub(r"\[\s*\]", "", repaired)
        repaired = re.sub(r"\s+", " ", repaired).strip()
        return repaired

    def retrieve_relevant_chunks(self, query: str, limit: int = 5, paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> List[tuple[Chunk, float]]:
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

        # 1. Run Query Intent Classifier if no explicit filters are passed
        if filters is None:
            query, filters = self._classify_intent_and_extract_filters(query)

        # Dense + FTS5 + RRF + Rerank
        expanded_queries = self._expand_query(query)
        
        # Determine dense retrieval limit per query.
        dense_limit = limit * 2 if len(expanded_queries) == 1 else limit
        
        all_dense_results = {}
        for variant in expanded_queries:
            variant_emb = self.emb_engine.get_embedding(variant)
            if filters:
                dense_res = self.vector_repo.search_similar_chunks(variant_emb, limit=dense_limit, filters=filters)
            else:
                dense_res = self.vector_repo.search_similar_chunks(variant_emb, limit=dense_limit)
            for chunk, score in dense_res:
                if chunk.id not in all_dense_results:
                    all_dense_results[chunk.id] = (chunk, score)
                else:
                    existing_chunk, existing_score = all_dense_results[chunk.id]
                    if score > existing_score:
                        all_dense_results[chunk.id] = (chunk, score)

        # 2. Run HyDE (Hypothetical Document Embeddings) if enabled
        if config.hyde_enabled:
            # Resolve hyde_responses parameter
            if hyde_responses is None:
                # Check sys.argv for --hyde
                import sys
                argv_hyde = None
                for idx, arg in enumerate(sys.argv):
                    if arg == "--hyde":
                        if idx + 1 < len(sys.argv):
                            try:
                                argv_hyde = int(sys.argv[idx + 1])
                            except ValueError:
                                pass
                    elif arg.startswith("--hyde="):
                        try:
                            argv_hyde = int(arg.split("=", 1)[1])
                        except ValueError:
                            pass
                if argv_hyde is not None:
                    hyde_responses = argv_hyde
                else:
                    hyde_responses = getattr(config, "hyde_count", 1)

            for _ in range(hyde_responses):
                try:
                    hypothetical = self.llm_engine.generate_response(
                        prompt=prompts.get_prompt("rag", "hyde", query=query),
                        max_tokens=config.hyde_max_tokens
                    )
                    con.debug(f"Generated hypothetical answer: {hypothetical}")
                    
                    hyp_emb = self.emb_engine.get_embedding(hypothetical)
                    if filters:
                        hyde_res = self.vector_repo.search_similar_chunks(hyp_emb, limit=limit * 2, filters=filters)
                    else:
                        hyde_res = self.vector_repo.search_similar_chunks(hyp_emb, limit=limit * 2)
                        
                    for chunk, score in hyde_res:
                        if chunk.id not in all_dense_results:
                            all_dense_results[chunk.id] = (chunk, score)
                        else:
                            existing_chunk, existing_score = all_dense_results[chunk.id]
                            if score > existing_score:
                                all_dense_results[chunk.id] = (chunk, score)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"HyDE generation failed: {e}")
                        
        dense_results = list(all_dense_results.values())
        # Sort dense results by score descending to assign proper ranks for RRF
        dense_results.sort(key=lambda x: x[1], reverse=True)

        if filters:
            fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2, filters=filters)
        else:
            fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2)
        
        if not dense_results and not fts5_results:
            return []
            
        id_to_chunk = {}
        for chunk, _ in dense_results:
            id_to_chunk[chunk.id] = chunk
        for chunk, _ in fts5_results:
            id_to_chunk[chunk.id] = chunk

        # Dynamic alpha blending based on FTS5 match strength
        fts_weight = 1.0
        if fts5_results:
            max_bm25 = max(score for _, score in fts5_results)
            if max_bm25 < 1.0: # Very weak keyword matches
                fts_weight = 0.2
            elif max_bm25 < 3.0:
                fts_weight = 0.5
        else:
            fts_weight = 0.0
            
        dense_weight = 1.0

        rrf_scores = {}
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + dense_weight * (1.0 / (60.0 + rank))
            
        for rank, (chunk, _) in enumerate(fts5_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + fts_weight * (1.0 / (60.0 + rank))
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        candidate_ids = sorted_ids[:limit * 2]
        candidates = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]
        
        if not candidates:
            return []
            
        try:
            reranker = self._get_reranker()
            pairs = [(query, c.text_content) for c in candidates]
            scores = reranker.predict(pairs)
            
            # Blend normalized Reranker score + normalized RRF score to prevent dense-only bias
            min_r = min(scores)
            max_r = max(scores)
            range_r = max_r - min_r if max_r > min_r else 1.0
            norm_r = [(s - min_r) / range_r for s in scores]
            
            rrf_vals = [rrf_scores[c.id] for c in candidates]
            min_rrf = min(rrf_vals)
            max_rrf = max(rrf_vals)
            range_rrf = max_rrf - min_rrf if max_rrf > min_rrf else 1.0
            norm_rrf = [(rrf_scores[c.id] - min_rrf) / range_rrf for c in candidates]
            
            scored_candidates = []
            for idx, c in enumerate(candidates):
                blended_score = 0.7 * norm_r[idx] + 0.3 * norm_rrf[idx]
                scored_candidates.append((c, blended_score, float(scores[idx])))
                
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [(chunk, raw_score) for chunk, _, raw_score in scored_candidates[:limit]]
        except Exception as e:
            con.warning(f"Reranking failed ({e}), falling back to RRF ranking.")
            return [(id_to_chunk[cid], rrf_scores[cid]) for cid in sorted_ids[:limit] if cid in id_to_chunk]

    def ask(self, query: str, limit: int = 5, history_str: str = "", paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> str:
        final_chunks = self.retrieve_relevant_chunks(query, limit, paper_id=paper_id, filters=filters, hyde_responses=hyde_responses)
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
        raw_response = self.llm_engine.generate_response(prompt)
        
        try:
            return self._validate_and_repair_citations(raw_response, trimmed_chunks)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Citation repair failed: {e}")
            return raw_response

    async def generate_stream(self, question: str, limit: int = 5, paper_id: Optional[str] = None, hyde_responses: Optional[int] = None) -> AsyncGenerator[dict, None]:
        import queue
        import threading

        try:
            final_chunks = await asyncio.to_thread(self.retrieve_relevant_chunks, question, limit, paper_id, None, hyde_responses)
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


