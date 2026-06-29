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

def _safe_float(val: Any, default: float) -> float:
    try:
        if val.__class__.__name__ in ("MagicMock", "Mock"):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

def _safe_int(val: Any, default: int) -> int:
    try:
        if val.__class__.__name__ in ("MagicMock", "Mock"):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def clean_reasoning_text(text: str) -> str:
    """
    Strips structured thinking/reasoning blocks and extracts the final answer.
    Handles formats like:
      - 1. _analysis..._ or ### 1. _analysis... or 1. _analysis..._analysis:
      - up to the answer section (5. _answer..._ or Final Answer:)
    """
    if not text:
        return text

    # 1. Try to find the last answer marker and take everything after it
    answer_markers = [
        r"(?:###\s*)?Final\s+Answer\s*:?\s*",
        r"(?:###\s*)?5\.\s*_(?:answer|status|reasoning|analysis|source_analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*\s*",
    ]
    combined_pattern = re.compile(
        r"|".join(f"(?:{p})" for p in answer_markers),
        re.IGNORECASE
    )
    
    matches = list(combined_pattern.finditer(text))
    if matches:
        last_match = matches[-1]
        candidate = text[last_match.end():].strip()
        # If there are still answer markers inside candidate, recurse
        if combined_pattern.search(candidate):
            return clean_reasoning_text(candidate)
        text = candidate
    else:
        # 2. Try to strip the entire block from "1. _analysis" to "5. _answer" or "Final Answer"
        text = re.sub(
            r"(?:###\s*)?[1-4]\.\s*_(?:analysis|start|reasoning|status|source_analysis)(?:\.\.\.)?.*?(?=(?:###\s*)?(?:5\.\s*_(?:answer|status|reasoning|analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?))",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    # 3. Clean up any remaining section headers/tags
    header_pattern = r"(?:###\s*)?[1-5]\.\s*_(?:analysis|start|reasoning|status|answer|source_analysis)(?:\.\.\.)?[_a-zA-Z0-9:]*"
    text = re.sub(header_pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:###\s*)?Final\s+Answer\s*:?", "", text, flags=re.IGNORECASE)
    
    # Mask source ID tags to preserve them during strip_thinking_tokens
    text = re.sub(r"<\|source_id\|>", "__SOURCE_ID_TAG__", text, flags=re.IGNORECASE)
    
    # Clean generic and specific technical tokens
    from src.llm_engine.base import strip_thinking_tokens
    text = strip_thinking_tokens(text)
    
    # Unmask source ID tags
    text = text.replace("__SOURCE_ID_TAG__", "<|source_id|>")
    
    return text.strip()


def parse_reasoning_response(raw_response: str) -> Tuple[str, str]:
    """
    Parses the raw LLM response to extract the status and the final answer.
    
    Returns:
        Tuple[str, str]: (status, answer)
    """
    import logging
    logger = logging.getLogger(__name__)

    if not raw_response or not isinstance(raw_response, str):
        return "UNKNOWN", "Error: Empty or incorrect response from model."

    # Extract status
    status = "UNKNOWN"
    status_match = re.search(r"<\|status_start\|>(.*?)<\|status_end\|>", raw_response, re.DOTALL)
    if status_match:
        status = status_match.group(1).strip()
    else:
        # Fallback if tag is unclosed
        status_unclosed = re.search(r"<\|status_start\|>(.*)", raw_response, re.DOTALL)
        if status_unclosed:
            content = status_unclosed.group(1).split("<|")[0].strip()
            status = content if content else "UNKNOWN"

    # If status is still UNKNOWN, try to extract from 4. _status... section
    if status == "UNKNOWN":
        status_sec_match = re.search(
            r"(?:###\s*)?4\.\s*_(?:status)(?:\.\.\.)?[_a-zA-Z0-9:]*\s*(.*?)(?=(?:###\s*)?(?:5\.\s*_(?:answer)(?:\.\.\.)?[_a-zA-Z0-9:]*|(?:###\s*)?Final\s+Answer\s*:?|$))",
            raw_response,
            re.IGNORECASE | re.DOTALL
        )
        if status_sec_match:
            status_text = status_sec_match.group(1).strip().upper()
            if "UNANSWERABLE" in status_text or "NOT ANSWERABLE" in status_text or "INSUFFICIENT" in status_text or "NOT_ANSWERABLE" in status_text:
                status = "UNANSWERABLE"
            elif "ANSWERABLE" in status_text or "SUFFICIENT" in status_text:
                status = "ANSWERABLE"
            elif "UNKNOWN" in status_text:
                status = "UNKNOWN"

    # Extract answer
    answer_match = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", raw_response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback if tag is unclosed
        answer_unclosed = re.search(r"<\|answer_start\|>(.*)", raw_response, re.DOTALL)
        if answer_unclosed:
            answer = answer_unclosed.group(1).strip()
        else:
            # Fallback to the raw response if no answer tags are present at all.
            # Strip reasoning/status/other tags to isolate text.
            answer = raw_response.strip()
            for tag in ["status", "query_analysis", "source_analysis", "reasoning"]:
                answer = re.sub(rf"<\|{tag}_start\|>.*?<\|{tag}_end\|>", "", answer, flags=re.DOTALL)
                answer = re.sub(rf"<\|{tag}_start\|>.*", "", answer, flags=re.DOTALL)
            
            # Clean reasoning markers and headers from the final answer
            answer = clean_reasoning_text(answer)
            
    logger.info(f"RAG reasoning status: {status}")
    return status, answer


TECHNICAL_TOKEN_RE = re.compile(
    r"(<\|.*?\|>|<<.*?>>|\[/?(?:[A-Z_]{2,}[A-Z0-9_-]*)\]|</?(?:s|pad|unk|turn)>|<\|im_start\|>|<\|im_end\|>|<\|im_sep\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\|eot_id\|>|<\|eom_id\|>|<\|endoftext\|>|<\|assistant\|>|<\|user\|>|<\|system\|>|<\|end\|>|\[INST\]|\[/INST\]|<s>|</s>|<start_of_turn>|<end_of_turn>|<<SYS>>|<</SYS>>|<pad>|<unk>)",
    re.IGNORECASE
)

class StreamTokenCleaner:
    def __init__(self):
        self.in_think = False
        self.buffer = ""

    def process_token(self, token: str) -> str:
        self.buffer += token
        output = ""
        
        while self.buffer:
            if self.in_think:
                # Look for end of think block
                idx = self.buffer.lower().find("</think>")
                if idx != -1:
                    # Found end of think block, discard it and everything before it
                    self.buffer = self.buffer[idx + 8:]
                    self.in_think = False
                else:
                    # End of think block not found yet. Keep only the last 7 characters
                    # to handle cases where "</think>" is split across token boundaries.
                    if len(self.buffer) > 7:
                        self.buffer = self.buffer[-7:]
                    break
            else:
                # Look for start of think block
                idx = self.buffer.lower().find("<think>")
                if idx != -1:
                    # Found start of think block. Emit everything before it.
                    output += self.buffer[:idx]
                    self.buffer = self.buffer[idx + 7:]
                    self.in_think = True
                else:
                    # Start tag not found. Keep at most 6 trailing characters
                    # in case "<think>" is split across token boundaries.
                    check_len = min(len(self.buffer), 6)
                    tail = self.buffer[-check_len:]
                    lt_idx = tail.rfind("<")
                    if lt_idx != -1:
                        split_idx = len(self.buffer) - check_len + lt_idx
                        output += self.buffer[:split_idx]
                        self.buffer = self.buffer[split_idx:]
                        break
                    else:
                        output += self.buffer
                        self.buffer = ""
                        break
        return output


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
        model_name = config.reranker_model_name
        con.model_msg(f"Loading reranker [bold]{model_name}[/bold] …")
        with con.suppress_stderr(), con.suppress_stdout():
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._reranker = CrossEncoder(model_name, device=device)
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

    def _get_scored_graph_lines(self, paper_ids: List[str], limit: Optional[int] = None) -> List[Tuple[str, float]]:
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
                            score = config.graph_weight_authored
                        elif edge_type == "CITES":
                            score = config.graph_weight_cites
                        elif edge_type == "MENTIONS_CONCEPT":
                            score = config.graph_weight_mentions_concept
                        else:
                            score = config.graph_weight_default

                    # Compact Cypher representation
                    if edge_type == "CITES" and props.get("raw_text"):
                        raw_text = props.get("raw_text")
                        ref_preview = raw_text if len(raw_text) < 100 else raw_text[:100] + "..."
                        ref_preview = ref_preview.replace('\n', ' ').strip()
                        line = f"- ({src_name}:{src_label})-[CITES {{preview: {json.dumps(ref_preview)}}}]->({tgt_name}:{tgt_label})"
                    else:
                        line = f"- ({src_name}:{src_label})-[{edge_type}]->({tgt_name}:{tgt_label})"

                    scored_lines.append((line, score))

        if limit is not None:
            scored_lines.sort(key=lambda x: x[1], reverse=True)
            scored_lines = scored_lines[:limit * 2]

        return scored_lines

    def build_context(self, similar_chunks: List[tuple[Chunk, float]], limit: Optional[int] = None) -> Tuple[str, str]:
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
            chunk_text = chunk.parent_text if isinstance(chunk.parent_text, str) else chunk.text_content
            doc_text = (
                f"Block {idx} (Score: {score:.3f}) | Paper: {title}{authors_str}{year_str} (Page {chunk.page_number}):\n"
                f"\"\"\"\n{chunk_text.strip()}\n\"\"\""
            )
            text_blocks.append(
                f"<|source_start|><|source_id|>{idx} {doc_text}<|source_end|>"
            )
        context_text = "\n\n".join(text_blocks)

        # 2. Format Graph Subgraph using helper
        if config.rag_components.get("graph_expansion", True):
            scored_lines = self._get_scored_graph_lines(paper_ids, limit=limit)
            graph_lines = [line for line, _ in scored_lines]
            context_graph = "\n".join(graph_lines) if graph_lines else "No direct graph relations found."
        else:
            context_graph = "Graph enrichment disabled."
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

        # 1. Prune graph connections first (cut from the tail after sorting descending by score)
        if total_tokens > tokens_limit and config.rag_components.get("graph_expansion", True):
            paper_ids = list({chunk.paper_id for chunk, _ in current_chunks})
            scored_lines = self._get_scored_graph_lines(paper_ids)
            # Sort by score descending (highest score first)
            scored_lines.sort(key=lambda x: x[1], reverse=True)
            
            while len(scored_lines) > 0 and total_tokens > tokens_limit:
                scored_lines.pop()
                current_graph = "\n".join([line for line, _ in scored_lines]) if scored_lines else "No direct graph relations found."
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
        current_papers = []
        for pid, chs in sorted_papers:
            copied_chs = []
            for c, s in chs:
                c_copy = copy(c)
                if isinstance(c_copy.parent_text, str):
                    c_copy.text_content = c_copy.parent_text
                    c_copy.parent_text = None
                copied_chs.append((c_copy, s))
            current_papers.append([pid, copied_chs])

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

            # Rebuild text, and keep graph at its minimum/pruned state
            current_text, _ = self.build_context(flat_active)
            current_graph = "No direct graph relations found." if config.rag_components.get("graph_expansion", True) else "Graph enrichment disabled."
            current_chunks = flat_active
            total_tokens = get_total_tokens(current_text, current_graph)

        return current_text, current_graph, current_chunks

    def _expand_query(self, query: str) -> List[str]:
        if not query or not isinstance(query, str) or not query.strip():
            return [query] if isinstance(query, str) else []

        try:
            max_expanded = _safe_int(config.max_expanded_queries, 3)
        except Exception:
            max_expanded = 3

        if max_expanded <= 1 or len(query) > 200:
            return [query]

        variants = []
        seen = {query.lower()}

        # 1. Short Query Concept/Synonym Lookup from Graph Ontology
        if config.rag_components.get("graph_ontology_lookup", True) and len(query.strip().split()) <= 2:
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
        if config.rag_components.get("llm_query_expansion", True) and len(variants) < max_expanded - 1:
            try:
                # Detect language & generate variants on the same language
                is_cyrillic = bool(re.search('[а-яА-ЯёЁ]', query))
                language = "Russian" if is_cyrillic else "English"
                prompt = prompts.get_prompt("rag", "query_expander", query=query, language=language)
                
                from src.llm_schemas import LLMQueryExpansionResponse
                from src.llm_engine import StructuredOutput

                max_tokens = min(_safe_int(config.llm_max_tokens, 1000), 200)
                structured = StructuredOutput(LLMQueryExpansionResponse)
                validated = structured.generate(self.llm_engine, prompt, max_tokens=max_tokens)
                parsed = validated.root

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
            "Example: 'articles from the last 2 years about convolutional networks'\n"
            f"Output: {{\"search_query\": \"convolutional networks\", \"year_start\": {current_year - 2}, \"year_end\": {current_year}, \"author\": null, \"venue\": null}}\n"
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
                    
            clean_q = parsed.get("search_query")
            if not clean_q or not isinstance(clean_q, str) or not clean_q.strip():
                clean_q = query
            else:
                clean_q = clean_q.strip()
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
        
        # 1. Parse bracket groups to support multi-citations like [1, 2, 99] or [Block 1; 2]
        bracket_regex = re.compile(r"\[([^\]]+)\]")
        
        def process_bracket(match):
            content = match.group(1)
            # Extract all digits (e.g., "Block 1, 2" -> ["1", "2"])
            nums = re.findall(r"\d+", content)
            valid_nums = []
            for n in nums:
                val = int(n)
                if 1 <= val <= max_idx:
                    valid_nums.append(str(val))
            
            # Reconstruct bracket if there are valid citation numbers
            if valid_nums:
                return f"[{', '.join(valid_nums)}]"
            return ""
            
        repaired = bracket_regex.sub(process_bracket, response)
        
        # 2. Parse standalone "Block X" mentions outside brackets
        block_regex = re.compile(r"\bBlock\s+(\d+)\b", re.IGNORECASE)
        repaired = block_regex.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= max_idx else "", repaired)
        
        # 3. Clean up punctuation spacing and double spaces after deletions
        repaired = re.sub(r"\s+([.,;!?])", r"\1", repaired)  # e.g., "fact ." -> "fact."
        repaired = re.sub(r"\s+", " ", repaired).strip()
        return repaired

    def retrieve_relevant_chunks(self, query: str, limit: int = 5, paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> List[tuple[Chunk, float]]:
        # Focused document RAG: cosine similarity search directly over document chunks in Python
        if not query or not isinstance(query, str) or not query.strip():
            return []

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
            if config.rag_components.get("reranker", True):
                try:
                    reranker = self._get_reranker()
                    pairs = [(query, c.text_content) for c in candidates]
                    scores = reranker.predict(pairs)
                    scored_candidates = list(zip(candidates, scores))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    return [(chunk, float(score)) for chunk, score in scored_candidates[:limit]]
                except Exception:
                    return scored[:limit]
            else:
                return scored[:limit]

        # 1. Run Query Intent Classifier if no explicit filters are passed
        if filters is None and config.rag_components.get("intent_classifier", True):
            query, filters = self._classify_intent_and_extract_filters(query)

        # Dense + FTS5 + RRF + Rerank
        expanded_queries = self._expand_query(query)
        
        # Determine dense retrieval limit per query.
        dense_limit = limit * 2 if len(expanded_queries) == 1 else limit
        
        all_dense_results = {}
        if config.rag_components.get("dense_search", True):
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
            if config.hyde_enabled and config.rag_components.get("hyde", True):
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
                        hyde_responses = _safe_int(getattr(config, "hyde_count", 1), 1)

                for _ in range(hyde_responses):
                    try:
                        hypothetical = self.llm_engine.generate_response(
                            prompt=prompts.get_prompt("rag", "hyde", query=query),
                            max_tokens=_safe_int(config.hyde_max_tokens, 300)
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
                            
        if config.rag_components.get("lexical_search", True):
            if filters:
                fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2, filters=filters)
            else:
                fts5_results = self.vector_repo.search_text_fts5(query, limit=limit * 2)
        else:
            fts5_results = []

        if hasattr(self, "current_trace") and self.current_trace is not None:
            dense_chunk_ids = list(all_dense_results.keys())
            lexical_chunk_ids = [c.id for c, _ in fts5_results]
            self.current_trace["seed_chunks_from_lexical_dense"] = {
                "lexical": lexical_chunk_ids,
                "dense": dense_chunk_ids
            }
            seed_papers = set()
            for chunk, _ in all_dense_results.values():
                if chunk.paper_id:
                    seed_papers.add(chunk.paper_id)
            for chunk, _ in fts5_results:
                if chunk.paper_id:
                    seed_papers.add(chunk.paper_id)
            self.current_trace["seed_paper_id_list"] = list(seed_papers)

        if config.rag_components.get("graph_neighbors_in_rrf", False):
            # Collect paper IDs
            paper_ids = set()
            for chunk, _ in all_dense_results.values():
                pid = getattr(chunk, "paper_id", None)
                if pid:
                    paper_ids.add(pid)
            for chunk, _ in fts5_results:
                pid = getattr(chunk, "paper_id", None)
                if pid:
                    paper_ids.add(pid)
            
            # Find neighbors
            neighbor_paper_ids = set()
            order = _safe_int(getattr(config, "b6_graph_neighbors_order", 2), 2)
            for pid in paper_ids:
                neighbors = self.graph_repo.get_neighbors(pid, max_depth=order)
                for src_id, src_label, _, tgt_id, tgt_label, _ in neighbors:
                    if src_label in ("Paper", "UserNote") and src_id not in paper_ids:
                        neighbor_paper_ids.add(src_id)
                    if tgt_label in ("Paper", "UserNote") and tgt_id not in paper_ids:
                        neighbor_paper_ids.add(tgt_id)

            if hasattr(self, "current_trace") and self.current_trace is not None:
                self.current_trace["graph_neighbor_paper_id_list"] = list(neighbor_paper_ids)
            
            # Load chunks and score them
            if neighbor_paper_ids:
                import numpy as np
                expanded_embs = []
                for variant in expanded_queries:
                    emb = self.emb_engine.get_embedding(variant)
                    expanded_embs.append(np.array(emb, dtype=np.float32))
                
                for neighbor_pid in neighbor_paper_ids:
                    neighbor_chunks = self.vector_repo.get_chunks_for_paper(neighbor_pid)
                    for chunk in neighbor_chunks:
                        cid = getattr(chunk, "id", None)
                        if cid and cid not in all_dense_results:
                            max_sim = -1.0
                            c_emb = getattr(chunk, "embedding", None)
                            if c_emb:
                                chunk_emb = np.array(c_emb, dtype=np.float32)
                                for q_emb in expanded_embs:
                                    if len(chunk_emb) == len(q_emb):
                                        sim = float(np.dot(chunk_emb, q_emb))
                                        if sim > max_sim:
                                            max_sim = sim
                            if max_sim == -1.0:
                                max_sim = 0.0
                            all_dense_results[cid] = (chunk, max_sim)

        dense_results = list(all_dense_results.values())
        # Sort dense results by score descending to assign proper ranks for RRF
        dense_results.sort(key=lambda x: x[1], reverse=True)
        
        if not dense_results and not fts5_results:
            return []
            
        id_to_chunk = {}
        for chunk, _ in dense_results:
            id_to_chunk[chunk.id] = chunk
        for chunk, _ in fts5_results:
            id_to_chunk[chunk.id] = chunk

        # Dynamic alpha blending based on FTS5 match strength
        if config.rag_components.get("dynamic_alpha_blending", True):
            fts_weight = _safe_float(config.dynamic_alpha_val_high, 1.0)
            if fts5_results:
                max_bm25 = max(score for _, score in fts5_results)
                if max_bm25 < _safe_float(config.dynamic_alpha_threshold_low, 1.0): # Very weak keyword matches
                    fts_weight = _safe_float(config.dynamic_alpha_val_low, 0.2)
                elif max_bm25 < _safe_float(config.dynamic_alpha_threshold_mid, 3.0):
                    fts_weight = _safe_float(config.dynamic_alpha_val_mid, 0.5)
            else:
                fts_weight = 0.0
        else:
            fts_weight = 1.0 if fts5_results else 0.0
            
        dense_weight = 1.0

        if config.rag_components.get("rrf", True):
            rrf_scores = {}
            for rank, (chunk, _) in enumerate(dense_results, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + dense_weight * (1.0 / (_safe_float(config.rrf_k, 60.0) + rank))
                
            for rank, (chunk, _) in enumerate(fts5_results, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + fts_weight * (1.0 / (_safe_float(config.rrf_k, 60.0) + rank))
                
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            candidate_ids = sorted_ids[:limit * 2]
            candidates = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]
        else:
            # Fallback when RRF is disabled: just concatenate dense and fts5 results, removing duplicates,
            # using their relative order.
            seen_ids = set()
            candidates = []
            for chunk, _ in dense_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    candidates.append(chunk)
            for chunk, _ in fts5_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    candidates.append(chunk)
            candidates = candidates[:limit * 2]
            rrf_scores = {c.id: 1.0 for c in candidates}
            sorted_ids = [c.id for c in candidates]
        
        if not candidates:
            return []
            
        returned_chunks = []
        if config.rag_components.get("reranker", True):
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
                    if config.rag_components.get("score_blending", True):
                        blended_score = _safe_float(config.score_blend_reranker_weight, 0.7) * norm_r[idx] + _safe_float(config.score_blend_rrf_weight, 0.3) * norm_rrf[idx]
                    else:
                        blended_score = float(scores[idx])
                    scored_candidates.append((c, blended_score, float(scores[idx])))
                    
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                returned_chunks = [(chunk, raw_score) for chunk, _, raw_score in scored_candidates[:limit]]
            except Exception as e:
                con.warning(f"Reranking failed ({e}), falling back to RRF ranking.")
                if config.rag_components.get("rrf", True):
                    returned_chunks = [(id_to_chunk[cid], rrf_scores[cid]) for cid in sorted_ids[:limit] if cid in id_to_chunk]
                else:
                    returned_chunks = [(c, 1.0) for c in candidates[:limit]]
        else:
            if config.rag_components.get("rrf", True):
                returned_chunks = [(id_to_chunk[cid], rrf_scores[cid]) for cid in sorted_ids[:limit] if cid in id_to_chunk]
            else:
                returned_chunks = [(c, 1.0) for c in candidates[:limit]]

        if hasattr(self, "current_trace") and self.current_trace is not None:
            self.current_trace["candidate_count_before_reranker"] = len(candidates) if 'candidates' in locals() else 0
            self.current_trace["candidate_count_after_reranker"] = len(returned_chunks)

        return returned_chunks

    def ask(self, query: str, limit: int = 5, history_str: str = "", paper_id: Optional[str] = None, filters: Optional[dict] = None, hyde_responses: Optional[int] = None) -> str:
        final_chunks = self.retrieve_relevant_chunks(query, limit, paper_id=paper_id, filters=filters, hyde_responses=hyde_responses)
        if not final_chunks:
            return "No relevant article chunks found in the database. Please index documents first."  # noqa: E501

        # Build initial context
        context_text, context_graph = self.build_context(final_chunks, limit=limit)

        wrapped_query = f"<|query_start|>{query}<|query_end|>"

        # Get system prompt for token counting
        if self.expander and config.rag_components.get("graph_expansion", True):
            system_prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block="", history_str=history_str, query=wrapped_query)
        else:
            system_prompt = prompts.get_prompt("rag", "ask_no_expander", context_text="", context_graph="", history_str=history_str, query=wrapped_query)

        model_max_context = getattr(config, "llm_model_max_context", 4096)

        if config.rag_components.get("context_trimming", True):
            trimmed_text, trimmed_graph, trimmed_chunks = self.trim_context(
                context_text=context_text,
                context_graph=context_graph,
                final_chunks=final_chunks,
                query=wrapped_query,
                history_str=history_str,
                system_prompt=system_prompt,
                model_max_context=model_max_context,
                reserved_tokens=500
            )
        else:
            trimmed_text, trimmed_graph, trimmed_chunks = context_text, context_graph, final_chunks

        if hasattr(self, "current_trace") and self.current_trace is not None:
            final_pids = list(set(c[0].paper_id if isinstance(c, tuple) else c.paper_id for c in trimmed_chunks))
            self.current_trace["final_context_paper_id_list"] = final_pids
            tokens = self.llm_engine.count_tokens(trimmed_text + trimmed_graph)
            if not isinstance(tokens, (int, float)):
                tokens = (len(trimmed_text) + len(trimmed_graph)) // 4
            self.current_trace["final_context_token_count"] = tokens
            graph_neighbors = self.current_trace.get("graph_neighbor_paper_id_list", [])
            self.current_trace["whether_graph_neighbor_chunk_survived_into_final_context"] = any(
                (c[0].paper_id if isinstance(c, tuple) else c.paper_id) in graph_neighbors for c in trimmed_chunks
            )

        if self.expander and config.rag_components.get("graph_expansion", True):
            if self.expander.reranker is None:
                self.expander.reranker = self._get_reranker()
            enrichment_block = self.expander.expand(query, trimmed_chunks)
            if not enrichment_block or enrichment_block == "No essential knowledge graph enrichment found.":
                con.warning("Graph expander found no essential facts. Falling back to raw retrieved context.")
                prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str=history_str, query=wrapped_query)
            else:
                prompt = prompts.get_prompt("rag", "ask_expander", enrichment_block=enrichment_block, history_str=history_str, query=wrapped_query)
        else:
            prompt = prompts.get_prompt("rag", "ask_no_expander", context_text=trimmed_text, context_graph=trimmed_graph, history_str=history_str, query=wrapped_query)

        con.search_msg("Generating answer …")
        raw_response = self.llm_engine.generate_response(prompt)
        self.last_raw_response = raw_response
        
        status, parsed_answer = parse_reasoning_response(raw_response)
        con.success(f"Reasoning Status: {status}")

        if config.rag_components.get("citation_repair", True):
            try:
                final_answer = self._validate_and_repair_citations(parsed_answer, trimmed_chunks)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Citation repair failed: {e}")
                final_answer = parsed_answer
        else:
            final_answer = parsed_answer

        if hasattr(self, "current_trace") and self.current_trace is not None:
            tokens = self.llm_engine.count_tokens(final_answer)
            if not isinstance(tokens, (int, float)):
                tokens = len(final_answer) // 4
            self.current_trace["answer_token_count"] = tokens

        return final_answer

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
            wrapped_question = f"<|query_start|>{question}<|query_end|>"
            if self.expander and config.rag_components.get("graph_expansion", True):
                if self.expander.reranker is None:
                    self.expander.reranker = await asyncio.to_thread(self._get_reranker)
                enrichment_block = await asyncio.to_thread(self.expander.expand, question, final_chunks)
                if not enrichment_block or enrichment_block == "No essential knowledge graph enrichment found.":
                    context_text, context_graph = await asyncio.to_thread(self.build_context, final_chunks, limit)
                    prompt = prompts.get_prompt("rag", "stream_no_expander", context_text=context_text, context_graph=context_graph, question=wrapped_question)
                else:
                    prompt = prompts.get_prompt("rag", "stream_expander", enrichment_block=enrichment_block, question=wrapped_question)
            else:
                context_text, context_graph = await asyncio.to_thread(self.build_context, final_chunks, limit)
                prompt = prompts.get_prompt("rag", "stream_no_expander", context_text=context_text, context_graph=context_graph, question=wrapped_question)

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
                        max_tokens=_safe_int(config.llm_max_tokens, 1000),
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

        cleaner = StreamTokenCleaner()
        while True:
            item = await asyncio.to_thread(token_queue.get)
            if item is None:
                break
            if isinstance(item, Exception):
                yield {"type": "error", "text": f"Generation failed: {item}"}
                return
            processed_item = cleaner.process_token(item)
            cleaned_item = TECHNICAL_TOKEN_RE.sub("", processed_item)
            if not cleaned_item and item:
                continue
            yield {"type": "token", "text": cleaned_item}

        yield {"type": "done"}


