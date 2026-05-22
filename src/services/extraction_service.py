"""
ExtractionService — Knowledge Extraction from Document Text.

Encapsulates the "try LLM → fallback to regex taxonomy scan" pattern
that was previously duplicated across index_pdf, index_epub, index_url,
and reindex_metadata in Indexer.

Also provides concept description lookup and LLM summary generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import config
from src.models import Paper, slugify
from src import console as con
from src.services.normalization_pipeline import NormalizationPipeline


@dataclass
class ExtractionResult:
    """Structured output from knowledge extraction on a document."""

    authors: List[str] = field(default_factory=list)
    """Author names discovered by LLM (not from parsers or Semantic Scholar)."""

    concepts: List[Dict[str, str]] = field(default_factory=list)
    """List of {name, description} dicts for MENTIONS_CONCEPT edges."""

    tags: List[str] = field(default_factory=list)
    """High-level topic/tag names for HAS_TAG edges."""

    via_llm: bool = False
    """True if concepts/tags were extracted by LLM; False if via regex fallback."""


class ExtractionService:
    """
    Extracts structured knowledge (authors, concepts, tags) from document text.

    Priority order:
        1. LLM-based extraction (if llm_engine is provided and use_llm=True)
        2. Regex keyword scan against taxonomy.yaml
        3. Empty result (graceful degradation)
    """

    def __init__(self, llm_engine: Any = None) -> None:
        self.llm_engine = llm_engine
        self.normalization_pipeline = NormalizationPipeline()

    @property
    def _tax(self) -> Dict[str, Any]:
        return config.taxonomy

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(
        self,
        title: str,
        abstract: str,
        full_text: str,
        use_llm: bool = True,
        trace_info: Optional[dict] = None,
    ) -> ExtractionResult:
        """
        Extracts authors, concepts, and tags from the provided text.

        Args:
            title:     Document title.
            abstract:  Document abstract (may be empty).
            full_text: Full document body text (used for keyword scan).
            use_llm:   Whether to attempt LLM-based extraction first.
            trace_info: Optional dictionary to collect timing and token metrics.

        Returns:
            ExtractionResult with authors, concepts, and tags.
        """
        llm_result = None
        if use_llm and self.llm_engine:
            llm_result = self._extract_via_llm(title, abstract, full_text, trace_info=trace_info)

        regex_result = self._extract_via_regex(title, abstract, full_text)

        if llm_result is not None:
            # Merge regex concepts/tags into LLM result
            # De-duplicate concepts by name (case-insensitive)
            seen_concepts = {c["name"].lower().strip() for c in llm_result.concepts}
            for c in regex_result.concepts:
                name_key = c["name"].lower().strip()
                if name_key not in seen_concepts:
                    llm_result.concepts.append(c)
                    seen_concepts.add(name_key)

            # De-duplicate tags (case-insensitive)
            seen_tags = {t.lower().strip() for t in llm_result.tags}
            for t in regex_result.tags:
                tag_key = t.lower().strip()
                if tag_key not in seen_tags:
                    llm_result.tags.append(t)
                    seen_tags.add(tag_key)

            return self._normalize_extraction_result(llm_result)

        return self._normalize_extraction_result(regex_result)

    def get_concept_description(self, name: str, trace_info: Optional[dict] = None) -> str:
        """
        Returns a one-sentence description for the given concept name.

        Priority:
            1. Taxonomy descriptions dict (case-insensitive match)
            2. LLM-generated description (if llm_engine available)
            3. Generic fallback string
        """
        descriptions: Dict[str, str] = self._tax.get("descriptions", {})
        for k, v in descriptions.items():
            if k.lower() == name.lower():
                return v

        if self.llm_engine:
            try:
                prompt = (
                    f"Provide a brief, one-sentence definition of the AI/ML concept "
                    f"or term: '{name}'. Do not write anything else. Keep it under 20 words."
                )
                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Concept description LLM calls"] = tokens_dict.get("Concept description LLM calls", 0) + self.llm_engine.count_tokens(prompt)
                desc = self.llm_engine.generate_response(prompt, task="extraction").strip()
                desc = re.sub(r'^["\']|["\']$', "", desc).strip()
                if desc:
                    return desc
            except Exception:
                pass

        return f"A key concept representing '{name}' within the AI/ML literature."

    def is_chunk_relevant(self, chunk_text: str, doc_title: str) -> bool:
        """
        Verifies if a specific transcript chunk is relevant/actual to the database.
        Checks for advertising, sponsor integrations, self-promotion, or generic intro/outro chat.
        """
        # 1. Rule-based fast check
        # Common sponsor names and self-promotion phrases (English and Russian)
        patterns = [
            r"\b(sponsor|squarespace|nordvpn|patreon|surfshark|brilliant\.org|expressvpn|audible|skillshare)\b",
            r"\b(raid shadow legends|athletic greens|hellofresh|ridge wallet|wix\.com|grammarly|honey browser)\b",
            r"\b(subscribe|bell icon|my channel|like and share|giveaway|promo code|discount link)\b",
            r"\b(click the link|use my code|curiositystream|nebula)\b",
            r"\b(спонсор|подпишитесь|поставьте лайк|жмите колокольчик|промокод|ссылка в описании|скидка)\b",
            r"\b(реклама|интеграция|наш партнер|подписка)\b"
        ]
        
        chunk_lower = chunk_text.lower()
        matched_rule = False
        for pattern in patterns:
            if re.search(pattern, chunk_lower):
                matched_rule = True
                break
        
        # If no keywords matched, we assume it's relevant (minimizing LLM overhead)
        if not matched_rule:
            return True
            
        # 2. If it matched a rule, let's ask the LLM to verify (if available) to prevent false positives
        if not self.llm_engine:
            # Fallback when LLM is unavailable: filter out chunks matching rule
            con.warning(f"Chunk matches sponsor/promo rule but LLM is not available for double-check. Skipping chunk: '{chunk_text[:60]}...'")
            return False
            
        try:
            from src.llm_schemas import LLMVerificationResponse
            import json
            
            prompt = (
                f"You are a validation assistant for a scientific and technical knowledge database.\n"
                f"Your task is to analyze the following video transcript chunk (from video: '{doc_title}')\n"
                f"and decide if it contains relevant educational, informational, or scientific concepts/details,\n"
                f"or if it is primarily an advertisement, sponsor plug, self-promotion (asking to subscribe, like, support on Patreon),\n"
                f"or irrelevant filler/intro/outro greetings.\n\n"
                f"Transcript chunk:\n\"{chunk_text}\"\n\n"
                f"Return relevant=true if it contains actual content, or relevant=false if it is promotional or filler."
            )
            response_raw = self.llm_engine.generate_json(prompt, schema_class=LLMVerificationResponse)
            response_json = json.loads(response_raw)
            relevant = response_json.get("relevant", True)
            reason = response_json.get("reason", "")
            if not relevant:
                con.warning(f"Skipping chunk (verified irrelevant by LLM: {reason}): '{chunk_text[:60]}...'")
                return False
        except Exception as e:
            # In case LLM fails, act defensively: accept the chunk to avoid losing data
            con.warning(f"Error during LLM verification of chunk: {e}. Accepting chunk defensively.")
            return True
            
        return True

    def generate_summary(
        self,
        paper: Paper,
        full_text: str,
        graph_repo: Any = None,
        trace_info: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Generates an LLM summary for a paper and optionally persists it.

        Args:
            paper:      The Paper object whose summary is to be generated.
            full_text:  Full document text (used as context, first 4000 chars).
            graph_repo: If provided, saves the updated paper after generating summary.
            trace_info: Optional dictionary to collect timing and token metrics.

        Returns:
            The generated summary string, or None if LLM is unavailable or fails.
        """
        if not self.llm_engine:
            return None

        # Check if this is a video
        source_type = paper.properties.get("source_type", "paper")
        if source_type == "video":
            con.dim(f"Generating structured summary for video [bold]{paper.title[:60]}[/bold] via LLM …")
            try:
                from src.llm_schemas import LLMVideoSummaryResponse
                import json
                
                sample_text = full_text[:6000] if full_text else ""
                prompt = (
                    f"Analyze the following video transcription text.\n"
                    f"Generate a detailed structured summary containing:\n"
                    f"1. A high-level overview/summary of the video (2-3 paragraphs).\n"
                    f"2. A list of key themes or topics discussed, with brief explanations.\n"
                    f"3. A detailed outline/notes structure of the video (chronological or logical breakdown).\n\n"
                    f"Video Title: {paper.title or paper.id}\n\n"
                    f"Transcript Content:\n{sample_text}\n"
                )
                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Summary Generation"] = tokens_dict.get("Summary Generation", 0) + self.llm_engine.count_tokens(prompt)
                
                summary_raw = self.llm_engine.generate_json(prompt, schema_class=LLMVideoSummaryResponse)
                summary_json = json.loads(summary_raw)
                
                overview = summary_json.get("overview", "")
                themes = summary_json.get("themes", [])
                outline = summary_json.get("outline", [])
                
                paper.properties["video_overview"] = overview
                paper.properties["video_themes"] = themes
                paper.properties["video_outline"] = outline
                
                # Also compile a fallback markdown summary in case a generic UI requests it
                markdown_summary = f"## 🎥 Обзор ролика\n\n{overview}\n\n"
                if themes:
                    markdown_summary += "## 🧠 Основные темы\n\n"
                    for theme in themes:
                        markdown_summary += f"- {theme}\n"
                    markdown_summary += "\n"
                if outline:
                    markdown_summary += "## 📝 Подробный конспект\n\n"
                    for outline_item in outline:
                        markdown_summary += f"- {outline_item}\n"
                
                paper.properties["summary"] = markdown_summary
                
                if graph_repo is not None:
                    graph_repo.save_paper(paper)
                con.success(f"Structured video summary generated for {(paper.title or paper.id)[:50]}")
                return markdown_summary
            except Exception as e:
                con.warning(f"Failed to generate structured video summary: {e}. Falling back to standard summary.")

        con.dim(f"Generating summary for [bold]{paper.title[:60]}[/bold] via LLM …")
        try:
            sample_text = full_text[:4000] if full_text else ""
            prompt = (
                f"Summarize the following document. Focus on key contributions, "
                f"methodologies, and findings.\n\n"
                f"Title: {paper.title or paper.id}\n"
                f"Abstract: {paper.abstract or ''}\n\n"
                f"Content snippet:\n{sample_text}\n\n"
                f"Provide a concise, professional markdown summary."
            )
            if trace_info is not None:
                tokens_dict = trace_info.setdefault("tokens", {})
                tokens_dict["Summary Generation"] = tokens_dict.get("Summary Generation", 0) + self.llm_engine.count_tokens(prompt)
            summary = self.llm_engine.generate_response(prompt, task="synthesis")
            if summary:
                paper.properties["summary"] = summary
                if graph_repo is not None:
                    graph_repo.save_paper(paper)
                con.success(f"Summary generated for {(paper.title or paper.id)[:50]}")
                return summary
        except Exception as e:
            con.warning(f"Failed to generate summary: {e}")

        return None

    def split_text_semantically(
        self, text: str, max_chunk_tokens: int, overlap_tokens: int
    ) -> List[str]:
        """Splits text into paragraph-aware chunks with overlap constraint."""
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        def _tokens(t: str) -> int:
            if self.llm_engine:
                return self.llm_engine.count_tokens(t)
            return len(t) // 4

        chunks = []
        current_chunk_paragraphs = []
        current_tokens = 0

        for paragraph in paragraphs:
            p_tokens = _tokens(paragraph)
            
            if p_tokens > max_chunk_tokens:
                if current_chunk_paragraphs:
                    chunks.append("\n\n".join(current_chunk_paragraphs))
                    current_chunk_paragraphs = []
                    current_tokens = 0
                
                sub_paragraphs = [sp.strip() for sp in paragraph.split("\n") if sp.strip()]
                for sp in sub_paragraphs:
                    sp_tokens = _tokens(sp)
                    if current_tokens + sp_tokens > max_chunk_tokens:
                        if current_chunk_paragraphs:
                            chunks.append("\n\n".join(current_chunk_paragraphs))
                            
                            overlap_paragraphs = []
                            overlap_tokens_count = 0
                            for op in reversed(current_chunk_paragraphs):
                                op_tokens = _tokens(op)
                                if overlap_tokens_count + op_tokens <= overlap_tokens:
                                    overlap_paragraphs.insert(0, op)
                                    overlap_tokens_count += op_tokens
                                else:
                                    break
                            current_chunk_paragraphs = overlap_paragraphs
                            current_tokens = overlap_tokens_count
                            
                        if sp_tokens > max_chunk_tokens:
                            chunks.append(sp)
                            current_chunk_paragraphs = []
                            current_tokens = 0
                            continue
                            
                    current_chunk_paragraphs.append(sp)
                    current_tokens += sp_tokens
                continue

            if current_tokens + p_tokens > max_chunk_tokens:
                if current_chunk_paragraphs:
                    chunks.append("\n\n".join(current_chunk_paragraphs))
                    
                    overlap_paragraphs = []
                    overlap_tokens_count = 0
                    for op in reversed(current_chunk_paragraphs):
                        op_tokens = _tokens(op)
                        if overlap_tokens_count + op_tokens <= overlap_tokens:
                            overlap_paragraphs.insert(0, op)
                            overlap_tokens_count += op_tokens
                        else:
                            break
                    current_chunk_paragraphs = overlap_paragraphs
                    current_tokens = overlap_tokens_count

            current_chunk_paragraphs.append(paragraph)
            current_tokens += p_tokens

        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_via_llm(
        self, title: str, abstract: str, full_text: str, trace_info: Optional[dict] = None
    ) -> Optional[ExtractionResult]:
        """Attempts LLM-based extraction. Uses Map-Reduce if text is too long."""
        if not self.llm_engine:
            return None

        total_text = f"{title}\n\n{abstract}\n\n{full_text}"
        limit = config.llm_extraction_input_limit
        threshold = int(0.85 * limit)
        
        def _get_tokens(text: str) -> int:
            cnt = self.llm_engine.count_tokens(text)
            if not isinstance(cnt, (int, float)):
                return len(text) // 4
            return int(cnt)
        
        total_tokens = 0
        if _get_tokens(total_text) > threshold:
            con.info("Input text exceeds 85% of context window limit. Using Map-Reduce extraction.")
            
            prefix = f"{title}\n\n{abstract}\n\n"
            prefix_tokens = _get_tokens(prefix)
            
            chunk_budget = threshold - prefix_tokens
            max_chunk_tokens = max(chunk_budget, 1000)
            overlap_tokens = int(max_chunk_tokens * 0.15)
            
            chunks = self.split_text_semantically(full_text, max_chunk_tokens, overlap_tokens)
            con.info(f"Split document into {len(chunks)} semantic chunks for map phase.")
            
            all_authors = []
            all_concepts = {}
            all_tags = []
            
            for idx, chunk in enumerate(chunks):
                con.dim(f"Extracting concepts from chunk {idx + 1}/{len(chunks)}...")
                chunk_text = f"{prefix}{chunk}"
                
                # Accumulate tokens for this chunk's LLM call
                total_tokens += _get_tokens(chunk_text)
                
                llm_data = self.llm_engine.extract_concepts_and_metadata(chunk_text)
                if not llm_data:
                    continue
                
                all_authors.extend(llm_data.get("authors", []))
                
                for item in llm_data.get("concepts", []):
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = item.get("description", "").strip()
                    slug = slugify(c_name)
                    if slug:
                        if slug in all_concepts:
                            if len(c_desc) > len(all_concepts[slug]["description"]):
                                all_concepts[slug] = {"name": c_name, "description": c_desc}
                        else:
                            all_concepts[slug] = {"name": c_name, "description": c_desc}
                            
                all_tags.extend(llm_data.get("tags", []))
                
            concepts = list(all_concepts.values())
            
            if trace_info is not None:
                tokens_dict = trace_info.setdefault("tokens", {})
                tokens_dict["Concept & Tag Extraction"] = tokens_dict.get("Concept & Tag Extraction", 0) + total_tokens
            
            return ExtractionResult(
                authors=list(set(all_authors)),
                concepts=concepts,
                tags=list(set(all_tags)),
                via_llm=True
            )
            
        else:
            try:
                # Accumulate tokens for the single call
                total_tokens = _get_tokens(total_text)
                
                llm_data = self.llm_engine.extract_concepts_and_metadata(total_text)
                if not llm_data:
                    return None

                raw_concepts = llm_data.get("concepts", [])
                concepts = []
                for item in raw_concepts:
                    c_name = item.get("name", "").strip()
                    if not c_name:
                        continue
                    c_desc = (
                        item.get("description", "").strip()
                        or self.get_concept_description(c_name, trace_info=trace_info)
                    )
                    concepts.append({"name": c_name, "description": c_desc})

                if trace_info is not None:
                    tokens_dict = trace_info.setdefault("tokens", {})
                    tokens_dict["Concept & Tag Extraction"] = tokens_dict.get("Concept & Tag Extraction", 0) + total_tokens

                return ExtractionResult(
                    authors=llm_data.get("authors", []),
                    concepts=concepts,
                    tags=llm_data.get("tags", []),
                    via_llm=True,
                )
            except Exception as e:
                con.warning(f"LLM extraction failed, falling back to regex: {e}")
                return None

    def _extract_via_regex(
        self, title: str, abstract: str, full_text: str
    ) -> ExtractionResult:
        """Regex keyword scan against the taxonomy."""
        text_to_scan = f"{title} {abstract} {full_text[:10000]}".lower()

        concepts: List[Dict[str, str]] = []
        seen_concept_names: set = set()
        for keyword, concept_name in self._tax.get("concepts", {}).items():
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_scan):
                if concept_name not in seen_concept_names:
                    seen_concept_names.add(concept_name)
                    c_desc = self.get_concept_description(concept_name)
                    concepts.append({"name": concept_name, "description": c_desc})

        tags: List[str] = []
        for keyword, tag_name in self._tax.get("topics", {}).items():
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_scan):
                if tag_name not in tags:
                    tags.append(tag_name)

        return ExtractionResult(concepts=concepts, tags=tags, via_llm=False)

    def _normalize_extraction_result(self, result: ExtractionResult) -> ExtractionResult:
        from src.llm_schemas import LLMExtractionResponse, LLMConcept
        
        concepts = [
            LLMConcept(name=c["name"], description=c.get("description", ""))
            for c in result.concepts
        ]
        
        response_model = LLMExtractionResponse(
            authors=result.authors,
            concepts=concepts,
            tags=result.tags
        )
        
        normalized_response = self.normalization_pipeline.normalize_extraction_response(response_model)
        
        normalized_concepts = [
            {"name": c.name, "description": c.description}
            for c in normalized_response.concepts
        ]
        
        return ExtractionResult(
            authors=normalized_response.authors,
            concepts=normalized_concepts,
            tags=normalized_response.tags,
            via_llm=result.via_llm
        )

