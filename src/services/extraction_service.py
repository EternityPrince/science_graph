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
from src.models import Paper
from src import console as con


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
    ) -> ExtractionResult:
        """
        Extracts authors, concepts, and tags from the provided text.

        Args:
            title:     Document title.
            abstract:  Document abstract (may be empty).
            full_text: Full document body text (used for keyword scan).
            use_llm:   Whether to attempt LLM-based extraction first.

        Returns:
            ExtractionResult with authors, concepts, and tags.
        """
        llm_result = None
        if use_llm and self.llm_engine:
            llm_result = self._extract_via_llm(title, abstract, full_text)

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

            return llm_result

        return regex_result

    def get_concept_description(self, name: str) -> str:
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
                desc = self.llm_engine.generate_response(prompt, task="extraction").strip()
                desc = re.sub(r'^["\']|["\']$', "", desc).strip()
                if desc:
                    return desc
            except Exception:
                pass

        return f"A key concept representing '{name}' within the AI/ML literature."

    def generate_summary(self, paper: Paper, full_text: str, graph_repo: Any = None) -> Optional[str]:
        """
        Generates an LLM summary for a paper and optionally persists it.

        Args:
            paper:      The Paper object whose summary is to be generated.
            full_text:  Full document text (used as context, first 4000 chars).
            graph_repo: If provided, saves the updated paper after generating summary.

        Returns:
            The generated summary string, or None if LLM is unavailable or fails.
        """
        if not self.llm_engine:
            return None

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

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_via_llm(
        self, title: str, abstract: str, full_text: str
    ) -> Optional[ExtractionResult]:
        """Attempts LLM-based extraction. Returns None on any failure."""
        try:
            sample = f"{title}\n\n{abstract}\n\n{full_text[:4000]}"
            llm_data = self.llm_engine.extract_concepts_and_metadata(sample)
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
                    or self.get_concept_description(c_name)
                )
                concepts.append({"name": c_name, "description": c_desc})

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
