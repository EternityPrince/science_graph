import re
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from src.services.extraction_service import ExtractionService

class CitationService:
    """Service to handle citation context extraction and classification."""
    
    def __init__(self, extractor: ExtractionService) -> None:
        """Initialize the CitationService with an ExtractionService instance.

        Args:
            extractor: The extraction service used to classify citation intents.
        """
        self.extractor = extractor

    def get_citation_context(
        self,
        full_text: str,
        ref_title: str,
        ref_author: Optional[str] = None,
        ref_year: Optional[int] = None
    ) -> str:
        """Extracts a surrounding sentence window context for a reference match.

        Args:
            full_text: The full text of the document.
            ref_title: The title of the referenced work.
            ref_author: The author of the referenced work.
            ref_year: The publication year of the referenced work.

        Returns:
            A string containing the matched sentence and its surrounding context,
            or an empty string if no match is found or arguments are invalid.
        """
        if not full_text or not ref_title:
            return ""
        
        try:
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
        except Exception:
            # Handle potential regex split errors on malformed strings
            return ""
        
        patterns = []
        if ref_title and len(ref_title) > 8:
            words = [re.escape(w) for w in ref_title.split()[:4] if len(w) > 2]
            if words:
                patterns.append(
                    re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)
                )
        if ref_author and ref_year:
            patterns.append(
                re.compile(
                    rf"\b{re.escape(ref_author)}.*\b{ref_year}\b", re.IGNORECASE
                )
            )
        elif ref_author:
            patterns.append(
                re.compile(rf"\b{re.escape(ref_author)}\b", re.IGNORECASE)
            )
            
        for idx, sent in enumerate(sentences):
            for pat in patterns:
                try:
                    if pat.search(sent):
                        start = max(0, idx - 1)
                        end = min(len(sentences), idx + 2)
                        return " ".join(sentences[start:end]).strip()
                except Exception:
                    continue
        return ""

    async def classify_cites_edges_async(
        self,
        cites_list: List[Dict[str, Any]],
        full_text: str
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Takes a list of citation dicts and classifies their citation intents.

        Args:
            cites_list: A list of dictionaries representing citations, each containing
                source_id, target_id, title, author, year, and properties.
            full_text: The full text of the document containing citations.

        Returns:
            A list of tuples of the form (source_id, target_id, "CITES", properties)
            where properties has context and intent updated.
        """
        if not cites_list:
            return []
            
        tasks = []
        metadata = []
        
        for cit in cites_list:
            ref_title = cit.get("title") or ""
            ref_author = cit.get("author")
            ref_year = cit.get("year")
            
            # Avoid potential KeyError if source_id or target_id are missing
            source_id = cit.get("source_id", "")
            target_id = cit.get("target_id", "")
            
            context = self.get_citation_context(
                full_text, ref_title, ref_author, ref_year
            )
            props = {**cit.get("properties", {})}
            if context:
                props["context"] = context
                tasks.append(
                    self.extractor.classify_citation_intent_async(
                        context, ref_title
                    )
                )
                metadata.append((source_id, target_id, props))
            else:
                props["intent"] = "BACKGROUND"
                metadata.append((source_id, target_id, props))
                tasks.append(asyncio.sleep(0, result="BACKGROUND"))
                
        intents = await asyncio.gather(*tasks)
        
        edges = []
        for (src, tgt, props), intent in zip(metadata, intents):
            props["intent"] = intent or "BACKGROUND"
            edges.append((src, tgt, "CITES", props))
        return edges
