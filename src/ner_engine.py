"""
NER Engine — extracts PERSON entities from text using a cached HuggingFace model.
Falls back to regex heuristics if the model is unavailable.
"""

import re
import os
from typing import List, Optional

# Regex to find human name patterns (2-4 capitalized words, allowing initials like "A.")
# Matches patterns like: "Ashish Vaswani", "Aidan N. Gomez", "Lukasz Kaiser"
_NAME_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){1,3})\b'
)

# Known non-person false positives for academic papers
_STOPWORDS = {
    "Abstract", "Introduction", "Related Work", "Conclusion", "References",
    "Figure", "Table", "Appendix", "Neural Network", "Deep Learning",
    "Machine Learning", "Language Model", "Attention Mechanism",
    "Neural Networks", "Natural Language", "Artificial Intelligence",
    "Google Brain", "Google Research", "University", "Institute",
    "Conference", "Workshop", "Journal", "Proceedings",
}


def _is_likely_name(text: str) -> bool:
    """Heuristic check: is this string a plausible human name?"""
    words = text.split()
    if len(words) < 2 or len(words) > 5:
        return False
    if any(w.lower() in {"and", "or", "the", "of", "in", "for", "with", "at"} for w in words):
        return False
    if any(char.isdigit() for char in text):
        return False
    if text in _STOPWORDS:
        return False
    if len(text) > 50:
        return False
    return True


def _is_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is already in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        return result is not None
    except Exception:
        return False


class NEREngine:
    """Extracts PERSON entities from text using bert-base-NER."""
    
    MODEL_ID = "dslim/bert-base-NER"
    
    def __init__(self):
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        from src import console as con
        from src.config import config

        # Set HF token if configured
        hf_token = config.hf_token
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

        is_cached = _is_model_cached(self.MODEL_ID)
        if not is_cached:
            con.info(f"Downloading NER model [bold]{self.MODEL_ID}[/bold] (one-time, ~400 MB)…")
        else:
            con.model_msg(f"Loading NER model [bold]{self.MODEL_ID}[/bold] from cache…")

        try:
            from transformers import pipeline
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use 'first' strategy — more reliable for multi-token names than 'simple'
                self._pipeline = pipeline(
                    "ner",
                    model=self.MODEL_ID,
                    aggregation_strategy="first",
                    token=hf_token or None,
                )
            con.success(f"NER model ready: [bold]{self.MODEL_ID}[/bold]")
        except Exception as e:
            con.warning(f"NER model could not load ({e}). Falling back to regex extraction.")
            self._pipeline = None

    def extract_persons(self, text: str, max_text: int = 4000) -> List[str]:
        """
        Extract person names from `text`.
        Uses HuggingFace NER pipeline when available, else falls back to regex.
        """
        snippet = text[:max_text]
        
        if self._pipeline is not None:
            return self._ner_extract(snippet)
        return self._regex_extract(snippet)

    def _ner_extract(self, text: str) -> List[str]:
        try:
            entities = self._pipeline(text)
            persons = []
            for ent in entities:
                if ent.get("entity_group") == "PER":
                    word = ent["word"].strip()
                    # Clean WordPiece subword artifacts: "##ish Vaswani" → "Vaswani" (partial)
                    # We rebuild the name from the original text span when possible
                    start, end = ent.get("start"), ent.get("end")
                    if start is not None and end is not None:
                        # Extract directly from source text to avoid subword issues
                        name = text[start:end].strip()
                    else:
                        name = re.sub(r'^#+', '', word).strip()
                    if _is_likely_name(name) and name not in persons:
                        persons.append(name)
            return persons
        except Exception:
            return self._regex_extract(text)

    def _regex_extract(self, text: str) -> List[str]:
        """Pure-regex fallback — looks for capitalized name patterns."""
        found = _NAME_PATTERN.findall(text)
        seen = set()
        result = []
        for name in found:
            if _is_likely_name(name) and name not in seen:
                seen.add(name)
                result.append(name)
        return result


# Module-level singleton, lazy-initialized
_ner_instance: Optional[NEREngine] = None


def get_ner_engine() -> NEREngine:
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = NEREngine()
    return _ner_instance


def extract_persons_from_text(text: str) -> List[str]:
    """Convenience wrapper — returns deduplicated list of person names."""
    return get_ner_engine().extract_persons(text)
