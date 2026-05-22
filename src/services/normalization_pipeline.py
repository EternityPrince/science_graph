"""
Normalization Pipeline Module.

Handles cleaning, translating (stub), alias resolution, lemmatization (via spaCy),
and slug generation for concepts, tags, and authors to ensure graph determinism.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import spacy
import spacy.cli
from src.models import slugify
from src.llm_schemas import LLMExtractionResponse, LLMConcept

logger = logging.getLogger(__name__)

# Default global acronyms and variation aliases mapped to canonical names.
DEFAULT_ALIASES: Dict[str, str] = {
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "cnn": "Convolutional Neural Network",
    "cnns": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "rnns": "Recurrent Neural Network",
    "nlp": "Natural Language Processing",
    "llm": "Large Language Model",
    "llms": "Large Language Model",
    "gan": "Generative Adversarial Network",
    "gans": "Generative Adversarial Network",
    "rl": "Reinforcement Learning",
    "sgd": "Stochastic Gradient Descent",
    "adam": "Adam Optimizer",
    "vit": "Vision Transformer",
    "vits": "Vision Transformer",
    "gemma": "Gemma Model",
    "bert": "BERT Model",
    "gpt": "GPT Model",
    "t5": "T5 Model",
    "lstm": "LSTM Network",
    "lstms": "LSTM Network",
}

_nlp: Optional[spacy.language.Language] = None
_spacy_attempted: bool = False


def get_spacy_nlp() -> Optional[spacy.language.Language]:
    """Lazy loaded spaCy model tokenizer & parser."""
    global _nlp, _spacy_attempted
    if _nlp is not None:
        return _nlp
    if _spacy_attempted:
        return None

    _spacy_attempted = True
    import os
    import sys
    from src.config import config
    model_name = config.spacy_model_name
    try:
        # Disable parser and NER components to make loading and lemmatization much faster
        _nlp = spacy.load(model_name, disable=["parser", "ner"])
    except OSError as e:
        is_path = os.path.sep in model_name or os.path.exists(model_name)
        if is_path:
            from src import console as con
            con.warning(
                f"spaCy model path '{model_name}' could not be loaded: {e}. "
                "Lemmatization will fall back to original text."
            )
            logger.warning(
                f"spaCy model path '{model_name}' could not be loaded: {e}. "
                "Lemmatization will fall back to original text."
            )
        else:
            try:
                from src import console as con
                con.model_msg(f"spaCy model '{model_name}' is not installed. Attempting to download...")
                
                # Check if running in a virtual environment
                # sys.prefix != sys.base_prefix is the true indicator for the current Python interpreter
                is_venv = sys.prefix != sys.base_prefix
                env_backup = {}
                
                # Backup current values of all relevant environment variables
                for var in ["VIRTUAL_ENV", "UV_SYSTEM_PYTHON", "PIP_BREAK_SYSTEM_PACKAGES"]:
                    if var in os.environ:
                        env_backup[var] = os.environ[var]
                
                if is_venv:
                    # Force VIRTUAL_ENV to match the current python interpreter's virtualenv prefix
                    os.environ["VIRTUAL_ENV"] = sys.prefix
                    # Ensure system python flags are NOT set (we want to install to the venv)
                    os.environ.pop("UV_SYSTEM_PYTHON", None)
                    os.environ.pop("PIP_BREAK_SYSTEM_PACKAGES", None)
                else:
                    # Not in virtualenv, we want to allow installing to the system python if using uv/pip
                    os.environ["UV_SYSTEM_PYTHON"] = "true"
                    os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "true"
                    # Remove any leftover VIRTUAL_ENV env var that could confuse uv
                    os.environ.pop("VIRTUAL_ENV", None)
                
                try:
                    # Always use spacy.cli.download – it uses pip or uv internally,
                    # which respects the environment variables configured above.
                    spacy.cli.download(model_name)
                finally:
                    # Restore original environment variables
                    for var in ["VIRTUAL_ENV", "UV_SYSTEM_PYTHON", "PIP_BREAK_SYSTEM_PACKAGES"]:
                        if var in env_backup:
                            os.environ[var] = env_backup[var]
                        else:
                            os.environ.pop(var, None)
                                
                _nlp = spacy.load(model_name, disable=["parser", "ner"])
                con.success(f"spaCy model '{model_name}' downloaded and loaded successfully.")
            except Exception as ex:
                from src import console as con
                con.warning(
                    f"spaCy model '{model_name}' is not installed and auto-download failed ({ex}). "
                    "Lemmatization will fall back to original text."
                )
                logger.warning(
                    f"spaCy model '{model_name}' is not installed and auto-download failed ({ex}). "
                    "Lemmatization will fall back to original text."
                )
    return _nlp


class NormalizationPipeline:
    """
    Decoupled pipeline for sanitizing, translating, lemmatizing, and resolving aliases
    for extracted graph nodes (Authors, Concepts, and Tags).
    """

    def __init__(self, aliases: Optional[Dict[str, str]] = None) -> None:
        self.aliases = aliases if aliases is not None else DEFAULT_ALIASES

    def translate_to_english(self, text: str) -> str:
        """
        Translates non-English concept names and tags to English.
        Currently implemented as a stub, returning text as-is.
        """
        # In future implementations, a translation library or API can be integrated here.
        return text

    def _lemmatize(self, text: str) -> str:
        """Converts words in the text to their base singular forms using spaCy."""
        nlp = get_spacy_nlp()
        if not nlp:
            return text

        # Lowercase text to help spaCy correctly identify singular nouns
        doc = nlp(text.strip().lower())
        lemmas = [token.lemma_ for token in doc]
        joined = " ".join(lemmas)
        import re
        return re.sub(r'\s*-\s*', '-', joined)

    def normalize_concept_name(self, name: str) -> str:
        """
        Performs full normalization on a concept name:
        1. English translation (stub)
        2. Alias/acronym lookup (case-insensitive)
        3. spaCy lemmatization (base singular uninflected form)
        4. Canonical Title Casing (preserving interior casing like acronyms if possible)
        """
        # Translate
        translated = self.translate_to_english(name.strip())
        
        # Check alias lookup (case-insensitive key comparison)
        lookup_key = translated.lower()
        if lookup_key in self.aliases:
            resolved = self.aliases[lookup_key]
        else:
            resolved = translated

        # Lemmatize
        lemmatized = self._lemmatize(resolved)

        return self._title_case(lemmatized)

    def _title_case(self, text: str) -> str:
        """Capitalizes each word, supporting sub-words separated by hyphens."""
        words = text.split()
        capitalized_words = []
        for w in words:
            if not w:
                continue
            parts = w.split('-')
            cap_parts = [p[0].upper() + p[1:] if p else "" for p in parts]
            capitalized_words.append("-".join(cap_parts))
        return " ".join(capitalized_words)

    def normalize_tag(self, tag: str) -> str:
        """
        Normalizes topic tags:
        1. English translation (stub)
        2. Alias resolution
        3. Title Case formatting
        """
        translated = self.translate_to_english(tag.strip())
        lookup_key = translated.lower()
        
        if lookup_key in self.aliases:
            resolved = self.aliases[lookup_key]
        else:
            resolved = translated
            
        return self._title_case(resolved)

    def normalize_author_name(self, name: str) -> str:
        """
        Cleans author name by removing leading/trailing spaces and formatting
        to Capital Case. Does NOT apply lemmatization or translation.
        """
        return self._title_case(name.strip())

    def normalize_description(self, description: str) -> str:
        """
        Sanitizes the description by stripping any thinking/reasoning tags and text before/within them.
        """
        if not description:
            return ""
        import re
        desc = description.strip()
        # Find first closing think/thought/reasoning tag and discard everything before it
        match = re.search(r'</(think|thought|reasoning)>', desc, re.IGNORECASE)
        if match:
            desc = desc[match.end():].strip()
        # Strip any remaining tags
        desc = re.sub(r'</?(think|thought|reasoning)>', '', desc, flags=re.IGNORECASE).strip()
        return desc

    def normalize_extraction_response(self, response: LLMExtractionResponse) -> LLMExtractionResponse:
        """
        Intercepts an LLMExtractionResponse Pydantic model and returns a new
        fully-normalized, deduplicated model prior to graph persistence.
        """
        # 1. Normalize authors (deduplicating case variations)
        normalized_authors: List[str] = []
        seen_authors = set()
        for author in response.authors:
            norm_author = self.normalize_author_name(author)
            if norm_author and norm_author.lower() not in seen_authors:
                seen_authors.add(norm_author.lower())
                normalized_authors.append(norm_author)

        # 2. Normalize and deduplicate concepts
        normalized_concepts: List[LLMConcept] = []
        seen_concepts = set()
        for concept in response.concepts:
            if not concept.name:
                continue
            norm_name = self.normalize_concept_name(concept.name)
            concept_slug = slugify(norm_name)
            
            if concept_slug and concept_slug not in seen_concepts:
                seen_concepts.add(concept_slug)
                normalized_concepts.append(
                    LLMConcept(
                        name=norm_name,
                        description=self.normalize_description(concept.description)
                    )
                )

        # 3. Normalize and deduplicate tags
        normalized_tags: List[str] = []
        seen_tags = set()
        for tag in response.tags:
            norm_tag = self.normalize_tag(tag)
            tag_slug = slugify(norm_tag)
            
            if tag_slug and tag_slug not in seen_tags:
                seen_tags.add(tag_slug)
                normalized_tags.append(norm_tag)

        return LLMExtractionResponse(
            authors=normalized_authors,
            concepts=normalized_concepts,
            tags=normalized_tags
        )
