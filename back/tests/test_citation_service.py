import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.citation_service import CitationService, CitationInput
from src.services.extraction_service import ExtractionService


def test_citation_service_context_extraction():
    """Test standard citation context extraction."""
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)
    text = (
        "This is first sentence. We use the method described in "
        "DeepLearning Book. Third sentence."
    )

    # Verify regex context extraction
    context = service.get_citation_context(text, "DeepLearning Book")
    assert "We use the method described in DeepLearning Book." in context


def test_citation_service_empty_inputs():
    """Test that empty inputs return empty strings / lists safely."""
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)

    assert service.get_citation_context("", "Title") == ""
    assert service.get_citation_context("Some text.", "") == ""


@pytest.mark.asyncio
async def test_classify_cites_edges_empty_inputs():
    """Test classifying empty cites list returns empty list."""
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)
    edges = await service.classify_cites_edges_async([], "Some text.")
    assert edges == []


def test_author_cleaning_formats():
    """Test cleaning of various author formats using _extract_primary_author."""
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)

    assert service._extract_primary_author("Goodfellow, I.") == "Goodfellow"
    assert service._extract_primary_author("Goodfellow, Ian") == "Goodfellow"
    assert service._extract_primary_author("I. Goodfellow") == "Goodfellow"
    assert service._extract_primary_author("Ian Goodfellow") == "Goodfellow"
    assert service._extract_primary_author("Goodfellow") == "Goodfellow"
    assert service._extract_primary_author("") is None
    assert service._extract_primary_author(None) is None


def test_short_titles():
    """Test matching for short titles and acronyms like BERT or ResNet."""
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)

    text = "Sentence one. The BERT model was evaluated on GLUE. Sentence three."
    context = service.get_citation_context(text, "BERT")
    assert "The BERT model was evaluated on GLUE." in context

    text = "We compare against ResNet. It achieves lower error."
    context = service.get_citation_context(text, "ResNet")
    assert "We compare against ResNet." in context


def test_sentence_splitting_anomalies():
    """Test sentence splitting with decimals, abbreviations, and et al."""
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)

    # Decimal "1.5" and abbreviation "e.g."
    text = (
        "Pre-sentence. Version 1.5 is released. We use a cheap model, "
        "e.g., Gemini Flash. It achieves high throughput. Post-sentence."
    )
    context = service.get_citation_context(text, "Gemini Flash")
    assert "We use a cheap model, e.g., Gemini Flash." in context
    # It should not include "Pre-sentence." or "Post-sentence." since they
    # are outside the window.
    assert "Pre-sentence." not in context
    assert "Post-sentence." not in context

    # "et al." abbreviation anomaly
    text = (
        "This is first sentence. Goodfellow et al. (2016) introduced a great "
        "framework. Third sentence."
    )
    context = service.get_citation_context(
        text, "DeepLearning Book", "Goodfellow", 2016
    )
    assert "Goodfellow et al. (2016) introduced a great framework." in context


@pytest.mark.asyncio
async def test_classify_cites_edges_async_properties_none():
    """Test classify_cites_edges_async handles properties=None safely."""
    extractor = MagicMock(spec=ExtractionService)

    async def mock_classify_citation_intent_async(
        context: str, ref_title: str
    ) -> str:
        return "USES_METHOD"

    extractor.classify_citation_intent_async = AsyncMock(
        side_effect=mock_classify_citation_intent_async
    )

    service = CitationService(extractor)

    # properties is None
    cites_list: list[CitationInput] = [
        {
            "source_id": "paper_a",
            "target_id": "paper_b",
            "title": "BERT",
            "properties": None,
        }
    ]

    text = "We use BERT for classification."
    edges = await service.classify_cites_edges_async(cites_list, text)

    assert len(edges) == 1
    edge = edges[0]
    assert edge[0] == "paper_a"
    assert edge[1] == "paper_b"
    assert edge[2] == "CITES"
    assert edge[3]["intent"] == "USES_METHOD"
    assert "context" in edge[3]

