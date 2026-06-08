import pytest
from unittest.mock import AsyncMock, MagicMock

from src.services.citation_service import CitationService, CitationInput
from src.services.extraction_service import ExtractionService


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Fixture for ExtractionService mock."""
    return MagicMock(spec=ExtractionService)


@pytest.fixture
def citation_service(mock_extractor: MagicMock) -> CitationService:
    """Fixture for CitationService."""
    return CitationService(mock_extractor)


def test_citation_service_context_extraction(
    citation_service: CitationService,
):
    """Test standard citation context extraction."""
    text = (
        "This is first sentence. We use the method described in "
        "DeepLearning Book. Third sentence."
    )

    # Verify regex context extraction
    context = citation_service.get_citation_context(text, "DeepLearning Book")
    assert "We use the method described in DeepLearning Book." in context


def test_citation_service_empty_inputs(citation_service: CitationService):
    """Test that empty inputs return empty strings / lists safely."""
    assert citation_service.get_citation_context("", "Title") == ""
    assert citation_service.get_citation_context("Some text.", "") == ""


@pytest.mark.asyncio
async def test_classify_cites_edges_empty_inputs(
    citation_service: CitationService,
):
    """Test classifying empty cites list returns empty list."""
    edges = await citation_service.classify_cites_edges_async([], "Some text.")
    assert edges == []


def test_author_cleaning_formats(citation_service: CitationService):
    """Test cleaning of various author formats using _extract_primary_author."""
    assert citation_service._extract_primary_author("Goodfellow, I.") == "Goodfellow"
    assert citation_service._extract_primary_author("Goodfellow, Ian") == "Goodfellow"
    assert citation_service._extract_primary_author("I. Goodfellow") == "Goodfellow"
    assert citation_service._extract_primary_author("Ian Goodfellow") == "Goodfellow"
    assert citation_service._extract_primary_author("Goodfellow") == "Goodfellow"
    assert citation_service._extract_primary_author("") is None
    assert citation_service._extract_primary_author("   ") is None
    assert citation_service._extract_primary_author(None) is None


def test_short_titles(citation_service: CitationService):
    """Test matching for short titles and acronyms like BERT or ResNet."""
    text = "Sentence one. The BERT model was evaluated on GLUE. Sentence three."
    context = citation_service.get_citation_context(text, "BERT")
    assert "The BERT model was evaluated on GLUE." in context

    text = "We compare against ResNet. It achieves lower error."
    context = citation_service.get_citation_context(text, "ResNet")
    assert "We compare against ResNet." in context


def test_title_with_short_words(citation_service: CitationService):
    """Test matching titles containing short words (e.g. 'is', 'all')."""
    text = "First. We refer to Attention is all you need for details. Third."
    context = citation_service.get_citation_context(text, "Attention is all you need")
    assert "We refer to Attention is all you need for details." in context


def test_author_year_order_independence(citation_service: CitationService):
    """Test matching when year appears before the author in the sentence."""
    text = "First. In 2017, Vaswani et al. introduced transformer. Third."
    context = citation_service.get_citation_context(
        text, "Attention is all you need", "Vaswani", 2017
    )
    assert "In 2017, Vaswani et al. introduced transformer." in context


def test_sentence_splitting_anomalies(citation_service: CitationService):
    """Test sentence splitting with decimals, abbreviations, and et al."""
    # Decimal "1.5", abbreviation "e.g.", exclamation mark "!"
    # and single-letter uppercase initials "A. Smith"
    text = (
        "Pre-sentence. Version 1.5 is released! We use a cheap model, "
        "e.g., Gemini Flash as suggested by A. Smith. It achieves high "
        "throughput. Post-sentence."
    )
    context = citation_service.get_citation_context(text, "Gemini Flash")
    assert (
        "We use a cheap model, e.g., Gemini Flash as suggested by A. Smith." in context
    )
    # It should not include "Pre-sentence." or "Post-sentence." since they
    # are outside the window.
    assert "Pre-sentence." not in context
    assert "Post-sentence." not in context

    # "et al." abbreviation anomaly
    text = (
        "This is first sentence. Goodfellow et al. (2016) introduced a great "
        "framework. Third sentence."
    )
    context = citation_service.get_citation_context(
        text, "DeepLearning Book", "Goodfellow", 2016
    )
    assert "Goodfellow et al. (2016) introduced a great framework." in context


@pytest.mark.asyncio
async def test_classify_cites_edges_async_properties_none(
    mock_extractor: MagicMock, citation_service: CitationService
):
    """Test classify_cites_edges_async handles properties=None safely."""
    mock_extractor.classify_citation_intent_async = AsyncMock(
        return_value="USES_METHOD"
    )

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
    edges = await citation_service.classify_cites_edges_async(cites_list, text)

    assert len(edges) == 1
    edge = edges[0]
    assert edge[0] == "paper_a"
    assert edge[1] == "paper_b"
    assert edge[2] == "CITES"
    assert edge[3]["intent"] == "USES_METHOD"
    assert "context" in edge[3]
