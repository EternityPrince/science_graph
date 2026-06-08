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
    context = citation_service.get_citation_context(
        text, "Attention is all you need"
    )
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
        "We use a cheap model, e.g., Gemini Flash as suggested by A. Smith."
        in context
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


def test_author_cleaning_formats_advanced(citation_service: CitationService):
    """Test cleaning of advanced author formats, including et al., and multiple authors."""
    assert citation_service._extract_primary_author("Vaswani et al.") == "Vaswani"
    assert citation_service._extract_primary_author("Goodfellow et al.") == "Goodfellow"
    assert citation_service._extract_primary_author("Vaswani and Bengio") == "Vaswani"
    assert citation_service._extract_primary_author("Vaswani & Bengio") == "Vaswani"
    assert (
        citation_service._extract_primary_author("Vaswani, A. and Bengio, Y.")
        == "Vaswani"
    )
    assert (
        citation_service._extract_primary_author("A. Vaswani and Y. Bengio")
        == "Vaswani"
    )
    assert (
        citation_service._extract_primary_author("Vaswani, A., Bengio, Y.") == "Vaswani"
    )
    assert citation_service._extract_primary_author("et al.") is None


def test_sentence_splitting_lowercase_abbreviations(
    citation_service: CitationService,
):
    """Test that lowercase abbreviations like fig., ref., eq., vs., sec. do not trigger sentence splits."""
    text = (
        "We show our results in fig. 1. This matches the equation in eq. 3. "
        "For details, see sec. 4. Our model vs. baselines is evaluated. "
        "This is the end."
    )
    # Match for "matches the equation" which is in the second sentence if split correctly.
    # If fig. 1 split the sentence, "matches the equation" would be part of a different segment.
    # The regex splits on ". " or similar.
    # Let's verify context extraction with a title mentioned in the text.
    context = citation_service.get_citation_context(text, "equation in eq. 3")
    assert (
        "We show our results in fig. 1. This matches the equation in eq. 3. For details, see sec. 4."
        in context
    )
    assert "Our model vs. baselines is evaluated." not in context
    assert "This is the end." not in context


@pytest.mark.asyncio
async def test_classify_cites_edges_async_invalid_inputs(
    citation_service: CitationService,
):
    """Test that invalid types in cites_list are safely ignored."""
    cites_list = [None, "invalid_str", {"source_id": "a", "target_id": "b"}]
    edges = await citation_service.classify_cites_edges_async(
        cites_list, "Some text."
    )
    # The dictionary one is valid, others are skipped.
    assert len(edges) == 1
    assert edges[0][0] == "a"


@pytest.mark.asyncio
async def test_classify_cites_edges_async_schema_consistency(
    citation_service: CitationService,
):
    """Test that properties dictionary always contains context and intent keys."""
    cites_list: list[CitationInput] = [
        {
            "source_id": "paper_a",
            "target_id": "paper_b",
            "title": "BERT",
            "properties": {"existing_key": 42},
        }
    ]
    # No match context will be found, so it falls back to BACKGROUND
    edges = await citation_service.classify_cites_edges_async(
        cites_list, "This text has nothing to do with it."
    )
    assert len(edges) == 1
    props = edges[0][3]
    assert props["existing_key"] == 42
    assert props["context"] == ""
    assert props["intent"] == "BACKGROUND"


def test_citation_service_invalid_argument_types(
    citation_service: CitationService,
):
    """Test that public methods handle unexpected types without raising exceptions."""
    # _extract_primary_author
    assert citation_service._extract_primary_author(123) is None
    assert citation_service._extract_primary_author([]) is None

    # get_citation_context
    assert citation_service.get_citation_context(123, "BERT") == ""
    assert citation_service.get_citation_context("Some text.", 456) == ""
    assert (
        citation_service.get_citation_context(
            "Some text.", "BERT", sentences=123
        )
        == ""
    )


@pytest.mark.asyncio
async def test_classify_cites_edges_async_resilience_to_exceptions(
    mock_extractor: MagicMock, citation_service: CitationService
):
    """Test that if a classification task raises an exception, the pipeline falls back to BACKGROUND."""

    # Mocking first call to raise an exception, second call to succeed
    mock_extractor.classify_citation_intent_async = AsyncMock(
        side_effect=[Exception("LLM Timeout"), "USES_METHOD"]
    )

    cites_list: list[CitationInput] = [
        {
            "source_id": "paper_a",
            "target_id": "paper_b",
            "title": "BERT",
        },
        {
            "source_id": "paper_a",
            "target_id": "paper_c",
            "title": "GPT",
        },
    ]

    text = "We use BERT. We use GPT."
    edges = await citation_service.classify_cites_edges_async(cites_list, text)

    assert len(edges) == 2

    # Check paper_b (raised exception, fallback to BACKGROUND)
    edge_b = [e for e in edges if e[1] == "paper_b"][0]
    assert edge_b[3]["intent"] == "BACKGROUND"
    assert "BERT" in edge_b[3]["context"]

    # Check paper_c (succeeded, should be USES_METHOD)
    edge_c = [e for e in edges if e[1] == "paper_c"][0]
    assert edge_c[3]["intent"] == "USES_METHOD"
    assert "GPT" in edge_c[3]["context"]


def test_sentence_splitting_academic_abbreviations(
    citation_service: CitationService,
):
    """Test sentence splitting lookbehinds with additional academic abbreviations."""
    text = (
        "See details in vol. 2. Read ch. 5 for more. Refer to no. 12! "
        "Also cf. the appendix. Check pp. 3-4 for details. This is etc. "
        "The end."
    )
    context = citation_service.get_citation_context(text, "appendix")
    # All of the abbreviations should be kept in the same sentence segment.
    # The segment with "appendix" starts from "Also cf. the appendix."
    # The preceding sentence is "Refer to no. 12!"
    # The succeeding sentence is "Check pp. 3-4 for details."
    assert "Also cf. the appendix." in context
    assert "no. 12!" in context
    assert "pp. 3-4" in context
    assert "vol. 2" not in context
    assert "ch. 5" not in context


def test_empirical_word_splitting(citation_service: CitationService):
    """Test that words ending in 'al.' (e.g. 'empirical.') split correctly."""
    text = "Pre-sentence. This result is empirical. Next sentence follows."
    context = citation_service.get_citation_context(text, "Next sentence")
    assert "This result is empirical. Next sentence follows." in context
    assert "Pre-sentence." not in context


def test_priority_author_year_over_title(citation_service: CitationService):
    """Test that Author + Year pattern takes precedence over Title pattern."""
    text = (
        "Pre-sentence. We use a generic title here. This is a buffer sentence. "
        "In 2017, Vaswani et al. introduced the transformer model. Post-sentence."
    )
    # Search with title matching the first sentence, and author+year matching the second.
    # Because Author+Year is prioritized, it should return the context of the second sentence.
    context = citation_service.get_citation_context(
        text,
        ref_title="generic title",
        ref_author="Vaswani",
        ref_year=2017,
    )
    assert (
        "In 2017, Vaswani et al. introduced the transformer model." in context
    )
    assert "We use a generic title here." not in context


def test_year_regex_escaping_safety(citation_service: CitationService):
    """Test that regex characters in year are escaped safely without crashing."""
    text = "We use some special year formats like [2017] in this text."
    context = citation_service.get_citation_context(
        text, ref_title="Some Title", ref_author="Special", ref_year="[2017]"
    )
    # Even if it matches nothing, it should not crash.
    assert isinstance(context, str)

    # Let's verify it can match when escaped correctly
    text2 = "Special author in year [2017] says hello."
    context2 = citation_service.get_citation_context(
        text2, ref_title="Some Title", ref_author="Special", ref_year="[2017]"
    )
    assert "Special author in year [2017] says hello." in context2


def test_comma_separated_author_lists(citation_service: CitationService):
    """Test that comma-separated lists of authors are parsed correctly."""
    assert (
        citation_service._extract_primary_author("A. Vaswani, Y. Bengio")
        == "Vaswani"
    )
    assert (
        citation_service._extract_primary_author(
            "A. Vaswani, Y. Bengio, and others"
        )
        == "Vaswani"
    )
