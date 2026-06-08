import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.citation_service import CitationService
from src.services.extraction_service import ExtractionService

@pytest.mark.asyncio
async def test_citation_service_context_extraction():
    extractor = MagicMock(spec=ExtractionService)
    extractor.classify_citation_intent_async = AsyncMock(return_value="METHODology")
    
    service = CitationService(extractor)
    text = (
        "This is first sentence. We use the method described in "
        "DeepLearning Book. Third sentence."
    )
    
    # Verify regex context extraction
    context = service.get_citation_context(text, "DeepLearning Book")
    assert "We use the method described in DeepLearning Book." in context


@pytest.mark.asyncio
async def test_citation_service_empty_inputs():
    extractor = MagicMock(spec=ExtractionService)
    service = CitationService(extractor)
    
    assert service.get_citation_context("", "Title") == ""
    assert service.get_citation_context("Some text.", "") == ""
    
    edges = await service.classify_cites_edges_async([], "Some text.")
    assert edges == []


@pytest.mark.asyncio
async def test_classify_cites_edges_async():
    extractor = MagicMock(spec=ExtractionService)
    
    async def mock_classify_citation_intent_async(context: str, ref_title: str) -> str:
        if "DeepLearning" in ref_title:
            return "USES_METHOD"
        return "BACKGROUND"
        
    extractor.classify_citation_intent_async = AsyncMock(
        side_effect=mock_classify_citation_intent_async
    )
    
    service = CitationService(extractor)
    
    cites_list = [
        {
            "source_id": "paper_a",
            "target_id": "paper_b",
            "title": "DeepLearning Book",
            "author": "Goodfellow",
            "year": 2016,
            "properties": {"some_prop": 1}
        },
        {
            "source_id": "paper_a",
            "target_id": "paper_c",
            "title": "Other Paper Title",
            "properties": {}
        }
    ]
    
    text = (
        "This is first sentence. Goodfellow (2016) introduced a great "
        "framework. Some unrelated text is written here directly."
    )
    
    edges = await service.classify_cites_edges_async(cites_list, text)
    
    assert len(edges) == 2
    
    # Check paper_b (should match author/year regex context)
    edge_b = [e for e in edges if e[1] == "paper_b"][0]
    assert edge_b[0] == "paper_a"
    assert edge_b[2] == "CITES"
    assert edge_b[3]["intent"] == "USES_METHOD"
    assert "Goodfellow (2016)" in edge_b[3]["context"]
    assert edge_b[3]["some_prop"] == 1
    
    # Check paper_c (no match, fallback to BACKGROUND)
    edge_c = [e for e in edges if e[1] == "paper_c"][0]
    assert edge_c[0] == "paper_a"
    assert edge_c[2] == "CITES"
    assert edge_c[3]["intent"] == "BACKGROUND"
    assert "context" not in edge_c[3]
