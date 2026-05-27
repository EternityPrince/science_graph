import unittest
from unittest.mock import patch
from src.services.metadata_enricher import MetadataEnricher
from src.models import Paper

class TestMetadataEnricher(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enricher = MetadataEnricher()

    def test_enrich_no_metadata_fields(self):
        """Test enrich returns None when paper has no identifiers."""
        paper = Paper(id="p1", title="", authors=[], year=None, doi=None)
        res = self.enricher.enrich(paper)
        self.assertIsNone(res)

    @patch("src.external_api.fetch_paper_metadata")
    def test_enrich_with_doi(self, mock_fetch):
        """Test enrich calls API with DOI if available."""
        mock_fetch.return_value = {"title": "Enriched DOI Paper"}
        paper = Paper(id="p1", title="Title", authors=[], year=None, doi="10.1234/5678")
        
        res = self.enricher.enrich(paper)
        self.assertEqual(res, {"title": "Enriched DOI Paper"})
        mock_fetch.assert_called_once_with(doi="10.1234/5678", arxiv_id=None, title="Title")

    @patch("src.external_api.fetch_paper_metadata")
    def test_enrich_with_arxiv_id(self, mock_fetch):
        """Test enrich calls API with arxiv_id from properties."""
        mock_fetch.return_value = {"title": "Enriched arXiv Paper"}
        paper = Paper(id="p1", title="Title", authors=[], year=None)
        paper.properties["arxiv_id"] = "2401.12345"
        
        res = self.enricher.enrich(paper)
        self.assertEqual(res, {"title": "Enriched arXiv Paper"})
        mock_fetch.assert_called_once_with(doi=None, arxiv_id="2401.12345", title="Title")

    @patch("src.external_api.fetch_paper_metadata")
    def test_enrich_with_title_fallback(self, mock_fetch):
        """Test enrich calls API with title only if DOI and arXiv ID are missing."""
        mock_fetch.return_value = {"title": "Enriched Title Paper"}
        paper = Paper(id="p1", title="My ML Paper", authors=[], year=None)
        
        res = self.enricher.enrich(paper)
        self.assertEqual(res, {"title": "Enriched Title Paper"})
        mock_fetch.assert_called_once_with(doi=None, arxiv_id=None, title="My ML Paper")

    @patch("src.external_api.fetch_paper_metadata")
    def test_enrich_exception_handling(self, mock_fetch):
        """Test enrich catches exceptions internally and returns None (non-blocking)."""
        mock_fetch.side_effect = Exception("Semantic Scholar API down")
        paper = Paper(id="p1", title="Title", authors=[], year=None, doi="10.1234/5678")
        
        res = self.enricher.enrich(paper)
        self.assertIsNone(res)  # Should catch exception and gracefully return None

    def test_apply_metadata_to_paper(self):
        """Test apply correctly updates Paper properties, references, and citations."""
        paper = Paper(
            id="p1",
            title="Old Title",
            authors=["Author One"],
            year=2020,
            doi=None
        )
        api_meta = {
            "title": "New Enriched Title",
            "authors": ["Author One", "Author Two"],
            "year": 2024,
            "abstract": "This is a new abstract.",
            "doi": "10.9999/enriched",
            "references": [{"title": "Ref A", "doi": "ref_doi_a"}],
            "citations": [{"title": "Cit A", "doi": "cit_doi_a"}]
        }
        
        enriched_paper, refs, cits = self.enricher.apply(paper, api_meta)
        
        # Verify in-place and return value update
        self.assertEqual(enriched_paper.title, "New Enriched Title")
        self.assertEqual(enriched_paper.authors, ["Author One", "Author Two"])
        self.assertEqual(enriched_paper.year, 2024)
        self.assertEqual(enriched_paper.abstract, "This is a new abstract.")
        self.assertEqual(enriched_paper.doi, "10.9999/enriched")
        
        self.assertEqual(refs, [{"title": "Ref A", "doi": "ref_doi_a"}])
        self.assertEqual(cits, [{"title": "Cit A", "doi": "cit_doi_a"}])

    def test_apply_missing_fields_no_overwrite(self):
        """Test apply doesn't clear fields on paper if they are missing in api_meta."""
        paper = Paper(
            id="p1",
            title="Old Title",
            authors=["Author One"],
            year=2020,
            doi="10.1234/existing"
        )
        # Empty api_meta (except title)
        api_meta = {"title": "New Title"}
        
        enriched_paper, refs, cits = self.enricher.apply(paper, api_meta)
        
        self.assertEqual(enriched_paper.title, "New Title")
        self.assertEqual(enriched_paper.authors, ["Author One"])
        self.assertEqual(enriched_paper.year, 2020)
        self.assertEqual(enriched_paper.doi, "10.1234/existing")
        self.assertEqual(refs, [])
        self.assertEqual(cits, [])

    async def test_enrich_async_no_metadata_fields(self):
        paper = Paper(id="p1", title="", authors=[], year=None, doi=None)
        res = await self.enricher.enrich_async(paper)
        self.assertIsNone(res)

    @patch("src.external_api.fetch_paper_metadata_async")
    async def test_enrich_async_with_doi(self, mock_fetch_async):
        mock_fetch_async.return_value = {"title": "Enriched DOI Async"}
        paper = Paper(id="p1", title="Title", authors=[], year=None, doi="10.1234/5678")
        
        res = await self.enricher.enrich_async(paper)
        self.assertEqual(res, {"title": "Enriched DOI Async"})
        mock_fetch_async.assert_called_once_with(doi="10.1234/5678", arxiv_id=None, title="Title")

    @patch("src.external_api.fetch_paper_metadata_async")
    async def test_enrich_async_exception_handling(self, mock_fetch_async):
        mock_fetch_async.side_effect = Exception("API error")
        paper = Paper(id="p1", title="Title", authors=[], year=None, doi="10.1234/5678")
        
        res = await self.enricher.enrich_async(paper)
        self.assertIsNone(res)

    async def test_enrich_async_fallback_to_sync_if_mocked(self):
        from unittest.mock import MagicMock
        self.enricher.enrich = MagicMock(return_value={"title": "Mocked Sync Enriched"})
        
        paper = Paper(id="p1", title="Title", authors=[], year=None, doi="10.1234/5678")
        res = await self.enricher.enrich_async(paper)
        self.assertEqual(res, {"title": "Mocked Sync Enriched"})
        self.enricher.enrich.assert_called_once_with(paper)
