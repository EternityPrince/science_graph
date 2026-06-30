import unittest
from unittest.mock import patch
from src.parsers.url_parser import UrlParser

class TestURLParser(unittest.TestCase):
    @patch("src.parsers.url_parser.requests.get")
    def test_parse_url_extracts_author_and_saves_content(self, mock_get):
        # Mock responses
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <head>
                <title>Как я обучил GPT с нуля на русском языке</title>
                <meta name="author" content="Vladimir Kasterin">
                <meta property="article:author" content="HabrAuthor">
            </head>
            <body>
                <main>
                    <h1>Как я обучил GPT с нуля на русском языке</h1>
                    <p>Это подробная статья про обучение языковой модели.</p>
                </main>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)

        paper, references, md_content = UrlParser().parse("https://habr.com/ru/articles/1037532/")
        
        self.assertEqual(paper.title, "Как я обучил GPT с нуля на русском языке")
        self.assertIn("Vladimir Kasterin", paper.authors)
        self.assertIn("HabrAuthor", paper.authors)
        self.assertIn("обучение языковой модели", md_content)

    @patch("src.parsers.url_parser.requests.get")
    def test_parse_url_author_css_and_profile_links(self, mock_get):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <head>
                <title>Some Tech Blog Post</title>
            </head>
            <body>
                <div class="tm-user-info__username">Alice_Dev</div>
                <a class="author-link" href="https://example.com/users/Bob_Coder/">Bob Link</a>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)

        paper, references, _ = UrlParser().parse("https://example.com/blog/1")
        self.assertIn("Alice_Dev", paper.authors)
        self.assertIn("Bob Link", paper.authors)

    @patch("src.parsers.url_parser.requests.get")
    def test_arxiv_id_and_doi_detection_from_url(self, mock_get):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass
        mock_get.return_value = MockResponse("<html><body>Content</body></html>")

        # Test arXiv detection from URL
        paper, _, _ = UrlParser().parse("https://arxiv.org/abs/2304.12345v1")
        self.assertEqual(paper.properties["arxiv_id"], "2304.12345v1")
        self.assertTrue(paper.id.startswith("arxiv_"))

        # Test DOI detection from URL
        paper_doi, _, _ = UrlParser().parse("https://doi.org/10.1000/xyz123")
        self.assertEqual(paper_doi.doi, "10.1000/xyz123")
        self.assertTrue(paper_doi.id.startswith("doi_"))

    @patch("src.parsers.url_parser.requests.get")
    def test_academic_metadata_tags_extraction(self, mock_get):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <head>
                <meta name="citation_title" content="A Great Academic Paper">
                <meta name="citation_author" content="First Author">
                <meta name="dc.creator" content="Second Creator">
                <meta name="citation_doi" content="10.1234/meta.doi.555">
                <meta name="citation_arxiv_id" content="2101.99999">
                <meta name="citation_date" content="2025/03/15">
                <meta name="citation_abstract" content="This is the citation abstract.">
            </head>
            <body>
                <article>
                    Some article content here.
                </article>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)

        paper, _, _ = UrlParser().parse("https://example.org/academic-page")
        self.assertEqual(paper.title, "A Great Academic Paper")
        self.assertIn("First Author", paper.authors)
        self.assertIn("Second Creator", paper.authors)
        self.assertEqual(paper.doi, "10.1234/meta.doi.555")
        self.assertEqual(paper.properties["arxiv_id"], "2101.99999")
        self.assertEqual(paper.year, 2025)
        self.assertEqual(paper.abstract, "This is the citation abstract.")

    @patch("src.parsers.url_parser.requests.get")
    def test_author_splitting_and_cleaning(self, mock_get):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <head>
                <meta name="author" content="John Doe, Jane Smith; Bob Jones and Alice Wonderland">
                <meta name="parsely-author" content="https://example.com/author/john-doe">
                <meta name="author-name" content="admin"> <!-- should be ignored -->
                <meta name="twitter:creator" content="moderator@example.com"> <!-- should be ignored -->
            </head>
            <body>
                <span itemprop="author">Itemprop Author</span>
                <span itemprop="creator"><span itemprop="name">Itemprop Name Creator</span></span>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)

        paper, _, _ = UrlParser().parse("https://example.com/blog-post")
        self.assertIn("John Doe", paper.authors)
        self.assertIn("Jane Smith", paper.authors)
        self.assertIn("Bob Jones", paper.authors)
        self.assertIn("Alice Wonderland", paper.authors)
        self.assertIn("john-doe", paper.authors) # extracted from URL parts[-1]
        self.assertIn("Itemprop Author", paper.authors)
        self.assertIn("Itemprop Name Creator", paper.authors)
        self.assertNotIn("admin", paper.authors)
        self.assertNotIn("moderator@example.com", paper.authors)

    @patch("src.external_api.fetch_paper_metadata")
    @patch("src.parsers.url_parser.requests.get")
    def test_semantic_scholar_enrichment(self, mock_get, mock_fetch):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass
        mock_get.return_value = MockResponse("<html><body>Content</body></html>")

        # Mock successful fetch from Semantic Scholar
        mock_fetch.return_value = {
            "title": "Enriched Paper Title",
            "authors": ["Semantic Scholar Author"],
            "year": 2026,
            "abstract": "Enriched abstract text",
            "doi": "10.9999/enriched.doi"
        }

        # Passing DOI via URL to trigger SS call
        paper, _, _ = UrlParser().parse("https://doi.org/10.9999/enriched.doi")
        self.assertEqual(paper.title, "Enriched Paper Title")
        self.assertEqual(paper.authors, ["Semantic Scholar Author"])
        self.assertEqual(paper.year, 2026)
        self.assertEqual(paper.abstract, "Enriched abstract text")
        self.assertEqual(paper.doi, "10.9999/enriched.doi")
        self.assertTrue(paper.properties["api_enriched"])

        # Test Semantic Scholar fetch failure (should fall back gracefully)
        mock_fetch.side_effect = Exception("SS API Error")
        paper_fail, _, _ = UrlParser().parse("https://doi.org/10.9999/enriched.doi")
        self.assertFalse(paper_fail.properties["api_enriched"])

    @patch("src.parsers.url_parser.requests.get")
    def test_link_extraction_filtering(self, mock_get):
        class MockResponse:
            def __init__(self, text, status_code=200):
                self.text = text
                self.status_code = status_code
            def raise_for_status(self):
                pass

        html_content = """
        <html>
            <body>
                <main>
                    <a href="https://example.com/target-link">Valid Link</a>
                    <a href="https://example.com/target-link#section">Same Link Anchor</a>
                    <a href="mailto:someone@example.com">Mail Link</a>
                    <a href="javascript:void(0)">JS Link</a>
                    <a href="https://example.com/current-url">Current Page Link</a>
                </main>
            </body>
        </html>
        """
        mock_get.return_value = MockResponse(html_content)

        paper, links, _ = UrlParser().parse("https://example.com/current-url")
        self.assertIn("https://example.com/target-link", links)
        self.assertNotIn("mailto:someone@example.com", links)
        self.assertNotIn("javascript:void(0)", links)
        self.assertNotIn("https://example.com/current-url", links)

    @patch("src.parsers.url_parser.requests.get")
    def test_fetch_error_raises_runtime_error(self, mock_get):
        mock_get.side_effect = Exception("Network Down")
        
        with self.assertRaises(RuntimeError) as context:
            UrlParser().parse("https://example.com")
            
        self.assertIn("Failed to fetch URL", str(context.exception))

if __name__ == "__main__":
    unittest.main()
