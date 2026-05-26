from unittest.mock import MagicMock, patch
import pytest

from src.parsers.epub_parser import EPUBParser
from src.models import Paper

class TestEPUBParser:
    @pytest.fixture
    def parser(self):
        return EPUBParser()

    def test_clean_html_strips_tags_and_html_entities(self):
        from src.parsers.epub_parser import _clean_html
        raw_html = "<html><body><h1>Hello World</h1><p>This is a paragraph with &nbsp; and &amp; and &lt;tag&gt;.</p></body></html>"
        cleaned = _clean_html(raw_html)
        assert cleaned == "Hello World This is a paragraph with and & and <tag>."

    @patch("ebooklib.epub.read_epub")
    def test_parse_epub_successfully(self, mock_read_epub, parser):
        # Create a mock EpubBook
        mock_book = MagicMock()
        
        def mock_get_metadata(namespace, name):
            meta_map = {
                "title": [("Test EPUB Book", {})],
                "creator": [("Author One, Author Two and Author Three", {})],
                "language": [("en", {})],
                "identifier": [("id12345", {})]
            }
            return meta_map.get(name, [])
            
        mock_book.get_metadata.side_effect = mock_get_metadata

        # Create mock chapter item
        mock_item = MagicMock()
        mock_item.get_content.return_value = b"<html><body><h1>Chapter 1</h1><p>This is the content of chapter one of our test EPUB book. It has more than fifty characters.</p></body></html>"
        
        import ebooklib
        mock_book.get_items_of_type.side_effect = lambda itype: [mock_item] if itype == ebooklib.ITEM_DOCUMENT else []

        mock_read_epub.return_value = mock_book

        # Parse dummy file path
        paper, links, full_text = parser.parse("dummy_path.epub")

        assert isinstance(paper, Paper)
        assert paper.title == "Test EPUB Book"
        assert paper.authors == ["Author One", "Author Two", "Author Three"]
        assert paper.properties["source_type"] == "book"
        assert paper.properties["epub_identifier"] == "id12345"
        assert paper.properties["language"] == "en"
        assert paper.properties["chapter_count"] == 1
        assert len(links) == 0
        assert "This is the content of chapter one" in full_text
        assert "test_epub_book" in paper.id
        
        mock_read_epub.assert_called_once_with("dummy_path.epub", options={"ignore_ncx": True})
