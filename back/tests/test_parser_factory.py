import pytest
from src.parsers.factory import ParserFactory
from src.parsers.pdf_parser import PDFParser
from src.parsers.marker_parser import MarkerPDFParser
from src.parsers.md_parser import MarkdownParser
from src.parsers.url_parser import UrlParser
from src.parsers.epub_parser import EPUBParser
from src.parsers.youtube_parser import YoutubeVideoParser

def test_parser_factory_urls():
    # YouTube URL
    p = ParserFactory.get_parser("https://youtube.com/watch?v=123")
    assert isinstance(p, YoutubeVideoParser)
    p2 = ParserFactory.get_parser("https://youtu.be/123")
    assert isinstance(p2, YoutubeVideoParser)

    # Standard URL
    p3 = ParserFactory.get_parser("https://example.com/paper")
    assert isinstance(p3, UrlParser)

def test_parser_factory_pdfs(monkeypatch):
    # Marker PDF parser
    p = ParserFactory.get_parser("test.pdf", pdf_parser_type="marker")
    assert isinstance(p, MarkerPDFParser)

    # Default PDF parser
    p2 = ParserFactory.get_parser("test.pdf", pdf_parser_type="pymupdf")
    assert isinstance(p2, PDFParser)

    # Fallback to config
    from src.config import config
    monkeypatch.setitem(config.data, "pdf_parser", "marker")
    p3 = ParserFactory.get_parser("test.pdf")
    assert isinstance(p3, MarkerPDFParser)

def test_parser_factory_other_extensions():
    # Markdown
    p = ParserFactory.get_parser("doc.md")
    assert isinstance(p, MarkdownParser)

    # EPUB
    p2 = ParserFactory.get_parser("book.epub")
    assert isinstance(p2, EPUBParser)

def test_parser_factory_unsupported():
    with pytest.raises(ValueError) as exc:
        ParserFactory.get_parser("unsupported.txt")
    assert "No parser available for source" in str(exc.value)
