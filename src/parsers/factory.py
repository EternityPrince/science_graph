from src.parsers.base import BaseParser
from src.parsers.pdf_parser import PDFParser
from src.parsers.md_parser import MarkdownParser
from src.parsers.url_parser import UrlParser
from src.parsers.epub_parser import EPUBParser

class ParserFactory:
    @staticmethod
    def get_parser(source: str) -> BaseParser:
        """
        Inspects the source string and returns the correct parser instance.
        """
        source_lower = source.lower().strip()
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            return UrlParser()
        elif source_lower.endswith(".pdf"):
            return PDFParser()
        elif source_lower.endswith(".md"):
            return MarkdownParser()
        elif source_lower.endswith(".epub"):
            return EPUBParser()
        else:
            raise ValueError(f"No parser available for source: {source}")
