import os
import tempfile
from unittest.mock import patch
import pytest
import fitz  # PyMuPDF

from src.parsers.pdf_parser import PDFParser
from src.models import Paper

class TestPDFParser:
    @pytest.fixture
    def parser(self):
        return PDFParser()

    def test_parse_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("non_existent_file.pdf")

    def test_parse_with_metadata(self, parser):
        # Create a temp PDF with metadata
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Abstract: This is a short abstract of the paper.\nIntroduction: We introduce stuff here.")
            page.insert_text((50, 100), "References\n[1] John Doe. A great paper. 2024.\n[2] Jane Smith. Another paper. 2025.")
            
            # Set metadata
            metadata = {
                "title": "Quantum Computation",
                "author": "Alice Smith and Bob Jones",
                "creationDate": "D:20240815120000Z"
            }
            doc.set_metadata(metadata)
            doc.save(pdf_path)
            doc.close()

            paper, references, full_text = parser.parse(pdf_path)

            assert isinstance(paper, Paper)
            assert paper.title == "Quantum Computation"
            assert paper.authors == ["Alice Smith", "Bob Jones"]
            assert paper.year == 2024
            assert paper.abstract == "This is a short abstract of the paper."
            assert len(references) == 2
            assert "John Doe. A great paper. 2024." in references
            assert "quantum_computation" in paper.id
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_parse_fallback_title_and_authors(self, parser):
        # Create a temp PDF without title metadata
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            # Insert author list after title
            text_lines = [
                "JOURNAL OF SCIENCE PREPRINT", # should be skipped as journal header
                "Attention mechanisms in deep learning networks", # title candidate
                "Ashish Vaswani, Noam Shazeer (Copyright)", # authors (Copyright skips title, not skipped in authors)
                "Abstract",
                "Introduction",
                "This paper discusses attention.",
                "Start of intro."
            ]
            page.insert_text((50, 50), "\n".join(text_lines))
            # Set empty metadata
            doc.set_metadata({"title": "", "author": ""})
            doc.save(pdf_path)
            doc.close()

            paper, references, full_text = parser.parse(pdf_path)

            # Heuristics should kick in
            assert paper.title == "Attention mechanisms in deep learning networks"
            assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
            assert paper.year == 2026 # fallback
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_parse_ner_authors_fallback(self, parser):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Very Complex Title\nSome unstructured text line with names in it\nAbstract\nAbstract text\nIntroduction")
            doc.set_metadata({"title": "Complex Title", "author": ""})
            doc.save(pdf_path)
            doc.close()

            with patch("src.ner_engine.extract_persons_from_text", return_value=["Guido van Rossum", "Linus Torvalds"]) as mock_ner:
                paper, references, full_text = parser.parse(pdf_path)
                mock_ner.assert_called_once()
                assert paper.authors == ["Guido van Rossum", "Linus Torvalds"]
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_parse_doi_and_abstract_fallbacks(self, parser):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            # Include DOI and abstract without 'Introduction' keyword
            page.insert_text((50, 50), "Title Here\nAuthors\nDOI: 10.1000/xyz123\nAbstract This paper shows that 1+1=2.")
            doc.set_metadata({"title": "Title Here"})
            doc.save(pdf_path)
            doc.close()

            paper, references, full_text = parser.parse(pdf_path)
            assert paper.doi == "10.1000/xyz123"
            assert paper.id == "10.1000/xyz123"
            assert "This paper shows that 1+1=2." in paper.abstract
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_parse_references_variations(self, parser):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            # Use 'Bibliography' instead of 'References'
            page.insert_text((50, 50), "Title\nAbstract\nSummary\nBibliography\n1. Author A. Book A. 2020.\n2. Author B. Book B. 2021.")
            doc.set_metadata({"title": "Title"})
            doc.save(pdf_path)
            doc.close()

            paper, references, full_text = parser.parse(pdf_path)
            assert len(references) == 2
            assert "Author A. Book A. 2020." in references
            assert "Author B. Book B. 2021." in references
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
