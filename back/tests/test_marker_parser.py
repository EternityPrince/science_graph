import unittest
from unittest.mock import patch


from src.models import Paper
from src.parsers.marker_parser import MarkerPDFParser, get_marker_models
import src.parsers.marker_parser as marker_module


class TestMarkerPDFParser(unittest.TestCase):
    def setUp(self):
        # Reset cached models before each test
        marker_module._marker_models = None

    def tearDown(self):
        # Clean up cached models after each test
        marker_module._marker_models = None

    @patch("marker.models.load_all_models")
    def test_get_marker_models_lazy_loading(self, mock_load_all_models):
        """Test get_marker_models loads models once and caches them."""
        mock_models = ["model1", "model2"]
        mock_load_all_models.return_value = mock_models

        # First call loads models
        models1 = get_marker_models()
        assert models1 == mock_models
        mock_load_all_models.assert_called_once()

        # Second call returns cached models
        models2 = get_marker_models()
        assert models2 == mock_models
        mock_load_all_models.assert_called_once()  # still called once

    @patch("os.path.exists")
    @patch("src.parsers.marker_parser.get_marker_models")
    @patch("marker.convert.convert_single_pdf")
    @patch("src.parsers.pdf_parser.PDFParser.parse")
    def test_marker_parser_success(self, mock_legacy_parse, mock_convert_pdf, mock_get_models, mock_exists):
        """Test successful Marker parsing with metadata integration."""
        mock_exists.return_value = True
        
        # Setup legacy parser mock output (for metadata)
        legacy_paper = Paper(
            id="test-paper",
            title="Test Paper Title",
            authors=["Alice", "Bob"],
            year=2024,
            doi="10.1234/test",
            abstract="This is the legacy abstract.",
            file_path="dummy.pdf"
        )
        mock_legacy_parse.return_value = (legacy_paper, ["Ref A", "Ref B"], "Raw Text")
        
        # Setup Marker mock output
        mock_models = ["mock_model"]
        mock_get_models.return_value = mock_models
        mock_convert_pdf.return_value = (
            "# Test Paper Title\n\nThis is Marker extracted text.\n\nReferences\n1. Vaswani et al. Attention is all you need. 2017.\n2. Devlin et al. BERT. 2018.",
            {"language": "Russian"}
        )

        parser = MarkerPDFParser()
        paper, references, full_text = parser.parse("dummy.pdf")

        # Verify Marker was called correctly
        mock_convert_pdf.assert_called_once_with(
            "dummy.pdf",
            mock_models,
            metadata={"language": "Russian"},
            parallel_factor=1
        )

        # Verify parsed paper fields (inherited from Fitz legacy parse)
        assert paper.id == "test-paper"
        assert paper.title == "Test Paper Title"
        assert paper.authors == ["Alice", "Bob"]
        assert paper.year == 2024
        assert paper.doi == "10.1234/test"
        assert paper.abstract == "This is the legacy abstract."
        assert paper.file_path == "dummy.pdf"

        # Verify references were extracted from Marker markdown
        assert references == [
            "Vaswani et al. Attention is all you need. 2017.",
            "Devlin et al. BERT. 2018."
        ]
        assert "This is Marker extracted text." in full_text

    @patch("os.path.exists")
    @patch("src.parsers.marker_parser.get_marker_models")
    @patch("marker.convert.convert_single_pdf")
    @patch("src.parsers.pdf_parser.PDFParser.parse")
    def test_marker_parser_fallback_on_failure(self, mock_legacy_parse, mock_convert_pdf, mock_get_models, mock_exists):
        """Test that Marker parser falls back to PDFParser text on any exception."""
        mock_exists.return_value = True
        
        # Setup legacy parser mock output (for metadata & text)
        legacy_paper = Paper(
            id="test-paper",
            title="Test Paper Title",
            authors=["Alice", "Bob"],
            year=2024,
            doi="10.1234/test",
            abstract="This is the legacy abstract.",
            file_path="dummy.pdf"
        )
        mock_legacy_parse.return_value = (legacy_paper, ["Ref A", "Ref B"], "Legacy Raw Text")
        
        # Simulate Marker failing
        mock_get_models.side_effect = Exception("Model loading failed!")

        parser = MarkerPDFParser()
        paper, references, full_text = parser.parse("dummy.pdf")

        # Verify fallback values are returned
        assert paper.id == "test-paper"
        assert paper.authors == ["Alice", "Bob"]
        assert references == ["Ref A", "Ref B"]
        assert full_text == "Legacy Raw Text"

    def test_extract_references_from_markdown(self):
        """Test _extract_references_from_markdown extracts various list patterns."""
        parser = MarkerPDFParser()
        
        # Numbered list format
        markdown1 = """
# Introduction
Text here...

# References
1. Vaswani et al. Attention is all you need. 2017.
2. Devlin et al. BERT. 2018.
        """
        refs1 = parser._extract_references_from_markdown(markdown1)
        assert refs1 == [
            "Vaswani et al. Attention is all you need. 2017.",
            "Devlin et al. BERT. 2018."
        ]

        # Bullet list format
        markdown2 = """
# Conclusion
Done.

## Bibliography
* Author A. Title A. 2020.
- Author B. Title B. 2021.
        """
        refs2 = parser._extract_references_from_markdown(markdown2)
        assert refs2 == [
            "Author A. Title A. 2020.",
            "Author B. Title B. 2021."
        ]

        # Square bracket numbered list format
        markdown3 = """
## Список литературы
[1] И. Иванов. Статья. 2022.
  [2]  П. Петров. Книга. 2023.
        """
        refs3 = parser._extract_references_from_markdown(markdown3)
        assert refs3 == [
            "И. Иванов. Статья. 2022.",
            "П. Петров. Книга. 2023."
        ]
