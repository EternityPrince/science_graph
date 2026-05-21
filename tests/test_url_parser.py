import unittest
from unittest.mock import patch
from src.parsers.url_parser import parse_url

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

        paper, md_content = parse_url("https://habr.com/ru/articles/1037532/")
        
        self.assertEqual(paper.title, "Как я обучил GPT с нуля на русском языке")
        # Should extract "Vladimir Kasterin" and "HabrAuthor"
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

        paper, _ = parse_url("https://example.com/blog/1")
        self.assertIn("Alice_Dev", paper.authors)
        self.assertIn("Bob Link", paper.authors)

if __name__ == "__main__":
    unittest.main()
