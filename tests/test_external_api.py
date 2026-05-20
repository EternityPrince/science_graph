import json
import unittest
from unittest.mock import patch, MagicMock
from src.external_api import fetch_paper_metadata, _normalize_response

class TestExternalAPI(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_by_doi(self, mock_urlopen):
        # 1. Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        raw_response_data = {
            "paperId": "abc123xyz",
            "title": "Attention Is All You Need",
            "year": 2017,
            "abstract": "The dominant sequence transduction models...",
            "authors": [
                {"authorId": "a1", "name": "Ashish Vaswani"},
                {"authorId": "a2", "name": "Noam Shazeer"}
            ],
            "externalIds": {"DOI": "10.1145/37565.37566"},
            "references": [
                {"title": "Deep Residual Learning for Image Recognition", "externalIds": {"DOI": "10.1109/CVPR.2016.90"}},
                {"title": "Neural Machine Translation by Jointly Learning to Align and Translate", "externalIds": {}}
            ],
            "citations": [
                {"title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "externalIds": {"DOI": "10.18653/v1/N19-1423"}}
            ]
        }
        mock_response.read.return_value = json.dumps(raw_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # 2. Call function
        result = fetch_paper_metadata(doi="10.1145/37565.37566")

        # 3. Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Attention Is All You Need")
        self.assertEqual(result["year"], 2017)
        self.assertEqual(result["doi"], "10.1145/37565.37566")
        self.assertIn("Ashish Vaswani", result["authors"])
        self.assertEqual(len(result["references"]), 2)
        self.assertEqual(result["references"][0]["title"], "Deep Residual Learning for Image Recognition")
        self.assertEqual(result["references"][0]["doi"], "10.1109/CVPR.2016.90")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["title"], "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")

    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_by_title(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        raw_response_data = {
            "data": [
                {
                    "paperId": "abc123xyz",
                    "title": "Attention Is All You Need",
                    "year": 2017,
                    "abstract": "The dominant sequence transduction models...",
                    "authors": [{"name": "Ashish Vaswani"}],
                    "externalIds": {"DOI": "10.1145/37565.37566"},
                    "references": [],
                    "citations": []
                }
            ]
        }
        mock_response.read.return_value = json.dumps(raw_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = fetch_paper_metadata(title="Attention Is All You Need")
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Attention Is All You Need")
        self.assertEqual(result["doi"], "10.1145/37565.37566")
