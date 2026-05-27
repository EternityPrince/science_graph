import json
import unittest
from unittest.mock import patch, MagicMock
from src.external_api import fetch_paper_metadata, _normalize_response

class TestExternalAPI(unittest.IsolatedAsyncioTestCase):
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

    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_by_arxiv_id(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        raw_response_data = {
            "paperId": "arxiv_1706.03762",
            "title": "Attention Is All You Need",
            "year": 2017,
            "abstract": "The dominant sequence transduction models...",
            "authors": [{"name": "Ashish Vaswani"}],
            "externalIds": {"ArXiv": "1706.03762"},
            "references": [],
            "citations": []
        }
        mock_response.read.return_value = json.dumps(raw_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = fetch_paper_metadata(arxiv_id="1706.03762")
        self.assertIsNotNone(result)
        self.assertEqual(result["title"], "Attention Is All You Need")
        self.assertEqual(result["year"], 2017)

    def test_normalize_response_with_null_fields(self):
        raw_data = {
            "title": "Null Fields Paper",
            "authors": None,
            "externalIds": None,
            "references": None,
            "citations": None,
            "year": 2026,
            "abstract": None
        }
        res = _normalize_response(raw_data)
        self.assertEqual(res["title"], "Null Fields Paper")
        self.assertEqual(res["authors"], [])
        self.assertEqual(res["references"], [])
        self.assertEqual(res["citations"], [])
        self.assertIsNone(res["doi"])
        self.assertEqual(res["year"], 2026)
        self.assertIsNone(res["abstract"])

    def test_fetch_paper_metadata_no_query(self):
        self.assertIsNone(fetch_paper_metadata())

    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_search_empty_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        self.assertIsNone(fetch_paper_metadata(title="No such paper title"))

    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_404(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", None, None
        )
        self.assertIsNone(fetch_paper_metadata(doi="10.1234/nonexistent"))

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_fetch_paper_metadata_429_retry_after(self, mock_sleep, mock_urlopen):
        import urllib.error
        headers = MagicMock()
        headers.get.side_effect = lambda key: "5" if key == "Retry-After" else None
        
        he = urllib.error.HTTPError("url", 429, "Too Many Requests", headers, None)
        
        mock_success = MagicMock()
        mock_success.status = 200
        mock_success.read.return_value = json.dumps({"title": "Success Paper"}).encode("utf-8")
        mock_success.__enter__.return_value = mock_success
        
        mock_urlopen.side_effect = [he, he, mock_success]
        
        res = fetch_paper_metadata(doi="10.1234/retry")
        self.assertEqual(res["title"], "Success Paper")
        self.assertEqual(mock_sleep.call_count, 2)
        sleep_args = [call[0][0] for call in mock_sleep.call_args_list]
        for s in sleep_args:
            self.assertTrue(4.0 <= s <= 6.0)

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_fetch_paper_metadata_5xx_retries(self, mock_sleep, mock_urlopen):
        import urllib.error
        he = urllib.error.HTTPError("url", 500, "Internal Server Error", None, None)
        
        mock_urlopen.side_effect = [he, he, he]
        
        res = fetch_paper_metadata(doi="10.1234/fail")
        self.assertIsNone(res)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_fetch_paper_metadata_generic_exception(self, mock_sleep, mock_urlopen):
        mock_urlopen.side_effect = Exception("network disconnect")
        
        res = fetch_paper_metadata(doi="10.1234/exception")
        self.assertIsNone(res)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("urllib.request.urlopen")
    async def test_fetch_paper_metadata_async(self, mock_urlopen):
        from src.external_api import fetch_paper_metadata_async
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"title": "Async Success"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = await fetch_paper_metadata_async(doi="10.1234/async")
        self.assertEqual(res["title"], "Async Success")

    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_prefix_cleaning(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"title": "Cleaned Paper"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Test "doi:" prefix
        fetch_paper_metadata(doi="doi:10.1234/test")
        args, kwargs = mock_urlopen.call_args
        self.assertIn("DOI:10.1234/test", args[0].full_url)

        # Test "arxiv:" prefix
        fetch_paper_metadata(arxiv_id="arxiv:1706.03762")
        args, kwargs = mock_urlopen.call_args
        self.assertIn("arXiv:1706.03762", args[0].full_url)

    @patch("urllib.request.urlopen")
    @patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "test-api-key"})
    def test_fetch_paper_metadata_api_key(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"title": "Keyed Paper"}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        fetch_paper_metadata(doi="10.1234/key")
        args, kwargs = mock_urlopen.call_args
        self.assertEqual(args[0].headers["X-api-key"], "test-api-key")

    @patch("urllib.request.urlopen")
    def test_fetch_paper_metadata_403(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 403, "Forbidden", None, None
        )
        self.assertIsNone(fetch_paper_metadata(doi="10.1234/forbidden"))

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_fetch_paper_metadata_429_invalid_retry_after(self, mock_sleep, mock_urlopen):
        import urllib.error
        headers = MagicMock()
        headers.get.side_effect = lambda key: "not-a-number" if key == "Retry-After" else None
        he = urllib.error.HTTPError("url", 429, "Too Many Requests", headers, None)
        
        mock_urlopen.side_effect = [he, he, he]
        self.assertIsNone(fetch_paper_metadata(doi="10.1234/retry-invalid"))
        self.assertEqual(mock_sleep.call_count, 2)

    def test_normalize_response_null_elements(self):
        raw_data = {
            "title": "Null Elements Paper",
            "authors": None,
            "externalIds": None,
            "references": [None, {"title": "Ref A", "externalIds": {"DOI": "10.1"}}],
            "citations": [None, {"title": "Cit A", "externalIds": {"DOI": "10.2"}}],
            "year": 2026,
            "abstract": None
        }
        res = _normalize_response(raw_data)
        self.assertEqual(len(res["references"]), 1)
        self.assertEqual(res["references"][0]["title"], "Ref A")
        self.assertEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["title"], "Cit A")


