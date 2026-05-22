import unittest
from unittest.mock import patch, MagicMock
from src.parsers.youtube_parser import YoutubeVideoParser

class TestYoutubeVideoParser(unittest.TestCase):
    @patch("src.parsers.youtube_parser.yt_dlp.YoutubeDL")
    @patch("src.parsers.youtube_parser.WhisperModel")
    @patch("src.parsers.youtube_parser.os.path.exists")
    @patch("src.parsers.youtube_parser.os.remove")
    def test_parse_youtube_video_success(self, mock_remove, mock_exists, mock_whisper, mock_ytdl):
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock yt_dlp
        mock_ytdl_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance
        mock_ytdl_instance.extract_info.return_value = {
            "title": "Test Video Title",
            "uploader": "Test Uploader",
            "description": "Check out this link: https://example.com/info and another link: http://test.org",
            "duration": 180,
            "upload_date": "20260522",
        }
        mock_ytdl_instance.prepare_filename.return_value = "yt_test_video.m4a"

        # Mock Whisper segments
        mock_whisper_instance = MagicMock()
        mock_whisper.return_value = mock_whisper_instance
        
        class MockSegment:
            def __init__(self, start, text):
                self.start = start
                self.text = text
                
        mock_segments = [
            MockSegment(0.0, "Hello and welcome."),
            MockSegment(65.0, "This is a segment at 1 minute 5 seconds."),
            MockSegment(3665.0, "This segment is over an hour long.")
        ]
        mock_whisper_instance.transcribe.return_value = (mock_segments, None)

        # Execute parser
        parser = YoutubeVideoParser()
        paper, links, full_text = parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        # Verify paper attributes
        self.assertEqual(paper.id, "yt_dQw4w9WgXcQ")
        self.assertEqual(paper.title, "Test Video Title")
        self.assertEqual(paper.authors, ["Test Uploader"])
        self.assertEqual(paper.year, 2026)
        self.assertEqual(paper.properties["source_type"], "video")
        self.assertEqual(paper.properties["video_id"], "dQw4w9WgXcQ")
        self.assertEqual(paper.properties["uploader"], "Test Uploader")
        self.assertEqual(paper.properties["duration"], 180)
        self.assertEqual(paper.properties["publish_date"], "20260522")
        
        # Verify timestamps in transcript
        self.assertIn("[00:00] Hello and welcome.", paper.properties["transcript"])
        self.assertIn("[01:05] This is a segment at 1 minute 5 seconds.", paper.properties["transcript"])
        self.assertIn("[01:01:05] This segment is over an hour long.", paper.properties["transcript"])

        # Verify links extracted from description
        self.assertIn("https://example.com/info", links)
        self.assertIn("http://test.org", links)

        # Verify full text content structure
        self.assertIn("# Test Video Title", full_text)
        self.assertIn("Uploader: Test Uploader", full_text)
        self.assertIn("## Transcript", full_text)

        # Ensure Whisper Model was loaded correctly
        mock_whisper.assert_called_once_with("/Users/vladimirkasterin/models/audio/faster-whisper-large-v3-turbo", device="cpu", compute_type="float32")

    @patch("src.parsers.youtube_parser.yt_dlp.YoutubeDL")
    def test_parse_youtube_video_fallback(self, mock_ytdl):
        # Force yt-dlp to raise an error
        mock_ytdl.side_effect = Exception("yt-dlp error")

        parser = YoutubeVideoParser()
        paper, links, full_text = parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        # Verify fallback metadata
        self.assertEqual(paper.id, "yt_dQw4w9WgXcQ")
        self.assertEqual(paper.title, "YouTube Video")
        self.assertEqual(paper.authors, ["Unknown Creator"])
        self.assertEqual(paper.properties["source_type"], "video")
        self.assertIn("Audio download failed", paper.properties["transcript"])

if __name__ == "__main__":
    unittest.main()
