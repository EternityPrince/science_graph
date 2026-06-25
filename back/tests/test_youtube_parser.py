import unittest
from unittest.mock import patch, MagicMock
from src.parsers.youtube_parser import YoutubeVideoParser

class TestYoutubeVideoParser(unittest.TestCase):
    @patch("yt_dlp.YoutubeDL")
    @patch("faster_whisper.WhisperModel")
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

    @patch("yt_dlp.YoutubeDL")
    def test_parse_youtube_video_fallback(self, mock_ytdl):
        # Force yt-dlp to raise an error
        mock_ytdl.side_effect = Exception("yt-dlp error")

        parser = YoutubeVideoParser()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        paper, links, full_text = parser.parse(url)

        # Verify fallback metadata has URL in title
        self.assertEqual(paper.id, "yt_dQw4w9WgXcQ")
        self.assertEqual(paper.title, f"YouTube Video ({url})")
        self.assertEqual(paper.authors, ["Unknown Creator"])
        self.assertEqual(paper.properties["source_type"], "video")
        self.assertIn("Audio download failed", paper.properties["transcript"])

    @patch("yt_dlp.YoutubeDL")
    def test_parse_publish_date_invalid(self, mock_ytdl):
        """Test fallback when publish_date cannot be parsed as integer year."""
        mock_ytdl_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance
        mock_ytdl_instance.extract_info.return_value = {
            "title": "Test Title",
            "uploader": "Test Uploader",
            "description": "Desc",
            "duration": 100,
            "upload_date": "invalid_date",
        }
        mock_ytdl_instance.prepare_filename.return_value = "yt_test.m4a"

        parser = YoutubeVideoParser()
        paper, links, full_text = parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIsNone(paper.year)

    @patch("yt_dlp.YoutubeDL")
    @patch("src.parsers.youtube_parser.os.path.exists")
    def test_parse_whisper_model_missing(self, mock_exists, mock_ytdl):
        """Test fallback when local Whisper model path does not exist."""
        # yt_dlp succeeds and returns temporary audio path
        mock_ytdl_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance
        mock_ytdl_instance.extract_info.return_value = {
            "title": "Test Title",
            "uploader": "Test Uploader",
            "description": "My Video Desc",
            "duration": 100,
            "upload_date": "20260522",
        }
        mock_ytdl_instance.prepare_filename.return_value = "yt_test.m4a"

        # os.path.exists returns True for temp_audio_path but False for the Whisper model path
        def side_effect(path):
            if "faster-whisper-large-v3-turbo" in path:
                return False
            return True
        mock_exists.side_effect = side_effect

        parser = YoutubeVideoParser()
        paper, links, full_text = parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("Transcription unavailable", paper.properties["transcript"])

    @patch("yt_dlp.YoutubeDL")
    @patch("faster_whisper.WhisperModel")
    @patch("src.parsers.youtube_parser.os.path.exists")
    def test_parse_whisper_transcription_fails(self, mock_exists, mock_whisper, mock_ytdl):
        """Test fallback when WhisperModel.transcribe raises an exception."""
        mock_exists.return_value = True
        mock_ytdl_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance
        mock_ytdl_instance.extract_info.return_value = {
            "title": "Test Title",
            "uploader": "Test Uploader",
            "description": "My Video Desc",
            "duration": 100,
            "upload_date": "20260522",
        }
        mock_ytdl_instance.prepare_filename.return_value = "yt_test.m4a"

        # Whisper transcribe raises exception
        mock_whisper_instance = MagicMock()
        mock_whisper.return_value = mock_whisper_instance
        mock_whisper_instance.transcribe.side_effect = Exception("Whisper transcription crash")

        parser = YoutubeVideoParser()
        paper, links, full_text = parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertIn("Transcription unavailable", paper.properties["transcript"])

    @patch("yt_dlp.YoutubeDL")
    @patch("faster_whisper.WhisperModel")
    @patch("src.parsers.youtube_parser.os.path.exists")
    @patch("src.parsers.youtube_parser.os.remove")
    def test_parse_os_remove_raises_error(self, mock_remove, mock_exists, mock_whisper, mock_ytdl):
        """Test fallback when cleaning up the temp audio file raises an exception."""
        mock_exists.return_value = True
        mock_ytdl_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_ytdl_instance
        mock_ytdl_instance.extract_info.return_value = {
            "title": "Test Title",
            "uploader": "Test Uploader",
            "description": "My Video Desc",
            "duration": 100,
            "upload_date": "20260522",
        }
        mock_ytdl_instance.prepare_filename.return_value = "yt_test.m4a"

        # Whisper succeeds
        mock_whisper_instance = MagicMock()
        mock_whisper.return_value = mock_whisper_instance
        mock_whisper_instance.transcribe.return_value = ([], None)

        # os.remove raises OSError
        mock_remove.side_effect = OSError("Permission denied")

        parser = YoutubeVideoParser()
        # Should not raise exception out of parse
        paper, links, full_text = parser.parse("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(paper.title, "Test Title")

if __name__ == "__main__":
    unittest.main()
