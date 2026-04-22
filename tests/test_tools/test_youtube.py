"""Tests for the YouTube transcript tool."""

from unittest.mock import patch

import pytest

from ai_researcher.tools.youtube import (
    YoutubeTranscriptError,
    youtube_transcript_reader,
)


class TestYoutubeTranscriptReader:
    """Tests for YouTube transcript fetching tool."""

    @patch("ai_researcher.tools.youtube.YouTubeTranscriptApi")
    def test_successful_transcript_fetch(self, mock_api_class):
        """Test a normal successful transcript extraction."""
        # Setup mock instance
        mock_api_instance = mock_api_class.return_value

        # Mock transcript data
        mock_transcript = [{"text": "Hello world"}, {"text": "Welcome to AI research"}]
        mock_api_instance.fetch.return_value = mock_transcript

        url = "https://www.youtube.com/watch?v=abc123def"
        result = youtube_transcript_reader.invoke({"url": url})

        assert "Hello world Welcome to AI research" in result
        mock_api_instance.fetch.assert_called_once()

    @patch("ai_researcher.tools.youtube.YouTubeTranscriptApi")
    def test_short_youtube_url(self, mock_api_class):
        """Test parsing of youtu.be short URLs."""
        mock_api_instance = mock_api_class.return_value
        mock_api_instance.fetch.return_value = [{"text": "short url test"}]

        url = "https://youtu.be/xyz789"
        result = youtube_transcript_reader.invoke({"url": url})

        assert "short url test" in result
        # Check that it called with the correct ID
        mock_api_instance.fetch.assert_called_with(
            "xyz789", languages=["en", "en-US", "en-GB"]
        )

    def test_invalid_youtube_url(self):
        """Test behavior when an invalid URL is provided."""
        url = "https://www.google.com"
        result = youtube_transcript_reader.invoke({"url": url})

        assert "Error: Invalid YouTube URL" in result

    @patch("ai_researcher.tools.youtube.YouTubeTranscriptApi")
    def test_transcript_truncation(self, mock_api_class):
        """Test that very long transcripts are truncated."""
        mock_api_instance = mock_api_class.return_value
        # Create a transcript that exceeds 30,000 characters
        long_text = "word " * 10000
        mock_api_instance.fetch.return_value = [{"text": long_text}]

        url = "https://youtube.com/watch?v=longvideo"
        result = youtube_transcript_reader.invoke({"url": url})

        assert len(result) <= 31000  # 30000 + truncation message
        assert "[TRANSCRIPT TRUNCATED FOR LENGTH]" in result

    @patch("ai_researcher.tools.youtube.YouTubeTranscriptApi")
    def test_api_failure_raises_custom_error(self, mock_api_class):
        """Test that API failures raise YoutubeTranscriptError."""
        mock_api_instance = mock_api_class.return_value
        mock_api_instance.fetch.side_effect = Exception("Transcripts disabled")
        # Ensure fallback list search also fails
        mock_api_instance.list.side_effect = Exception("No transcripts")

        url = "https://youtube.com/watch?v=error"
        with pytest.raises(YoutubeTranscriptError) as excinfo:
            youtube_transcript_reader.invoke({"url": url})
        assert "Failed to extract transcript" in str(excinfo.value)
        assert excinfo.value.video_id == "error"
