"""YouTube Transcript downloading tool."""

import urllib.parse
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.tools import tool

from ai_researcher.exceptions import ToolError
from ai_researcher.logging import get_logger

logger = get_logger(__name__)

class YoutubeTranscriptError(ToolError):
    def __init__(self, message: str = "", video_id: str = "", *args, **kwargs):
        self.video_id = video_id
        super().__init__(message, *args, **kwargs)

@tool
def youtube_transcript_reader(url: str) -> str:
    """Extracts the full text transcript from a YouTube video URL.
    Use this to summarize keynotes, conferences, or tutorials.
    
    Args:
        url: The full YouTube URL to extract text from.
    """
    logger.info("Extracting YouTube transcript for: '%s'", url)
    
    try:
        parsed_url = urllib.parse.urlparse(url)
        video_id = ""
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            video_id = parsed_url.path[1:]
        elif parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                video_id = urllib.parse.parse_qs(parsed_url.query)['v'][0]
        
        if not video_id:
            raise ValueError("Could not parse a valid YouTube video ID from the URL.")
            
    except Exception as e:
        logger.warning("Invalid YouTube URL provided: %s", e)
        return "Error: Invalid YouTube URL provided. Ensure it is a valid youtube.com or youtu.be link."

    try:
        ytt_api = YouTubeTranscriptApi()
        
        # Try to find the best available transcript
        # Priority: English manual → English auto → any language manual → any language auto
        transcript = None
        transcript_lang = "en"
        
        try:
            # First, try fetching English directly
            transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
        except Exception:
            logger.info("No English transcript found, searching for alternatives...")
            try:
                # List all available transcripts and pick the best one
                transcript_list = ytt_api.list(video_id)
                
                # Try to find any transcript (manual or auto-generated)
                available = list(transcript_list)
                if available:
                    best = available[0]
                    transcript_lang = best.language_code if hasattr(best, 'language_code') else best.get('language_code', 'unknown')
                    transcript = ytt_api.fetch(video_id, languages=[transcript_lang])
                    logger.info("Using %s transcript (language: %s)", 
                               "auto-generated" if getattr(best, 'is_generated', False) else "manual",
                               transcript_lang)
            except Exception as list_err:
                logger.warning("Could not list transcripts: %s", list_err)
                # Last resort: try without any language filter
                transcript = ytt_api.fetch(video_id)
        
        if transcript is None:
            return "Error: No transcript available for this video in any language."
        
        # Determine if we got dictionaries or dataclasses (API version dependent)
        full_text = " ".join([
            segment.text if hasattr(segment, 'text') else segment["text"] 
            for segment in transcript
        ])
        
        # Add language note if non-English
        if transcript_lang not in ('en', 'en-US', 'en-GB'):
            full_text = f"[NOTE: This transcript is in '{transcript_lang}'. Summarize the content as best you can.]\n\n{full_text}"
        
        # 12k TPM safely allows ~30,000 characters
        if len(full_text) > 30000:
            logger.warning("Truncating YouTube transcript from %d to 30000 chars", len(full_text))
            full_text = full_text[:30000] + "\n\n... [TRANSCRIPT TRUNCATED FOR LENGTH] ..."
            
        logger.info("Successfully extracted %d characters from transcript (lang=%s)", len(full_text), transcript_lang)
        return full_text
        
    except Exception as e:
        logger.exception("Failed to get transcript for %s", video_id)
        raise YoutubeTranscriptError(
            message=f"Failed to extract transcript: {e!s}",
            video_id=video_id
        ) from e
