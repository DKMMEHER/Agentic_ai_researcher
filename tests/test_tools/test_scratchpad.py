"""Tests for the Scratchpad note-taking tool."""

from ai_researcher.tools.scratchpad import save_research_note


class TestSaveResearchNote:
    """Tests for the save_research_note tool."""

    def test_saves_note_returns_confirmation(self):
        """Test that saving a note returns a success message."""
        result = save_research_note.invoke(
            {"note": "Transformers use self-attention for sequence modeling."}
        )
        assert "successfully saved" in result.lower()

    def test_saves_long_note(self):
        """Test that a very long note is accepted without error."""
        long_note = "A" * 5000
        result = save_research_note.invoke({"note": long_note})
        assert "successfully saved" in result.lower()

    def test_saves_empty_note(self):
        """Test that an empty note is handled gracefully."""
        result = save_research_note.invoke({"note": ""})
        assert "successfully saved" in result.lower()
