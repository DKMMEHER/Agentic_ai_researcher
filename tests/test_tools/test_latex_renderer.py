"""Tests for the LaTeX renderer tool."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ai_researcher.tools.latex_renderer import render_latex_pdf, _get_tectonic_command, _extract_title_slug
from ai_researcher.exceptions import LatexRenderError


class TestGetTectonicCommand:
    """Tests for tectonic availability check."""

    @patch("ai_researcher.tools.latex_renderer.Path.exists", return_value=False)
    @patch("ai_researcher.tools.latex_renderer.shutil.which")
    def test_tectonic_found_on_path(self, mock_which, mock_exists):
        mock_which.return_value = "/usr/bin/tectonic"
        result = _get_tectonic_command()
        assert result == "/usr/bin/tectonic"

    @patch("ai_researcher.tools.latex_renderer.Path.exists", return_value=False)
    @patch("ai_researcher.tools.latex_renderer.shutil.which")
    def test_tectonic_not_found_raises(self, mock_which, mock_exists):
        mock_which.return_value = None
        with pytest.raises(LatexRenderError, match="tectonic is not installed"):
            _get_tectonic_command()


class TestExtractTitleSlug:
    """Tests for the filename slug extraction from LaTeX titles."""

    def test_simple_title(self):
        latex = r"\title{My Research Paper}"
        assert _extract_title_slug(latex) == "My_Research_Paper"

    def test_textbf_title(self):
        """Regression test: \\textbf{} in title should be stripped."""
        latex = r"\title{\textbf{100 Data Science Interview Questions}}"
        slug = _extract_title_slug(latex)
        assert "textbf" not in slug.lower(), f"textbf should be stripped, got: {slug}"
        assert "100_Data_Science" in slug

    def test_textit_title(self):
        latex = r"\title{\textit{Attention Is All You Need}}"
        slug = _extract_title_slug(latex)
        assert "textit" not in slug.lower()
        assert "Attention" in slug

    def test_no_title_returns_fallback(self):
        latex = r"\documentclass{article}\begin{document}Hello\end{document}"
        assert _extract_title_slug(latex) == "research_paper"

    def test_long_title_truncated(self):
        long_title = "A" * 100
        latex = rf"\title{{{long_title}}}"
        slug = _extract_title_slug(latex)
        assert len(slug) <= 60


class TestRenderLatexPdf:
    """Tests for LaTeX rendering."""

    @patch("ai_researcher.tools.latex_renderer._get_tectonic_command", return_value="/usr/bin/tectonic")
    @patch("ai_researcher.tools.latex_renderer.subprocess.run")
    def test_successful_render(self, mock_run, mock_tectonic, tmp_path, monkeypatch):
        """Test successful LaTeX to PDF rendering."""
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

        from ai_researcher.config import get_settings
        get_settings.cache_clear()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create the expected PDF file so the existence check passes
        def create_pdf_side_effect(*args, **kwargs):
            for f in tmp_path.glob("*.tex"):
                pdf_name = f.stem + ".pdf"
                (tmp_path / pdf_name).touch()
            return mock_result

        mock_run.side_effect = create_pdf_side_effect

        latex = r"\documentclass{article}\title{Test}\begin{document}Hello\end{document}"
        result = render_latex_pdf.invoke({"latex_content": latex})
        assert "SUCCESS" in result

        get_settings.cache_clear()

    @patch("ai_researcher.tools.latex_renderer._get_tectonic_command", return_value="/usr/bin/tectonic")
    @patch("ai_researcher.tools.latex_renderer.subprocess.run")
    def test_compilation_failure_returns_error(self, mock_run, mock_tectonic, tmp_path, monkeypatch):
        """Test that compilation errors return an error string."""
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

        from ai_researcher.config import get_settings
        get_settings.cache_clear()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Undefined control sequence"
        mock_run.return_value = mock_result

        result = render_latex_pdf.invoke(
            {"latex_content": r"\invalidcommand"}
        )
        assert "ERROR" in result

        get_settings.cache_clear()
