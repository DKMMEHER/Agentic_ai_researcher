import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from ai_researcher.config import get_settings
from ai_researcher.exceptions import LatexRenderError
from ai_researcher.logging import get_logger

logger = get_logger(__name__)


def _extract_title_slug(latex_content: str) -> str:
    """Extract a filename-safe slug from the LaTeX \\title{} command.

    Falls back to 'research_paper' if no title is found.
    """
    # Extract the title. We use a more robust regex that looks for the closing bracket
    # while allowing for one level of nested braces (common for \textbf{\textit{...}})
    match = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})+)\}", latex_content)
    if match:
        title = match.group(1)
        # Strip LaTeX commands like \textbf{...} but keep the content inside
        title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
        title = re.sub(r"[^\w\s-]", "", title).strip()
        title = re.sub(r"[\s]+", "_", title)
        return title[:60] if title else "research_paper"
    return "research_paper"


def _get_tectonic_command() -> str:
    """Find the tectonic binary, checking local project root first then system PATH.

    Returns:
        The command name or path to tectonic.

    Raises:
        LatexRenderError: If tectonic is not found.
    """
    # Check current directory for tectonic.exe (portable install)
    local_tectonic = Path("tectonic.exe")
    if local_tectonic.exists():
        return str(local_tectonic.resolve())

    # Check system PATH
    sys_tectonic = shutil.which("tectonic")
    if sys_tectonic:
        return sys_tectonic

    raise LatexRenderError(
        message=(
            "tectonic is not installed or not on PATH. "
            "Install it from: https://tectonic-typesetting.github.io/\n"
            "  • Windows: winget install tectonic-typesetting.Tectonic\n"
            "  • macOS:   brew install tectonic\n"
            "  • Linux:   conda install -c conda-forge tectonic"
        )
    )


@tool
def render_latex_pdf(latex_content: str) -> str:
    """Render a LaTeX document to PDF.

    Args:
        latex_content: The LaTeX document content as a string.

    Returns:
        Absolute path to the generated PDF file.
    """
    # Auto-fix: Sometimes the AI double-escapes backslashes in JSON (e.g. \\documentclass)
    # This regex cleans up known structural macros to prevent "There's no line here to end" errors.
    latex_content = re.sub(
        r"\\\\(documentclass|usepackage|begin|end|section|subsection|item|url|textbf|textit|title|author|date|maketitle|abstract)",
        r"\\\1",
        latex_content,
    )

    tectonic_cmd = _get_tectonic_command()

    settings = get_settings()
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    title_slug = _extract_title_slug(latex_content)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tex_filename = f"{title_slug}_{timestamp}.tex"
    pdf_filename = f"{title_slug}_{timestamp}.pdf"

    tex_file = output_dir / tex_filename
    pdf_file = output_dir / pdf_filename

    logger.info("Writing LaTeX to %s", tex_file)

    try:
        tex_file.write_text(latex_content, encoding="utf-8")

        logger.info("Running tectonic (%s) to compile PDF...", tectonic_cmd)
        result = subprocess.run(
            [tectonic_cmd, tex_filename, "--outdir", str(output_dir)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for compilation
        )

        if result.returncode != 0:
            logger.error("Tectonic compilation failed. Sending error back to agent...")
            return (
                f"ERROR: LaTeX compilation failed due to syntax errors.\n"
                f"Please fix the following errors and carefully call the tool again.\n\n"
                f"Tectonic output:\n{result.stderr}"
            )

        if not pdf_file.exists():
            return (
                "ERROR: Tectonic compilation appeared to succeed, but no PDF was generated! "
                "Ensure your LaTeX structure is completely valid and call the tool again."
            )

        logger.info("Successfully generated PDF at %s", pdf_file)
        return f"🎉 SUCCESS: LaTeX document compiled successfully! The PDF is available at: {pdf_file}"

    except subprocess.TimeoutExpired:
        return "ERROR: LaTeX compilation timed out after 120 seconds. Is the document too complex or stuck in a loop?"
    except Exception as e:
        return f"ERROR: Unexpected system error during LaTeX rendering: {e}. Please attempt to fix or simplify the document."
