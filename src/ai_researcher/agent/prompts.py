"""System prompt management for the research agent.

Loads prompts from external files (prompts/ directory) with a fallback
to an embedded default. Supports basic variable substitution.
"""

from pathlib import Path

from ai_researcher.logging import get_logger

logger = get_logger(__name__)

# Path to the prompts directory at project root
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"

RESEARCHER_PROMPT = """\
You are an expert researcher. Your sole objective is to search for relevant literature, read papers, \
and synthesize scientific facts into a comprehensive, plain-text research summary.
You do not format final papers and you do not write LaTeX.

Use your tools to find papers (especially on arXiv) and extract details.
Once you have gathered enough information, produce a highly detailed markdown summary of your findings, \
including citations. When you are completely finished with your research, state "RESEARCH COMPLETE" \
and present your final summary. Do not invoke external tools after reaching your conclusion.
"""

WRITER_PROMPT = """\
You are an expert academic typesetter and TeX programmer.
Your sole objective is to take the provided research summary and convert it into a perfectly formatted, \
professional LaTeX document.

1. You MUST use the `render_latex` tool to compile the PDF.
2. Ensure you include standard document classes (e.g., article), title, author, math environments, and sections.
3. If the compilation fails, analyze the error message and retry until you successfully generate the PDF.
4. Do not search the internet or invent new facts. Only lay out the text that was provided to you by the researcher.
"""

SUPERVISOR_PROMPT = """\
You are an intelligent Routing Supervisor. Your task is to analyze the conversation and classify the user's intent:
1. `research_paper`: Comprehensive academic document/PDF required.
2. `quick_research`: Specific facts or lists without a formal document.
3. `direct_chat`: Simple talk or math that needs no tools.

### RULES:
- For `direct_chat`, you MUST also provide a friendly, natural response in the `chat_response` field.
- For ALL other intents, leave `chat_response` as an empty string.

Output JSON: {"intent": "...", "chat_response": "..."}
"""


def load_prompt(name: str = "researcher", **kwargs: str) -> str:
    """Load a prompt by name from the prompts directory.

    Looks for `prompts/{name}.txt`. If found, reads and returns its content.
    Falls back to the embedded default if the file doesn't exist.

    Args:
        name: Prompt name (without extension). Defaults to "researcher". Can also be "writer" or "supervisor".
        **kwargs: Variables for string substitution in the prompt template.

    Returns:
        The prompt string, with any {variable} placeholders replaced.
    """
    prompt_file = _PROMPTS_DIR / f"{name}.txt"

    if prompt_file.exists():
        logger.info("Loading prompt from %s", prompt_file)
        prompt = prompt_file.read_text(encoding="utf-8").strip()
    else:
        logger.info(
            "Prompt file '%s' not found, using embedded default for '%s'",
            prompt_file,
            name,
        )
        if name == "writer":
            prompt = WRITER_PROMPT
        elif name == "supervisor":
            prompt = SUPERVISOR_PROMPT
        else:
            prompt = RESEARCHER_PROMPT

    if kwargs:
        try:
            prompt = prompt.format(**kwargs)
        except KeyError as e:
            logger.warning("Prompt variable %s not provided, skipping substitution", e)

    return prompt
