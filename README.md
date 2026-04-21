# 🔬 AI Researcher

An **agentic AI system** built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [Google Gemini](https://ai.google.dev/) that can search arXiv papers, analyze research, and generate publication-ready LaTeX papers — all through an interactive conversation.

> ⚠️ **For educational purposes only.** This project demonstrates agentic AI patterns and is not intended for production academic research.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **arXiv Search** | Search for recently published papers by topic via the arXiv API |
| 📖 **PDF Reader** | Download and extract text from research papers |
| 📝 **Paper Writer** | Generate LaTeX research papers with mathematical equations |
| 🤖 **Agentic Workflow** | LangGraph-based agent that autonomously plans and executes research steps |
| 💬 **Interactive Chat** | Streamlit web UI and CLI for conversational interaction |
| 🔄 **Memory** | Conversation checkpointing for multi-turn research sessions |

---

## 🏗️ Architecture

```
src/ai_researcher/
├── config.py              # Centralized config (pydantic-settings)
├── logging.py             # Structured logging
├── exceptions.py          # Custom exception hierarchy
├── cli.py                 # CLI entry point (terminal + Streamlit)
│
├── agent/
│   ├── state.py           # LangGraph state definition
│   ├── graph.py           # Workflow graph builder
│   └── prompts.py         # Externalized system prompts
│
├── tools/
│   ├── arxiv.py           # arXiv paper search
│   ├── pdf_reader.py      # PDF text extraction
│   └── latex_renderer.py  # LaTeX → PDF compilation
│
├── models/
│   └── schemas.py         # Pydantic data models
│
└── ui/
    └── streamlit_app.py   # Streamlit web interface
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **[tectonic](https://tectonic-typesetting.github.io/)** (for LaTeX PDF generation)
- **Google API Key** with Gemini access

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd Agentic_ai_researcher

# Install with uv (recommended)
uv sync

# Or install dev dependencies too
uv sync --all-extras
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run

**CLI Mode** (terminal chat):

```bash
uv run ai-researcher --mode cli
```

**Web UI** (Streamlit):

```bash
uv run ai-researcher --mode ui
```

---

## ⚙️ Configuration

All settings are managed via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *required* | Google API key for Gemini |
| `MODEL_NAME` | `gemini-2.5-pro` | Model to use |
| `MODEL_TEMPERATURE` | `0.7` | Sampling temperature |
| `MAX_ARXIV_RESULTS` | `5` | Max papers per search |
| `PDF_REQUEST_TIMEOUT` | `30` | PDF download timeout (seconds) |
| `OUTPUT_DIR` | `output` | Where generated PDFs are saved |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `THREAD_ID` | `default-thread` | Conversation thread ID |

---

## 🧪 Development

### Run Tests

```bash
uv run pytest tests/ -v --cov=ai_researcher
```

### Lint & Format

```bash
uv run ruff check src/ tests/    # Lint
uv run ruff format src/ tests/   # Format
uv run mypy src/                  # Type check
```

### Using Make (Linux/macOS)

```bash
make test       # Run tests
make lint        # Lint + type check
make format      # Auto-format
make run         # CLI mode
make run-ui      # Streamlit mode
```

---

## 📁 Project Structure

```
Agentic_ai_researcher/
├── src/ai_researcher/    # Main package (see Architecture above)
├── tests/                # Test suite
│   ├── test_tools/       # Tool unit tests
│   └── test_agent/       # Agent unit tests
├── prompts/              # Externalized system prompts
├── output/               # Generated PDFs (gitignored)
├── pyproject.toml        # Project config, deps, tool config
├── Makefile              # Dev commands
├── .env.example          # Environment variable template
└── README.md             # This file
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `uv sync --all-extras`
4. Make your changes and add tests
5. Run `make lint` and `make test`
6. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.
