# 🔬 AI Researcher

An **agentic AI system** built with [LangGraph](https://langchain-ai.github.io/langgraph/) and [Google Gemini / Groq](https://ai.google.dev/) that can search academic papers, scrape the web, read PDFs, and generate publication-ready LaTeX papers — all through an interactive conversation.

> ⚠️ **For educational purposes only.** This project demonstrates advanced multi-agentic AI patterns and is not intended for out-of-the-box production academic research.

---

## 📈 Recent Updates

- **CI/CD Automation**: Fully automated deployment pipeline to Google Cloud Run via GitHub Actions (`ci.yml` and `cd.yml`).
- **Unified Container**: Consolidated the FastAPI backend and Streamlit UI into a single Cloud Run-compatible Docker container with dynamic port binding.
- **Test Suite Stabilization**: Achieved highly stable E2E testing by implementing robust module-level mocks for LLM instantiation and isolating the LangGraph checkpointer.
- **GCP Ready**: Added comprehensive instructions and configurations for deploying using Google Workload Identity or Service Accounts, Artifact Registry, and Cloud Run.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Intent-Driven Routing** | A Supervisor agent intelligently decides whether you need quick answers, deep research, or just a chat. |
| 🧰 **Massive Tool Suite** | Over 12+ integrated tools including Google Scholar, PubMed, ArXiv, Wikipedia, YouTube transcripts, and Web Searches. |
| 📖 **PDF Ingestion (RAG)** | Download PDFs, extract text into a Chroma vector database, and read specific chunks natively. |
| 📝 **LaTeX Paper Writer** | Converts research summaries directly into compilable `.tex` files and compiles them into PDFs. |
| 🤖 **LangGraph Workflow** | A state-graph agent that autonomously plans, selects tools, and manages errors/retries. |
| 🚦 **Guardrails** | Built-in circuit breakers to prevent infinite LLM loops and runaway API costs. |
| 💻 **Modern UI/UX** | Includes both a FastAPI driven Streamlit Web UI and a fully-featured terminal CLI. |
| 🔄 **Stateful Memory** | SQLite and in-memory conversation checkpointing for multi-turn research sessions. |

---

## 🛠️ Integrated Tools

The Researcher agent has access to the following abilities:

- **arxiv.py**: Search the arXiv database for recent pre-prints.
- **google_scholar.py**: Search via Serper API for Google Scholar results.
- **pubmed.py**: Retrieve biomedical literature from NCBI PubMed.
- **semantic_scholar.py**: Graph-based academic paper search and citation tracking.
- **pdf_reader.py**: Download, parse text, and extract visual figures from PDFs.
- **query_pdf.py**: RAG-based lookup against ingested PDF vector databases.
- **wikipedia_tool.py**: Pull encyclopedic summaries.
- **youtube.py**: Extract transcripts from YouTube video links.
- **web_search.py**: Advanced internet search via Tavily.
- **summarizer.py**: Distill large blocks of text into key findings.
- **scratchpad.py**: Temporary agent memory for multi-step reasoning.
- **latex_renderer.py**: Tectonic-powered LaTeX PDF compiler.

---

## 🏗️ Architecture & Project Structure

```text
Agentic_ai_researcher/
├── src/
│   └── ai_researcher/         # Core application code
│       ├── agent/             # Graph logic, state, and supervisor routing
│       │   ├── checkpointer.py
│       │   ├── graph.py
│       │   ├── guardrails.py
│       │   ├── prompts.py
│       │   ├── state.py
│       │   └── supervisor.py
│       ├── models/            # Pydantic data schemas
│       ├── server/            # FastAPI backend logic (ports 8000)
│       │   └── main.py
│       ├── tools/             # 12+ Agent tools (API integrations, utilities)
│       ├── ui/                # Streamlit frontend & client (port 8501)
│       ├── cli.py             # CLI entry points
│       └── config.py          # Centralized configuration (pydantic-settings)
├── tests/                     # Pytest suite
│   ├── eval/                  # LangSmith evaluations and datasets
│   ├── test_agent/            # Agent logic tests
│   ├── test_server/           # FastAPI backend tests
│   └── test_tools/            # Tools unit tests
├── prompts/                   # Externalized system prompts (.txt)
├── .github/workflows/         # CI/CD pipelines (GitHub Actions)
├── Dockerfile                 # Docker container instructions
├── docker-compose.yml         # Multi-container local execution setup
├── entrypoint.sh              # Startup orchestration script
├── Makefile                   # Development commands and shortcuts
├── pyproject.toml             # Dependency and project config (uv)
└── output/                    # Generated PDFs/checkpoints (ignored)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **[tectonic](https://tectonic-typesetting.github.io/)** (for LaTeX PDF generation)
- API Keys for Google Gemini or Groq

### 1. Clone & Install

```bash
git clone https://github.com/DKMMEHER/Agentic_ai_researcher.git
cd Agentic_ai_researcher

# Install with uv (recommended)
uv sync --all-extras
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY / GROQ_API_KEY
```

### 3. Run

**CLI Mode** (terminal chat):
```bash
uv run ai-researcher --mode cli
```

**Web UI** (Streamlit & FastAPI):
```bash
# This starts both the backend and frontend
./entrypoint.sh 
# OR
make run-ui
```

---

## ⚙️ Configuration

All settings are managed via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *optional* | Needed if using LLaMA models via Groq |
| `GOOGLE_API_KEY` | *optional* | Needed if using Gemini |
| `TAVILY_API_KEY` | *optional* | Needed for Web Search |
| `SERPER_API_KEY` | *optional* | Needed for Google Scholar |
| `MODEL_NAME` | `gemini-2.5-flash` | Model to use |
| `MODEL_TEMPERATURE` | `0.7` | Sampling temperature |
| `CHECKPOINT_BACKEND` | `sqlite` | Persistence type (`sqlite` or `memory`) |
| `OUTPUT_DIR` | `output` | Where generated PDFs are saved |

---

## 🧪 Development

### Run Tests

Verify the entire system using the local pytest suite.
```bash
make test       # Uses uv to run tests
make lint       # Ruff formatting and linting
```

### Evaluating with LangSmith
The `tests/eval/` directory contains logic to benchmark the LLM Agent's trajectory, RAG context retrieval, and decision-making accuracy against known datasets.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `uv sync --all-extras`
4. Make your changes and ensure `make test` runs cleanly.
5. Submit a pull request.

---

## 🚀 GCP Deployment (CI/CD)

The project includes a GitHub Actions workflow for automated deployment to **Google Cloud Run**.

### 1. GCP Setup

1. **Enable APIs**: Enable the Cloud Run, Artifact Registry, and Cloud Build APIs in your GCP Project.
2. **Artifact Registry**: Create a Docker repository named `ai-researcher` in your preferred region.
3. **Service Account**: Create a Service Account with the following roles:
   - `roles/run.admin`
   - `roles/storage.admin` (for Artifact Registry)
   - `roles/iam.serviceAccountUser`
   - `roles/artifactregistry.writer`
4. **Service Account Key**: Generate a JSON key for this service account.

### 2. GitHub Secrets

Add the following secrets to your GitHub repository (`Settings -> Secrets and variables -> Actions`):

| Secret | Description |
|--------|-------------|
| `GCP_PROJECT_ID` | Your Google Cloud Project ID |
| `GCP_SA_KEY` | The JSON key of your Service Account |
| `GCP_REGION` | The GCP region (e.g., `us-central1`) |
| `GROQ_API_KEY` | Your Groq API Key |
| `GOOGLE_API_KEY` | Your Google Gemini API Key |
| `TAVILY_API_KEY` | Your Tavily API Key |
| `SERPER_API_KEY` | Your Serper API Key |

### 3. Automated Flow

1. Push to `main` or `master`.
2. The `CI` workflow runs tests.
3. Upon success, the `CD` workflow builds the Docker image, pushes it to GAR, and deploys it to Cloud Run.
4. The service will be available at the URL provided in the GitHub Action logs.

---

## 📄 License

This project is licensed under the MIT License.
