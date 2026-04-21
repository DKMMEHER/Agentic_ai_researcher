"""Shared test fixtures and configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key-for-testing")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-for-testing")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key-for-testing")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key-for-testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OUTPUT_DIR", "test_output")
    monkeypatch.setenv("MODEL_NAME", "gemini-1.5-flash")


@pytest.fixture
def sample_arxiv_xml():
    """Sample arXiv API XML response for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex
    recurrent or convolutional neural networks.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
    <link type="application/pdf" href="http://arxiv.org/pdf/1706.03762v7"/>
  </entry>
  <entry>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <summary>We introduce a new language representation model called BERT.</summary>
    <author><name>Jacob Devlin</name></author>
    <category term="cs.CL"/>
    <link type="application/pdf" href="http://arxiv.org/pdf/1810.04805v2"/>
  </entry>
</feed>"""


@pytest.fixture
def sample_paper_entries():
    """Parsed paper entries for testing."""
    return [
        {
            "title": "Attention Is All You Need",
            "summary": "The dominant sequence transduction models...",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "categories": ["cs.CL", "cs.LG"],
            "pdf": "http://arxiv.org/pdf/1706.03762v7",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "summary": "We introduce a new language representation model called BERT.",
            "authors": ["Jacob Devlin"],
            "categories": ["cs.CL"],
            "pdf": "http://arxiv.org/pdf/1810.04805v2",
        },
    ]
