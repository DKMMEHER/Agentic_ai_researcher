# AI Researcher: LangGraph Architecture

This diagram shows the complete node and edge structure of the compiled agent graph.
You can preview this file natively in VS Code or GitHub to see the drawn diagram.

```mermaid
graph TD
    classDef startend fill:#2B303A,color:white,stroke:#555,stroke-width:2px;
    classDef agent fill:#4A90E2,color:white,stroke:#357ABD,stroke-width:2px,rx:10px,ry:10px;
    classDef tools fill:#E94E77,color:white,stroke:#C0392B,stroke-width:2px,rx:10px,ry:10px;

    Start(((START))):::startend
    Agent["🧠 Agent Node<br><i>Llama-3 LLM</i>"]:::agent
    Tools[["🛠️ Tools Node<br><i>(arXiv, PDF, Tavily, DuckDuckGo, PubMed, SemanticScholar, YouTube)</i>"]]:::tools
    End(((END))):::startend

    Start -->|Initializes State| Agent

    Agent -.->|<i>evaluates 'should_continue' logic</i>| Agent

    Agent -- "Tool Call Requested" --> Tools
    Tools -- "JSON Result Returned" --> Agent

    Agent -- "Final Answer Drafted" --> End

```
