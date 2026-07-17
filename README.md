# Compliance Advisory & Triage Agent

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://capstoneprojectagenticai-n4cur585zhvuhnpmwuupjb.streamlit.app/)

A production-quality AI application for compliance question answering, triage, and audit — built as a college capstone project.

🔗 **Live app:** https://capstoneprojectagenticai-n4cur585zhvuhnpmwuupjb.streamlit.app/

## Features

- **RAG pipeline** over PDF policy documents (GDPR, AML, Vendor, Security, etc.)
- **LangGraph routing agent** classifies topic, risk level, and owner
- **Governance layer** — refuses hallucination, enforces confidence threshold
- **Human escalation** for HIGH-risk queries
- **Immutable JSON audit log** for every interaction
- **Streamlit dark-mode dashboard** with theme toggle
- **Evaluation page** with PASS/FAIL test suite

## Project Structure

```
capstone_project/
├── app/
│   ├── rag/
│   │   ├── loader.py          # PDF ingestion
│   │   ├── chunker.py         # Text splitting
│   │   ├── embeddings.py      # SentenceTransformers wrapper
│   │   └── retriever.py       # ChromaDB retrieval
│   ├── agents/
│   │   ├── routing_agent.py   # LangGraph topic/risk routing
│   │   └── compliance_agent.py# Main answer generation
│   ├── governance/
│   │   ├── escalation.py      # Escalation logic
│   │   ├── refusal.py         # Out-of-corpus refusal
│   │   └── confidence.py      # Confidence scoring
│   ├── audit/
│   │   └── logger.py          # Audit trail (JSON + SQLite)
│   └── ui/
│       └── components.py      # Reusable Streamlit widgets
├── data/
│   ├── policies/              # Drop your PDFs here
│   └── chroma_db/             # Auto-generated vector store
├── logs/
│   └── audit.json             # Auto-generated audit trail
├── .streamlit/
│   └── config.toml            # Theme configuration
├── main.py                    # Streamlit entry point
├── requirements.txt
└── .env.example
```

## Setup

### 1. Prerequisites

- Python 3.11
- An [OpenRouter](https://openrouter.ai) API key

### 2. Install dependencies

```bash
cd capstone_project
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENROUTER_API_KEY
```

### 4. Add policy documents

Copy your PDF files into `data/policies/`:

```
data/policies/
├── GDPR.pdf
├── FATF_AML.pdf
├── CRF_Governance.pdf
├── PasswordPolicy.pdf
├── IncidentResponse.pdf
└── VendorPolicy.pdf
```

The app will **automatically ingest** them into ChromaDB on first launch.

### 5. Run the application

```bash
streamlit run main.py
```

## Usage

1. Open the app in your browser (default: http://localhost:8501)
2. The sidebar shows document and vector DB status
3. Type a compliance question in the chat input
4. The agent returns: Answer · Sources · Confidence · Topic · Risk · Owner · Escalation status
5. All queries are logged automatically to `logs/audit.json`
6. Use the **Evaluation** page to run the built-in PASS/FAIL test suite
7. Download the full audit log from the **Audit History** page

## Example Questions

- What is the retention period for customer records?
- Can we store EU customer data in the US?
- What are customer due diligence requirements?
- Can we ignore GDPR for one client?
- What password length is required?

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit 1.35 |
| Orchestration | LangGraph |
| LLM | OpenRouter (Mistral-7B) |
| RAG | LangChain + ChromaDB |
| Embeddings | SentenceTransformers |
| Data models | Pydantic v2 |
| Audit store | JSON + SQLite |

## License

MIT — for educational use.
