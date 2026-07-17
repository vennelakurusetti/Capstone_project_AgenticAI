# Compliance Advisory & Triage Agent

A production-quality AI application for compliance question answering, triage, and audit — built as a college capstone project.

## Features

- **RAG pipeline** over PDF policy documents (GDPR, AML, Vendor, Security, etc.)
- **LangGraph routing agent** classifies topic, risk level, and owner
- **Governance layer** — refuses hallucination, enforces confidence threshold
- **Human escalation** for HIGH-risk queries
- **Immutable JSON audit log** for every interaction
- **Streamlit dark-mode dashboard** with theme toggle
- **Evaluation page** with PASS/FAIL test suite

## How the knowledge base works

The vector database is built **once, locally** using `ingest.py`. The generated
`data/chroma_db/` is committed to GitHub and shipped with the repo. The deployed
app simply loads it on startup — no embedding or rebuilding ever happens in production.

```
Local machine                        Render / any server
─────────────                        ───────────────────
python ingest.py                     streamlit run main.py
  └─ reads data/policies/*.pdf         └─ loads data/chroma_db/  ← from git
  └─ builds data/chroma_db/            └─ answers queries
  └─ git commit + push
```

> If `data/chroma_db/` is missing on startup the app shows a clear error
> with instructions instead of trying to rebuild automatically.

## Project Structure

```
capstone_project/
├── app/
│   ├── rag/
│   │   ├── loader.py          # PDF ingestion helpers
│   │   ├── chunker.py         # Text splitting
│   │   ├── embeddings.py      # SentenceTransformers wrapper
│   │   └── retriever.py       # ChromaDB load + retrieval
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
│   ├── policies/              # PDFs — gitignored, keep locally
│   └── chroma_db/             # Built by ingest.py — COMMIT THIS
├── logs/                      # Runtime logs — gitignored
├── .streamlit/
│   └── config.toml            # Theme + server config
├── ingest.py                  # Run locally to build the vector DB
├── main.py                    # Streamlit entry point (load-only)
├── render.yaml                # Render deployment config
├── requirements.txt
└── .env.example               # Copy to .env and fill in secrets
```

---

## Local Development

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
# Edit .env — set OPENROUTER_API_KEY at minimum
```

### 4. Add policy PDFs

Copy your PDF files into `data/policies/`:

```
data/policies/
├── GDPR_Regulation.pdf
├── FATF_AML_Guidelines.pdf
├── CRF-Governance-Risk-Model.pdf
├── SANS_Password_Construction_Standard.pdf
├── sans_Log_Management_Policy.pdf
└── vendor-policy.pdf
```

PDFs are gitignored — they stay on your local machine only.

### 5. Build the knowledge base

```bash
python ingest.py
```

This scans `data/policies/`, embeds all chunks, and writes the ChromaDB to
`data/chroma_db/`. **Run this once** before starting the app. Re-run any time
you add or replace PDFs.

### 6. Run locally

```bash
streamlit run main.py
```

Open http://localhost:8501.

---

## Deploying to Render

### One-time setup

1. Build the vector DB locally if you haven't already:

   ```bash
   python ingest.py
   ```

2. Commit `data/chroma_db/` to git (**this is intentional**):

   ```bash
   git add data/chroma_db/
   git commit -m "Add pre-built ChromaDB vector store"
   git push
   ```

3. Connect your GitHub repo to [Render](https://render.com).  
   Render will auto-detect `render.yaml` and configure the service.

4. In the Render dashboard → your service → **Environment** tab, add:

   | Key | Value |
   |-----|-------|
   | `OPENROUTER_API_KEY` | your key from openrouter.ai |

5. Trigger a deploy. Render will run `pip install -r requirements.txt`
   then `streamlit run main.py`. The app loads `data/chroma_db/` from the
   repo — no embedding happens on the server.

### Adding new documents later

1. Add PDFs to `data/policies/` locally.
2. Re-run `python ingest.py` (this rebuilds `data/chroma_db/` from scratch).
3. Commit and push `data/chroma_db/`:

   ```bash
   git add data/chroma_db/
   git commit -m "Rebuild ChromaDB with updated policy documents"
   git push
   ```

4. Render auto-deploys on push — the new DB is live immediately.

---

## Usage

1. Open the app in your browser
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
| Deployment | Render |

## License

MIT — for educational use.
