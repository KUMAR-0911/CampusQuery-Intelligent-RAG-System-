# CampusQuery — Intelligent RAG System

An **Agentic Hybrid-RAG** (Retrieval-Augmented Generation) system for career and campus document analysis. Upload resumes, job descriptions, syllabi, and academic documents — then ask intelligent questions grounded in your document contents with full user authorization and persistent context.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                          │
│               React + Vite Frontend  (Port 5173)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP / REST (JWT Cookies & Axios)
┌──────────────────────────▼──────────────────────────────────────┐
│                        Backend API                             │
│                  FastAPI Backend  (Port 8000)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                     ┌─────▼─────┐
                     │ Guardrails │  ← HF Llama-3.1-8B (input/output safety)
                     └─────┬─────┘
                           │
               ┌───────────▼──────────────────────┐
               │  Step 1: Pre-Retrieval Rewriter  │  ← Qwen3-0.6B (featherless-ai)
               │  • Intent Detection              │    (TARGETED / SKILL_GAP /
               │  • Context Query Expansion       │     JOB_MATCH / RESUME_SCORE)
               └───────────┬──────────────────────┘
                           │ Rewritten Query + Intent
               ┌───────────▼──────────────────────┐
               │  Step 2: Direct Retrieval &      │  ← PostgreSQL (pgvector)
               │          Cross-Encoder Rerank    │  ← ms-marco-MiniLM-L-6-v2
               └───────────┬──────────────────────┘
                           │ Grounded Context Chunks
               ┌───────────▼──────────────────────┐
               │  Step 3: Reasoning Answer Model  │  ← Portkey → Groq (qwen3:32b-a3b)
               │  • Grounded Context Reasoning    │  • Strict Symbol Sanitation:
               │  • Post-Processing Text Cleaner  │    No `--`, no `*`, no `#`
               └───────────┬──────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │       Persistent Memory & Database    │
        │  PostgreSQL (cloud) — User & Memory   │
        │  • User accounts, JWT Auth & OTP     │
        │  • Durable facts & rolling summary   │
        │  • Chat history with async compaction│
        └──────────────────────────────────────┘
```

---

## Tech Stack

### Frontend (`/frontend`)
- **Framework**: React 19 + Vite
- **Routing**: React Router DOM v7
- **HTTP Client**: Axios with interceptors & HTTP-only JWT cookies
- **Icons**: Lucide React
- **Styling**: Modern CSS design system

### Backend (`/backend`)
- **API Framework**: FastAPI + Uvicorn
- **Auth**: JWT Tokens (HttpOnly cookies), Argon2 / Bcrypt hashing
- **Document Processing**: Affinda AI Document & Resume Parser API (with high-fidelity local fallback for PDF, DOCX, TXT, MD)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: PostgreSQL (`pgvector`)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Cross-encoder reranker)
- **Reasoning Model**: `qwen3:32b-a3b` (`qwen/qwen3-30b-a3b`) via Portkey
- **Query Rewriter & Intent Detection**: `Qwen/Qwen3-0.6B` (Hugging Face Inference provider: `featherless-ai`)
- **Guardrails**: `meta-llama/Llama-3.1-8B-Instruct` (Hugging Face Inference provider: `featherless-ai`)
- **Background Summariser**: `Qwen/Qwen2.5-Coder-3B-Instruct` (Hugging Face Inference provider: `nscale`)

---

## Project Structure

```
CampusQuery-Intelligent-RAG-System/
├── README.md               # Unified project documentation
├── .gitignore              # Git ignore configuration
│
├── backend/                # Python FastAPI Backend
│   ├── main.py             # FastAPI entry point & API endpoints
│   ├── config.py           # Configuration manager (reads backend/.env)
│   ├── auth.py             # JWT authentication & password hashing
│   ├── models.py           # Database user models & schema manager
│   ├── database.py         # SQLAlchemy database connection engine
│   ├── retrieval_agent.py  # LangChain agent with tool calling
│   ├── guardrails.py       # Safety & relevance guardrail classifier
│   ├── postgres_memory.py  # User-scoped persistent conversation memory
│   ├── memory_summarizer.py# Background memory compaction worker
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # Backend environment secrets & configuration
│   └── RETRIVAL/          # Document processing & ingestion pipeline
│       ├── pipeline.py     # End-to-end ingest & query orchestration
│       ├── chunking.py     # Hybrid chunking strategies
│       ├── cleaning.py     # Text cleaning & normalization
│       ├── embedding.py    # Batch vector embedding generation
│       ├── enrich_metadata.py # Metadata enrichment
│       └── pgvectorstore.py# PostgreSQL pgvector similarity search & reranking
│
└── frontend/               # React + Vite Frontend
    ├── package.json        # Node.js dependencies & scripts
    ├── vite.config.js      # Vite configuration & dev proxy
    ├── .env                # Frontend environment variables
    ├── public/
    └── src/
        ├── App.jsx         # App routing & protected route wrappers
        ├── main.jsx        # React root entry point
        ├── api/            # Axios API client with interceptors
        ├── context/        # Auth and Toast notification context
        └── pages/          # Pages (Login, Register, Dashboard, Documents, Chat, etc.)
```

---

## Eager Model Loading & Startup

When the backend application starts (`uvicorn main:app`), **all AI models are pre-loaded immediately** during startup (`lifespan` handler) to prevent first-query latency:

1. **Dense Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
2. **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
3. **Guardrails Model**: `meta-llama/Llama-3.1-8B-Instruct` (Provider: `featherless-ai`)
4. **Query Rewriter & Intent Classifier**: `Qwen/Qwen3-0.6B` (Provider: `featherless-ai`)
5. **Reasoning & Answer Model**: `qwen3:32b-a3b` (`qwen/qwen3-30b-a3b`) via Portkey
6. **Background Summarizer**: `Qwen/Qwen2.5-Coder-3B-Instruct` (Provider: `nscale`)

---

## Pre-Retrieval Pipeline & Formatting Rules

### 1. Pre-Retrieval Intent Detection & Query Rewriting
Before any vector retrieval occurs, the system invokes **Qwen/Qwen3-0.6B** (hosted on Featherless AI):
- **Intent Classification**: Evaluates recent conversation history and classifies the user prompt into one of four distinct categories:
  1. `RESUME_SCORE`: Overall ATS score, profile readiness, or full resume evaluation.
  2. `JOB_MATCH`: Comparing candidate qualifications against a job description.
  3. `SKILL_GAP`: Identifying missing technologies, tools, or learning paths.
  4. `TARGETED`: Specific fact queries (e.g. graduation year, GPA, specific company, tool).
- **Query Rewriting**: Expands vague user phrases (e.g. *"rate me"*) into targeted keywords and relevant resume sections for high-precision semantic matching.
- **Dynamic Retrieval Tuning**: The detected intent automatically scales the retrieval parameters:
  - `TARGETED`: `top_k=5`, `rerank_top_n=10`
  - `SKILL_GAP`: `top_k=10`, `rerank_top_n=20`
  - `JOB_MATCH`: `top_k=15`, `rerank_top_n=30`
  - `RESUME_SCORE`: `top_k=20`, `rerank_top_n=40`

### 2. Strict Symbol Sanitation (No `--`, `*`, or `#`)
The reasoning model is strictly constrained to output clean, professional plain text:
- **No Asterisks (`*`)**: No bold markdown (`**text**`), no italic markdown (`*text*`), and no bullet asterisks (`* item`).
- **No Hashtags (`#`)**: No markdown headers (`# Header`, `### Section`). Uppercase titles with colons (e.g. `TECH STACK:`) are used instead.
- **No Double Hyphens (`--`)**: No horizontal line dividers (`---`) or double hyphens.
- **Lists**: Standard numbered lists (`1.`, `2.`, `3.`) or clean line breaks.
- **Dual-Layer Stripping**: Both backend post-processing (`clean_markdown_text` in `main.py` and `_clean_output_text` in `retrieval_agent.py`) and frontend sanitization strip any stray symbols.

---

## Environment Setup & Configurations

The project maintains **independent configuration files** for the backend and frontend.

### 1. Backend Configuration (`/backend/.env`)

Configure environment variables in `backend/.env`:

Key environment variables:
| Variable | Description |
|----------|-------------|
| `PORTKEY_API_KEY` / `PORTKEY_CONFIG_SLUG` | Portkey API credentials for routing the reasoning agent |
| `GROQ_MODEL` | Reasoning agent model: `qwen3:32b-a3b` (routed via Portkey) |
| `HF_TOKEN` | Hugging Face Token for remote inference |
| `HF_GUARDRAIL_MODEL` / `HF_INFERENCE_PROVIDER` | Guardrail model: `meta-llama/Llama-3.1-8B-Instruct` (provider: `featherless-ai`) |
| `QUERY_REWRITER_MODEL` / `QUERY_REWRITER_PROVIDER` | Query rewriter: `Qwen/Qwen3-0.6B` (provider: `featherless-ai`) |
| `SUMMARIZER_MODEL` / `SUMMARIZER_INFERENCE_PROVIDER` | Async background summarizer: `Qwen/Qwen2.5-Coder-3B-Instruct` (provider: `nscale`) |
| `RERANKER_MODEL` | Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `POSTGRES_URL` | PostgreSQL connection string with `pgvector` enabled |
| `CORS_ORIGINS` | Allowed frontend origins (e.g. `http://localhost:5173,http://127.0.0.1:5173`) |
| `JWT_SECRET_KEY` | Secret key for signing JWT tokens |
| `CAMPUSQUERY_SYSTEM_PROMPT` | Main reasoning assistant system prompt with strict formatting constraints |
| `RESUME_ANALYSER_GUARDRAIL_PROMPT` | Safety & relevance classification prompt for guardrails |
| `QUERY_REWRITER_PROMPT` | Context-aware query expansion & intent classification prompt |
| `MEMORY_EXTRACTION_PROMPT` | Asynchronous background memory compaction & factual extraction prompt |

### 2. Frontend Configuration (`/frontend/.env`)

Configure environment variables in `frontend/.env`:


Key environment variables:
| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Target FastAPI backend URL | `http://localhost:8000` |

---

## Running the Application

### 1. Start the FastAPI Backend

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows (PowerShell)      
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Start Uvicorn backend server
uvicorn main:app --reload --port 8000
```

The backend server will run at: `http://localhost:8000`  
API documentation is accessible at: `http://localhost:8000/docs`

---

### 2. Start the React Frontend

Open a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

The frontend application will run at: `http://localhost:5173`

---

## API Endpoints Overview

### Authentication Endpoints
- `POST /register`: Register a new user account.
- `POST /verify-otp`: Verify registration or reset OTP.
- `POST /login`: Authenticate and set HttpOnly JWT access/refresh cookies.
- `POST /refresh`: Refresh access token using refresh cookie.
- `POST /logout`: Clear authentication cookies.
- `GET /me`: Fetch authenticated user profile.

### Document & RAG Endpoints
- `POST /upload`: Upload and ingest documents into pgvector table.
- `POST /chat`: Send a question to the agentic RAG pipeline.
- `GET /health`: Health check endpoint.

---

## License

Educational and Research Purposes.
