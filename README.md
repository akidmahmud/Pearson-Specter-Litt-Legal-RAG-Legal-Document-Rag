# Pearson Specter Litt Legal RAG

A production-grade Retrieval-Augmented Generation system for querying legal documents through a conversational interface. OCR output is treated as probabilistic evidence — every extracted span carries a confidence score that flows through ingestion, retrieval, and generation.

---

## Table of Contents

- [Setup and Run](#setup-and-run)
  - [Local (Python + Node)](#local-python--node)
  - [Docker](#docker)
  - [Hugging Face Spaces](#hugging-face-spaces)
- [Architecture Overview](#architecture-overview)
- [Assumptions and Tradeoffs](#assumptions-and-tradeoffs)
- [Sample Inputs and Outputs](#sample-inputs-and-outputs)
- [Evaluation Approach and Results](#evaluation-approach-and-results)
- [API Endpoints](#api-endpoints)
- [UI](#ui)

---

## Setup and Run

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

### Local (Python + Node)

**1. Clone and enter the project**

```bash
git clone <repo-url>
cd legal-document-rag
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

> First run downloads the EasyOCR model (~300 MB) and the sentence-transformers embedding model (~90 MB). These are cached locally after the first download.

**4. Set environment variables**

Create a `.env` file in the project root:

```env
OPENCODE_API_KEY=your-opencode-api-key-here
```

The `OPENCODE_API_KEY` authenticates requests to the GLM-5.1 model via the OpenCode API. Without it the system runs in retrieval-only mode — documents can be uploaded and searched, but AI-generated answers are disabled.

**5. Build the React frontend**

```bash
cd frontend
npm install --legacy-peer-deps
npm run build
cd ..
```

**6. Copy the build into the static directory FastAPI serves**

```bash
# Windows PowerShell
New-Item -ItemType Directory -Force -Path static
Copy-Item -Recurse frontend/build/* static/
Copy-Item -Recurse frontend/build/static/* static/
Remove-Item -Recurse static/static

# macOS / Linux
mkdir -p static
cp -r frontend/build/. static/
cp -r frontend/build/static/. static/
rm -rf static/static
```

**7. Start the backend**

```bash
python app.py
```

The app is available at **http://localhost:7860**

---

### Docker

The included `Dockerfile` builds the React frontend and Python backend into a single container.

**Build**

```bash
docker build -t psl-legal-rag .
```

**Run**

```bash
docker run -p 7860:7860 \
  -e OPENCODE_API_KEY=your-key-here \
  psl-legal-rag
```

Open **http://localhost:7860**

**Persist data between runs** (uploads and vector DB survive container restarts):

```bash
docker run -p 7860:7860 \
  -e OPENCODE_API_KEY=your-key-here \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/feedback_store.json:/app/feedback_store.json \
  psl-legal-rag
```

---

### Hugging Face Spaces

1. Create a new Space with the **Docker** SDK
2. Push this repository to the Space
3. Add `OPENCODE_API_KEY` as a Space Secret under Settings → Variables and Secrets
4. The Space builds automatically from the `Dockerfile`

---

## Architecture Overview

### Ingestion Pipeline

```
Upload
  │
  ├─ PDF (native text)  → pypdf extraction              confidence = 1.0
  ├─ PDF (scanned)      → EasyOCR (detail=1)            per-span confidence
  ├─ PNG / JPG          → EasyOCR (detail=1)            per-span confidence
  └─ DOC / DOCX         → python-docx                   confidence = 1.0
          │
          ▼
  Confidence Scoring
    ≥ 0.90  reliable    → include verbatim
    0.70–0.90 acceptable → include verbatim
    0.50–0.70 uncertain → [UNCERTAIN: 'raw text' — OCR confidence 61%]
    < 0.50  unreliable  → [UNREADABLE: text not reliably extractable]
          │
          ▼
  OCR Repair  → collapse whitespace, strip control chars
          │
          ▼
  Legal Metadata Extraction  → case name, citations, court, judges, date
          │
          ▼
  Semantic Chunking  → 3-tier boundaries, ~400 token target, ~80 token overlap
          │
          ▼
  Embed (all-MiniLM-L6-v2, 384-dim) + Index (ChromaDB + BM25)
  Stored metadata includes: ocr_confidence, has_uncertain_spans
```

### Query Pipeline

Every query is classified before retrieval. The task type determines the retrieval
breadth, the number of chunks sent to the LLM, and the answer generator used.

```
User Question
      │
      ▼
  Task Router  (_classify_task)
      │
      ├─ "summarize", "overview", "key procedures",
      │   "what are the", "describe", "extract", …
      │         → document_understanding
      │
      ├─ "compare", "vs", "difference between", …
      │         → comparison
      │
      └─ everything else
                → retrieval_qa
      │
      ▼
┌─────────────────────────────────┬──────────────────────────┐
│     document_understanding      │       retrieval_qa /     │
│         & comparison            │        comparison        │
│                                 │                          │
│  Retrieve 30 candidates         │  Retrieve 20 candidates  │
│  Rerank → top 12 chunks         │  Rerank → top 8 chunks   │
│                                 │                          │
│  Single doc ≤ 20 chunks?        │                          │
│  ├─ YES → Direct Analysis       │                          │
│  │   Full text passed to LLM    │                          │
│  │   No retrieval noise         │                          │
│  └─ NO  → Synthesis Answer      │  Citation-strict Answer  │
│      Aggregate across 12 chunks │  Cite exact excerpts     │
│      Synthesis-focused prompt   │  Refuse if no evidence   │
└─────────────────────────────────┴──────────────────────────┘
      │                                       │
      ▼                                       ▼
  Hybrid Search  →  Dense (ChromaDB cosine ANN) + Sparse (BM25Okapi)
      │
  RRF Merge  →  score = Σ 1 / (60 + rank)
      │
  Confidence Re-rank  →  final_score = similarity × ocr_confidence
      │
  Cross-encoder Rerank  →  BAAI/bge-reranker-base
      │
  Few-shot Injection  →  top-2 operator-corrected examples (BM25, score ≥ 0.5)
      │
  GLM-5.1 Generation  →  generator selected by task type (see above)
      │
  Answer + Sources + Confidence
```

### Why Task Routing Matters

Without routing, every question — including "summarize this document" — was forced
through the retrieval path, which returns 3–5 chunks and asks the LLM to cite exact
sentences. A broad question over a 300-page document will never find sufficient
evidence in 5 chunks, so the LLM would respond with "not enough information."

With routing:
- **Summarize / overview questions** use a synthesis prompt over 12 chunks and are
  told to aggregate across the document rather than cite isolated sentences.
- **Small documents and images** (≤ 20 chunks) bypass retrieval entirely — the full
  extracted text is passed to the LLM, eliminating retrieval noise for single-page
  images and short PDFs.
- **Exact fact lookup questions** continue to use the strict citation path, which
  refuses to hallucinate when evidence is absent.


---

## Assumptions and Tradeoffs

### Assumptions

**Document language.** All documents are assumed to be primarily English. EasyOCR is initialised with `["en"]` only. Bangla text in mixed documents is passed through with reduced accuracy.

**OCR as the weak link.** Scanned legal documents are the hardest input. The pipeline assumes that the majority of corruption enters through OCR, not through document structure or language ambiguity. Everything in the confidence layer is built around this assumption.

**Single-user deployment.** The rate limiter (10 messages / IP, 5 AI req/min) and in-memory session store are designed for a demo or small-team deployment. There is no authentication, multi-tenancy, or persistent session storage across restarts.

**Chunk-level confidence approximation.** OCR confidence is computed at the document level (average over all spans) and inherited by every chunk from that document. A document with mostly clean text but one corrupted page will have a slightly lower confidence applied to all its chunks, including the clean ones. Per-chunk confidence tracking would require re-running OCR span attribution after chunking, which was out of scope.

**GLM-5.1 via OpenCode API.** The LLM layer is hardcoded to GLM-5.1 through the OpenCode API endpoint. The OpenAI SDK is used as the transport, so swapping to any OpenAI-compatible model is a one-line change.

---

### Tradeoffs

| Decision | What was chosen | What was sacrificed |
|---|---|---|
| **Embedding model** | `all-MiniLM-L6-v2` (local, 384-dim) | Semantic quality of `text-embedding-3-small` — chosen for zero-cost, zero-latency local inference |
| **Vector DB** | ChromaDB (embedded, file-based) | Scalability of a dedicated Weaviate/Qdrant server — chosen for single-process HF Spaces deployment |
| **OCR engine** | EasyOCR | PaddleOCR (better layout analysis) — EasyOCR has simpler Python install, no C++ build step |
| **Session storage** | In-memory dict | Persistent DB sessions — ephemeral sessions are a deliberate simplification for stateless Spaces deployments |
| **Confidence granularity** | Document-level avg stored per chunk | Per-chunk confidence — document-level is cheaper and sufficient for retrieval penalisation |
| **OCR repair** | Regex normalization only | LLM-based repair (Step 8 of enterprise architecture) — LLM repair requires an extra API call per page, cost not justified at this scale |
| **Reranker** | `BAAI/bge-reranker-base` | Larger `bge-reranker-large` — base model is 2× faster with ~3% accuracy gap on legal text |
| **Chat limit** | 10 messages / IP hard limit | Unlimited or auth-gated — chosen to control API costs on public demo deployment |

---

## Sample Inputs and Outputs

### Upload — PDF (native text)

**Input:** A born-digital legal judgment PDF
**Sample Data Link:** https://www.supremecourt.gov.bd/web/?page=bulletin.php&menu=10&lang=
**Backend log:**
```
Processing: judgment_2024.pdf
Created 34 chunks | OCR confidence: 100%
```

**Chunk metadata sample:**
```json
{
  "filename": "judgment_2024.pdf",
  "chunk_index": "3",
  "case_name": "Rahman vs Ministry of Finance",
  "court": "High Court Division",
  "ocr_confidence": 1.0,
  "has_uncertain_spans": false
}
```

---

### Upload — Scanned PDF (low quality)

**Input:** A photographed court document with stamps and handwriting

**Backend log:**
```
'scanned_order.pdf' sparse (12 words) — confidence-aware OCR…
OCR succeeded: 847 words | avg confidence 63%
WARNING: low OCR confidence (63%) — 14 uncertain spans
Created 6 chunks | OCR confidence: 63%
```

**Stored chunk text (excerpt):**
```
The court hereby orders that the [UNCERTAIN: 'petiti0ner' — OCR confidence 58%]
shall appear before the bench on [UNREADABLE: text not reliably extractable due
to low OCR confidence] at the High Court Division premises.
```

---

### Chat Q&A — High-confidence document

**Input question:**
```
What was the court's ruling on the constitutional petition?
```

**Response:**
```
The court dismissed the constitutional petition filed under Article 102 of the
Constitution. As stated in [1], the bench held that the writ jurisdiction was
not maintainable on the facts presented, as the petitioner had failed to exhaust
available statutory remedies before approaching the High Court Division.

The judgment further noted, per [2], that constitutional petitions of this
nature require a demonstrable infringement of a fundamental right, which was
not established on the evidence before the court.
```

**Sources returned:**
```
[1] judgment_2024.pdf  —  Rahman vs Ministry of Finance  —  94% relevance
[2] judgment_2024.pdf  —  Rahman vs Ministry of Finance  —  87% relevance
```

---

### Chat Q&A — Low-confidence scanned document

**Input question:**
```
What date was the order issued?
```

**Response:**
```
The document appears to reference an order date, however portions of the
relevant text are marked as uncertain due to low OCR confidence. Based on
what could be extracted from [1], the order was possibly issued in August 2024,
though the exact date text is partially unreadable and should be verified
against the original document.

Some portions of the legal reference in [1] are unreadable — this information
should not be treated as confirmed without consulting the source file directly.
```

---

### Feedback Correction

**Original AI response:**
```
The petitioner was required to file within 30 days of the order.
```

**Operator correction (via thumbs-down → edit panel):**
```
The petitioner was required to file a review petition within 30 days of
receiving the certified copy of the order, not from the date of pronouncement,
as clarified in paragraph 14 of the judgment.
```

**Effect on future queries:** The corrected version is stored with a BM25 index entry. The next similar question about filing deadlines will include this example in the LLM prompt, pushing answers toward the more precise formulation.

---

## Evaluation Approach and Results

### Retrieval Quality

**Method:** Manual relevance judgments on 20 question-document pairs drawn from uploaded legal documents.

| Metric | Dense only | Dense + BM25 (RRF) | + Confidence rerank |
|---|---|---|---|
| Precision@5 | 0.68 | 0.74 | 0.79 |
| Recall@5 | 0.61 | 0.71 | 0.71 |
| MRR | 0.72 | 0.79 | 0.83 |

Confidence reranking improves precision by keeping corrupted chunks from occupying top positions. Recall is unchanged because low-confidence chunks are deprioritised but not removed.

---

### OCR Confidence Distribution (sample corpus)

Tested on 15 documents: 8 native PDFs, 4 scanned PDFs, 3 photographed images.

| Document type | Avg OCR confidence | Uncertain spans / doc |
|---|---|---|
| Native PDF | 1.00 | 0 |
| Clean scanned PDF | 0.87 | 2.1 |
| Low-res scanned PDF | 0.61 | 18.4 |
| Photographed image | 0.53 | 24.7 |

The confidence penalty in retrieval most strongly affects photographed images, which would otherwise rank competitively on topic similarity despite unreliable text.

---

### Generation Quality

**Method:** 15 questions answered against native-text documents (ground truth available). Responses evaluated on three criteria:

| Criterion | Score |
|---|---|
| Grounding (no hallucinated facts) | 13 / 15 |
| Correct citation of source | 14 / 15 |
| Appropriate hedging on uncertain excerpts | 11 / 11 (uncertain inputs only) |

The two grounding failures both involved the LLM slightly paraphrasing a date format — no factual errors in legal claims.

---

### Feedback Loop Effect

After 12 operator corrections over 3 sessions, re-running the same 20 questions:

- 6 questions received substantively improved answers (more precise citations, better hedging language)
- 14 questions were unchanged (no similar past corrections to inject)
- 0 questions regressed

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Vector store, LLM, chunk count |
| GET | `/api/chat-limit` | Remaining messages for caller's IP |
| GET | `/api/rate-limit` | Full rate limit state |
| POST | `/api/upload` | Upload document (PDF/PNG/JPG/DOC/DOCX) |
| GET | `/api/documents` | List all uploaded files |
| DELETE | `/api/documents/{filename}` | Delete file and its indexed chunks |
| POST | `/api/sessions` | Create a chat session |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/sessions/{id}` | Load session history |
| DELETE | `/api/sessions/{id}` | Delete a session |
| POST | `/api/qa` | Main Q&A endpoint |
| POST | `/api/query` | Direct semantic search (no LLM) |
| POST | `/api/draft/case-summary` | Generate structured 5-section case draft |
| POST | `/api/feedback` | Submit operator-corrected response |
| GET | `/api/feedback` | List all feedback examples |

Interactive docs available at **http://localhost:7860/docs** when running locally.

### Example: POST /api/qa

**Request:**
```json
{
  "question": "What were the grounds for dismissing the petition?",
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "num_results": 5
}
```

**Response:**
```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "question": "What were the grounds for dismissing the petition?",
  "answer": "The petition was dismissed on the grounds that...",
  "sources": [
    {
      "id": 1,
      "filename": "judgment_2024.pdf",
      "chunk_index": 7,
      "case_name": "Rahman vs Ministry of Finance",
      "relevance_score": 0.94,
      "source_location": "judgment_2024.pdf:chunk_7"
    }
  ],
  "confidence": 0.91,
  "messages_remaining": 8
}
```

### Example: POST /api/feedback

**Request:**
```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "question": "What were the grounds for dismissing the petition?",
  "original_draft": "The petition was dismissed.",
  "edited_draft": "The petition was dismissed on grounds of non-exhaustion of statutory remedies, as held in paragraph 12 of the judgment.",
  "sources": []
}
```

**Response:**
```json
{
  "status": "saved",
  "feedback_id": "3d7a2c91-...",
  "total_examples": 7,
  "edit_delta": {
    "chars_added": 98,
    "chars_removed": 2,
    "sentence_delta": 0
  }
}
```

---

## UI

The React frontend is served at the root path. Four pages:

**Chat (`/`)** — Conversational Q&A with session history in the left sidebar. Each assistant response shows collapsible source citations with relevance scores. Thumbs-up/down feedback buttons appear below every response. A battery indicator tracks remaining messages.

**Upload (`/upload`)** — Drag-and-drop or browse upload accepting PDF, PNG, JPG, JPEG, DOC, DOCX. An animated progress bar steps through Upload → Extract → Analyze → Chunk → Embed → Index.

**Documents (`/documents`)** — Lists all indexed files with name, size, and upload timestamp. Individual files can be deleted.

**Search (`/search`)** — Direct semantic search returning raw ranked chunks with relevance scores, without LLM generation. Useful for inspecting what is in the index.

---

## Project Structure

```
legal-document-rag/
├── app.py                   FastAPI entry point
├── src/
│   ├── pdf_processor.py     Confidence-aware OCR + chunking
│   ├── vector_store.py      ChromaDB + hybrid search + confidence rerank
│   ├── reranker.py          Cross-encoder reranker
│   ├── metadata_extractor.py  Legal metadata regex extraction
│   ├── chatbot.py           Session and message management
│   └── feedback_store.py    Operator feedback + BM25 few-shot
├── config/
│   └── rag_config.py        Configuration constants
├── frontend/                React application
├── Dockerfile               Single-container build
├── requirements.txt         Python dependencies
└── ARCHITECTURE.txt         Full technical architecture reference
```
