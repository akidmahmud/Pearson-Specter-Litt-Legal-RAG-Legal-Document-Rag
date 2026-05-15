"""
Pearson Specter Litt Legal RAG System - Hugging Face Spaces Entry Point
Serves both FastAPI backend and React frontend as static files
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src and config directories to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "config"))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import time
import uuid
import json as _json
import re as _re
import uvicorn

from vector_store import VectorStoreManager
from pdf_processor import PDFProcessor

ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.doc', '.docx'}
from metadata_extractor import LegalMetadataExtractor
from chatbot import Chatbot, ChatbotConfig
from reranker import Reranker
from feedback_store import FeedbackStore
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(
    title="Pearson Specter Litt Legal Document RAG",
    description="Search and query legal documents with Pearson Specter Litt",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== RATE LIMITING ==============

class RateLimiter:
    """Simple in-memory rate limiter to prevent API abuse"""

    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.ai_requests: Dict[str, list] = defaultdict(list)

        # HARD LIMIT: Total chat messages per IP (permanent, doesn't reset)
        self.chat_usage: Dict[str, int] = defaultdict(int)
        self.CHAT_LIMIT = 10  # Total messages allowed per IP

        # Rate limits (per minute)
        self.general_limit = 30
        self.ai_limit = 5
        self.window = 60

    def _clean_old_requests(self, request_list: list, window: int) -> list:
        """Remove requests older than the window"""
        cutoff = time.time() - window
        return [t for t in request_list if t > cutoff]

    def check_chat_limit(self, client_ip: str) -> tuple[bool, dict]:
        """Check if user has remaining chat messages. Returns (allowed, info)"""
        used = self.chat_usage[client_ip]
        remaining = max(0, self.CHAT_LIMIT - used)

        if used >= self.CHAT_LIMIT:
            return False, {
                "allowed": False,
                "used": used,
                "limit": self.CHAT_LIMIT,
                "remaining": 0,
                "message": "You have reached the maximum limit of 10 messages. Thank you for trying Pearson Specter Litt Legal Assistant!"
            }

        return True, {
            "allowed": True,
            "used": used,
            "limit": self.CHAT_LIMIT,
            "remaining": remaining
        }

    def increment_chat_usage(self, client_ip: str):
        """Increment chat usage for an IP"""
        self.chat_usage[client_ip] += 1

    def get_chat_remaining(self, client_ip: str) -> int:
        """Get remaining chat messages for an IP"""
        return max(0, self.CHAT_LIMIT - self.chat_usage[client_ip])

    def check_rate_limit(self, client_ip: str, is_ai_request: bool = False) -> tuple[bool, dict]:
        """Check if request is within rate limits. Returns (allowed, info)"""
        current_time = time.time()

        if is_ai_request:
            self.ai_requests[client_ip] = self._clean_old_requests(
                self.ai_requests[client_ip], self.window
            )

            if len(self.ai_requests[client_ip]) >= self.ai_limit:
                remaining = 0
                reset_time = int(self.ai_requests[client_ip][0] + self.window - current_time)
                return False, {
                    "allowed": False,
                    "limit": self.ai_limit,
                    "remaining": remaining,
                    "reset_in_seconds": max(0, reset_time),
                    "message": f"AI query limit exceeded. Try again in {reset_time} seconds."
                }

            self.ai_requests[client_ip].append(current_time)
            remaining = self.ai_limit - len(self.ai_requests[client_ip])
            return True, {
                "allowed": True,
                "limit": self.ai_limit,
                "remaining": remaining,
                "reset_in_seconds": self.window
            }
        else:
            self.requests[client_ip] = self._clean_old_requests(
                self.requests[client_ip], self.window
            )

            if len(self.requests[client_ip]) >= self.general_limit:
                remaining = 0
                reset_time = int(self.requests[client_ip][0] + self.window - current_time)
                return False, {
                    "allowed": False,
                    "limit": self.general_limit,
                    "remaining": remaining,
                    "reset_in_seconds": max(0, reset_time),
                    "message": f"Rate limit exceeded. Try again in {reset_time} seconds."
                }

            self.requests[client_ip].append(current_time)
            remaining = self.general_limit - len(self.requests[client_ip])
            return True, {
                "allowed": True,
                "limit": self.general_limit,
                "remaining": remaining,
                "reset_in_seconds": self.window
            }

    def get_client_ip(self, request: Request) -> str:
        """Get client IP from request"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

# Global rate limiter instance
rate_limiter = RateLimiter()

# Pydantic Models
class QueryRequest(BaseModel):
    question: str
    num_results: int = 5
    use_ai_answer: bool = False

class QueryResponse(BaseModel):
    question: str
    results: List[dict]
    ai_answer: Optional[str] = None
    total_results: int
    source_citations: List[dict] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    num_results: int = 5

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []
    chat_history: List[ChatMessage] = []

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    chunks_added: int = 0

class SystemStatus(BaseModel):
    status: str
    vector_store_connected: bool
    total_documents: int
    total_chunks: int
    llm_ready: bool = False

# QA and Session Models
class QARequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    num_results: int = 5

class QAResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: List[dict] = []
    confidence: float = 0.0
    messages_remaining: int = 10

class SessionCreateRequest(BaseModel):
    title: str = "New Chat"

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    question_count: int = 0

class SessionListResponse(BaseModel):
    sessions: List[dict]
    total: int

class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    original_draft: str
    edited_draft: str
    sources: List[dict] = []

class CaseSummaryRequest(BaseModel):
    question: str
    num_results: int = 7
    session_id: Optional[str] = None

class DraftSection(BaseModel):
    title: str
    content: str

class CaseSummaryResponse(BaseModel):
    draft_id: str
    question: str
    sections: List[DraftSection]
    sources: List[dict]
    generated_at: str

class SessionHistoryResponse(BaseModel):
    session_id: str
    title: str
    messages: List[dict]
    metadata: dict = {}

# Global instances
vector_store = None
pdf_processor = None
metadata_extractor = None
llm_client = None
reranker = None
feedback_store = None
chatbot = None

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Chunking config
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MIN_CHUNK_SIZE = 200

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global vector_store, pdf_processor, metadata_extractor, llm_client, reranker, feedback_store, chatbot

    print("Initializing Pearson Specter Litt Legal RAG System...")

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize components
    vector_store = VectorStoreManager(persist_directory=str(CHROMA_DIR))
    pdf_processor = PDFProcessor("")
    metadata_extractor = LegalMetadataExtractor()

    # Initialize OpenCode API client
    api_key = os.getenv("OPENCODE_API_KEY")
    if api_key and api_key.strip() and not api_key.startswith("your"):
        llm_client = OpenAI(
            api_key=api_key,
            base_url="https://opencode.ai/zen/go/v1"
        )
        print("GLM-5.1 model initialized via OpenCode API")
    else:
        print("WARNING: OPENCODE_API_KEY not set — AI answers disabled")

    # Initialize reranker
    try:
        reranker = Reranker("BAAI/bge-reranker-base")
    except Exception as e:
        print(f"WARNING: Reranker failed to load ({e}) — falling back to raw retrieval")
        reranker = None

    # Initialize feedback store
    feedback_store = FeedbackStore(str(BASE_DIR / "feedback_store.json"))

    # Initialize chatbot
    chatbot_config = ChatbotConfig(
        max_context_messages=10,
        temperature=0.3,
        max_tokens=500,
        top_k_results=5,
        enable_source_citations=True
    )
    chatbot = Chatbot(config=chatbot_config)

    print("Pearson Specter Litt Legal RAG System ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if vector_store:
        vector_store.close()
    print("Shutdown complete")


# ============== API ENDPOINTS ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Pearson Specter Litt Legal RAG",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/rate-limit")
async def get_rate_limit_status(request: Request):
    """Get current rate limit status for the client"""
    client_ip = rate_limiter.get_client_ip(request)

    # Clean old requests
    rate_limiter.requests[client_ip] = rate_limiter._clean_old_requests(
        rate_limiter.requests[client_ip], rate_limiter.window
    )
    rate_limiter.ai_requests[client_ip] = rate_limiter._clean_old_requests(
        rate_limiter.ai_requests[client_ip], rate_limiter.window
    )

    return {
        "general": {
            "limit": rate_limiter.general_limit,
            "used": len(rate_limiter.requests[client_ip]),
            "remaining": rate_limiter.general_limit - len(rate_limiter.requests[client_ip]),
            "window_seconds": rate_limiter.window
        },
        "ai_queries": {
            "limit": rate_limiter.ai_limit,
            "used": len(rate_limiter.ai_requests[client_ip]),
            "remaining": rate_limiter.ai_limit - len(rate_limiter.ai_requests[client_ip]),
            "window_seconds": rate_limiter.window
        },
        "chat_messages": {
            "limit": rate_limiter.CHAT_LIMIT,
            "used": rate_limiter.chat_usage[client_ip],
            "remaining": rate_limiter.get_chat_remaining(client_ip)
        }
    }

@app.get("/api/chat-limit")
async def get_chat_limit_status(request: Request):
    """Get remaining chat messages for the client"""
    client_ip = rate_limiter.get_client_ip(request)
    used = rate_limiter.chat_usage[client_ip]
    remaining = rate_limiter.get_chat_remaining(client_ip)

    return {
        "limit": rate_limiter.CHAT_LIMIT,
        "used": used,
        "remaining": remaining,
        "exhausted": remaining <= 0
    }

@app.get("/api/status")
async def get_status() -> SystemStatus:
    """Get system status"""
    try:
        connected = vector_store is not None and vector_store.collection is not None
        total_chunks = vector_store.get_document_count() if connected else 0
        unique_files = len(vector_store.get_unique_filenames()) if connected else 0

        return SystemStatus(
            status="ready" if connected else "disconnected",
            vector_store_connected=connected,
            total_documents=unique_files,
            total_chunks=total_chunks,
            llm_ready=llm_client is not None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _extract_text_from_image(file_path: str) -> tuple:
    """Returns (annotated_text, ocr_meta) using the global pdf_processor."""
    if pdf_processor is None:
        raise ValueError("PDF processor not initialised")
    return pdf_processor.extract_with_confidence(file_path)


def _extract_text_from_doc(file_path: str) -> tuple:
    """Returns (text, ocr_meta). Word docs have no OCR uncertainty."""
    try:
        from docx import Document
    except ImportError:
        raise ValueError("python-docx not installed. Run: pip install python-docx")
    try:
        doc = Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        return text, {"avg_confidence": 1.0, "uncertain_spans": []}
    except Exception as e:
        raise ValueError(
            f"Failed to read Word document: {e}. "
            "If this is an old .doc file, convert it to .docx first."
        )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a document (PDF, image, or Word)"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Save file
        file_path = DATA_DIR / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        print(f"Processing: {file.filename}")

        # Extract text with OCR confidence metadata
        try:
            if ext == '.pdf':
                text, ocr_meta = pdf_processor.extract_with_confidence(str(file_path))
            elif ext in {'.png', '.jpg', '.jpeg'}:
                text, ocr_meta = _extract_text_from_image(str(file_path))
            else:  # .doc / .docx
                text, ocr_meta = _extract_text_from_doc(str(file_path))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no extractable text")

        avg_conf = ocr_meta.get("avg_confidence", 1.0)
        uncertain_count = len(ocr_meta.get("uncertain_spans", []))
        if avg_conf < 0.70:
            print(f"WARNING: low OCR confidence ({avg_conf:.0%}) — {uncertain_count} uncertain spans")

        # Extract metadata
        doc_metadata = metadata_extractor.extract_all_metadata(text, file.filename)

        # Chunk text
        chunks = pdf_processor.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE)
        print(f"Created {len(chunks)} chunks | OCR confidence: {avg_conf:.0%}")

        # Prepare documents for batch insert
        documents = []
        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                doc_id = f"{file.filename}_{idx}_{uuid.uuid4().hex[:8]}"
                documents.append({
                    'id': doc_id,
                    'text': chunk,
                    'metadata': {
                        'filename': file.filename,
                        'filepath': str(file_path),
                        'source': 'User Upload',
                        'year': str(datetime.now().year),
                        'chunk_index': str(idx),
                        'case_name': doc_metadata.get('case_name', ''),
                        'case_number': doc_metadata.get('case_number', ''),
                        'court': doc_metadata.get('court', ''),
                        'judges': doc_metadata.get('judges', []),
                        'judgment_date': doc_metadata.get('judgment_date', ''),
                        'citations': doc_metadata.get('citations', []),
                        'subject_matter': doc_metadata.get('subject_matter', []),
                        'ocr_confidence': avg_conf,
                        'has_uncertain_spans': uncertain_count > 0,
                    }
                })

        # Add to vector store
        chunks_added = vector_store.add_documents_batch(documents)

        return UploadResponse(
            filename=file.filename,
            status="success",
            message=f"Processed {file.filename}",
            chunks_added=chunks_added
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, req: Request):
    """Query the document collection"""
    try:
        # Rate limiting - use AI limit if requesting AI answer
        client_ip = rate_limiter.get_client_ip(req)
        allowed, info = rate_limiter.check_rate_limit(client_ip, is_ai_request=request.use_ai_answer)
        if not allowed:
            raise HTTPException(status_code=429, detail=info)

        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Search: retrieve broad candidates, then rerank
        raw_results = vector_store.search(request.question, limit=20)
        results = (
            reranker.rerank(request.question, raw_results, top_k=request.num_results)
            if reranker else raw_results[:request.num_results]
        )

        # Format results
        formatted_results = []
        source_citations = []

        for idx, result in enumerate(results):
            formatted_result = {
                "text": result.get('text', ''),
                "filename": result.get('filename', ''),
                "source": result.get('source', ''),
                "year": result.get('year', ''),
                "chunk_index": int(result.get('chunk_index', 0)),
                "relevance_score": 1 - float(result.get('distance', 0)),
                "case_name": result.get('case_name', ''),
                "case_number": result.get('case_number', ''),
                "court": result.get('court', ''),
                "judges": result.get('judges', '').split(', ') if result.get('judges') else [],
                "judgment_date": result.get('judgment_date', ''),
                "citations": result.get('citations', '').split(', ') if result.get('citations') else [],
                "subject_matter": result.get('subject_matter', '').split(', ') if result.get('subject_matter') else [],
            }
            formatted_results.append(formatted_result)

            source_citations.append({
                "id": idx + 1,
                "filename": result.get('filename', ''),
                "filepath": result.get('filepath', ''),
                "chunk_index": int(result.get('chunk_index', 0)),
                "case_name": result.get('case_name', ''),
                "relevance_score": formatted_result["relevance_score"],
                "source_location": f"{result.get('filename', '')}:chunk_{result.get('chunk_index', 0)}"
            })

        # Generate AI answer if requested
        ai_answer = None
        if request.use_ai_answer and llm_client and formatted_results:
            ai_answer = generate_answer(request.question, formatted_results)

        return QueryResponse(
            question=request.question,
            results=formatted_results,
            ai_answer=ai_answer,
            total_results=len(formatted_results),
            source_citations=source_citations
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

def generate_answer(question: str, results: List[dict]) -> str:
    """Generate AI answer using GLM-5.1"""
    try:
        context_parts = []
        for idx, r in enumerate(results[:3], 1):
            source_location = f"{r.get('filename', 'Unknown')}:chunk_{r.get('chunk_index', 0)}"
            context_parts.append(f"[Source {idx}: {source_location}]\n{r['text']}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a legal document assistant. Your ONLY knowledge source is the excerpts below.

STRICT RULES:
- Answer using ONLY information present in the excerpts. Do NOT use outside knowledge.
- If the excerpts do not contain enough information to answer, say exactly: "The uploaded documents do not contain sufficient information to answer this question."
- Cite the source tag (e.g. [Source 1]) for every factual claim you make.
- Never guess, infer beyond the text, or fill gaps from general legal knowledge.

Document excerpts:
{context}

Question: {question}

Answer (citing sources):"""

        response = llm_client.chat.completions.create(
            model="glm-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )

        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message, "reasoning_content", "") or "No answer generated."
        return content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Multi-turn chat endpoint"""
    try:
        # Rate limiting - always AI request
        client_ip = rate_limiter.get_client_ip(req)
        allowed, info = rate_limiter.check_rate_limit(client_ip, is_ai_request=True)
        if not allowed:
            raise HTTPException(status_code=429, detail=info)

        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get last user message
        last_message = None
        for msg in reversed(request.messages):
            if msg.role.lower() == "user":
                last_message = msg.content
                break

        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Search: retrieve broad candidates, then rerank
        raw_results = vector_store.search(last_message, limit=20)
        results = (
            reranker.rerank(last_message, raw_results, top_k=request.num_results)
            if reranker else raw_results[:request.num_results]
        )

        # Format results
        formatted_results = []
        sources = []

        for idx, result in enumerate(results):
            formatted_result = {
                "text": result.get('text', ''),
                "filename": result.get('filename', ''),
                "chunk_index": int(result.get('chunk_index', 0)),
                "relevance_score": 1 - float(result.get('distance', 0)),
            }
            formatted_results.append(formatted_result)

            sources.append({
                "id": idx + 1,
                "filename": result.get('filename', ''),
                "chunk_index": int(result.get('chunk_index', 0)),
                "relevance_score": formatted_result["relevance_score"],
                "source_location": f"{result.get('filename', '')}:chunk_{result.get('chunk_index', 0)}"
            })

        # Generate response
        ai_response = None
        if llm_client and formatted_results:
            ai_response = generate_chat_answer(request.messages, formatted_results)

        chat_history = [ChatMessage(role=msg.role, content=msg.content) for msg in request.messages]
        if ai_response:
            chat_history.append(ChatMessage(role="assistant", content=ai_response))

        return ChatResponse(
            response=ai_response or "No response generated",
            sources=sources,
            chat_history=chat_history
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def generate_chat_answer(messages: List[ChatMessage], results: List[dict]) -> str:
    """Generate chat answer using GLM-5.1"""
    try:
        context_parts = []
        for idx, r in enumerate(results[:3], 1):
            context_parts.append(f"[Source {idx}]\n{r['text']}")
        context = "\n\n".join(context_parts)

        system_prompt = f"""You are a legal document assistant. Your ONLY knowledge source is the excerpts below.

STRICT RULES:
- Answer using ONLY information present in the excerpts. Do NOT use outside knowledge.
- If the excerpts do not contain enough information, say: "The uploaded documents do not contain sufficient information to answer this question."
- Cite the source tag (e.g. [Source 1]) for every factual claim.
- Never guess, infer beyond the text, or fill gaps from general legal knowledge.

Document excerpts:
{context}"""

        openai_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = "user" if msg.role.lower() == "user" else "assistant"
            openai_messages.append({"role": role, "content": msg.content})

        response = llm_client.chat.completions.create(
            model="glm-5.1",
            messages=openai_messages,
            temperature=0.3,
            max_tokens=2000
        )

        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message, "reasoning_content", "") or "No answer generated."
        return content
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/api/documents")
async def list_documents():
    """List uploaded documents"""
    try:
        documents = []
        if DATA_DIR.exists():
            for f in DATA_DIR.iterdir():
                if f.suffix.lower() in ALLOWED_EXTENSIONS:
                    documents.append({
                        "filename": f.name,
                        "size_bytes": f.stat().st_size,
                        "uploaded_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    try:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete from vector store first
        vector_deleted = False
        if vector_store:
            vector_deleted = vector_store.delete_by_filename(filename)
            if not vector_deleted:
                print(f"Warning: No chunks found in vector store for {filename}")

        # Delete file
        file_path.unlink()

        message = f"Deleted {filename}"
        if not vector_deleted:
            message += " (no chunks were in vector store)"

        return {"status": "success", "message": message}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== SESSION & QA ENDPOINTS ==============

# In-memory session storage (for HF Spaces simplicity)
chat_sessions = {}

@app.post("/api/qa", response_model=QAResponse)
async def question_answer(request: QARequest, req: Request):
    """Question-Answer endpoint with session management"""
    try:
        client_ip = rate_limiter.get_client_ip(req)

        # Check HARD chat limit first (10 total messages per IP)
        chat_allowed, chat_info = rate_limiter.check_chat_limit(client_ip)
        if not chat_allowed:
            raise HTTPException(status_code=429, detail=chat_info["message"])

        # Then check rate limiting
        allowed, info = rate_limiter.check_rate_limit(client_ip, is_ai_request=True)
        if not allowed:
            raise HTTPException(status_code=429, detail=info)

        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")

        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Create or get session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "session_id": session_id,
                "title": "New Chat",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": [],
                "metadata": {"question_count": 0}
            }

        session = chat_sessions[session_id]
        session["updated_at"] = datetime.now().isoformat()
        session["metadata"]["question_count"] = session["metadata"].get("question_count", 0) + 1

        # Add user message
        session["messages"].append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.now().isoformat()
        })

        # Search: retrieve broad candidates, then rerank
        raw_results = vector_store.search(request.question, limit=20)
        results = (
            reranker.rerank(request.question, raw_results, top_k=request.num_results)
            if reranker else raw_results[:request.num_results]
        )

        # Format sources
        sources = []
        for idx, result in enumerate(results):
            sources.append({
                "id": idx + 1,
                "filename": result.get('filename', ''),
                "chunk_index": int(result.get('chunk_index', 0)),
                "case_name": result.get('case_name', ''),
                "relevance_score": 1 - float(result.get('distance', 0)),
                "source_location": f"{result.get('filename', '')}:chunk_{result.get('chunk_index', 0)}"
            })

        # Generate AI answer
        ai_answer = "Unable to generate answer from available documents"
        confidence = 0.0

        if llm_client and results:
            examples = feedback_store.get_relevant_examples(request.question) if feedback_store else []
            ai_answer = generate_qa_answer(request.question, results, examples=examples)
            confidence = sum(s['relevance_score'] for s in sources) / len(sources) if sources else 0.0

        # Add assistant response
        session["messages"].append({
            "role": "assistant",
            "content": ai_answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })

        # Update session title based on first question
        if session["title"] == "New Chat" and request.question:
            session["title"] = request.question[:50] + ("..." if len(request.question) > 50 else "")

        # Increment chat usage AFTER successful response
        rate_limiter.increment_chat_usage(client_ip)
        remaining = rate_limiter.get_chat_remaining(client_ip)

        return QAResponse(
            session_id=session_id,
            question=request.question,
            answer=ai_answer,
            sources=sources,
            confidence=round(confidence, 2),
            messages_remaining=remaining
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

def _ocr_confidence_label(conf: float) -> str:
    if conf >= 0.90:
        return "OCR quality: HIGH"
    if conf >= 0.70:
        return "OCR quality: ACCEPTABLE"
    if conf >= 0.50:
        return "OCR quality: LOW — treat uncertain spans with caution"
    return "OCR quality: VERY LOW — treat all extracted text as potentially unreliable"


def generate_qa_answer(question: str, results: List[dict], examples: List[dict] = None) -> str:
    """Generate a confidence-aware Q&A answer using GLM-5.1."""
    try:
        context_parts = []
        for idx, r in enumerate(results[:3], 1):
            case_name = r.get('case_name', '')
            case_info = f" ({case_name})" if case_name else ""
            ocr_conf = float(r.get('ocr_confidence', 1.0))
            conf_label = _ocr_confidence_label(ocr_conf)
            context_parts.append(
                f"[{idx}]{case_info} [{conf_label}]\n{r.get('text', '')}"
            )

        context = "\n\n".join(context_parts)

        # Build few-shot prefix from operator-improved past drafts
        few_shot = ""
        if examples:
            few_shot = "OPERATOR-IMPROVED EXAMPLES — use these as a quality and style reference:\n\n"
            for i, ex in enumerate(examples, 1):
                few_shot += (
                    f"Example {i}:\n"
                    f"Question: {ex['question']}\n"
                    f"Operator-approved draft:\n{ex['edited_draft'][:600]}\n\n"
                )
            few_shot += "---\n\n"

        prompt = f"""{few_shot}You are a legal document assistant. Your ONLY knowledge source is the excerpts below.

OCR UNCERTAINTY RULES — read carefully:
- Excerpts are labeled with their OCR quality (HIGH / ACCEPTABLE / LOW / VERY LOW).
- Spans marked [UNCERTAIN: '...' — OCR confidence X%] were poorly recognised — do NOT treat them as confirmed fact.
- Spans marked [UNREADABLE: ...] could not be extracted — do NOT speculate about their content.
- For LOW or VERY LOW quality excerpts, hedge every claim: use "the document appears to...", "it seems...", "portions of the text suggest...".
- Never reconstruct or guess content that is marked uncertain or unreadable.

STRICT RULES:
- Answer using ONLY information present in the excerpts. Do NOT use outside knowledge.
- For HIGH/ACCEPTABLE excerpts: cite directly (e.g. "As stated in [1]...").
- For LOW/VERY LOW excerpts: acknowledge the uncertainty explicitly before citing.
- If excerpts do not contain sufficient information, say exactly: "The uploaded documents do not contain sufficient information to answer this question."
- Never infer beyond the text or fill gaps from general legal knowledge.
- Be concise and precise — aim for 2–4 paragraphs.

Legal Document Excerpts:
{context}

Question: {question}

Answer (grounded in excerpts only):"""

        response = llm_client.chat.completions.create(
            model="glm-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )

        content = response.choices[0].message.content
        if not content:
            content = getattr(response.choices[0].message, "reasoning_content", "") or "No answer generated."
        return content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def _parse_draft_sections(content: str) -> List[DraftSection]:
    """
    Parse the LLM JSON response into DraftSection objects.
    Falls back gracefully if the model returns malformed output.
    """
    SECTION_KEYS = {
        "case_overview":       "Case Overview",
        "key_facts":           "Key Facts",
        "legal_issues":        "Legal Issues",
        "supporting_evidence": "Supporting Evidence",
        "conclusion":          "Conclusion",
    }

    # Strip markdown code fences the model sometimes adds
    clean = _re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=_re.MULTILINE)
    clean = _re.sub(r'\s*```$', '', clean, flags=_re.MULTILINE)

    parsed = None
    # Attempt 1: direct parse
    try:
        parsed = _json.loads(clean)
    except Exception:
        pass

    # Attempt 2: find the first {...} block
    if parsed is None:
        m = _re.search(r'\{[\s\S]*\}', clean)
        if m:
            try:
                parsed = _json.loads(m.group())
            except Exception:
                pass

    if parsed:
        sections = []
        for key, title in SECTION_KEYS.items():
            sections.append(DraftSection(
                title=title,
                content=parsed.get(key, "Insufficient information in the provided documents.").strip()
            ))
        return sections

    # Last resort: return raw text under Case Overview
    return [DraftSection(title="Case Overview", content=content.strip())]


def generate_case_summary(
    question: str,
    results: List[dict],
    examples: List[dict] = None
) -> List[DraftSection]:
    """
    Generate a structured case fact summary grounded in retrieved chunks.
    Sections: Case Overview, Key Facts, Legal Issues, Supporting Evidence, Conclusion.
    """
    context_parts = []
    for idx, r in enumerate(results[:5], 1):
        case_name = r.get("case_name", "")
        label = f"[{idx}]" + (f" {case_name}" if case_name else "")
        context_parts.append(f"{label}\n{r.get('text', '')}")
    context = "\n\n".join(context_parts)

    # Few-shot prefix from operator-approved past drafts
    few_shot = ""
    if examples:
        few_shot = "OPERATOR-APPROVED EXAMPLES — match this quality and citation style:\n\n"
        for i, ex in enumerate(examples, 1):
            few_shot += (
                f"Example {i}:\n"
                f"Topic: {ex['question']}\n"
                f"Approved draft excerpt:\n{ex['edited_draft'][:500]}\n\n"
            )
        few_shot += "---\n\n"

    prompt = f"""{few_shot}You are a legal analyst at Pearson Specter Litt.

Using ONLY the document excerpts below, produce a structured case fact summary.

STRICT RULES:
- Use ONLY information present in the excerpts. Do NOT use outside legal knowledge.
- Cite excerpt numbers [1], [2], [3], etc. for every factual claim.
- If a section cannot be supported by the excerpts, write exactly:
  "Insufficient information in the provided documents."
- Respond with ONLY a valid JSON object — no markdown, no explanation, no text outside the JSON.

Document Excerpts:
{context}

Topic: {question}

Output this exact JSON structure:
{{
  "case_overview": "2-3 sentences: what is this case or document about, who are the parties, which court, what date. Cite sources.",
  "key_facts": "Numbered list of the most important facts established in the documents. Each fact must cite a source [n].",
  "legal_issues": "The legal questions or issues raised. Cite sources [n].",
  "supporting_evidence": "Key passages or provisions quoted or paraphrased from the excerpts, with citations [n].",
  "conclusion": "How the matter was decided or resolved, or state explicitly if the excerpts do not contain a conclusion."
}}"""

    response = llm_client.chat.completions.create(
        model="glm-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2500
    )

    content = response.choices[0].message.content
    if not content:
        content = getattr(response.choices[0].message, "reasoning_content", "") or ""

    return _parse_draft_sections(content)


@app.post("/api/draft/case-summary", response_model=CaseSummaryResponse)
async def draft_case_summary(request: CaseSummaryRequest, req: Request):
    """
    Generate a structured case fact summary draft grounded in uploaded documents.

    Sections returned:
      - Case Overview
      - Key Facts
      - Legal Issues
      - Supporting Evidence
      - Conclusion

    Each section cites the source excerpt(s) that support it.
    Submit the operator-edited version to POST /api/feedback to improve future drafts.
    """
    try:
        client_ip = rate_limiter.get_client_ip(req)
        allowed, info = rate_limiter.check_rate_limit(client_ip, is_ai_request=True)
        if not allowed:
            raise HTTPException(status_code=429, detail=info)

        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        if not llm_client:
            raise HTTPException(status_code=503, detail="LLM not initialized — check OPENCODE_API_KEY")
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question/topic cannot be empty")

        # Retrieve: broad candidates → rerank to top num_results
        raw_results = vector_store.search(request.question, limit=20)
        results = (
            reranker.rerank(request.question, raw_results, top_k=request.num_results)
            if reranker else raw_results[:request.num_results]
        )

        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found. Upload documents first.")

        # Format sources for response
        sources = [
            {
                "id": idx + 1,
                "filename": r.get("filename", ""),
                "chunk_index": int(r.get("chunk_index", 0)),
                "case_name": r.get("case_name", ""),
                "relevance_score": round(1 - float(r.get("distance", 0)), 3),
                "source_location": f"{r.get('filename', '')}:chunk_{r.get('chunk_index', 0)}",
            }
            for idx, r in enumerate(results)
        ]

        # Few-shot from operator feedback store
        examples = feedback_store.get_relevant_examples(request.question) if feedback_store else []

        # Generate structured draft
        sections = generate_case_summary(request.question, results, examples=examples)

        return CaseSummaryResponse(
            draft_id=str(uuid.uuid4()),
            question=request.question,
            sections=sections,
            sources=sources,
            generated_at=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft generation failed: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Operator submits an edited draft.
    The store saves the (question, original, edited) triple and extracts
    a reusable signal. Future similar questions will receive this as a
    few-shot example, gradually improving draft quality.
    """
    if not feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")

    if request.original_draft == request.edited_draft:
        return {"status": "skipped", "reason": "No changes detected between original and edited draft"}

    entry = feedback_store.add_feedback(
        question=request.question,
        original_draft=request.original_draft,
        edited_draft=request.edited_draft,
        sources=request.sources,
        session_id=request.session_id,
    )
    return {
        "status": "saved",
        "feedback_id": entry["id"],
        "total_examples": feedback_store.count(),
        "edit_delta": entry["edit_delta"],
    }


@app.get("/api/feedback")
async def list_feedback():
    """List all stored operator edits (for inspection / audit)."""
    if not feedback_store:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    return {
        "total": feedback_store.count(),
        "examples": feedback_store.list_all(),
    }


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session"""
    try:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        chat_sessions[session_id] = {
            "session_id": session_id,
            "title": request.title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "metadata": {"question_count": 0}
        }

        return SessionResponse(
            session_id=session_id,
            title=request.title,
            created_at=now,
            updated_at=now,
            message_count=0,
            question_count=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all chat sessions"""
    try:
        sessions = []
        for sid, session in chat_sessions.items():
            sessions.append({
                "session_id": session["session_id"],
                "title": session["title"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "message_count": len(session["messages"]),
                "question_count": session["metadata"].get("question_count", 0)
            })

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)

        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.get("/api/sessions/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Get conversation history for a specific session"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = chat_sessions[session_id]

        return SessionHistoryResponse(
            session_id=session["session_id"],
            title=session["title"],
            messages=session["messages"],
            metadata=session["metadata"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        del chat_sessions[session_id]

        return {"status": "success", "message": f"Session {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


# ============== STATIC FILE SERVING ==============

# Mount static files if build directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_root():
    """Serve React app"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Pearson Specter Litt Legal RAG API", "docs": "/docs"}

@app.get("/{full_path:path}")
async def serve_spa(request: Request, full_path: str):
    """Serve React SPA for client-side routing"""
    # Don't serve API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    # Try to serve static file
    static_file = STATIC_DIR / full_path
    if static_file.exists() and static_file.is_file():
        return FileResponse(str(static_file))

    # Fall back to index.html for SPA routing
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))

    return {"message": "Pearson Specter Litt Legal RAG API", "docs": "/docs"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
