"""
PDF Processing with OCR fallback + Token-aware Semantic Chunking.

Extraction strategy:
  1. pypdf  — fast, exact for text-layer PDFs (digital / born-digital).
  2. OCR    — fallback via PyMuPDF page rendering + EasyOCR when pypdf
              returns too few words (scanned pages, image-only PDFs,
              handwritten notes, low-resolution scans).

Chunking strategy:
  Split-then-merge with a three-tier boundary hierarchy:
    legal section markers  →  paragraph breaks  →  sentences
  Chunks are sized by token count (tiktoken cl100k_base), not characters,
  so every chunk fits consistently inside the embedding model, reranker,
  and LLM context window.
"""

import re
import os
import io
from typing import List, Dict, Optional
from pypdf import PdfReader
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except ImportError:
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# ── Chunking parameters ────────────────────────────────────────────────────────
TOKEN_LIMIT    = 400
OVERLAP_TOKENS = 80
MIN_TOKENS     = 40

_LEGAL_BOUNDARY = re.compile(
    r'(?m)^(?:'
    r'\d+[\.\)]\s'
    r'|\([a-zA-Z]\)\s'
    r'|Article\s+\d'
    r'|Section\s+\d'
    r'|WHEREAS'
    r'|SCHEDULE'
    r'|ANNEXURE'
    r'|ORDER'
    r'|JUDGMENT'
    r')'
)

# Words below this count in pypdf output → likely a scanned/image PDF
_SPARSE_WORD_THRESHOLD = 100

# ── OCR confidence thresholds ─────────────────────────────────────────────────
OCR_CONF_RELIABLE   = 0.90   # use as-is
OCR_CONF_ACCEPTABLE = 0.70   # acceptable — include verbatim
OCR_CONF_UNCERTAIN  = 0.50   # uncertain  — annotate
# below 0.50 = unreliable  — mark as unreadable

_EMPTY_OCR_META: dict = {"avg_confidence": 0.0, "uncertain_spans": []}
_NATIVE_PDF_META: dict = {"avg_confidence": 1.0, "uncertain_spans": []}


def _annotate_span(text: str, confidence: float) -> str:
    """Replace low-confidence OCR spans with inline uncertainty annotations."""
    if confidence >= OCR_CONF_ACCEPTABLE:
        return text
    if confidence >= OCR_CONF_UNCERTAIN:
        preview = text.strip()[:60]
        return f"[UNCERTAIN: '{preview}' — OCR confidence {confidence:.0%}]"
    return "[UNREADABLE: text not reliably extractable due to low OCR confidence]"


def _repair_ocr_text(text: str) -> str:
    """
    Lightweight normalization of common OCR artifacts.
    Collapses stray whitespace and removes control characters.
    Raw original text is preserved separately upstream.
    """
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


class PDFProcessor:
    """Handles PDF text extraction (with OCR fallback) and semantic chunking."""

    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        self._ocr_reader = None   # lazy-initialised on first OCR call

    # ── Extraction ─────────────────────────────────────────────────────────────

    def _pypdf_extract(self, pdf_path: str) -> str:
        """Text-layer extraction for digital PDFs."""
        try:
            reader = PdfReader(pdf_path)
            pages = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
            return "\n".join(pages)
        except Exception as e:
            print(f"pypdf error ({Path(pdf_path).name}): {e}")
            return ""

    def _ensure_ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            print("Initialising EasyOCR (first run downloads ~300 MB)…")
            self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("EasyOCR ready")

    def _ocr_extract_with_confidence(self, source_path: str) -> tuple:
        """
        OCR a PDF or image file with per-span confidence scoring.

        Accepts PDF (pages rendered via PyMuPDF) or any image file.
        Returns (annotated_text, ocr_meta) where:
          ocr_meta = {
            "avg_confidence": float,          — 0.0–1.0
            "uncertain_spans": [              — spans below OCR_CONF_UNCERTAIN
              {"text": str, "confidence": float}, ...
            ]
          }
        Low-confidence spans are replaced inline with [UNCERTAIN:...] /
        [UNREADABLE:...] annotations so downstream LLMs see the uncertainty.
        """
        try:
            import numpy as np
            from PIL import Image, ImageEnhance
            self._ensure_ocr_reader()
        except ImportError as e:
            print(f"OCR dependencies missing: {e}")
            return "", dict(_EMPTY_OCR_META)

        try:
            ext = Path(source_path).suffix.lower()
            images = []

            if ext == '.pdf':
                import fitz
                doc = fitz.open(source_path)
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    images.append(("page", Image.open(io.BytesIO(pix.tobytes("png")))))
                doc.close()
            else:
                images.append(("image", Image.open(source_path)))

            pages = []
            all_confidences: List[float] = []
            uncertain_spans: List[dict] = []

            for page_num, (kind, img) in enumerate(images):
                img = img.convert("L")
                img = ImageEnhance.Contrast(img).enhance(1.5)

                # detail=1 → [(bbox, text, confidence), ...]
                results = self._ocr_reader.readtext(
                    img if isinstance(img, type(img)) else img,
                    detail=1,
                    paragraph=False,
                )

                page_parts = []
                for _, text, conf in results:
                    text = text.strip()
                    if not text:
                        continue
                    all_confidences.append(conf)
                    if conf < OCR_CONF_UNCERTAIN:
                        uncertain_spans.append({
                            "text": text[:120],
                            "confidence": round(conf, 3),
                        })
                    page_parts.append(_annotate_span(text, conf))

                page_text = " ".join(page_parts)
                if page_text.strip():
                    prefix = f"\n--- Page {page_num + 1} (OCR) ---\n" if kind == "page" else ""
                    pages.append(f"{prefix}{page_text}")

            avg_conf = (
                sum(all_confidences) / len(all_confidences)
                if all_confidences else 0.0
            )
            return "\n".join(pages), {
                "avg_confidence": round(avg_conf, 3),
                "uncertain_spans": uncertain_spans[:20],
            }

        except Exception as e:
            print(f"OCR error ({Path(source_path).name}): {e}")
            return "", dict(_EMPTY_OCR_META)

    def _ocr_extract(self, pdf_path: str) -> str:
        """Backward-compatible wrapper — returns plain text only."""
        text, _ = self._ocr_extract_with_confidence(pdf_path)
        return text

    def extract_with_confidence(self, file_path: str) -> tuple:
        """
        Public API: extract text with OCR confidence metadata.

        Returns (text, ocr_meta):
          - Native-text PDFs  → avg_confidence 1.0, no uncertain spans
          - Scanned PDFs      → OCR text annotated with [UNCERTAIN/UNREADABLE]
                                spans + per-document avg_confidence
          - Image files       → same as scanned PDFs
          - DOCX/DOC          → delegated to caller (_extract_text_from_doc)

        Confidence metadata is stored in vector store chunk metadata and used
        to penalise low-quality chunks during retrieval.
        """
        ext = Path(file_path).suffix.lower()

        # Non-PDF images are OCR-only
        if ext in {'.png', '.jpg', '.jpeg'}:
            text, meta = self._ocr_extract_with_confidence(file_path)
            return _repair_ocr_text(text), meta

        # PDF: try native text layer first
        text = self._pypdf_extract(file_path)
        word_count = len(text.split())

        if word_count >= _SPARSE_WORD_THRESHOLD:
            return text, dict(_NATIVE_PDF_META)

        # Sparse → confidence-aware OCR
        name = Path(file_path).name
        print(f"'{name}' sparse ({word_count} words) — confidence-aware OCR…")
        ocr_text, ocr_meta = self._ocr_extract_with_confidence(file_path)
        ocr_words = len(ocr_text.split())

        if ocr_words > word_count:
            print(f"OCR succeeded: {ocr_words} words | avg confidence {ocr_meta['avg_confidence']:.0%}")
            return _repair_ocr_text(ocr_text), ocr_meta

        print(f"OCR did not improve extraction — using pypdf output")
        return text, dict(_NATIVE_PDF_META)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF.

        Tries pypdf first. If the result is too sparse (< 100 words —
        typical of a scanned or image-only page), falls back to OCR.
        Uses whichever path returned more content.
        """
        text = self._pypdf_extract(pdf_path)
        word_count = len(text.split())

        if word_count < _SPARSE_WORD_THRESHOLD:
            name = Path(pdf_path).name
            print(f"'{name}' text sparse ({word_count} words) — attempting OCR…")
            ocr_text = self._ocr_extract(pdf_path)
            ocr_words = len(ocr_text.split())
            if ocr_words > word_count:
                print(f"OCR succeeded: {ocr_words} words extracted")
                return ocr_text
            print(f"OCR did not improve extraction ({ocr_words} words) — using pypdf output")

        return text

    def process_all_pdfs(self) -> List[Dict[str, str]]:
        documents = []
        if not os.path.exists(self.pdf_directory):
            print(f"Directory not found: {self.pdf_directory}")
            return documents

        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith(".pdf")]
        print(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            print(f"Processing: {pdf_file}")
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                documents.append({
                    "filename": pdf_file,
                    "filepath": pdf_path,
                    "text": text,
                    "source": "PSL 2015",
                    "year": "2015",
                })
        return documents

    # ── Semantic unit splitting ────────────────────────────────────────────────

    def _split_into_units(self, text: str) -> List[str]:
        segments = _LEGAL_BOUNDARY.split(text)
        headers  = _LEGAL_BOUNDARY.findall(text)

        reassembled = [segments[0]] if segments[0].strip() else []
        for header, segment in zip(headers, segments[1:]):
            reassembled.append(header + segment)

        paragraphs = []
        for seg in reassembled:
            parts = re.split(r'\n{2,}', seg)
            paragraphs.extend(p.strip() for p in parts if p.strip())

        units = []
        for para in paragraphs:
            if _count_tokens(para) <= TOKEN_LIMIT:
                units.append(para)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                units.extend(s.strip() for s in sentences if s.strip())
        return units

    # ── Chunk assembly ─────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1500,   # kept for API compat; token constants take precedence
        overlap: int = 300,
        min_chunk_size: int = 200,
    ) -> List[str]:
        """Token-aware semantic chunker."""
        units = self._split_into_units(text)
        if not units:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = _count_tokens(unit)

            # Unit too big on its own — hard-split by words
            if unit_tokens > TOKEN_LIMIT:
                words = unit.split()
                sub: List[str] = []
                sub_tokens = 0
                for word in words:
                    wt = _count_tokens(word + " ")
                    if sub_tokens + wt > TOKEN_LIMIT and sub:
                        chunk_str = " ".join(sub).strip()
                        if _count_tokens(chunk_str) >= MIN_TOKENS:
                            chunks.append(chunk_str)
                        sub = sub[-int(OVERLAP_TOKENS / 3):]
                        sub_tokens = _count_tokens(" ".join(sub))
                    sub.append(word)
                    sub_tokens += wt
                if sub:
                    leftover = " ".join(sub).strip()
                    if leftover:
                        current.append(leftover)
                        current_tokens += _count_tokens(leftover)
                continue

            if current_tokens + unit_tokens > TOKEN_LIMIT and current:
                chunk_str = "\n\n".join(current).strip()
                if _count_tokens(chunk_str) >= MIN_TOKENS:
                    chunks.append(chunk_str)

                # Overlap: carry tail units into next chunk
                overlap_units: List[str] = []
                overlap_total = 0
                for u in reversed(current):
                    ut = _count_tokens(u)
                    if overlap_total + ut > OVERLAP_TOKENS:
                        break
                    overlap_units.insert(0, u)
                    overlap_total += ut
                current = overlap_units
                current_tokens = overlap_total

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunk_str = "\n\n".join(current).strip()
            if _count_tokens(chunk_str) >= MIN_TOKENS:
                chunks.append(chunk_str)

        return chunks
