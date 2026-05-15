"""
Operator Feedback Store.

Captures operator edits (original draft → improved draft), stores them
persistently, and surfaces the most relevant past examples as few-shot
context when generating future drafts.

Improvement loop:
  1.  System generates a draft.
  2.  Operator edits it and submits via POST /api/feedback.
  3.  Store saves the (question, original, edited) triple.
  4.  On the next similar question, BM25 finds the closest past edit.
  5.  That edit is injected into the prompt as a quality example.
  6.  The model implicitly learns the operator's preferred style and precision.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r'\b[\w.-]+\b', text.lower())
    return [t for t in tokens if len(t) > 1]


class FeedbackStore:
    MAX_EXAMPLES = 200   # cap to avoid unbounded growth
    MIN_BM25_SCORE = 0.5  # minimum relevance to inject as few-shot

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.examples: List[Dict] = []
        self._load()

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_feedback(
        self,
        question: str,
        original_draft: str,
        edited_draft: str,
        sources: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Store one operator edit."""
        entry = {
            "id": str(uuid.uuid4()),
            "session_id": session_id or "",
            "question": question,
            "original_draft": original_draft,
            "edited_draft": edited_draft,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat(),
            "edit_delta": self._summarise_delta(original_draft, edited_draft),
        }
        self.examples.append(entry)
        if len(self.examples) > self.MAX_EXAMPLES:
            self.examples = self.examples[-self.MAX_EXAMPLES:]
        self._save()
        return entry

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_relevant_examples(self, question: str, top_k: int = 2) -> List[Dict]:
        """
        Return the top_k most relevant past edited drafts for this question.
        Uses BM25 over (question + edited_draft) so both topical and stylistic
        signals are considered.
        """
        if not self.examples:
            return []

        corpus = [
            _tokenize(ex["question"] + " " + ex["edited_draft"])
            for ex in self.examples
        ]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(question))

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        return [
            self.examples[i]
            for i, score in ranked[:top_k]
            if score >= self.MIN_BM25_SCORE
        ]

    def list_all(self) -> List[Dict]:
        return [
            {
                "id": ex["id"],
                "session_id": ex.get("session_id", ""),
                "question_preview": ex["question"][:120],
                "timestamp": ex["timestamp"],
                "edit_delta": ex.get("edit_delta", {}),
            }
            for ex in reversed(self.examples)  # newest first
        ]

    def count(self) -> int:
        return len(self.examples)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _summarise_delta(original: str, edited: str) -> Dict:
        """
        Extract a lightweight reusable signal from the edit:
        how many characters were added/removed and rough sentence counts.
        This metadata is stored alongside the example and surfaced in
        GET /api/feedback so operators can see which edits were substantive.
        """
        added = max(0, len(edited) - len(original))
        removed = max(0, len(original) - len(edited))
        orig_sentences = len(re.findall(r'[.!?]', original))
        edit_sentences = len(re.findall(r'[.!?]', edited))
        return {
            "chars_added": added,
            "chars_removed": removed,
            "sentence_delta": edit_sentences - orig_sentences,
        }

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.examples = json.load(f)
                print(f"Feedback store loaded: {len(self.examples)} examples")
            except Exception as e:
                print(f"Feedback store load error: {e}")
                self.examples = []

    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.examples, f, indent=2, ensure_ascii=False)
