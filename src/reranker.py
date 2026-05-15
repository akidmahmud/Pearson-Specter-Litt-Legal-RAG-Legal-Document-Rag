"""
Cross-encoder reranker using BAAI/bge-reranker-base.
Retrieves broad candidates from hybrid search, then deeply scores
each (query, chunk) pair to surface the most relevant results.
"""

from typing import List, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"Loading reranker: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        print("Reranker ready")

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Score every (query, chunk) pair and return the top_k by score."""
        if not results:
            return results

        pairs = [(query, r.get("text", "")) for r in results]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
