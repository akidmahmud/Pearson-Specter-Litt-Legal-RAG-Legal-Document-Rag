"""
ChromaDB Vector Store Manager with Hybrid Retrieval (Dense + BM25)
"""

import os
import re
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class VectorStoreManager:
    """Manages ChromaDB vector store with hybrid dense + BM25 retrieval"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "legal_documents"
        self.client = None
        self.collection = None
        self.local_model = None

        # BM25 state
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_doc_ids: List[str] = []
        self.bm25_doc_texts: Dict[str, Dict] = {}  # id -> {text, metadata}

        self._init_embedding_model()
        self.connect()
        self._rebuild_bm25_index()

    def _init_embedding_model(self):
        print("Initializing local embeddings (all-MiniLM-L6-v2)...")
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Local embedding model ready")

    def connect(self) -> bool:
        try:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            print(f"ChromaDB connected. Collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Failed to connect to ChromaDB: {e}")
            return False

    # ── Embedding ──────────────────────────────────────────────────────────────

    def generate_embedding(self, text: str) -> List[float]:
        if not self.local_model:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.local_model.encode(text).tolist()

    # ── BM25 ───────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize for BM25 — keeps legal refs like 'article', '102', 'section'"""
        tokens = re.findall(r'\b[\w.-]+\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from all documents currently in ChromaDB"""
        try:
            if not self.collection or self.collection.count() == 0:
                self.bm25_index = None
                self.bm25_doc_ids = []
                self.bm25_doc_texts = {}
                return

            results = self.collection.get(include=["documents", "metadatas"])
            if not results or not results['ids']:
                return

            self.bm25_doc_ids = results['ids']
            self.bm25_doc_texts = {}
            tokenized_corpus = []

            for i, doc_id in enumerate(results['ids']):
                text = results['documents'][i] if results['documents'] else ""
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                self.bm25_doc_texts[doc_id] = {'text': text, 'metadata': metadata}
                tokenized_corpus.append(self._tokenize(text))

            self.bm25_index = BM25Okapi(tokenized_corpus)
            print(f"BM25 index built: {len(self.bm25_doc_ids)} chunks")
        except Exception as e:
            print(f"BM25 index build error: {e}")
            self.bm25_index = None

    def _bm25_search(self, query: str, limit: int) -> List[Dict]:
        """Keyword search via BM25"""
        if not self.bm25_index or not self.bm25_doc_ids:
            return []
        try:
            tokens = self._tokenize(query)
            if not tokens:
                return []

            scores = self.bm25_index.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

            results = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                doc_id = self.bm25_doc_ids[idx]
                doc_data = self.bm25_doc_texts.get(doc_id, {})
                result = {
                    'id': doc_id,
                    'text': doc_data.get('text', ''),
                    'distance': 0.0,  # placeholder; RRF uses rank not score
                }
                result.update(doc_data.get('metadata', {}))
                results.append(result)

            return results
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

    def _dense_search(self, query: str, limit: int) -> List[Dict]:
        """Dense vector search via ChromaDB"""
        try:
            total = self.collection.count()
            if total == 0:
                return []

            query_embedding = self.generate_embedding(query)
            n = min(limit, total)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"]
            )

            formatted = []
            if results and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    result = {
                        'id': doc_id,
                        'text': results['documents'][0][i] if results['documents'] else "",
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    }
                    if results['metadatas'] and results['metadatas'][0]:
                        result.update(results['metadatas'][0][i])
                    formatted.append(result)

            return formatted
        except Exception as e:
            print(f"Dense search error: {e}")
            return []

    @staticmethod
    def _rrf_merge(dense: List[Dict], bm25: List[Dict], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion: score = Σ 1/(k + rank)"""
        scores: Dict[str, float] = {}
        pool: Dict[str, Dict] = {}

        for rank, result in enumerate(dense):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            pool[doc_id] = result

        for rank, result in enumerate(bm25):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in pool:
                pool[doc_id] = result

        return [pool[doc_id] for doc_id in sorted(scores, key=lambda x: scores[x], reverse=True)]

    # ── Public search ──────────────────────────────────────────────────────────

    @staticmethod
    def _confidence_rerank(results: List[Dict]) -> List[Dict]:
        """
        Re-rank results by  semantic_similarity × ocr_confidence.

        Chunks from low-quality OCR documents are penalised so that
        corrupted evidence is not injected into LLM context ahead of
        clean text.  Documents without an ocr_confidence field (e.g.
        native-text PDFs) default to 1.0 — no penalty.

        Adds 'confidence_adjusted_score' to each result dict.
        """
        for r in results:
            ocr_conf = float(r.get('ocr_confidence', 1.0))
            distance  = float(r.get('distance', 0.0))
            similarity = max(0.0, 1.0 - distance)
            r['confidence_adjusted_score'] = round(similarity * ocr_conf, 4)
        return sorted(
            results,
            key=lambda x: x.get('confidence_adjusted_score', 0.0),
            reverse=True,
        )

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Hybrid search: dense + BM25 → RRF → OCR-confidence re-rank"""
        try:
            candidates = limit * 3

            dense_results = self._dense_search(query, candidates)
            bm25_results  = self._bm25_search(query, candidates)

            if not bm25_results:
                merged = dense_results
            elif not dense_results:
                merged = bm25_results
            else:
                merged = self._rrf_merge(dense_results, bm25_results)

            # Penalise chunks from low-confidence OCR documents
            merged = self._confidence_rerank(merged)
            return merged[:limit]
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

    # ── Write operations ───────────────────────────────────────────────────────

    def add_document(self, doc_id: str, text: str, metadata: Dict) -> bool:
        try:
            embedding = self.generate_embedding(text)
            clean_metadata = self._clean_metadata(metadata)

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[clean_metadata]
            )
            self._rebuild_bm25_index()
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False

    def add_documents_batch(self, documents: List[Dict]) -> int:
        added = 0
        ids, embeddings, texts, metadatas = [], [], [], []

        for doc in documents:
            try:
                embedding = self.generate_embedding(doc['text'])
                ids.append(doc['id'])
                embeddings.append(embedding)
                texts.append(doc['text'])
                metadatas.append(self._clean_metadata(doc.get('metadata', {})))
                added += 1
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {e}")

        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                self._rebuild_bm25_index()
            except Exception as e:
                print(f"Error in batch add: {e}")
                added = 0

        return added

    def delete_by_filename(self, filename: str) -> bool:
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=[]
            )
            if results and results['ids']:
                count = len(results['ids'])
                self.collection.delete(where={"filename": filename})
                self._rebuild_bm25_index()
                print(f"Deleted {count} chunks for {filename}")
                return True

            print(f"No chunks found for {filename}")
            return False
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_metadata(metadata: Dict) -> Dict:
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                clean[key] = ", ".join(str(v) for v in value) if value else ""
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        return clean

    def get_document_count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0

    def get_unique_filenames(self) -> List[str]:
        try:
            results = self.collection.get(include=["metadatas"])
            filenames = set()
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if 'filename' in metadata:
                        filenames.add(metadata['filename'])
            return list(filenames)
        except:
            return []

    def close(self):
        pass

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._rebuild_bm25_index()
            print("Collection reset successfully")
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False
