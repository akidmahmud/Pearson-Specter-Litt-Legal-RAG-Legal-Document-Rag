"""
Weaviate Manager for handling vector database operations
"""
import sys
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Optional
import rag_config as config


class WeaviateManager:
    """Manages Weaviate database operations for legal documents"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.use_openai = config.USE_OPENAI_EMBEDDINGS and config.OPENAI_API_KEY

        # Initialize embedding model
        if self.use_openai:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.embedding_model = None
            print(f"Using OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
            self.openai_client = None
            print(f"Using local embeddings: {config.LOCAL_EMBEDDING_MODEL}")

    def connect(self):
        """Connect to Weaviate instance"""
        try:
            # Connect to local Weaviate instance
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080
            )
            print("Connected to Weaviate successfully")
            return True
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            print("Make sure Weaviate is running. You can start it with Docker:")
            print("docker-compose up -d (from the PSL_RAG directory)")
            return False

    def create_schema(self):
        """Create the schema for legal documents"""
        try:
            # Delete collection if it exists
            if self.client.collections.exists(config.COLLECTION_NAME):
                self.client.collections.delete(config.COLLECTION_NAME)
                print(f"Deleted existing collection: {config.COLLECTION_NAME}")

            # Create collection with properties including metadata
            self.collection = self.client.collections.create(
                name=config.COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own vectors
                properties=[
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="The text content of the document chunk"
                    ),
                    Property(
                        name="filename",
                        data_type=DataType.TEXT,
                        description="Name of the PDF file"
                    ),
                    Property(
                        name="filepath",
                        data_type=DataType.TEXT,
                        description="Full path to the PDF file"
                    ),
                    Property(
                        name="source",
                        data_type=DataType.TEXT,
                        description="Source of the document (e.g., SCOB 2015)"
                    ),
                    Property(
                        name="year",
                        data_type=DataType.TEXT,
                        description="Year of the law report"
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Index of the chunk within the document"
                    ),
                    # Metadata fields for legal cases
                    Property(
                        name="case_name",
                        data_type=DataType.TEXT,
                        description="Name of the legal case (parties involved)"
                    ),
                    Property(
                        name="case_number",
                        data_type=DataType.TEXT,
                        description="Case/Appeal number"
                    ),
                    Property(
                        name="court",
                        data_type=DataType.TEXT,
                        description="Court information"
                    ),
                    Property(
                        name="judges",
                        data_type=DataType.TEXT_ARRAY,
                        description="Names of judges"
                    ),
                    Property(
                        name="judgment_date",
                        data_type=DataType.TEXT,
                        description="Date of judgment"
                    ),
                    Property(
                        name="citations",
                        data_type=DataType.TEXT_ARRAY,
                        description="Legal citations"
                    ),
                    Property(
                        name="subject_matter",
                        data_type=DataType.TEXT_ARRAY,
                        description="Legal topics/subject matter"
                    ),
                ]
            )
            print(f"Created collection: {config.COLLECTION_NAME}")
            return True
        except Exception as e:
            print(f"Failed to create schema: {e}")
            return False

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI or sentence-transformers"""
        try:
            if self.use_openai:
                # Use OpenAI embeddings
                response = self.openai_client.embeddings.create(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    input=text
                )
                return response.data[0].embedding
            else:
                # Use local sentence-transformers
                embedding = self.embedding_model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
        except Exception as e:
            print(f"\n  ERROR generating embedding: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, any]], chunk_size: int = 1500,
                      chunk_overlap: int = 300):
        """
        Add documents to Weaviate with chunking, embeddings, and metadata extraction

        Args:
            documents: List of document dictionaries
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not self.collection:
            self.collection = self.client.collections.get(config.COLLECTION_NAME)

        total_chunks = 0

        # Import required modules
        from pdf_processor import PDFProcessor
        from metadata_extractor import LegalMetadataExtractor

        processor = PDFProcessor("")
        metadata_extractor = LegalMetadataExtractor()

        for doc in documents:
            print(f"Processing document: {doc['filename']}")

            # Extract metadata from the full document text
            doc_metadata = metadata_extractor.extract_all_metadata(doc['text'], doc['filename'])
            print(f"  Extracted metadata: {metadata_extractor.format_metadata_for_display(doc_metadata)}")

            chunks = processor.chunk_text(doc['text'], chunk_size, chunk_overlap, config.MIN_CHUNK_SIZE)

            # Batch process embeddings for efficiency
            print(f"  Processing {len(chunks)} chunks in batches...")

            batch_size = 100  # Process 100 chunks at a time
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                print(f"    Batch {batch_start//batch_size + 1}: Processing chunks {batch_start+1}-{batch_end}/{len(chunks)}")

                # Process batch
                for idx_in_batch, chunk in enumerate(batch_chunks):
                    if not chunk.strip():
                        continue

                    actual_idx = batch_start + idx_in_batch

                    # Generate embedding
                    vector = self.generate_embedding(chunk)

                    # Create data object with metadata
                    data_object = {
                        "text": chunk,
                        "filename": doc['filename'],
                        "filepath": doc['filepath'],
                        "source": doc['source'],
                        "year": doc['year'],
                        "chunk_index": actual_idx,
                        # Add extracted metadata
                        "case_name": doc_metadata.get('case_name') or "",
                        "case_number": doc_metadata.get('case_number') or "",
                        "court": doc_metadata.get('court') or "",
                        "judges": doc_metadata.get('judges') or [],
                        "judgment_date": doc_metadata.get('judgment_date') or "",
                        "citations": doc_metadata.get('citations') or [],
                        "subject_matter": doc_metadata.get('subject_matter') or [],
                    }

                    # Insert into Weaviate
                    self.collection.data.insert(
                        properties=data_object,
                        vector=vector
                    )

                    total_chunks += 1

            print(f"  ✓ Completed: Added {len(chunks)} chunks from {doc['filename']}")

        print(f"\nTotal chunks added to Weaviate: {total_chunks}")

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant documents using vector similarity

        Args:
            query: Search query
            limit: Number of results to return

        Returns:
            List of relevant document chunks with metadata
        """
        if not self.collection:
            self.collection = self.client.collections.get(config.COLLECTION_NAME)

        # Generate query embedding
        query_vector = self.generate_embedding(query)

        # Perform vector search
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for obj in response.objects:
            results.append({
                "text": obj.properties["text"],
                "filename": obj.properties["filename"],
                "filepath": obj.properties.get("filepath", ""),
                "source": obj.properties["source"],
                "year": obj.properties["year"],
                "chunk_index": obj.properties["chunk_index"],
                "distance": obj.metadata.distance,
                # Include metadata in results
                "case_name": obj.properties.get("case_name", ""),
                "case_number": obj.properties.get("case_number", ""),
                "court": obj.properties.get("court", ""),
                "judges": obj.properties.get("judges", []),
                "judgment_date": obj.properties.get("judgment_date", ""),
                "citations": obj.properties.get("citations", []),
                "subject_matter": obj.properties.get("subject_matter", []),
            })

        return results

    def delete_by_filename(self, filename: str) -> bool:
        """
        Delete all chunks/documents with a specific filename from Weaviate

        Args:
            filename: The filename to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use where filter to find and delete all objects with matching filename
            where = {
                "path": ["filename"],
                "operator": "Equal",
                "valueString": filename
            }

            # Delete all objects matching the filename
            result = self.collection.data.delete_many(
                where=where
            )

            print(f"Deleted {result.number_deleted} chunks for file: {filename}")
            return True
        except Exception as e:
            print(f"Error deleting chunks for {filename}: {e}")
            return False

    def get_all_filenames(self) -> List[str]:
        """
        Get all unique filenames currently in Weaviate

        Returns:
            List of filenames in the database
        """
        try:
            response = self.collection.query.fetch_objects(
                limit=10000,
                return_metadata=MetadataQuery(count=True)
            )

            filenames = set()
            for obj in response.objects:
                filename = obj.properties.get("filename")
                if filename:
                    filenames.add(filename)

            return sorted(list(filenames))
        except Exception as e:
            print(f"Error fetching filenames from Weaviate: {e}")
            return []

    def cleanup_orphaned_chunks(self, valid_filenames: List[str]) -> Dict:
        """
        Remove chunks from Weaviate that don't exist in the provided list of valid filenames

        Args:
            valid_filenames: List of filenames that should exist in the database

        Returns:
            Dict with cleanup statistics
        """
        try:
            # Get all filenames in Weaviate
            weaviate_filenames = self.get_all_filenames()

            # Find orphaned filenames
            orphaned = [f for f in weaviate_filenames if f not in valid_filenames]

            deleted_count = 0
            for filename in orphaned:
                where = {
                    "path": ["filename"],
                    "operator": "Equal",
                    "valueString": filename
                }
                result = self.collection.data.delete_many(where=where)
                deleted_count += result.number_deleted
                print(f"Cleaned up {result.number_deleted} chunks for orphaned file: {filename}")

            return {
                "orphaned_files": orphaned,
                "chunks_deleted": deleted_count,
                "status": "success"
            }
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return {
                "orphaned_files": [],
                "chunks_deleted": 0,
                "status": "error",
                "error": str(e)
            }

    def close(self):
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
            print("Weaviate connection closed")
